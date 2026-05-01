import pytest

from app.embeddings.embedding_contract import EmbeddingBatch, EmbeddingRecord
from app.vectorstore.qdrant_schema import (
    QdrantCollectionConfig,
    create_local_qdrant_client,
    ensure_collection,
)
from app.vectorstore.qdrant_upsert import (
    build_qdrant_point_id,
    embedding_record_to_qdrant_point,
    upsert_embedding_batch,
)


def _record(unit_id: str, vector: list[float] | None) -> EmbeddingRecord:
    return EmbeddingRecord(
        unit_id=unit_id,
        unit_index=0,
        source_kind="paragraph",
        document_family="FS_FCIS_14.7.0.0.0$ASNB",
        release_label="R24",
        content_hash=f"hash-{unit_id}",
        artifact_version="v1",
        cache_key=f"cache-{unit_id}",
        text=f"Text for {unit_id}",
        embedding_model="text-embedding-3-large",
        embedding_status="embedded" if vector is not None else "pending",
        vector=vector,
    )


def test_build_qdrant_point_id_is_deterministic() -> None:
    record = _record("unit-1", [0.1, 0.2])

    assert build_qdrant_point_id(record) == build_qdrant_point_id(record)


def test_embedding_record_to_qdrant_point_preserves_payload() -> None:
    record = _record("unit-1", [0.1, 0.2])

    point = embedding_record_to_qdrant_point(record)

    assert point.vector == [0.1, 0.2]
    assert point.payload["unit_id"] == "unit-1"
    assert point.payload["document_family"] == "FS_FCIS_14.7.0.0.0$ASNB"
    assert point.payload["release_label"] == "R24"
    assert point.payload["text"] == "Text for unit-1"


def test_embedding_record_to_qdrant_point_rejects_missing_vector() -> None:
    record = _record("unit-1", None)

    try:
        embedding_record_to_qdrant_point(record)
    except ValueError as exc:
        assert "without vector" in str(exc)
    else:
        raise AssertionError("Expected ValueError for record without vector")


def test_upsert_embedding_batch_indexes_only_vector_records() -> None:
    client = create_local_qdrant_client()
    config = QdrantCollectionConfig(
        collection_name="test_functional_specs",
        vector_size=2,
    )
    ensure_collection(client, config)

    batch = EmbeddingBatch(
        document_name="example.docx",
        total_records=2,
        records=[
            _record("unit-1", [0.1, 0.2]),
            _record("unit-2", None),
        ],
    )

    summary = upsert_embedding_batch(client, config.collection_name, batch)

    assert summary.attempted_records == 2
    assert summary.upserted_points == 1
    assert summary.skipped_records == 1
    assert client.count(config.collection_name).count == 1

    points, _ = client.scroll(
        collection_name=config.collection_name,
        limit=10,
        with_payload=True,
        with_vectors=True,
    )
    assert len(points) == 1
    assert points[0].payload["unit_id"] == "unit-1"
    assert points[0].vector == pytest.approx([0.4472135955, 0.894427191])
