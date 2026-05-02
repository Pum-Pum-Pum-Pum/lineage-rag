from app.embeddings.embedding_contract import EmbeddingBatch, EmbeddingRecord
from app.vectorstore.qdrant_schema import (
    QdrantCollectionConfig,
    create_local_qdrant_client,
    ensure_collection,
)
from app.vectorstore.qdrant_search import build_metadata_filter, search_vectors
from app.vectorstore.qdrant_upsert import upsert_embedding_batch


def _record(
    unit_id: str,
    vector: list[float],
    text: str,
    document_family: str = "FS_FCIS_14.7.0.0.0$ASNB",
    release_label: str = "R24",
    source_kind: str = "paragraph",
) -> EmbeddingRecord:
    return EmbeddingRecord(
        unit_id=unit_id,
        unit_index=0,
        source_kind=source_kind,
        document_family=document_family,
        release_label=release_label,
        content_hash=f"hash-{unit_id}",
        artifact_version="v1",
        cache_key=f"cache-{unit_id}",
        text=text,
        embedding_model="text-embedding-3-large",
        embedding_status="embedded",
        vector=vector,
    )


def test_search_vectors_returns_expected_nearest_payload() -> None:
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
            _record("unit-close", [1.0, 0.0], "Close match"),
            _record("unit-far", [0.0, 1.0], "Far match"),
        ],
    )
    upsert_embedding_batch(client, config.collection_name, batch)

    results = search_vectors(
        client=client,
        collection_name=config.collection_name,
        query_vector=[1.0, 0.0],
        limit=1,
    )

    assert len(results) == 1
    assert results[0].payload["unit_id"] == "unit-close"
    assert results[0].payload["text"] == "Close match"
    assert results[0].score > 0.99


def test_search_vectors_filters_by_document_family_release_and_source_kind() -> None:
    client = create_local_qdrant_client()
    config = QdrantCollectionConfig(
        collection_name="test_functional_specs",
        vector_size=2,
    )
    ensure_collection(client, config)

    batch = EmbeddingBatch(
        document_name="example.docx",
        total_records=3,
        records=[
            _record(
                "unit-r24-paragraph",
                [1.0, 0.0],
                "R24 paragraph",
                document_family="FS_FCIS_14.7.0.0.0$ASNB",
                release_label="R24",
                source_kind="paragraph",
            ),
            _record(
                "unit-r24-table",
                [1.0, 0.0],
                "R24 table",
                document_family="FS_FCIS_14.7.0.0.0$ASNB",
                release_label="R24",
                source_kind="table",
            ),
            _record(
                "unit-r2-paragraph",
                [1.0, 0.0],
                "R2 paragraph",
                document_family="FS_FCIS_14.4.0.0.0$ASNB",
                release_label="R2",
                source_kind="paragraph",
            ),
        ],
    )
    upsert_embedding_batch(client, config.collection_name, batch)

    results = search_vectors(
        client=client,
        collection_name=config.collection_name,
        query_vector=[1.0, 0.0],
        limit=5,
        document_family="FS_FCIS_14.7.0.0.0$ASNB",
        release_label="R24",
        source_kind="table",
    )

    assert len(results) == 1
    assert results[0].payload["unit_id"] == "unit-r24-table"
    assert results[0].payload["source_kind"] == "table"


def test_search_vectors_rejects_invalid_inputs() -> None:
    client = create_local_qdrant_client()

    try:
        search_vectors(client, "missing", [], limit=1)
    except ValueError as exc:
        assert "must not be empty" in str(exc)
    else:
        raise AssertionError("Expected ValueError for empty query vector")

    try:
        search_vectors(client, "missing", [1.0, 0.0], limit=0)
    except ValueError as exc:
        assert "greater than 0" in str(exc)
    else:
        raise AssertionError("Expected ValueError for invalid limit")


def test_build_metadata_filter_returns_none_when_no_filters_are_provided() -> None:
    assert build_metadata_filter() is None
