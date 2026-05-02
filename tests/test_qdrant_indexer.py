from app.embeddings.embedding_artifact_writer import write_embedding_batch_to_json
from app.embeddings.embedding_contract import EmbeddingBatch, EmbeddingRecord
from app.vectorstore.qdrant_indexer import (
    build_embedding_batch_from_cache,
    index_embedding_cache_directory,
)
from app.vectorstore.qdrant_schema import QdrantCollectionConfig, create_local_qdrant_client


def _record(unit_id: str, vector: list[float]) -> EmbeddingRecord:
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
        embedding_status="embedded",
        vector=vector,
    )


def test_build_embedding_batch_from_cache(tmp_path) -> None:
    cache_dir = tmp_path / "embeddings"
    batch = EmbeddingBatch(
        document_name="example.docx",
        total_records=2,
        records=[
            _record("unit-2", [0.0, 1.0]),
            _record("unit-1", [1.0, 0.0]),
        ],
    )
    write_embedding_batch_to_json(batch, cache_dir)

    loaded_batch = build_embedding_batch_from_cache(cache_dir)

    assert loaded_batch.document_name == "embedding_cache"
    assert loaded_batch.total_records == 2
    assert [record.cache_key for record in loaded_batch.records] == ["cache-unit-1", "cache-unit-2"]


def test_index_embedding_cache_directory_indexes_cached_vectors(tmp_path) -> None:
    cache_dir = tmp_path / "embeddings"
    batch = EmbeddingBatch(
        document_name="example.docx",
        total_records=2,
        records=[
            _record("unit-1", [1.0, 0.0]),
            _record("unit-2", [0.0, 1.0]),
        ],
    )
    write_embedding_batch_to_json(batch, cache_dir)

    client = create_local_qdrant_client()
    config = QdrantCollectionConfig(
        collection_name="test_functional_specs",
        vector_size=2,
    )

    summary = index_embedding_cache_directory(client, config, cache_dir)

    assert summary.attempted_records == 2
    assert summary.upserted_points == 2
    assert summary.skipped_records == 0
    assert client.count(config.collection_name).count == 2
