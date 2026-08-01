from pathlib import Path

import pytest

from app.embeddings.embedding_artifact_writer import write_embedding_batch_to_json
from app.embeddings.embedding_contract import EmbeddingBatch, EmbeddingRecord
from app.vectorstore.qdrant_schema import (
    QdrantCollectionConfig,
    create_local_qdrant_client,
    ensure_collection,
)
from app.vectorstore.qdrant_upsert import upsert_embedding_batch
from scripts.check_qdrant_index import verify_embedding_artifacts


def _record() -> EmbeddingRecord:
    return EmbeddingRecord(
        unit_id="FS_ASNB_R25::paragraph_1",
        unit_index=1,
        source_kind="paragraph",
        document_family="FS_ASNB",
        release_label="R25",
        content_hash="content-hash-1",
        artifact_version="v1",
        cache_key="cache-key-1",
        text="Verified teller-report evidence.",
        embedding_model="text-embedding-3-large",
        embedding_status="embedded",
        vector=[1.0, 0.0],
    )


def _artifact(tmp_path: Path) -> tuple[Path, EmbeddingBatch]:
    batch = EmbeddingBatch(
        document_name="FS_ASNB_R25.docx",
        total_records=1,
        records=[_record()],
    )
    artifact_path = write_embedding_batch_to_json(batch, tmp_path / "embeddings")
    return artifact_path, batch


def test_verify_embedding_artifacts_confirms_exact_qdrant_payload(tmp_path: Path) -> None:
    artifact_path, batch = _artifact(tmp_path)
    client = create_local_qdrant_client()
    config = QdrantCollectionConfig(collection_name="functional_specs", vector_size=2)
    ensure_collection(client, config)
    upsert_embedding_batch(client, config.collection_name, batch)

    verified = verify_embedding_artifacts(
        client=client,
        collection_name=config.collection_name,
        artifact_paths=[artifact_path],
    )

    assert verified == 1


def test_verify_embedding_artifacts_rejects_missing_qdrant_point(tmp_path: Path) -> None:
    artifact_path, _ = _artifact(tmp_path)
    client = create_local_qdrant_client()
    config = QdrantCollectionConfig(collection_name="functional_specs", vector_size=2)
    ensure_collection(client, config)

    with pytest.raises(RuntimeError, match="missing expected embedding points"):
        verify_embedding_artifacts(
            client=client,
            collection_name=config.collection_name,
            artifact_paths=[artifact_path],
        )
