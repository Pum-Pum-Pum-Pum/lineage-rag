from __future__ import annotations

from pathlib import Path

from qdrant_client import QdrantClient

from app.embeddings.embedding_cache import load_embedding_records
from app.embeddings.embedding_contract import EmbeddingBatch
from app.vectorstore.qdrant_schema import QdrantCollectionConfig, ensure_collection
from app.vectorstore.qdrant_upsert import QdrantUpsertSummary, upsert_embedding_batch


def build_embedding_batch_from_cache(cache_directory: str | Path) -> EmbeddingBatch:
    """Build an index batch retaining every persisted citeable embedding record."""

    records = sorted(
        load_embedding_records(cache_directory),
        key=lambda record: (
            record.document_id,
            record.document_family,
            record.release_label,
            record.unit_id,
            record.cache_key,
        ),
    )

    return EmbeddingBatch(
        document_name="embedding_cache",
        total_records=len(records),
        records=records,
    )


def index_embedding_cache_directory(
    client: QdrantClient,
    collection_config: QdrantCollectionConfig,
    cache_directory: str | Path,
) -> QdrantUpsertSummary:
    """Index all persisted embedding records from a cache directory into Qdrant."""

    ensure_collection(client, collection_config)
    batch = build_embedding_batch_from_cache(cache_directory)
    return upsert_embedding_batch(client, collection_config.collection_name, batch)
