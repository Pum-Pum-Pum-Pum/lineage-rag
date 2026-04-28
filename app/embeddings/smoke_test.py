from __future__ import annotations

from app.embeddings.embedding_contract import EmbeddingBatch


def limit_embedding_batch(batch: EmbeddingBatch, limit: int) -> EmbeddingBatch:
    """Return a small embedding batch for a controlled real-API smoke test."""

    if limit <= 0:
        raise ValueError("Smoke-test limit must be greater than 0")

    limited_records = batch.records[:limit]
    return EmbeddingBatch(
        document_name=batch.document_name,
        total_records=len(limited_records),
        records=limited_records,
    )
