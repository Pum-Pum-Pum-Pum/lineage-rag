from __future__ import annotations

from app.embeddings.embedding_contract import EmbeddingBatch


SUPPORTED_SOURCE_KINDS = {"paragraph", "table"}


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


def filter_embedding_batch_by_source_kind(
    batch: EmbeddingBatch,
    source_kind: str | None,
) -> EmbeddingBatch:
    """Filter an embedding batch by source kind before smoke-test limiting."""

    if source_kind is None:
        return batch

    if source_kind not in SUPPORTED_SOURCE_KINDS:
        raise ValueError(
            "source_kind must be one of: "
            f"{sorted(SUPPORTED_SOURCE_KINDS)}"
        )

    filtered_records = [
        record for record in batch.records if record.source_kind == source_kind
    ]

    return EmbeddingBatch(
        document_name=batch.document_name,
        total_records=len(filtered_records),
        records=filtered_records,
    )
