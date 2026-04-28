from __future__ import annotations

from dataclasses import dataclass

from app.embeddings.embedding_contract import EmbeddingBatch


@dataclass(frozen=True)
class EmbeddingCacheMetrics:
    total_records: int
    cached_count: int
    embedded_count: int
    cache_miss_count: int
    cache_hit_rate: float


def calculate_cache_hit_rate(batch: EmbeddingBatch) -> float:
    """Calculate cache hit rate safely for one embedding batch."""

    if batch.total_records == 0:
        return 0.0
    return batch.cached_count / batch.total_records


def summarize_embedding_cache_metrics(batch: EmbeddingBatch) -> EmbeddingCacheMetrics:
    """Summarize lightweight cache metrics for one embedding batch."""

    return EmbeddingCacheMetrics(
        total_records=batch.total_records,
        cached_count=batch.cached_count,
        embedded_count=batch.embedded_count,
        cache_miss_count=batch.cache_miss_count,
        cache_hit_rate=calculate_cache_hit_rate(batch),
    )
