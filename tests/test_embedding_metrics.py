from app.embeddings.embedding_contract import EmbeddingBatch
from app.embeddings.embedding_metrics import (
    calculate_cache_hit_rate,
    summarize_embedding_cache_metrics,
)


def test_calculate_cache_hit_rate() -> None:
    batch = EmbeddingBatch(
        document_name="example.docx",
        total_records=10,
        records=[],
        cached_count=7,
        embedded_count=3,
        cache_miss_count=3,
    )

    assert calculate_cache_hit_rate(batch) == 0.7


def test_calculate_cache_hit_rate_empty_batch() -> None:
    batch = EmbeddingBatch(
        document_name="empty.docx",
        total_records=0,
        records=[],
    )

    assert calculate_cache_hit_rate(batch) == 0.0


def test_summarize_embedding_cache_metrics() -> None:
    batch = EmbeddingBatch(
        document_name="example.docx",
        total_records=4,
        records=[],
        cached_count=1,
        embedded_count=3,
        cache_miss_count=3,
    )

    metrics = summarize_embedding_cache_metrics(batch)

    assert metrics.total_records == 4
    assert metrics.cached_count == 1
    assert metrics.embedded_count == 3
    assert metrics.cache_miss_count == 3
    assert metrics.cache_hit_rate == 0.25
