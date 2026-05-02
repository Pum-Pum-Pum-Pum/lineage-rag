from app.embeddings.embedding_contract import EmbeddingBatch, EmbeddingRecord
from app.embeddings.smoke_test import filter_embedding_batch_by_source_kind, limit_embedding_batch


def _record(index: int, source_kind: str = "paragraph") -> EmbeddingRecord:
    return EmbeddingRecord(
        unit_id=f"unit-{index}",
        unit_index=index,
        source_kind=source_kind,
        document_family="FS_FCIS_14.7.0.0.0$ASNB",
        release_label="R24",
        content_hash=f"content-hash-{index}",
        artifact_version="v1",
        cache_key=f"cache-key-{index}",
        text=f"Text {index}",
        embedding_model="text-embedding-3-large",
        embedding_status="pending",
        vector=None,
    )


def test_limit_embedding_batch_keeps_only_requested_records() -> None:
    batch = EmbeddingBatch(
        document_name="example.docx",
        total_records=3,
        records=[_record(0), _record(1), _record(2)],
    )

    limited = limit_embedding_batch(batch, limit=2)

    assert limited.document_name == "example.docx"
    assert limited.total_records == 2
    assert [record.unit_id for record in limited.records] == ["unit-0", "unit-1"]


def test_limit_embedding_batch_rejects_invalid_limit() -> None:
    batch = EmbeddingBatch(
        document_name="example.docx",
        total_records=0,
        records=[],
    )

    try:
        limit_embedding_batch(batch, limit=0)
    except ValueError as exc:
        assert "greater than 0" in str(exc)
    else:
        raise AssertionError("Expected ValueError for invalid smoke-test limit")


def test_filter_embedding_batch_by_source_kind() -> None:
    batch = EmbeddingBatch(
        document_name="example.docx",
        total_records=3,
        records=[_record(0, "paragraph"), _record(1, "table"), _record(2, "table")],
    )

    filtered = filter_embedding_batch_by_source_kind(batch, source_kind="table")

    assert filtered.document_name == "example.docx"
    assert filtered.total_records == 2
    assert [record.source_kind for record in filtered.records] == ["table", "table"]


def test_filter_embedding_batch_by_source_kind_rejects_invalid_source_kind() -> None:
    batch = EmbeddingBatch(
        document_name="example.docx",
        total_records=0,
        records=[],
    )

    try:
        filter_embedding_batch_by_source_kind(batch, source_kind="image")
    except ValueError as exc:
        assert "source_kind must be one of" in str(exc)
    else:
        raise AssertionError("Expected ValueError for invalid source kind")
