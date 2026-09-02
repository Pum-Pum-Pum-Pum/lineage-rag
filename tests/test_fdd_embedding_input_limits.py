from __future__ import annotations

from app.embeddings.client import embed_batch
from app.embeddings.embedding_contract import EmbeddingBatch, EmbeddingRecord, validate_embedding_batch_inputs
from app.ingestion.embedded_workbook_chunker import _bounded_row_groups
from app.ingestion.embedding_input_limits import (
    MAX_EMBEDDING_INPUT_BYTES,
    MAX_SOURCE_UNIT_BYTES,
    split_text_by_utf8_bytes,
    utf8_byte_length,
)
from app.ingestion.embedded_workbook_extractor import ExtractedWorkbookCell, ExtractedWorkbookRow


class _NoCallEmbeddings:
    def __init__(self) -> None:
        self.calls = 0

    def create(self, **_kwargs):
        self.calls += 1
        raise AssertionError("provider must not be called for an oversized input")


class _NoCallClient:
    def __init__(self) -> None:
        self.embeddings = _NoCallEmbeddings()


def _record(text: str) -> EmbeddingRecord:
    return EmbeddingRecord(
        unit_id="fixture::oversized",
        unit_index=0,
        source_kind="embedded_workbook",
        document_family="FS_ASNB",
        release_label="R4",
        content_hash="hash",
        artifact_version="v1",
        cache_key="cache-key",
        text=text,
        embedding_model="text-embedding-3-large",
        embedding_status="pending",
    )


def test_utf8_split_is_lossless_and_each_piece_is_bounded() -> None:
    text = ("field=value \U0001f642\n" * 2_000).strip()

    pieces = split_text_by_utf8_bytes(text, MAX_SOURCE_UNIT_BYTES)

    assert "".join(pieces) == text
    assert all(utf8_byte_length(piece) <= MAX_SOURCE_UNIT_BYTES for piece in pieces)


def test_single_oversized_workbook_row_is_split_with_same_row_provenance() -> None:
    row = ExtractedWorkbookRow(
        row_number=7,
        cells=(ExtractedWorkbookCell(reference="A7", value="x" * (MAX_SOURCE_UNIT_BYTES * 2)),),
    )

    groups = _bounded_row_groups((row,), rows_per_unit=25)

    assert len(groups) > 1
    assert all(start == end == 7 for start, end, _ in groups)
    assert all(utf8_byte_length(text) <= MAX_SOURCE_UNIT_BYTES for _, _, text in groups)


def test_embedding_preflight_rejects_oversized_input_before_provider_call() -> None:
    batch = EmbeddingBatch(
        document_name="fixture.docx",
        total_records=1,
        records=[_record("x" * (MAX_EMBEDDING_INPUT_BYTES + 1))],
    )
    client = _NoCallClient()

    try:
        validate_embedding_batch_inputs(batch)
    except ValueError as error:
        assert "fixture::oversized" in str(error)
    else:
        raise AssertionError("Expected exact-input preflight to fail")

    try:
        embed_batch(batch, client=client)
    except ValueError:
        pass
    else:
        raise AssertionError("Expected embed_batch to fail before the provider call")
    assert client.embeddings.calls == 0
