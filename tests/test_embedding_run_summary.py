import json
from pathlib import Path

from app.embeddings.embedding_contract import EmbeddingBatch, EmbeddingRecord
from app.embeddings.embedding_run_summary import (
    build_embedding_run_summary,
    write_embedding_run_summary_to_json,
)


def _record(unit_id: str, model: str = "text-embedding-3-large") -> EmbeddingRecord:
    return EmbeddingRecord(
        unit_id=unit_id,
        unit_index=0,
        source_kind="paragraph",
        document_family="FS_FCIS_14.7.0.0.0$ASNB",
        release_label="R24",
        content_hash=f"hash-{unit_id}",
        artifact_version="v1",
        cache_key=f"cache-{unit_id}",
        text="Example text",
        embedding_model=model,
        embedding_status="embedded",
        vector=[0.1, 0.2],
    )


def test_build_embedding_run_summary() -> None:
    batch = EmbeddingBatch(
        document_name="FS_FCIS_14.7.0.0.0$ASNB_R24_Test.docx",
        total_records=4,
        cached_count=1,
        embedded_count=3,
        cache_miss_count=3,
        records=[_record("1"), _record("2"), _record("3"), _record("4")],
    )

    summary = build_embedding_run_summary(batch)

    assert summary.document_name == "FS_FCIS_14.7.0.0.0$ASNB_R24_Test.docx"
    assert summary.total_records == 4
    assert summary.cached_count == 1
    assert summary.embedded_count == 3
    assert summary.cache_miss_count == 3
    assert summary.cache_hit_rate == 0.25
    assert summary.embedding_models == ["text-embedding-3-large"]
    assert summary.artifact_versions == ["v1"]


def test_write_embedding_run_summary_to_json(tmp_path: Path) -> None:
    batch = EmbeddingBatch(
        document_name="FS_FCIS_14.7.0.0.0$ASNB_R24_Test.docx",
        total_records=2,
        cached_count=1,
        embedded_count=1,
        cache_miss_count=1,
        records=[_record("1"), _record("2")],
    )
    summary = build_embedding_run_summary(batch)

    output_file = write_embedding_run_summary_to_json(summary, tmp_path / "summaries")

    assert output_file.exists()
    assert output_file.name == "FS_FCIS_14.7.0.0.0$ASNB_R24_Test.embedding_summary.json"

    payload = json.loads(output_file.read_text(encoding="utf-8"))
    assert payload["total_records"] == 2
    assert payload["cached_count"] == 1
    assert payload["embedded_count"] == 1
    assert payload["cache_hit_rate"] == 0.5


def test_write_embedding_run_summary_to_json_supports_file_stem_suffix(tmp_path: Path) -> None:
    batch = EmbeddingBatch(
        document_name="example.docx",
        total_records=0,
        records=[],
    )
    summary = build_embedding_run_summary(batch)

    output_file = write_embedding_run_summary_to_json(
        summary,
        tmp_path / "summaries",
        file_stem_suffix=".table",
    )

    assert output_file.name == "example.table.embedding_summary.json"


def test_build_embedding_run_summary_empty_batch() -> None:
    batch = EmbeddingBatch(
        document_name="empty.docx",
        total_records=0,
        records=[],
    )

    summary = build_embedding_run_summary(batch)

    assert summary.cache_hit_rate == 0.0
    assert summary.embedding_models == []
    assert summary.artifact_versions == []
