from __future__ import annotations

from datetime import UTC, datetime

import pytest

from app.code_indexing.models import CodeIndexArtifact, CodeIndexRecord
from app.code_indexing.reuse_verification import verify_embedding_reuse
from app.code_ingestion.snapshot_models import (
    CodeFileManifestEntry,
    CodeSnapshotManifest,
    SnapshotDiff,
    SnapshotRequest,
)
from app.code_ingestion.plsql_models import SourceMap


def _record(*, source_path: str, cache_key: str, status: str) -> CodeIndexRecord:
    return CodeIndexRecord(
        unit_id=f"unit-{source_path}", point_id=f"point-{source_path}", unit_index=0,
            snapshot_id="fci-custom-r3-new", module_id="fci-custom", source_path=source_path,
            source_kind="routine", display_name="spTest", source_map=SourceMap(
            source_path=source_path, start_offset=0, end_offset=10, start_line=1, end_line=1
            ), parser_state="full_parse", conditional_state="known", citation_text="BEGIN; END;",
        embedding_text="BEGIN; END;", content_sha256="a" * 64, cache_key=cache_key,
        embedding_model="text-embedding-3-large", embedding_status=status,
        vector=(1.0, 2.0),
    )


def _artifact(snapshot_id: str, records: tuple[CodeIndexRecord, ...]) -> CodeIndexArtifact:
    return CodeIndexArtifact(
        status="embedded", snapshot_id=snapshot_id, snapshot_content_sha256="b" * 64,
        parse_generation="v15", analysis_policy_sha256="c" * 64,
        module_id="fci-custom", embedding_model="text-embedding-3-large",
        vector_dimension=2, total_records=len(records), artifact_identity_sha256="d" * 64,
        records=records,
    )


def _snapshot() -> CodeSnapshotManifest:
    files = (
        CodeFileManifestEntry(path="old.sql", extension=".sql", source_handler="text", sha256="1" * 64,
                              normalized_text_sha256="1" * 64, size_bytes=1, encoding="utf-8", line_count=1),
        CodeFileManifestEntry(path="new.sql", extension=".sql", source_handler="text", sha256="2" * 64,
                              normalized_text_sha256="2" * 64, size_bytes=1, encoding="utf-8", line_count=1),
    )
    return CodeSnapshotManifest(
        snapshot_id="fci-custom-r3-new", snapshot_content_sha256="e" * 64,
        created_at_utc=datetime.now(UTC), ingestion_policy_schema_version="v1",
        ingestion_policy_sha256="f" * 64,
        request=SnapshotRequest(module_set="fci-custom", svn_revision="3", application_build="x", reviewer="SME", base_snapshot_id="fci-custom-r2-base"),
        files=files,
        diff=SnapshotDiff(base_snapshot_id="fci-custom-r2-base", added=("new.sql",), unchanged=("old.sql",)),
    )


def test_reuse_verification_allows_new_embeddings_only_on_changed_paths() -> None:
    base = _artifact("fci-custom-r2-base", (_record(source_path="old.sql", cache_key="a" * 64, status="embedded"),))
    candidate = _artifact("fci-custom-r3-new", (
        _record(source_path="old.sql", cache_key="a" * 64, status="cached"),
        _record(source_path="new.sql", cache_key="b" * 64, status="embedded"),
    ))

    report = verify_embedding_reuse(snapshot=_snapshot(), base_artifact=base, embedded_artifact=candidate)

    assert report["status"] == "pass"
    assert report["cached_records"] == 1
    assert report["embedded_records"] == 1


def test_reuse_verification_rejects_cache_miss_on_unchanged_source() -> None:
    base = _artifact("fci-custom-r2-base", (_record(source_path="old.sql", cache_key="a" * 64, status="embedded"),))
    candidate = _artifact("fci-custom-r3-new", (
        _record(source_path="old.sql", cache_key="c" * 64, status="embedded"),
        _record(source_path="new.sql", cache_key="b" * 64, status="embedded"),
    ))

    with pytest.raises(RuntimeError, match="Unchanged source paths"):
        verify_embedding_reuse(snapshot=_snapshot(), base_artifact=base, embedded_artifact=candidate)
