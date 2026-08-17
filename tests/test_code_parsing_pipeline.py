from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.code_ingestion.code_parsing_pipeline import (
    PARSER_GENERATION_DIRECTORY,
    parse_code_snapshot,
)
from app.code_ingestion.code_analysis_models import CodeStaticAnalysisArtifact
from app.code_ingestion.plsql_models import CodeRetrievalArtifact, PlSqlFileParseArtifact
from app.code_ingestion.plsql_models import CodeParseStageManifest
from app.code_ingestion.snapshot_builder import build_code_snapshot


def _build_snapshot(tmp_path: Path, source_text: str):
    return _build_snapshot_files(tmp_path, {"pkg_customer_custom.sql": source_text})


def _build_snapshot_files(tmp_path: Path, files: dict[str, str]):
    intake = tmp_path / "intake"
    source = intake / "source"
    source.mkdir(parents=True)
    for relative_path, source_text in files.items():
        path = source / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source_text, encoding="utf-8", newline="")
    (intake / "snapshot_request.json").write_text(
        json.dumps(
            {
                "module_set": "fci-custom",
                "svn_revision": "153",
                "application_build": "14.7.1",
                "reviewer": "Phase 2 SME",
            }
        ),
        encoding="utf-8",
    )
    snapshot_root = tmp_path / "snapshots"
    manifest = build_code_snapshot(intake, snapshot_root)
    return snapshot_root / manifest.snapshot_id, manifest


def test_snapshot_parse_publishes_verified_parse_and_retrieval_artifacts(tmp_path: Path) -> None:
    snapshot_directory, snapshot = _build_snapshot(
        tmp_path,
        """CREATE OR REPLACE PACKAGE BODY pkg_customer_custom AS
  c_country CONSTANT VARCHAR2(2) := 'MY';
  PROCEDURE update_customer IS BEGIN DBMS_OUTPUT.PUT_LINE(c_country); END;
END pkg_customer_custom;
/
""",
    )
    staging_root = tmp_path / "staging"

    manifest = parse_code_snapshot(snapshot_directory, staging_root)

    target = staging_root / snapshot.snapshot_id / PARSER_GENERATION_DIRECTORY
    assert manifest.status == "complete"
    assert manifest.state_counts["full_parse"] == 1
    assert (target / "parse_stage_manifest.json").is_file()
    parsed = PlSqlFileParseArtifact.model_validate_json(
        (target / manifest.parse_artifacts[0]).read_text(encoding="utf-8")
    )
    retrieval = CodeRetrievalArtifact.model_validate_json(
        (target / manifest.retrieval_artifacts[0]).read_text(encoding="utf-8")
    )
    analysis = CodeStaticAnalysisArtifact.model_validate_json(
        (target / manifest.analysis_artifacts[0]).read_text(encoding="utf-8")
    )
    assert parsed.source_sha256 == snapshot.files[0].sha256
    assert any(unit.source_kind == "procedure" for unit in retrieval.units)
    assert analysis.source_path == snapshot.files[0].path
    assert analysis.analysis_policy_sha256 == manifest.analysis_policy_sha256
    assert not list(target.glob(".workers*"))

    with pytest.raises(FileExistsError, match="will not be overwritten"):
        parse_code_snapshot(snapshot_directory, staging_root)


def test_invalid_source_degrades_explicitly_instead_of_disappearing(tmp_path: Path) -> None:
    snapshot_directory, snapshot = _build_snapshot_files(
        tmp_path,
        {"invalid.ddl": "not valid PL/SQL\n" * 250},
    )

    manifest = parse_code_snapshot(snapshot_directory, tmp_path / "staging")

    assert manifest.status == "complete_with_degradation"
    assert manifest.state_counts["fallback_parse"] == 1
    target = tmp_path / "staging" / snapshot.snapshot_id / PARSER_GENERATION_DIRECTORY
    retrieval = CodeRetrievalArtifact.model_validate_json(
        (target / manifest.retrieval_artifacts[0]).read_text(encoding="utf-8")
    )
    assert retrieval.total_units >= 2
    assert all(unit.source_kind == "fallback_chunk" for unit in retrieval.units)


def test_tampered_snapshot_fails_before_any_parse_generation(tmp_path: Path) -> None:
    snapshot_directory, snapshot = _build_snapshot(tmp_path, "SELECT 1 FROM dual;\n")
    (snapshot_directory / "source/pkg_customer_custom.sql").write_text(
        "SELECT 2 FROM dual;\n",
        encoding="utf-8",
    )
    staging_root = tmp_path / "staging"

    with pytest.raises(RuntimeError, match="Immutable snapshot source verification failed"):
        parse_code_snapshot(snapshot_directory, staging_root)

    assert not (staging_root / snapshot.snapshot_id).exists()


def test_invalid_resource_boundaries_do_not_write(tmp_path: Path) -> None:
    snapshot_directory, snapshot = _build_snapshot(tmp_path, "SELECT 1 FROM dual;\n")
    staging_root = tmp_path / "staging"

    with pytest.raises(ValueError, match="greater than zero"):
        parse_code_snapshot(snapshot_directory, staging_root, timeout_seconds=0)

    assert not (staging_root / snapshot.snapshot_id).exists()


def test_symbol_collision_publishes_diagnostics_but_fails_stage_gate(tmp_path: Path) -> None:
    duplicate = "CREATE OR REPLACE PROCEDURE duplicate_proc_custom(p_id NUMBER) IS BEGIN NULL; END; /\n"
    snapshot_directory, snapshot = _build_snapshot_files(
        tmp_path,
        {
            "a/duplicate_proc_custom.prc": duplicate,
            "b/duplicate_proc_custom.prc": duplicate,
        },
    )
    staging_root = tmp_path / "staging"

    manifest = parse_code_snapshot(snapshot_directory, staging_root)

    assert manifest.status == "failed"
    target = staging_root / snapshot.snapshot_id / PARSER_GENERATION_DIRECTORY
    analyses = [
        CodeStaticAnalysisArtifact.model_validate_json(
            (target / relative_path).read_text(encoding="utf-8")
        )
        for relative_path in manifest.analysis_artifacts
    ]
    assert all(
        any(item.code == "overload_symbol_collision" for item in artifact.diagnostics)
        for artifact in analyses
    )


def test_stage_manifest_rejects_missing_file_accounting() -> None:
    with pytest.raises(ValueError, match="state counts must equal file_count"):
        CodeParseStageManifest(
            status="complete",
            snapshot_id="snapshot-1",
            snapshot_content_sha256="a" * 64,
            analysis_policy_sha256="b" * 64,
            file_count=2,
            state_counts={
                "full_parse": 1,
                "segmented_parse": 0,
                "fallback_parse": 0,
                "failed": 0,
            },
            parse_artifacts=("parse/a.json", "parse/b.json"),
            retrieval_artifacts=("retrieval/a.json", "retrieval/b.json"),
            analysis_artifacts=("analysis/a.json", "analysis/b.json"),
            timeout_seconds=120,
            memory_limit_bytes=1024,
            max_segment_characters=500,
            max_retrieval_unit_characters=6_000,
            retrieval_overlap_characters=400,
        )


def test_unchanged_parse_and_retrieval_are_reused_into_new_generation(tmp_path: Path) -> None:
    snapshot_directory, snapshot = _build_snapshot(
        tmp_path,
        """CREATE OR REPLACE PACKAGE BODY pkg_reuse_custom AS
  PROCEDURE run IS BEGIN NULL; END;
END;
/
""".replace("pkg_reuse_custom", "pkg_customer_custom"),
    )
    staging_root = tmp_path / "staging"
    first = parse_code_snapshot(
        snapshot_directory,
        staging_root,
        generation_directory="plsql_antlr_4_13_2_analysis_v7",
    )
    first_stage = staging_root / snapshot.snapshot_id / first.parser_generation

    second = parse_code_snapshot(
        snapshot_directory,
        staging_root,
        generation_directory="plsql_antlr_4_13_2_analysis_v8",
        reuse_generation_directory=first_stage,
    )

    assert second.reused_from_generation == "plsql_antlr_4_13_2_analysis_v7"
    assert second.reused_parse_file_count == 1
    assert second.parse_reuse_records[0].reused is True
    second_stage = staging_root / snapshot.snapshot_id / second.parser_generation
    assert PlSqlFileParseArtifact.model_validate_json(
        (first_stage / first.parse_artifacts[0]).read_text(encoding="utf-8")
    ) == PlSqlFileParseArtifact.model_validate_json(
        (second_stage / second.parse_artifacts[0]).read_text(encoding="utf-8")
    )


def test_reuse_contract_mismatch_fails_without_temporary_publication(tmp_path: Path) -> None:
    snapshot_directory, snapshot = _build_snapshot(
        tmp_path,
        """CREATE OR REPLACE PACKAGE BODY pkg_customer_custom AS
  PROCEDURE run IS BEGIN NULL; END;
END;
/
""",
    )
    staging_root = tmp_path / "staging"
    first = parse_code_snapshot(
        snapshot_directory,
        staging_root,
        generation_directory="plsql_antlr_4_13_2_analysis_v7",
    )
    first_stage = staging_root / snapshot.snapshot_id / first.parser_generation

    with pytest.raises(ValueError, match="max_segment_characters"):
        parse_code_snapshot(
            snapshot_directory,
            staging_root,
            generation_directory="plsql_antlr_4_13_2_analysis_v8",
            reuse_generation_directory=first_stage,
            max_segment_characters=501,
        )

    parent = staging_root / snapshot.snapshot_id
    assert not (parent / "plsql_antlr_4_13_2_analysis_v8").exists()
    assert not list(parent.glob(".code-parse-*"))
