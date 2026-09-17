from __future__ import annotations

from pathlib import Path

import pytest

from app.knowledge_updates.fdd_sources import plan_additive_fdd_update
from app.knowledge_updates.models import ChangeSet, make_release_manifest
from app.knowledge_updates.storage import Run


def _doc(directory: Path, name: str, content: bytes) -> Path:
    path = directory / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


def test_additive_fdd_plan_retains_baseline_and_classifies_new_source(tmp_path: Path):
    base, update = tmp_path / "base", tmp_path / "update"
    _doc(base, "FS_FCIS_14.7.0.0.0$ASNB_R1_Existing_v1.0.docx", b"old")
    _doc(update, "FS_FCIS_14.7.0.0.0$ASNB_R2_New_v1.0.docx", b"new")
    plan = plan_additive_fdd_update(base_generation="functional_specs_v9", target_generation="functional_specs_v10",
                                    baseline_directory=base, update_directory=update)
    assert {item.path for item in plan.target_sources} == {
        "FS_FCIS_14.7.0.0.0$ASNB_R1_Existing_v1.0.docx",
        "FS_FCIS_14.7.0.0.0$ASNB_R2_New_v1.0.docx",
    }
    assert plan.changes.added == ("FS_FCIS_14.7.0.0.0$ASNB_R2_New_v1.0.docx",)
    assert plan.changes.unchanged == ("FS_FCIS_14.7.0.0.0$ASNB_R1_Existing_v1.0.docx",)


def test_supersession_is_not_also_classified_as_removal(tmp_path: Path):
    base, update = tmp_path / "base", tmp_path / "update"
    _doc(base, "FS_FCIS_14.4.0.0.0$ASNB_R4_REST API Services_v2.30.docx", b"v230")
    _doc(update, "FS_FCIS_14.4.0.0.0$ASNB_R4_REST API Services_v2.31.docx", b"v231")
    plan = plan_additive_fdd_update(base_generation="functional_specs_v9", target_generation="functional_specs_v10",
                                    baseline_directory=base, update_directory=update)
    assert plan.changes.removed == ()
    assert plan.changes.superseded == ("FS_FCIS_14.4.0.0.0$ASNB_R4_REST API Services_v2.30.docx",)


def test_same_path_changed_fdd_requires_explicit_replacement_manifest(tmp_path: Path):
    base, update = tmp_path / "base", tmp_path / "update"
    name = "FS_FCIS_14.7.0.0.0$ASNB_R2_Changed_v1.0.docx"
    _doc(base, name, b"old"); _doc(update, name, b"new")
    with pytest.raises(ValueError, match="Conflicting FDD source identity"):
        plan_additive_fdd_update(base_generation="functional_specs_v9", target_generation="functional_specs_v10",
                                 baseline_directory=base, update_directory=update)
    plan = plan_additive_fdd_update(base_generation="functional_specs_v9", target_generation="functional_specs_v10",
                                    baseline_directory=base, update_directory=update, replacements=[name])
    assert plan.changes.modified == (name,)


def test_knowledge_state_namespace_never_reads_code_update_state(tmp_path: Path):
    code = tmp_path / "data/code_updates/Release_006"
    code.mkdir(parents=True)
    (code / "state.json").write_text('{"schema_version":"code_update_run_v1","run_id":"Release_006","status":"COMPLETE","steps":{}}', encoding="utf-8")
    run = Run(tmp_path, "Release_006")
    assert run.directory == tmp_path / "data/knowledge_updates/Release_006"
    assert run.state["schema_version"] == "knowledge_update_run_v1"
    assert run.state["status"] == "NEW"


def test_release_manifest_binds_the_complete_combination():
    value = make_release_manifest(run_id="Release_006", mode="both", fdd_generation="functional_specs_v10",
        fdd_collection="functional_specs_v10", fdd_processed_directory="data/indexes/functional_specs_v10/processed",
        fdd_stage_directory="data/knowledge_updates/Release_006/fdd_stage", fdd_stage_manifest_sha256="a" * 64,
        code_snapshot_id="fcis-custom-r6-abc", code_collection="code_custom_update_release_006", code_artifact_path="data/x.json",
        code_artifact_identity_sha256="b" * 64, code_analysis_directory="data/analysis", lineage_path="data/lineage.json",
        lineage_identity_sha256="c" * 64, deferred_lineage_path="data/deferred.json", review_identity_sha256="d" * 64,
        evaluation_report_sha256="e" * 64, runtime_files_sha256="f" * 64)
    assert value.manifest_identity_sha256
    assert value.fdd_generation == "functional_specs_v10"
