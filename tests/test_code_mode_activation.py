import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from app.activation.code_modes import (
    ACTIVATION_RUNTIME_FILES,
    build_activation_approval,
    build_activation_request,
    build_execution_evidence,
    evaluate_activation_preflight,
    initialize_disabled_baseline,
    switch_code_modes,
)


def _settings(tmp_path: Path, **overrides):
    artifact = tmp_path / "code-artifact.json"
    lineage = tmp_path / "lineage.json"
    artifact.write_text(json.dumps({"artifact_identity_sha256": "artifact-id"}), encoding="utf-8")
    lineage.write_text(json.dumps({"artifact_identity_sha256": "lineage-id"}), encoding="utf-8")
    values = {
        "root_dir": tmp_path,
        "code_modes_enabled": False,
        "qdrant_collection_name": "functional_specs_v5",
        "fdd_generation": "functional_specs_v5",
        "processed_dir": tmp_path / "processed-v5",
        "code_qdrant_collection_name": "code_custom_r1_v2",
        "code_qdrant_local_path": tmp_path / "qdrant-code",
        "code_index_artifact_path": artifact,
        "code_analysis_directory": tmp_path / "analysis-v12",
        "fdd_code_lineage_artifact_path": lineage,
    }
    values.update(overrides)
    for relative in ACTIVATION_RUNTIME_FILES:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(relative, encoding="utf-8")
    (tmp_path / ".env").write_text("CODE_MODES_ENABLED=false\n", encoding="utf-8")
    return SimpleNamespace(**values)


def _readiness(tmp_path: Path) -> Path:
    path = tmp_path / "readiness.json"
    path.write_text(json.dumps({
        "report_identity_sha256": "readiness-id",
        "summary": {"activation_ready": True, "activation_performed": False},
    }), encoding="utf-8")
    return path


def test_request_binds_explicit_runtime_and_evidence_identities(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    request = build_activation_request(
        settings=settings,
        readiness_report_path=_readiness(tmp_path),
        requested_by="operator",
    )

    assert request.status == "pending_approval"
    assert request.target_configuration["CODE_MODES_ENABLED"] == "true"
    assert request.rollback_configuration["CODE_MODES_ENABLED"] == "false"
    assert request.target_configuration["CODE_QDRANT_COLLECTION_NAME"] == "code_custom_r1_v2"
    assert len(request.evidence_identities["runtime_contract_sha256"]) == 64
    assert "app/fdd_code_lineage/paid_evaluation.py" in ACTIVATION_RUNTIME_FILES
    assert "app/fdd_code_lineage/combined_answer.py" in ACTIVATION_RUNTIME_FILES
    assert "scripts/run_code_modes_activation_smoke.py" in ACTIVATION_RUNTIME_FILES
    assert "OPENAI_API_KEY" not in request.model_dump_json()


def test_preflight_blocks_without_approval_and_accepts_bound_approval(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    readiness = _readiness(tmp_path)
    request = build_activation_request(settings=settings, readiness_report_path=readiness, requested_by="operator")

    pending = evaluate_activation_preflight(
        request=request, settings=settings, readiness_report_path=readiness,
        env_path=tmp_path / ".env",
    )
    approval = build_activation_approval(request=request, approved_by="approver")
    approved = evaluate_activation_preflight(
        request=request, settings=settings, readiness_report_path=readiness,
        env_path=tmp_path / ".env", approval=approval
    )

    assert pending.ready_to_apply is False
    assert approved.ready_to_apply is True


def test_preflight_rejects_configuration_drift(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    readiness = _readiness(tmp_path)
    request = build_activation_request(settings=settings, readiness_report_path=readiness, requested_by="operator")
    approval = build_activation_approval(request=request, approved_by="approver")
    drifted = _settings(tmp_path, code_qdrant_collection_name="wrong-generation")

    report = evaluate_activation_preflight(
        request=request, settings=drifted, readiness_report_path=readiness,
        env_path=tmp_path / ".env", approval=approval
    )

    assert report.ready_to_apply is False
    assert next(item for item in report.checks if item.name == "runtime_configuration").passed is False


def test_atomic_switch_is_dry_run_by_default_and_reversible(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    readiness = _readiness(tmp_path)
    request = build_activation_request(settings=settings, readiness_report_path=readiness, requested_by="operator")
    approval = build_activation_approval(request=request, approved_by="approver")
    env = tmp_path / ".env"
    env.write_text("OPENAI_API_KEY=secret\nCODE_MODES_ENABLED=false\n", encoding="utf-8")

    dry_run = switch_code_modes(
        env_path=env, action="activate", request=request, approval=approval,
        preflight=evaluate_activation_preflight(
            request=request, settings=settings, readiness_report_path=readiness,
            env_path=env, approval=approval
        ),
    )
    assert dry_run.applied is False
    assert "CODE_MODES_ENABLED=false" in env.read_text(encoding="utf-8")
    activated = switch_code_modes(
        env_path=env, action="activate", request=request, approval=approval,
        preflight=evaluate_activation_preflight(
            request=request, settings=settings, readiness_report_path=readiness,
            env_path=env, approval=approval
        ), apply=True
    )
    assert activated.applied is True
    assert "CODE_MODES_ENABLED=true" in env.read_text(encoding="utf-8")
    switch_code_modes(
        env_path=env, action="rollback", request=request, approval=approval,
        preflight=evaluate_activation_preflight(
            request=request,
            settings=SimpleNamespace(**{**settings.__dict__, "code_modes_enabled": True}),
            readiness_report_path=readiness, env_path=env,
            approval=approval, action="rollback"
        ), apply=True
    )
    final = env.read_text(encoding="utf-8")
    assert "CODE_MODES_ENABLED=false" in final
    assert "OPENAI_API_KEY=secret" in final


def test_switch_rejects_wrong_approval_or_ambiguous_env(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    readiness = _readiness(tmp_path)
    request = build_activation_request(settings=settings, readiness_report_path=readiness, requested_by="operator")
    rejected = build_activation_approval(
        request=request, approved_by="approver", decision="rejected"
    )
    env = tmp_path / ".env"
    env.write_text("CODE_MODES_ENABLED=false\n", encoding="utf-8")

    with pytest.raises(PermissionError):
        switch_code_modes(
            env_path=env, action="activate", request=request, approval=rejected,
            preflight=evaluate_activation_preflight(
                request=request, settings=settings, readiness_report_path=readiness,
                env_path=env
            ),
        )
    approved = build_activation_approval(request=request, approved_by="approver")
    preflight = evaluate_activation_preflight(
        request=request, settings=settings, readiness_report_path=readiness,
        env_path=env, approval=approved
    )
    env.write_text(
        "CODE_MODES_ENABLED=false\nCODE_MODES_ENABLED=false\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="exactly one"):
        switch_code_modes(
            env_path=env, action="activate", request=request, approval=approved,
            preflight=preflight,
        )


def test_disabled_baseline_is_dry_run_safe_and_refuses_enabled_state(tmp_path: Path) -> None:
    env = tmp_path / ".env"
    env.write_text("OPENAI_API_KEY=secret\n", encoding="utf-8")

    dry_run = initialize_disabled_baseline(env_path=env)
    assert dry_run.before == "missing"
    assert dry_run.applied is False
    assert "CODE_MODES_ENABLED" not in env.read_text(encoding="utf-8")
    applied = initialize_disabled_baseline(env_path=env, apply=True)
    assert applied.applied is True
    assert "OPENAI_API_KEY=secret" in env.read_text(encoding="utf-8")
    assert "CODE_MODES_ENABLED=false" in env.read_text(encoding="utf-8")
    assert initialize_disabled_baseline(env_path=env, apply=True).changed is False
    env.write_text("CODE_MODES_ENABLED=true\n", encoding="utf-8")
    with pytest.raises(ValueError, match="enabled"):
        initialize_disabled_baseline(env_path=env, apply=True)


def test_execution_evidence_requires_smoke_authority_and_all_runtime_gates(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    request = build_activation_request(
        settings=settings,
        readiness_report_path=_readiness(tmp_path),
        requested_by="operator",
    )
    activation_only = build_activation_approval(
        request=request, approved_by="approver"
    )
    incomplete = build_execution_evidence(
        request=request,
        approval=activation_only,
        configuration_applied=True,
        service_restart_confirmed=True,
        effective_code_modes_enabled=True,
        code_readiness_passed=True,
        combined_readiness_passed=True,
        smoke_trace_ids=("trace-code", "trace-combined"),
        rollback_owner="operator",
        rollback_rehearsed=True,
    )
    assert incomplete.activation_complete is False

    full_approval = build_activation_approval(
        request=request,
        approved_by="approver",
        paid_smoke_authorized=True,
        internal_evidence_disclosure_authorized=True,
    )
    complete = build_execution_evidence(
        request=request,
        approval=full_approval,
        configuration_applied=True,
        service_restart_confirmed=True,
        effective_code_modes_enabled=True,
        code_readiness_passed=True,
        combined_readiness_passed=True,
        smoke_trace_ids=("trace-code", "trace-combined"),
        rollback_owner="operator",
        rollback_rehearsed=True,
    )
    assert complete.activation_complete is True
    assert len(complete.evidence_identity_sha256) == 64
