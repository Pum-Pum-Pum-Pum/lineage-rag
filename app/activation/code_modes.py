from __future__ import annotations

import hashlib
import json
import os
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


ACTIVATION_RUNTIME_FILES = (
    "app/activation/code_modes.py",
    "app/api/routes/query.py",
    "app/api/routes/readiness.py",
    "app/api/routes/conversations.py",
    "app/services/knowledge_mode_orchestration.py",
    "app/schemas/query_api.py",
    "app/schemas/conversation_api.py",
    "app/fdd_code_lineage/paid_evaluation.py",
    "app/fdd_code_lineage/combined_answer.py",
    "scripts/run_code_modes_activation_smoke.py",
)


class FrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ActivationRequest(FrozenModel):
    schema_version: Literal["code_modes_activation_request_v1"]
    created_at_utc: datetime
    requested_by: str = Field(min_length=1)
    requested_modes: tuple[Literal["code", "combined"], ...]
    target_configuration: dict[str, str]
    rollback_configuration: dict[str, str]
    evidence_identities: dict[str, str]
    status: Literal["pending_approval"] = "pending_approval"
    request_identity_sha256: str


class ActivationApproval(FrozenModel):
    schema_version: Literal["code_modes_activation_approval_v1"]
    request_identity_sha256: str
    decision: Literal["approved", "rejected"]
    approved_by: str = Field(min_length=1)
    decided_at_utc: datetime
    allowed_actions: tuple[Literal["activate", "rollback"], ...]
    paid_smoke_authorized: bool = False
    internal_evidence_disclosure_authorized: bool = False
    approval_identity_sha256: str


class PreflightCheck(FrozenModel):
    name: str
    passed: bool
    detail: str


class ActivationPreflightReport(FrozenModel):
    schema_version: Literal["code_modes_activation_preflight_v1"]
    created_at_utc: datetime
    request_identity_sha256: str
    checks: tuple[PreflightCheck, ...]
    ready_to_apply: bool


class ConfigSwitchResult(FrozenModel):
    action: Literal["activate", "rollback"]
    before: str
    after: str
    applied: bool
    request_identity_sha256: str
    approval_identity_sha256: str
    parent_directory_fsynced: bool


class DisabledBaselineResult(FrozenModel):
    before: Literal["missing", "false"]
    after: Literal["false"] = "false"
    changed: bool
    applied: bool
    parent_directory_fsynced: bool


class ActivationExecutionEvidence(FrozenModel):
    schema_version: Literal["code_modes_activation_execution_v1"]
    created_at_utc: datetime
    request_identity_sha256: str
    approval_identity_sha256: str
    configuration_applied: bool
    service_restart_confirmed: bool
    effective_code_modes_enabled: bool
    code_readiness_passed: bool
    combined_readiness_passed: bool
    smoke_trace_ids: tuple[str, ...] = ()
    rollback_owner: str = Field(min_length=1)
    rollback_rehearsed: bool
    activation_complete: bool
    evidence_identity_sha256: str


def build_activation_request(*, settings, readiness_report_path: Path, requested_by: str) -> ActivationRequest:
    readiness = _read_json(readiness_report_path)
    if not readiness.get("summary", {}).get("activation_ready"):
        raise ValueError("Readiness report is not activation-ready")
    artifact_path = Path(settings.code_index_artifact_path)
    lineage_path = Path(settings.fdd_code_lineage_artifact_path)
    artifact = _read_json(artifact_path)
    lineage = _read_json(lineage_path)
    root_dir = Path(settings.root_dir)
    payload = {
        "schema_version": "code_modes_activation_request_v1",
        "created_at_utc": datetime.now(UTC),
        "requested_by": requested_by.strip(),
        "requested_modes": ("code", "combined"),
        "target_configuration": _runtime_configuration(settings, enabled=True),
        "rollback_configuration": _runtime_configuration(settings, enabled=False),
        "evidence_identities": {
            "readiness_report_sha256": _file_sha256(readiness_report_path),
            "readiness_report_identity": str(readiness["report_identity_sha256"]),
            "code_artifact_sha256": _file_sha256(artifact_path),
            "code_artifact_identity": str(artifact["artifact_identity_sha256"]),
            "lineage_artifact_sha256": _file_sha256(lineage_path),
            "lineage_artifact_identity": str(lineage["artifact_identity_sha256"]),
            "runtime_contract_sha256": _canonical_sha256(
                {
                    name: _file_sha256(root_dir / name)
                    for name in ACTIVATION_RUNTIME_FILES
                }
            ),
        },
        "status": "pending_approval",
    }
    serialized = ActivationRequest.model_construct(
        **payload, request_identity_sha256="pending"
    ).model_dump(mode="json", exclude={"request_identity_sha256"})
    identity = _canonical_sha256(serialized)
    return ActivationRequest(**payload, request_identity_sha256=identity)


def build_activation_approval(
    *, request: ActivationRequest, approved_by: str,
    decision: Literal["approved", "rejected"] = "approved",
    allowed_actions: tuple[Literal["activate", "rollback"], ...] = ("activate", "rollback"),
    paid_smoke_authorized: bool = False,
    internal_evidence_disclosure_authorized: bool = False,
) -> ActivationApproval:
    payload = {
        "schema_version": "code_modes_activation_approval_v1",
        "request_identity_sha256": request.request_identity_sha256,
        "decision": decision,
        "approved_by": approved_by.strip(),
        "decided_at_utc": datetime.now(UTC),
        "allowed_actions": allowed_actions,
        "paid_smoke_authorized": paid_smoke_authorized,
        "internal_evidence_disclosure_authorized": internal_evidence_disclosure_authorized,
    }
    serialized = ActivationApproval.model_construct(
        **payload, approval_identity_sha256="pending"
    ).model_dump(mode="json", exclude={"approval_identity_sha256"})
    return ActivationApproval(
        **payload,
        approval_identity_sha256=_canonical_sha256(serialized),
    )


def evaluate_activation_preflight(
    *, request: ActivationRequest, settings, readiness_report_path: Path,
    env_path: Path,
    approval: ActivationApproval | None = None,
    action: Literal["activate", "rollback"] = "activate",
) -> ActivationPreflightReport:
    expected_enabled = action == "rollback"
    expected_flag = "true" if expected_enabled else "false"
    checks = [
        _check("request_identity", _request_identity(request) == request.request_identity_sha256,
               "Activation request identity is valid."),
        _check("readiness_report", _file_sha256(readiness_report_path) == request.evidence_identities["readiness_report_sha256"],
               "Readiness report bytes match the request."),
        _check("runtime_configuration", _runtime_configuration(settings, enabled=True) == request.target_configuration,
               "Current runtime identities match the requested target."),
        _check("safe_start_state", bool(settings.code_modes_enabled) == expected_enabled,
               f"Code modes are currently {'enabled' if expected_enabled else 'disabled'}."),
        _check("explicit_activation_key", _env_flag_values(env_path) == [expected_flag],
               f".env contains exactly one CODE_MODES_ENABLED={expected_flag} entry."),
        _check("approval", _approval_valid(request, approval),
               "A hash-bound approval is present." if approval else "Approval is still required."),
    ]
    return ActivationPreflightReport(
        schema_version="code_modes_activation_preflight_v1",
        created_at_utc=datetime.now(UTC),
        request_identity_sha256=request.request_identity_sha256,
        checks=tuple(checks),
        ready_to_apply=all(item.passed for item in checks),
    )


def switch_code_modes(
    *, env_path: Path, action: Literal["activate", "rollback"],
    request: ActivationRequest, approval: ActivationApproval,
    preflight: ActivationPreflightReport, apply: bool = False,
) -> ConfigSwitchResult:
    if not _approval_valid(request, approval) or action not in approval.allowed_actions:
        raise PermissionError("A valid approval for this action is required")
    if (
        not preflight.ready_to_apply
        or preflight.request_identity_sha256 != request.request_identity_sha256
    ):
        raise PermissionError("A passing preflight for this request is required")
    text = env_path.read_text(encoding="utf-8")
    lines = text.splitlines(keepends=True)
    indexes = [i for i, line in enumerate(lines) if line.strip().upper().startswith("CODE_MODES_ENABLED=")]
    if len(indexes) != 1:
        raise ValueError(".env must contain exactly one CODE_MODES_ENABLED entry")
    index = indexes[0]
    before = lines[index].split("=", 1)[1].strip().lower()
    expected = "false" if action == "activate" else "true"
    after = "true" if action == "activate" else "false"
    if before != expected:
        raise ValueError(f"Refusing {action}: expected CODE_MODES_ENABLED={expected}")
    newline = "\r\n" if lines[index].endswith("\r\n") else "\n"
    lines[index] = f"CODE_MODES_ENABLED={after}{newline}"
    parent_fsynced = False
    if apply:
        parent_fsynced = _atomic_write(env_path, "".join(lines))
    return ConfigSwitchResult(
        action=action,
        before=before,
        after=after,
        applied=apply,
        request_identity_sha256=request.request_identity_sha256,
        approval_identity_sha256=approval.approval_identity_sha256,
        parent_directory_fsynced=parent_fsynced,
    )


def initialize_disabled_baseline(*, env_path: Path, apply: bool = False) -> DisabledBaselineResult:
    text = env_path.read_text(encoding="utf-8") if env_path.is_file() else ""
    values = _env_flag_values(env_path)
    if len(values) > 1:
        raise ValueError(".env contains duplicate CODE_MODES_ENABLED entries")
    if values == ["true"]:
        raise ValueError("Refusing to replace an enabled code-mode flag")
    if values and values != ["false"]:
        raise ValueError("CODE_MODES_ENABLED must be true or false")
    before: Literal["missing", "false"] = "false" if values else "missing"
    changed = before == "missing"
    parent_fsynced = False
    if apply and changed:
        separator = "" if not text or text.endswith(("\n", "\r")) else "\n"
        parent_fsynced = _atomic_write(
            env_path, f"{text}{separator}CODE_MODES_ENABLED=false\n"
        )
    return DisabledBaselineResult(
        before=before,
        changed=changed,
        applied=apply and changed,
        parent_directory_fsynced=parent_fsynced,
    )


def build_execution_evidence(
    *, request: ActivationRequest, approval: ActivationApproval,
    configuration_applied: bool, service_restart_confirmed: bool,
    effective_code_modes_enabled: bool, code_readiness_passed: bool,
    combined_readiness_passed: bool, smoke_trace_ids: tuple[str, ...],
    rollback_owner: str, rollback_rehearsed: bool,
) -> ActivationExecutionEvidence:
    if not _approval_valid(request, approval):
        raise PermissionError("Valid activation approval is required")
    smoke_authorized = (
        approval.paid_smoke_authorized
        and approval.internal_evidence_disclosure_authorized
    )
    complete = all((
        configuration_applied,
        service_restart_confirmed,
        effective_code_modes_enabled,
        code_readiness_passed,
        combined_readiness_passed,
        bool(smoke_trace_ids),
        smoke_authorized,
        rollback_rehearsed,
    ))
    payload = {
        "schema_version": "code_modes_activation_execution_v1",
        "created_at_utc": datetime.now(UTC),
        "request_identity_sha256": request.request_identity_sha256,
        "approval_identity_sha256": approval.approval_identity_sha256,
        "configuration_applied": configuration_applied,
        "service_restart_confirmed": service_restart_confirmed,
        "effective_code_modes_enabled": effective_code_modes_enabled,
        "code_readiness_passed": code_readiness_passed,
        "combined_readiness_passed": combined_readiness_passed,
        "smoke_trace_ids": smoke_trace_ids,
        "rollback_owner": rollback_owner.strip(),
        "rollback_rehearsed": rollback_rehearsed,
        "activation_complete": complete,
    }
    serialized = ActivationExecutionEvidence.model_construct(
        **payload, evidence_identity_sha256="pending"
    ).model_dump(mode="json", exclude={"evidence_identity_sha256"})
    return ActivationExecutionEvidence(
        **payload, evidence_identity_sha256=_canonical_sha256(serialized)
    )


def approval_identity(payload: dict) -> str:
    return _canonical_sha256({k: v for k, v in payload.items() if k != "approval_identity_sha256"})


def _approval_valid(request: ActivationRequest, approval: ActivationApproval | None) -> bool:
    return bool(
        approval
        and approval.decision == "approved"
        and approval.request_identity_sha256 == request.request_identity_sha256
        and approval_identity(approval.model_dump(mode="json")) == approval.approval_identity_sha256
    )


def _runtime_configuration(settings, *, enabled: bool) -> dict[str, str]:
    return {
        "CODE_MODES_ENABLED": str(enabled).lower(),
        "QDRANT_COLLECTION_NAME": str(settings.qdrant_collection_name),
        "FDD_GENERATION": str(settings.fdd_generation),
        "PROCESSED_DIR": str(settings.processed_dir),
        "CODE_QDRANT_COLLECTION_NAME": str(settings.code_qdrant_collection_name),
        "CODE_QDRANT_LOCAL_PATH": str(settings.code_qdrant_local_path),
        "CODE_INDEX_ARTIFACT_PATH": str(settings.code_index_artifact_path),
        "CODE_ANALYSIS_DIRECTORY": str(settings.code_analysis_directory),
        "FDD_CODE_LINEAGE_ARTIFACT_PATH": str(settings.fdd_code_lineage_artifact_path),
    }


def _env_flag_values(path: Path) -> list[str]:
    if not path.is_file():
        return []
    values: list[str] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped.upper().startswith("CODE_MODES_ENABLED="):
            values.append(stripped.split("=", 1)[1].strip().lower())
    return values


def _request_identity(request: ActivationRequest) -> str:
    payload = request.model_dump(mode="json", exclude={"request_identity_sha256"})
    return _canonical_sha256(payload)


def _check(name: str, passed: bool, detail: str) -> PreflightCheck:
    return PreflightCheck(name=name, passed=passed, detail=detail)


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_sha256(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str).encode()
    return hashlib.sha256(encoded).hexdigest()


def _atomic_write(path: Path, text: str) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        return _fsync_parent_directory(path.parent)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def _fsync_parent_directory(directory: Path) -> bool:
    flags = getattr(os, "O_DIRECTORY", None)
    if flags is None:
        return False
    descriptor = None
    try:
        descriptor = os.open(directory, flags)
        os.fsync(descriptor)
        return True
    except OSError:
        return False
    finally:
        if descriptor is not None:
            os.close(descriptor)
