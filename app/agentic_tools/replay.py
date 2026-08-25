from __future__ import annotations

import hashlib
import json
from pathlib import Path

from app.fdd_code_lineage.paid_evaluation import CODE_SYSTEM_PROMPT


def build_case_replay_authorization(
    *,
    case_id: str,
    reviewed_manifest: Path,
    prior_review_ledger: Path,
    prior_trace: Path,
    local_uat_report: Path,
    answer_model: str,
    approval_note: str,
) -> dict:
    if not approval_note.strip():
        raise ValueError("Replay approval note is required")
    values = {
        "schema_version": "bounded_tool_case_replay_authorization_v1",
        "case_id": case_id,
        "reviewed_manifest": str(reviewed_manifest),
        "reviewed_manifest_sha256": _sha256(reviewed_manifest),
        "prior_review_ledger": str(prior_review_ledger),
        "prior_review_ledger_sha256": _sha256(prior_review_ledger),
        "prior_trace": str(prior_trace),
        "prior_trace_sha256": _sha256(prior_trace),
        "local_uat_report": str(local_uat_report),
        "local_uat_report_sha256": _sha256(local_uat_report),
        "code_system_prompt_sha256": hashlib.sha256(
            CODE_SYSTEM_PROMPT.encode("utf-8")
        ).hexdigest(),
        "answer_model": answer_model,
        "maximum_answer_requests": 1,
        "maximum_query_embedding_requests": 0,
        "automatic_retries": 0,
        "paid_use_authorized": True,
        "internal_evidence_disclosure_authorized": True,
        "approval_source": "chat_confirmation",
        "approval_note": approval_note.strip(),
    }
    canonical = json.dumps(values, sort_keys=True, separators=(",", ":"))
    values["authorization_identity_sha256"] = hashlib.sha256(
        canonical.encode("utf-8")
    ).hexdigest()
    return values


def write_authorization_no_overwrite(value: dict, path: Path) -> Path:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite replay authorization: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        handle.write(json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True))
    return path


def validate_case_replay_authorization(value: dict) -> None:
    identity = value.get("authorization_identity_sha256")
    unsigned = dict(value)
    unsigned.pop("authorization_identity_sha256", None)
    canonical = json.dumps(unsigned, sort_keys=True, separators=(",", ":"))
    expected = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    if identity != expected:
        raise ValueError("Replay authorization identity mismatch")
    if value.get("maximum_answer_requests") != 1:
        raise ValueError("Replay authorization must permit exactly one answer request")
    if value.get("maximum_query_embedding_requests") != 0:
        raise ValueError("Replay authorization must permit zero query embeddings")
    if value.get("automatic_retries") != 0:
        raise ValueError("Replay authorization must disable retries")
    if not value.get("paid_use_authorized") or not value.get(
        "internal_evidence_disclosure_authorized"
    ):
        raise PermissionError("Replay paid use and disclosure are not authorized")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
