from __future__ import annotations

import hashlib
import json
import re
from datetime import UTC, datetime
from pathlib import Path

from app.agentic_tools.uat import load_manual_uat_cases


BATCH_PATTERN = re.compile(r"^- Batch identity: `(?P<value>[0-9a-f]{64})`\r?$", re.MULTILINE)


def promote_manual_uat_global_acceptance(
    *,
    draft_manifest: Path,
    batch_report: Path,
    review_packet: Path,
    reviewed_manifest: Path,
    ledger_file: Path,
    reviewer: str,
    approval_note: str,
    paid_use_authorized: bool,
    internal_evidence_disclosure_authorized: bool,
) -> dict:
    cases = load_manual_uat_cases(draft_manifest)
    if not approval_note.strip():
        raise ValueError("A durable UAT approval note is required")
    batch = json.loads(batch_report.read_text(encoding="utf-8"))
    packet_bytes = review_packet.read_bytes()
    packet = packet_bytes.decode("utf-8")
    manifest_hash = _sha256(draft_manifest.read_bytes())
    if batch.get("manifest_sha256") != manifest_hash:
        raise ValueError("UAT batch is not bound to the draft manifest")
    batch_identity = str(batch.get("batch_identity_sha256", ""))
    match = BATCH_PATTERN.search(packet)
    if match is None or match.group("value") != batch_identity:
        raise ValueError("UAT packet is not bound to the batch report")
    batch_ids = {item["case_id"] for item in batch.get("cases", [])}
    manifest_ids = {case.case_id for case in cases}
    if batch_ids != manifest_ids:
        raise ValueError("UAT packet/batch scope does not match the manifest")
    if not all(item.get("diagnostic_passed") for item in batch.get("cases", [])):
        raise ValueError("Global acceptance requires all UAT diagnostics to pass")
    reviewed_text = "".join(
        case.model_copy(
            update={"review_status": "reviewed", "sme_reviewed": True}
        ).model_dump_json()
        + "\n"
        for case in cases
    )
    ledger = {
        "schema_version": "manual_bounded_tool_uat_review_ledger_v1",
        "reviewer": reviewer.strip(),
        "reviewed_at": datetime.now(UTC).isoformat(),
        "approval_source": "chat_confirmation",
        "approval_note": approval_note.strip(),
        "draft_manifest_sha256": manifest_hash,
        "batch_report_sha256": _sha256(batch_report.read_bytes()),
        "batch_identity_sha256": batch_identity,
        "review_packet_sha256": _sha256(packet_bytes),
        "reviewed_manifest_sha256": _sha256(reviewed_text.encode("utf-8")),
        "case_count": len(cases),
        "paid_use_authorized": paid_use_authorized,
        "internal_evidence_disclosure_authorized": (
            internal_evidence_disclosure_authorized
        ),
        "maximum_answer_requests": len(cases),
        "maximum_query_embedding_requests": 0,
        "automatic_retries": 0,
    }
    identity_input = json.dumps(ledger, sort_keys=True, separators=(",", ":"))
    ledger["ledger_identity_sha256"] = _sha256(identity_input.encode("utf-8"))
    outputs = {
        reviewed_manifest: reviewed_text,
        ledger_file: json.dumps(ledger, indent=2, sort_keys=True) + "\n",
    }
    existing = [str(path) for path in outputs if path.exists()]
    if existing:
        raise FileExistsError(f"Refusing to overwrite UAT review outputs: {existing}")
    for path, content in outputs.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="") as handle:
            handle.write(content)
    persisted = load_manual_uat_cases(reviewed_manifest)
    if not all(case.sme_reviewed and case.review_status == "reviewed" for case in persisted):
        raise RuntimeError("Reviewed UAT manifest failed validation")
    return ledger


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()
