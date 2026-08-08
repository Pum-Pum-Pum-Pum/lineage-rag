from __future__ import annotations

import fnmatch
import hashlib
import json
from pathlib import Path
from typing import Any


ALLOWED_STATUSES = frozenset(
    {
        "benchmark_revised_pending_replay",
        "paid_replay_pending_semantic_review",
        "blocked_missing_source",
        "accepted_no_action",
        "accepted_after_paid_replay",
    }
)


def build_remediation_report(
    *,
    review_ledger_path: Path,
    remediation_plan_path: Path,
    artifact_directory: Path,
) -> dict[str, Any]:
    ledger = json.loads(review_ledger_path.read_text(encoding="utf-8"))
    plan = json.loads(remediation_plan_path.read_text(encoding="utf-8"))
    reviewed_case_ids = {
        str(decision["case_id"]) for decision in ledger.get("decisions", [])
    }
    artifact_names = sorted(path.name for path in artifact_directory.glob("*.json"))
    if not artifact_names:
        raise ValueError(f"No lexical artifacts found in {artifact_directory}")

    actions: list[dict[str, Any]] = []
    seen: set[str] = set()
    for raw_action in plan.get("actions", []):
        case_id = str(raw_action.get("case_id", "")).strip()
        status = str(raw_action.get("status", "")).strip()
        if not case_id or case_id in seen:
            raise ValueError(f"Blank or duplicate remediation case_id: {case_id!r}")
        if case_id not in reviewed_case_ids:
            raise ValueError(f"Remediation case was not present in SME review: {case_id}")
        if status not in ALLOWED_STATUSES:
            raise ValueError(f"Unsupported remediation status for {case_id}: {status}")
        seen.add(case_id)

        required_patterns = [
            str(value).strip() for value in raw_action.get("required_artifact_patterns", [])
        ]
        matched = {
            pattern: [name for name in artifact_names if fnmatch.fnmatchcase(name, pattern)]
            for pattern in required_patterns
        }
        missing_patterns = [pattern for pattern, names in matched.items() if not names]
        if status == "blocked_missing_source" and not missing_patterns:
            raise ValueError(
                f"{case_id} is marked blocked_missing_source but all required artifacts exist"
            )
        if status != "blocked_missing_source" and missing_patterns:
            raise ValueError(
                f"{case_id} has missing required artifacts but status is {status}: "
                f"{missing_patterns}"
            )

        action = dict(raw_action)
        action["artifact_matches"] = matched
        action["missing_artifact_patterns"] = missing_patterns
        actions.append(action)

    unresolved_review_ids = {
        str(decision["case_id"])
        for decision in ledger.get("decisions", [])
        if str(decision.get("verdict")) != "accepted"
    }
    if seen != unresolved_review_ids:
        raise ValueError(
            "Remediation plan must cover every unresolved SME decision exactly; "
            f"missing={sorted(unresolved_review_ids - seen)}, "
            f"extra={sorted(seen - unresolved_review_ids)}"
        )

    non_blocking_statuses = {"accepted_no_action", "accepted_after_paid_replay"}
    blocking = [
        action["case_id"]
        for action in actions
        if action["status"] not in non_blocking_statuses
    ]
    return {
        "schema_version": "fdd_sme_remediation_v1",
        "review_ledger": str(review_ledger_path),
        "review_ledger_sha256": _sha256(review_ledger_path),
        "remediation_plan": str(remediation_plan_path),
        "remediation_plan_sha256": _sha256(remediation_plan_path),
        "artifact_directory": str(artifact_directory),
        "artifact_count": len(artifact_names),
        "phase_1_gate_status": "pending_material_remediation" if blocking else "passed",
        "blocking_case_ids": blocking,
        "actions": actions,
    }


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
