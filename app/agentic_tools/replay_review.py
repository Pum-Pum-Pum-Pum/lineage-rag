from __future__ import annotations

import hashlib
import json
import re
from datetime import UTC, datetime
from pathlib import Path

from app.agentic_tools.replay import validate_case_replay_authorization


_CASE = re.compile(r"^##[ \t]+\d+\.[ \t]+(?P<case_id>[a-z0-9][a-z0-9-]*)[ \t]*$", re.MULTILINE)
_VERDICT = re.compile(r"^SME verdict:[ \t]*(?P<value>[^\r\n]*)$", re.MULTILINE)
_RATIONALE = re.compile(r"^SME rationale:[ \t]*(?P<value>[^\r\n]*)$", re.MULTILINE)
_FOLLOW_UP = re.compile(r"^Required follow-up:[ \t]*(?P<value>[^\r\n]*)$", re.MULTILINE)
_STRUCTURAL = re.compile(
    r"^Structural result:[ \t]*\*\*(?P<value>pass|fail)\*\*[ \t]*$",
    re.MULTILINE,
)


def build_replay_review_ledger(
    *,
    run_state_path: Path,
    review_file: Path,
    authorization_path: Path,
    prior_review_ledger_path: Path,
    reviewer: str,
    global_acceptance_note: str,
) -> dict:
    reviewer = reviewer.strip()
    fallback = global_acceptance_note.strip()
    if not reviewer or not fallback:
        raise ValueError("Reviewer and global acceptance note are required")

    run = _json(run_state_path)
    authorization = _json(authorization_path)
    prior = _json(prior_review_ledger_path)
    validate_case_replay_authorization(authorization)
    _validate_bindings(
        run=run,
        authorization=authorization,
        authorization_path=authorization_path,
        prior=prior,
        prior_review_ledger_path=prior_review_ledger_path,
    )

    decision = parse_replay_review_packet(review_file.read_text(encoding="utf-8"))
    case_id = run["case_id"]
    if decision["case_id"] != case_id:
        raise ValueError("Replay run and SME packet case IDs differ")
    observed_structural = bool(run["structural_passed"])
    if decision["structural"] != ("pass" if observed_structural else "fail"):
        raise ValueError("SME packet and replay structural results differ")
    if decision["verdict"] != "accepted":
        raise ValueError("Replay remediation closes only after explicit SME acceptance")

    prior_decisions = {item["case_id"]: item for item in prior["decisions"]}
    old = prior_decisions.get(case_id)
    if old is None or old.get("sme_verdict") != "corrected":
        raise ValueError("Prior ledger does not contain the matching correction")
    prior_accepted = int(prior["summary"]["semantic_acceptances"])
    prior_total = int(prior["summary"]["total_cases"])
    if prior_accepted != prior_total - 1:
        raise ValueError("Prior ledger is not the expected one-correction gate")

    trace_path = run_state_path.parent / run["trace"]
    rationale = decision["rationale"] or fallback
    ledger = {
        "schema_version": "paid_bounded_tool_case_replay_review_ledger_v1",
        "reviewer": reviewer,
        "reviewed_at": datetime.now(UTC).isoformat(),
        "case_id": case_id,
        "run_state": str(run_state_path),
        "run_state_sha256": _sha256(run_state_path),
        "replay_trace": str(trace_path),
        "replay_trace_sha256": _sha256(trace_path),
        "review_file": str(review_file),
        "review_file_sha256": _sha256(review_file),
        "authorization": str(authorization_path),
        "authorization_sha256": _sha256(authorization_path),
        "authorization_identity_sha256": authorization[
            "authorization_identity_sha256"
        ],
        "prior_review_ledger": str(prior_review_ledger_path),
        "prior_review_ledger_sha256": _sha256(prior_review_ledger_path),
        "prior_review_ledger_identity_sha256": prior["ledger_identity_sha256"],
        "decision": {
            "prior_sme_verdict": old["sme_verdict"],
            "prior_structural_passed": bool(old["observed_structural_passed"]),
            "replay_structural_passed": observed_structural,
            "replay_sme_verdict": decision["verdict"],
            "semantic_accepted": True,
            "rationale": rationale,
            "rationale_source": "packet" if decision["rationale"] else "global_acceptance_note",
            "required_follow_up": decision["required_follow_up"],
        },
        "summary": {
            "original_suite_total_cases": prior_total,
            "original_suite_prior_semantic_acceptances": prior_accepted,
            "effective_semantic_acceptances_after_replay": prior_accepted + 1,
            "semantic_review_status": "accepted_after_targeted_replay",
            "case_remediation_closed": True,
            "activation_authorized": False,
            "additional_paid_requests_authorized": 0,
        },
    }
    unsigned = json.dumps(ledger, sort_keys=True, separators=(",", ":"))
    ledger["ledger_identity_sha256"] = hashlib.sha256(unsigned.encode()).hexdigest()
    return ledger


def parse_replay_review_packet(markdown: str) -> dict[str, str]:
    headings = list(_CASE.finditer(markdown))
    if len(headings) != 1:
        raise ValueError("Replay SME packet must contain exactly one case")
    heading = headings[0]
    section = markdown[heading.end() :]
    return {
        "case_id": heading.group("case_id"),
        "verdict": _field(_VERDICT, section, "verdict").casefold(),
        "rationale": _field(_RATIONALE, section, "rationale", allow_blank=True),
        "required_follow_up": _field(
            _FOLLOW_UP, section, "required follow-up", allow_blank=True
        ),
        "structural": _field(_STRUCTURAL, section, "structural result"),
    }


def write_replay_review_ledger_no_overwrite(ledger: dict, path: Path) -> Path:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite replay review ledger: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        handle.write(json.dumps(ledger, indent=2, ensure_ascii=False, sort_keys=True))
    return path


def _validate_bindings(
    *,
    run: dict,
    authorization: dict,
    authorization_path: Path,
    prior: dict,
    prior_review_ledger_path: Path,
) -> None:
    if run.get("status") != "completed_pending_sme_review":
        raise ValueError("Replay is not complete and pending SME review")
    if run.get("answer_requests_completed") != 1:
        raise ValueError("Replay did not complete exactly one answer request")
    if run.get("query_embedding_requests_completed") != 0:
        raise ValueError("Replay unexpectedly used a query embedding")
    if run.get("automatic_openai_retries") != 0:
        raise ValueError("Replay unexpectedly used automatic retries")
    if run.get("authorization_sha256") != _sha256(authorization_path):
        raise ValueError("Run-state authorization hash mismatch")
    if run.get("authorization_identity_sha256") != authorization.get(
        "authorization_identity_sha256"
    ):
        raise ValueError("Run-state authorization identity mismatch")
    if authorization.get("case_id") != run.get("case_id"):
        raise ValueError("Authorization and replay case IDs differ")
    if authorization.get("prior_review_ledger_sha256") != _sha256(
        prior_review_ledger_path
    ):
        raise ValueError("Authorization prior-review-ledger hash mismatch")
    if prior.get("ledger_identity_sha256") is None:
        raise ValueError("Prior review ledger has no identity")


def _field(
    pattern: re.Pattern[str], section: str, label: str, *, allow_blank: bool = False
) -> str:
    match = pattern.search(section)
    if match is None:
        raise ValueError(f"Missing SME {label}")
    value = match.group("value").strip()
    if not value and not allow_blank:
        raise ValueError(f"Blank SME {label}")
    return value


def _json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
