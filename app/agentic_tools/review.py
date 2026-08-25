from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Sequence

from app.agentic_tools.evaluation import BoundedToolEvalCase, load_eval_cases


SECTION_PATTERN = re.compile(
    r"^##\s+\d+\.\s+(?P<case_id>[a-z0-9][a-z0-9-]+)\s*\r?$", re.MULTILINE
)
VERDICT_PATTERN = re.compile(
    r"^SME verdict:[ \t]*(?P<value>[^\r\n]*)[ \t]*\r?$", re.MULTILINE
)
CORRECTION_PATTERN = re.compile(
    r"^SME corrected expectation:[ \t]*(?P<value>[^\r\n]*)[ \t]*\r?$",
    re.MULTILINE,
)
REPORT_PATTERN = re.compile(
    r"^- Report identity: `(?P<value>[0-9a-f]{64})`\r?$", re.MULTILINE
)


@dataclass(frozen=True)
class BoundedToolReviewDecision:
    case_id: str
    verdict: str
    rationale: str
    rationale_source: str


def parse_accepted_review_packet(
    markdown: str, *, approval_note: str
) -> tuple[BoundedToolReviewDecision, ...]:
    headings = list(SECTION_PATTERN.finditer(markdown))
    if not headings:
        raise ValueError("Review packet contains no bounded-tool cases")
    decisions: list[BoundedToolReviewDecision] = []
    seen: set[str] = set()
    for index, heading in enumerate(headings):
        case_id = heading.group("case_id")
        if case_id in seen:
            raise ValueError(f"Duplicate review case ID: {case_id}")
        seen.add(case_id)
        end = headings[index + 1].start() if index + 1 < len(headings) else len(markdown)
        section = markdown[heading.end() : end]
        verdict = _field(VERDICT_PATTERN, section, case_id).casefold()
        correction = _field(CORRECTION_PATTERN, section, case_id, allow_blank=True)
        if verdict != "accepted":
            raise ValueError(f"Only accepted unchanged cases can be promoted: {case_id}")
        if correction:
            raise ValueError(f"Corrected expectation requires a new manifest: {case_id}")
        decisions.append(
            BoundedToolReviewDecision(
                case_id=case_id,
                verdict=verdict,
                rationale=approval_note.strip(),
                rationale_source="chat_confirmation",
            )
        )
    if not approval_note.strip():
        raise ValueError("A durable approval note is required")
    return tuple(decisions)


def promote_reviewed_manifest(
    *,
    draft_manifest: Path,
    review_packet: Path,
    reviewed_manifest: Path,
    ledger_file: Path,
    reviewer: str,
    approval_note: str,
) -> dict:
    cases = load_eval_cases(draft_manifest)
    packet_bytes = review_packet.read_bytes()
    packet_text = packet_bytes.decode("utf-8")
    draft_hash = _sha256(draft_manifest.read_bytes())
    if f"Manifest SHA-256: `{draft_hash}`" not in packet_text:
        raise ValueError("Review packet is not bound to the draft manifest")
    report_match = REPORT_PATTERN.search(packet_text)
    if report_match is None:
        raise ValueError("Review packet report identity is missing")
    decisions = parse_accepted_review_packet(packet_text, approval_note=approval_note)
    if {item.case_id for item in decisions} != {item.case_id for item in cases}:
        raise ValueError("Review packet scope does not exactly match the draft manifest")
    reviewed_cases = tuple(
        case.model_copy(update={"review_status": "reviewed", "sme_reviewed": True})
        for case in cases
    )
    manifest_text = "".join(case.model_dump_json() + "\n" for case in reviewed_cases)
    reviewed_hash = _sha256(manifest_text.encode("utf-8"))
    ledger = {
        "schema_version": "bounded_tool_eval_review_ledger_v1",
        "reviewer": reviewer.strip(),
        "reviewed_at": datetime.now(UTC).isoformat(),
        "approval_source": "chat_confirmation",
        "approval_note": approval_note.strip(),
        "draft_manifest": str(draft_manifest),
        "draft_manifest_sha256": draft_hash,
        "review_packet": str(review_packet),
        "review_packet_sha256": _sha256(packet_bytes),
        "evaluated_report_identity_sha256": report_match.group("value"),
        "reviewed_manifest": str(reviewed_manifest),
        "reviewed_manifest_sha256": reviewed_hash,
        "summary": {"total_cases": len(cases), "accepted_cases": len(decisions)},
        "decisions": [asdict(item) for item in decisions],
    }
    identity_input = json.dumps(ledger, sort_keys=True, separators=(",", ":"))
    ledger["ledger_identity_sha256"] = _sha256(identity_input.encode("utf-8"))
    outputs = {
        reviewed_manifest: manifest_text,
        ledger_file: json.dumps(ledger, indent=2, sort_keys=True) + "\n",
    }
    existing = [str(path) for path in outputs if path.exists()]
    if existing:
        raise FileExistsError(f"Refusing to overwrite reviewed outputs: {existing}")
    for path, content in outputs.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    persisted = load_eval_cases(reviewed_manifest)
    if not all(case.review_status == "reviewed" and case.sme_reviewed for case in persisted):
        raise RuntimeError("Persisted reviewed manifest failed validation")
    return ledger


def _field(
    pattern: re.Pattern[str], section: str, case_id: str, *, allow_blank: bool = False
) -> str:
    match = pattern.search(section)
    if match is None:
        raise ValueError(f"Review field is missing for case: {case_id}")
    value = match.group("value").strip()
    if not value and not allow_blank:
        raise ValueError(f"Review field is blank for case: {case_id}")
    return value


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()
