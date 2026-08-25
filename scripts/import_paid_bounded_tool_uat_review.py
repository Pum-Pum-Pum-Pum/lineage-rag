from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from datetime import UTC, datetime
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


SECTION = re.compile(r"^##\s+\d+\.\s+(?P<case_id>[a-z0-9][a-z0-9-]+)\s*\r?$", re.MULTILINE)
VERDICT = re.compile(r"^SME verdict:\s*(?P<value>[^\r\n]*)", re.MULTILINE)
RATIONALE = re.compile(r"^SME rationale:\s*(?P<value>[^\r\n]*)", re.MULTILINE)
FOLLOW_UP = re.compile(r"^Required follow-up:\s*(?P<value>[^\r\n]*)", re.MULTILINE)
STRUCTURAL = re.compile(r"^Structural result:\s*\*\*(?P<value>pass|fail)\*\*", re.MULTILINE)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Import paid bounded-tool SME decisions without changing machine results."
    )
    parser.add_argument("--run-state", type=Path, required=True)
    parser.add_argument("--review-file", type=Path, required=True)
    parser.add_argument("--reviewer", required=True)
    parser.add_argument("--global-acceptance-note", required=True)
    parser.add_argument("--output-file", type=Path, required=True)
    args = parser.parse_args()
    run = json.loads(args.run_state.read_text(encoding="utf-8"))
    packet = _parse_packet(args.review_file.read_text(encoding="utf-8"))
    run_cases = {item["case_id"]: item for item in run["cases"]}
    if set(packet) != set(run_cases):
        raise ValueError("Paid run and SME packet scopes differ")
    decisions = []
    accepted = 0
    corrected = 0
    trace_hashes = {}
    for case_id, observed in run_cases.items():
        submitted = packet[case_id]
        verdict = submitted["verdict"]
        if verdict not in {"accepted", "corrected"}:
            raise ValueError(f"Unresolved SME verdict for {case_id}: {verdict}")
        accepted += int(verdict == "accepted")
        corrected += int(verdict == "corrected")
        trace = args.run_state.parent / observed["trace"]
        trace_hashes[case_id] = _sha256(trace)
        rationale = submitted["rationale"] or args.global_acceptance_note.strip()
        if not rationale:
            raise ValueError(f"SME rationale is unavailable for {case_id}")
        decisions.append(
            {
                "case_id": case_id,
                "observed_structural_passed": bool(observed["structural_passed"]),
                "packet_displayed_structural_result": submitted["structural"],
                "sme_verdict": verdict,
                "semantic_accepted": verdict == "accepted",
                "rationale": rationale,
                "rationale_source": (
                    "packet" if submitted["rationale"] else "global_acceptance_note"
                ),
                "required_follow_up": submitted["required_follow_up"],
            }
        )
    ledger = {
        "schema_version": "paid_bounded_tool_uat_review_ledger_v1",
        "reviewer": args.reviewer.strip(),
        "reviewed_at": datetime.now(UTC).isoformat(),
        "run_state": str(args.run_state),
        "run_state_sha256": _sha256(args.run_state),
        "review_file": str(args.review_file),
        "review_file_sha256": _sha256(args.review_file),
        "trace_sha256": trace_hashes,
        "summary": {
            "total_cases": len(run_cases),
            "observed_structural_passes": sum(
                bool(item["structural_passed"]) for item in run_cases.values()
            ),
            "semantic_acceptances": accepted,
            "semantic_corrections_required": corrected,
            "semantic_review_status": (
                "remediation_required" if corrected else "accepted"
            ),
            "activation_authorized": False,
            "additional_paid_requests_authorized": 0,
        },
        "decisions": decisions,
    }
    canonical = json.dumps(ledger, sort_keys=True, separators=(",", ":"))
    ledger["ledger_identity_sha256"] = hashlib.sha256(canonical.encode()).hexdigest()
    if args.output_file.exists():
        raise FileExistsError(f"Refusing to overwrite paid UAT review ledger: {args.output_file}")
    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    with args.output_file.open("w", encoding="utf-8", newline="") as handle:
        handle.write(json.dumps(ledger, indent=2, ensure_ascii=False, sort_keys=True))
    print(f"semantic_acceptances={accepted}/{len(run_cases)}")
    print(f"semantic_corrections_required={corrected}")
    print(f"ledger_identity_sha256={ledger['ledger_identity_sha256']}")
    return 0


def _parse_packet(markdown: str) -> dict[str, dict[str, str]]:
    headings = list(SECTION.finditer(markdown))
    decisions = {}
    for index, heading in enumerate(headings):
        case_id = heading.group("case_id")
        end = headings[index + 1].start() if index + 1 < len(headings) else len(markdown)
        section = markdown[heading.end() : end]
        if case_id in decisions:
            raise ValueError(f"Duplicate SME case: {case_id}")
        decisions[case_id] = {
            "verdict": _field(VERDICT, section, case_id).casefold(),
            "rationale": _field(RATIONALE, section, case_id, allow_blank=True),
            "required_follow_up": _field(FOLLOW_UP, section, case_id, allow_blank=True),
            "structural": _field(STRUCTURAL, section, case_id),
        }
    if not decisions:
        raise ValueError("SME packet contains no cases")
    return decisions


def _field(pattern, section: str, case_id: str, *, allow_blank: bool = False) -> str:
    match = pattern.search(section)
    if match is None:
        raise ValueError(f"Missing SME field for {case_id}")
    value = match.group("value").strip()
    if not value and not allow_blank:
        raise ValueError(f"Blank SME field for {case_id}")
    return value


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
