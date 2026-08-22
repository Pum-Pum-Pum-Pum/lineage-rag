from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import UTC, datetime
from pathlib import Path


SECTION = re.compile(
    r"^##\s+\d+\.\s+(?P<case_id>[a-z0-9][a-z0-9-]+)\s*$", re.MULTILINE
)
VERDICT = re.compile(r"^SME verdict:\s*(?P<value>[^\r\n]*)", re.MULTILINE)
RATIONALE = re.compile(r"^SME rationale:\s*(?P<value>[^\r\n]*)", re.MULTILINE)
FOLLOW_UP = re.compile(r"^Required follow-up:\s*(?P<value>[^\r\n]*)", re.MULTILINE)
STRUCTURAL = re.compile(r"^Structural result:\s*\*\*(?P<value>pass|fail)\*\*", re.MULTILINE)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Import semantic SME decisions without rewriting machine results."
    )
    parser.add_argument("--run-state", type=Path, required=True)
    parser.add_argument("--review-file", type=Path, required=True)
    parser.add_argument("--reviewer", required=True)
    parser.add_argument("--global-approval-note", required=True)
    parser.add_argument("--accepted-override", action="append", default=[])
    parser.add_argument("--output-file", type=Path, required=True)
    args = parser.parse_args()

    run = json.loads(args.run_state.read_text(encoding="utf-8"))
    packet = args.review_file.read_text(encoding="utf-8")
    packet_decisions = _parse_packet(packet)
    overrides = _parse_overrides(args.accepted_override)
    run_cases = {item["case_id"]: item for item in run["cases"]}
    if set(packet_decisions) != set(run_cases):
        raise ValueError("SME packet and paid run case scopes differ")
    if not set(overrides).issubset(run_cases):
        raise ValueError("SME override names an unknown case")

    decisions = []
    trace_hashes = {}
    structural_passes = 0
    semantic_acceptances = 0
    display_drifts = 0
    for case_id, observed in run_cases.items():
        trace = _resolve_trace(args.run_state.parent, observed["trace"])
        trace_hashes[case_id] = _sha256(trace)
        submitted = packet_decisions[case_id]
        observed_pass = bool(observed["structural_passed"])
        structural_passes += int(observed_pass)
        displayed_pass = submitted["displayed_structural_result"] == "pass"
        display_drift = displayed_pass != observed_pass
        display_drifts += int(display_drift)
        verdict = overrides.get(case_id, submitted["verdict"])
        semantic_accepted = verdict == "accepted"
        semantic_acceptances += int(semantic_accepted)
        rationale = submitted["rationale"] or args.global_approval_note.strip()
        if not rationale:
            raise ValueError(f"Missing SME rationale for {case_id}")
        decisions.append(
            {
                "case_id": case_id,
                "observed_structural_passed": observed_pass,
                "packet_displayed_structural_result": submitted[
                    "displayed_structural_result"
                ],
                "packet_display_drift": display_drift,
                "submitted_verdict": submitted["verdict"],
                "normalized_semantic_verdict": verdict,
                "semantic_accepted": semantic_accepted,
                "rationale": rationale,
                "rationale_source": (
                    "packet" if submitted["rationale"] else "global_approval_note"
                ),
                "required_follow_up": submitted["required_follow_up"],
                "verdict_override_source": (
                    "chat_confirmation" if case_id in overrides else None
                ),
            }
        )

    if semantic_acceptances != len(run_cases):
        raise ValueError("Every paid answer must be semantically accepted for this ledger")
    ledger = {
        "schema_version": "code_combined_answer_review_ledger_v1",
        "reviewer": args.reviewer.strip(),
        "reviewed_at": datetime.now(UTC).isoformat(),
        "run_state": str(args.run_state),
        "run_state_sha256": _sha256(args.run_state),
        "review_file": str(args.review_file),
        "review_file_sha256": _sha256(args.review_file),
        "trace_sha256": trace_hashes,
        "summary": {
            "total_cases": len(run_cases),
            "observed_structural_passes": structural_passes,
            "semantic_acceptances": semantic_acceptances,
            "packet_structural_display_drifts": display_drifts,
            "semantic_review_status": "accepted",
            "activation_authorized": False,
        },
        "decisions": decisions,
    }
    canonical = json.dumps(ledger, sort_keys=True, separators=(",", ":"))
    ledger["ledger_identity_sha256"] = hashlib.sha256(canonical.encode()).hexdigest()
    if args.output_file.exists():
        raise FileExistsError(f"Refusing to overwrite review ledger: {args.output_file}")
    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    args.output_file.write_text(
        json.dumps(ledger, indent=2, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
    print(f"structural_passes={structural_passes}/{len(run_cases)}")
    print(f"semantic_acceptances={semantic_acceptances}/{len(run_cases)}")
    print(f"packet_structural_display_drifts={display_drifts}")
    print(f"ledger_identity_sha256={ledger['ledger_identity_sha256']}")
    return 0


def _parse_packet(markdown: str) -> dict[str, dict[str, str]]:
    headings = list(SECTION.finditer(markdown))
    decisions = {}
    for index, heading in enumerate(headings):
        case_id = heading.group("case_id")
        if case_id in decisions:
            raise ValueError(f"Duplicate SME case: {case_id}")
        end = headings[index + 1].start() if index + 1 < len(headings) else len(markdown)
        section = markdown[heading.end() : end]
        decisions[case_id] = {
            "verdict": _field(VERDICT, section, case_id).casefold(),
            "rationale": _field(RATIONALE, section, case_id, blank=True),
            "required_follow_up": _field(FOLLOW_UP, section, case_id, blank=True),
            "displayed_structural_result": _field(STRUCTURAL, section, case_id),
        }
    if not decisions:
        raise ValueError("SME packet contains no cases")
    return decisions


def _parse_overrides(values: list[str]) -> dict[str, str]:
    result = {}
    for value in values:
        case_id, separator, verdict = value.partition("=")
        if not separator or verdict.casefold() != "accepted" or not case_id:
            raise ValueError("Accepted overrides must use case-id=accepted")
        result[case_id] = "accepted"
    return result


def _field(pattern: re.Pattern[str], section: str, case_id: str, blank: bool = False) -> str:
    match = pattern.search(section)
    if match is None:
        raise ValueError(f"Missing review field for {case_id}")
    value = match.group("value").strip()
    if not value and not blank:
        raise ValueError(f"Blank review field for {case_id}")
    return value


def _resolve_trace(base: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else base / path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
