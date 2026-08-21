from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.fdd_code_lineage.evaluation import load_code_combined_eval_cases


SECTION_PATTERN = re.compile(
    r"^##\s+\d+\.\s+(?P<case_id>[a-z0-9][a-z0-9-]+)\s*$", re.MULTILINE
)
VERDICT_PATTERN = re.compile(
    r"^SME verdict:[ \t]*(?P<value>[^\r\n]*)[ \t]*\r?$", re.MULTILINE
)
CORRECTION_PATTERN = re.compile(
    r"^SME corrected expectations:[ \t]*(?P<value>[^\r\n]*)[ \t]*\r?$",
    re.MULTILINE,
)
RATIONALE_PATTERN = re.compile(
    r"^SME rationale:[ \t]*(?P<value>[^\r\n]*)[ \t]*\r?$", re.MULTILINE
)


@dataclass(frozen=True)
class ReviewDecision:
    case_id: str
    verdict: str
    rationale: str
    rationale_source: str


def parse_review_packet(
    markdown: str, *, global_approval_note: str
) -> list[ReviewDecision]:
    headings = list(SECTION_PATTERN.finditer(markdown))
    if not headings:
        raise ValueError("Review packet contains no recognized case headings")
    decisions: list[ReviewDecision] = []
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
        rationale = _field(RATIONALE_PATTERN, section, case_id, allow_blank=True)
        if verdict != "accepted":
            raise ValueError(
                f"Only accepted cases can be promoted without corrected JSON: {case_id}"
            )
        if correction:
            raise ValueError(
                f"Accepted case contains corrected expectations that were not applied: {case_id}"
            )
        rationale_source = "packet"
        if not rationale:
            rationale = global_approval_note.strip()
            rationale_source = "global_approval_note"
        if not rationale:
            raise ValueError(f"Review rationale is blank for case: {case_id}")
        decisions.append(
            ReviewDecision(
                case_id=case_id,
                verdict=verdict,
                rationale=rationale,
                rationale_source=rationale_source,
            )
        )
    return decisions


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Promote accepted code/combined draft eval cases into separate reviewed "
            "manifests and a hash-bound review ledger."
        )
    )
    parser.add_argument("--eval-file", type=Path, action="append", required=True)
    parser.add_argument("--review-file", type=Path, required=True)
    parser.add_argument("--reviewer", required=True)
    parser.add_argument("--global-approval-note", required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--ledger-file", type=Path, required=True)
    args = parser.parse_args(argv)

    cases_by_file = {
        path: load_code_combined_eval_cases(path) for path in args.eval_file
    }
    all_cases = [case for cases in cases_by_file.values() for case in cases]
    case_ids = [case.case_id for case in all_cases]
    if len(case_ids) != len(set(case_ids)):
        raise ValueError("Draft manifests contain duplicate case IDs")
    packet_bytes = args.review_file.read_bytes()
    packet_text = packet_bytes.decode("utf-8")
    _verify_packet_bindings(packet_text, args.eval_file)
    decisions = parse_review_packet(
        packet_text, global_approval_note=args.global_approval_note
    )
    decision_ids = {item.case_id for item in decisions}
    if decision_ids != set(case_ids):
        raise ValueError(
            "Review scope does not exactly match draft manifests; "
            f"missing={sorted(set(case_ids) - decision_ids)}, "
            f"extra={sorted(decision_ids - set(case_ids))}"
        )

    outputs: dict[Path, str] = {}
    reviewed_manifest_bindings: dict[str, str] = {}
    for source, cases in cases_by_file.items():
        name = (
            source.name.replace("_draft.jsonl", "_reviewed.jsonl")
            if source.name.endswith("_draft.jsonl")
            else f"{source.stem}_reviewed.jsonl"
        )
        target = args.output_directory / name
        content = "".join(
            case.model_copy(
                update={"sme_reviewed": True, "review_status": "reviewed"}
            ).model_dump_json()
            + "\n"
            for case in cases
        )
        outputs[target] = content
        reviewed_manifest_bindings[str(target)] = hashlib.sha256(
            content.encode("utf-8")
        ).hexdigest()

    ledger = {
        "schema_version": "code_combined_eval_review_ledger_v1",
        "reviewer": args.reviewer.strip(),
        "reviewed_at": datetime.now(UTC).isoformat(),
        "review_file": str(args.review_file),
        "review_file_sha256": hashlib.sha256(packet_bytes).hexdigest(),
        "source_manifests": {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in args.eval_file
        },
        "reviewed_manifests": reviewed_manifest_bindings,
        "summary": {
            "total_cases": len(decisions),
            "accepted_cases": len(decisions),
            "corrected_cases": 0,
            "removed_cases": 0,
            "review_status": "reviewed",
        },
        "decisions": [asdict(item) for item in decisions],
    }
    ledger_without_identity = json.dumps(
        ledger, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    )
    ledger["ledger_identity_sha256"] = hashlib.sha256(
        ledger_without_identity.encode("utf-8")
    ).hexdigest()
    outputs[args.ledger_file] = json.dumps(
        ledger, indent=2, ensure_ascii=False, sort_keys=True
    )

    existing = [str(path) for path in outputs if path.exists()]
    if existing:
        raise FileExistsError(f"Refusing to overwrite reviewed outputs: {existing}")
    for path, content in outputs.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")

    for path in reviewed_manifest_bindings:
        load_code_combined_eval_cases(Path(path))
    print(f"reviewed_cases={len(decisions)}")
    print(f"reviewed_manifests={len(reviewed_manifest_bindings)}")
    print(f"review_file_sha256={ledger['review_file_sha256']}")
    print(f"ledger_identity_sha256={ledger['ledger_identity_sha256']}")
    print(f"ledger={args.ledger_file}")
    return 0


def _verify_packet_bindings(markdown: str, paths: list[Path]) -> None:
    for path in paths:
        expected_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        if expected_hash not in markdown:
            raise ValueError(f"Review packet is not bound to manifest: {path}")


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


if __name__ == "__main__":
    raise SystemExit(main())
