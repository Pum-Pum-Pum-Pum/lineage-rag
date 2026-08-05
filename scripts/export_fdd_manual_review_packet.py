from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT_DIR = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class ManualReviewEntry:
    case_id: str
    question: str
    expected_claims: list[str]
    expected_document_ids: list[str]
    expected_release_labels: list[str]
    actual_answer: str
    is_answered: bool
    refusal_reason: str | None
    citation_document_ids: list[str | None]
    citation_release_labels: list[str | None]
    structural_failures: list[str]
    trace_path: str


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export failed FDD draft-evaluation cases into an SME manual-review Markdown packet."
    )
    parser.add_argument(
        "--report-file",
        type=Path,
        required=True,
        help="Completed FDD grounded-evaluation report JSON.",
    )
    parser.add_argument(
        "--eval-file",
        type=Path,
        default=ROOT_DIR / "data" / "evaluations" / "fdd_grounded_eval_v1.jsonl",
        help="Source evaluation JSONL containing expected claims and evidence requirements.",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        required=True,
        help="Markdown packet path to create.",
    )
    return parser.parse_args(argv)


def load_failed_review_entries(report_path: Path, eval_path: Path) -> list[ManualReviewEntry]:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    eval_cases = _load_eval_cases(eval_path)
    trace_directories = _resolve_trace_directories(report, report_path)
    entries: list[ManualReviewEntry] = []

    for result in report.get("cases", []):
        if result.get("structural_passed"):
            continue
        case_id = str(result["case_id"])
        expected = eval_cases.get(case_id)
        if expected is None:
            raise RuntimeError(f"Failed report case is absent from evaluation manifest: {case_id}")
        trace_path = _find_one_trace(case_id, trace_directories)
        entries.append(
            ManualReviewEntry(
                case_id=case_id,
                question=str(expected["question"]),
                expected_claims=list(expected["expected_claims"]),
                expected_document_ids=list(expected["required_citation_document_ids"]),
                expected_release_labels=list(expected["expected_release_labels"]),
                actual_answer=str(result["answer"]),
                is_answered=bool(result["is_answered"]),
                refusal_reason=result.get("refusal_reason"),
                citation_document_ids=list(result.get("citation_document_ids", [])),
                citation_release_labels=list(result.get("citation_release_labels", [])),
                structural_failures=list(result.get("failures", [])),
                trace_path=str(trace_path),
            )
        )
    return entries


def write_manual_review_packet(entries: list[ManualReviewEntry], output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# FDD v4 draft baseline — SME manual review packet",
        "",
        "This packet contains only deterministic evaluation failures. It is not a release approval, "
        "and its preliminary signals are not SME verdicts.",
        "",
        "For every case, review the cited source text in the trace, then choose one outcome:",
        "`expected_case_incorrect`, `retrieval_or_release_gap`, `citation_contract_gap`, "
        "`correct_safe_refusal`, or `other`.",
        "",
    ]
    for entry in entries:
        lines.extend(
            [
                f"## {entry.case_id}",
                "",
                f"Question: {entry.question}",
                "",
                "Expected claims:",
                *_bullets(entry.expected_claims),
                "",
                f"Expected document IDs: {', '.join(entry.expected_document_ids) or '(none)'}",
                f"Expected releases: {', '.join(entry.expected_release_labels) or '(none)'}",
                "",
                f"Answer state: {'answered' if entry.is_answered else 'refused'}",
                f"Refusal reason: {entry.refusal_reason or '(none)'}",
                f"Returned document IDs: {', '.join(str(value) for value in entry.citation_document_ids) or '(none)'}",
                f"Returned releases: {', '.join(str(value) for value in entry.citation_release_labels) or '(none)'}",
                "",
                "Deterministic failures:",
                *_bullets(entry.structural_failures),
                "",
                "Actual answer:",
                "",
                entry.actual_answer,
                "",
                f"Trace: `{entry.trace_path}`",
                "",
                "SME verdict: `pending`",
                "SME rationale: `pending`",
                "Required follow-up: `pending`",
                "",
            ]
        )
    output_path.write_text("\n".join(lines), encoding="utf-8")
    return output_path


def _load_eval_cases(path: Path) -> dict[str, dict[str, Any]]:
    cases: dict[str, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        case = json.loads(line)
        cases[str(case["case_id"])] = case
    return cases


def _resolve_trace_directories(report: dict[str, Any], report_path: Path) -> list[Path]:
    metadata = report.get("metadata", {})
    directories = [report_path.parent / str(metadata["run_id"]) / "answer_traces"]
    resume_directory = metadata.get("resume_trace_directory")
    if resume_directory:
        directories.append(Path(resume_directory))
    return directories


def _find_one_trace(case_id: str, directories: list[Path]) -> Path:
    matching_paths = [
        path
        for directory in directories
        if directory.is_dir()
        for path in directory.glob(f"*-{case_id}.json")
    ]
    if len(matching_paths) != 1:
        raise RuntimeError(
            f"Expected exactly one trace for failed case {case_id}, found {len(matching_paths)}."
        )
    return matching_paths[0]


def _bullets(values: list[str]) -> list[str]:
    return [f"- {value}" for value in values] or ["- (none)"]


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    entries = load_failed_review_entries(args.report_file, args.eval_file)
    output_path = write_manual_review_packet(entries, args.output_file)
    print(f"manual_review_cases={len(entries)}")
    print(f"packet={output_path}")


if __name__ == "__main__":
    main()
