from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.fdd_code_lineage.evaluation import load_code_combined_eval_cases


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render immutable SME review material for code/combined eval cases."
    )
    parser.add_argument("--eval-file", type=Path, action="append", required=True)
    parser.add_argument("--output-file", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.output_file.exists():
        raise FileExistsError(f"SME review packet already exists: {args.output_file}")
    cases = []
    seen: set[str] = set()
    for path in args.eval_file:
        for case in load_code_combined_eval_cases(path):
            if case.case_id in seen:
                raise ValueError(f"Duplicate case ID across manifests: {case.case_id}")
            seen.add(case.case_id)
            cases.append(case)

    lines = [
        "# Code and combined RAG evaluation SME review packet",
        "",
        "Release labels are intentionally absent from ordinary user questions. ",
        "Expected document IDs remain hidden evaluation metadata.",
        "",
        f"- Cases: `{len(cases)}`",
        "- Status: `draft`",
        "- External API calls used to create this packet: `0`",
        "- Manifest bindings:",
    ]
    lines.extend(
        f"  - `{path}`: `{hashlib.sha256(path.read_bytes()).hexdigest()}`"
        for path in args.eval_file
    )
    lines.extend(
        [
            "",
            "For each case, verify that the expected source identity and requested answer state are correct. ",
            "Do not mark a case reviewed merely because lexical retrieval passed.",
            "",
        ]
    )
    for number, case in enumerate(cases, start=1):
        lines.extend(
            [
                f"## {number}. {case.case_id}",
                "",
                f"- Mode: `{case.mode}`",
                f"- Question: {case.question}",
                f"- Analysis kind: `{case.analysis_kind}`",
                f"- Should abstain: `{str(case.should_abstain).lower()}`",
                f"- Expected code paths: `{list(case.expected_code_paths)}`",
                f"- Expected code symbols: `{list(case.expected_code_symbols)}`",
                f"- Expected FDD document IDs: `{list(case.expected_fdd_document_ids)}`",
                f"- Reviewed lineage required: `{str(case.require_reviewed_lineage).lower()}`",
                f"- Expected unknown kinds: `{list(case.expected_unknown_kinds)}`",
                f"- Rationale: {case.rationale}",
                "",
                "SME verdict: accepted | corrected | remove",
                "SME corrected expectations:",
                "SME rationale:",
                "",
            ]
        )
    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    args.output_file.write_text("\n".join(lines), encoding="utf-8")
    print(f"packet={args.output_file}")
    print(f"cases={len(cases)}")
    print("external_api_calls=0")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
