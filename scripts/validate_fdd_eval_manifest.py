from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.llm.fdd_eval_manifest_validation import (
    validate_fdd_eval_manifest,
    write_fdd_eval_manifest_report,
    write_pending_sme_review_packet,
)
from app.llm.fdd_grounded_evaluation import load_fdd_grounded_eval_cases


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate an FDD evaluation manifest against promoted lexical artifacts."
    )
    parser.add_argument(
        "--eval-file",
        type=Path,
        default=Path("data/evaluations/fdd_grounded_eval_v2.jsonl"),
    )
    parser.add_argument(
        "--artifact-directory",
        type=Path,
        default=Path("data/indexes/functional_specs_v5/processed"),
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        default=Path("data/exports/evaluations/fdd-v4-user-question-manifest-validation.json"),
    )
    parser.add_argument(
        "--review-packet",
        type=Path,
        default=Path("data/exports/evaluations/fdd-v4-user-question-pending-sme-review.md"),
    )
    parser.add_argument(
        "--allow-unreviewed",
        action="store_true",
        help="Write a diagnostic report even when SME review is pending; never marks the manifest release-ready.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cases = load_fdd_grounded_eval_cases(args.eval_file)
    report = validate_fdd_eval_manifest(
        cases=cases,
        manifest_path=args.eval_file,
        artifact_directory=args.artifact_directory,
    )
    report_path = write_fdd_eval_manifest_report(report, args.output_file)
    packet_path = write_pending_sme_review_packet(cases, args.review_packet)

    print(f"cases={report.total_cases}")
    print(f"reviewed={report.reviewed_cases}")
    print(f"pending_review={report.pending_review_cases}")
    print(f"errors={len(report.errors)}")
    print(f"gate_blockers={len(report.gate_blockers)}")
    print(f"release_gate_ready={str(report.release_gate_ready).lower()}")
    print(f"report={report_path}")
    print(f"review_packet={packet_path}")

    if report.errors:
        return 2
    if report.gate_blockers and not args.allow_unreviewed:
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
