from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.core.config import get_settings
from app.core.logging import configure_logging, get_logger
from app.retrieval.retrieval_error_analysis import (
    analyze_retrieval_comparison_report,
    load_retrieval_comparison_report,
    write_retrieval_error_analysis_report_to_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze dense-vs-lexical retrieval comparison failure modes."
    )
    parser.add_argument(
        "--comparison-report",
        default="data/eval/generated/retrieval_comparison_report.json",
        help="Path to the dense-vs-lexical comparison report JSON.",
    )
    parser.add_argument(
        "--output-file",
        default="data/eval/generated/retrieval_error_analysis_report.json",
        help="Path where retrieval error analysis JSON should be written.",
    )
    parser.add_argument(
        "--include-both-pass",
        action="store_true",
        help="Include both-pass cases in the analysis report. Defaults to failure-focused output only.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = get_settings()
    configure_logging(settings.log_level)
    logger = get_logger("retrieval_error_analysis")

    comparison_report = load_retrieval_comparison_report(args.comparison_report)
    analysis_report = analyze_retrieval_comparison_report(
        comparison_report,
        include_both_pass=args.include_both_pass,
    )
    output_path = write_retrieval_error_analysis_report_to_json(
        analysis_report,
        args.output_file,
    )

    logger.info(
        "Retrieval error analysis complete | analyzed=%s | high=%s | medium=%s | low=%s | info=%s | labels=%s",
        analysis_report.analyzed_case_count,
        analysis_report.high_severity_count,
        analysis_report.medium_severity_count,
        analysis_report.low_severity_count,
        analysis_report.info_severity_count,
        analysis_report.label_counts,
    )
    for case in analysis_report.cases:
        logger.info(
            "Case %s | outcome=%s | severity=%s | labels=%s | action=%s",
            case.case_id,
            case.comparison_outcome,
            case.severity,
            case.root_cause_labels,
            case.recommended_next_action,
        )
    logger.info("Wrote retrieval error analysis report: %s", output_path)


if __name__ == "__main__":
    main()