from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.core.config import get_settings
from app.core.logging import configure_logging, get_logger
from app.retrieval.evaluation import load_retrieval_eval_cases
from app.retrieval.lexical_search import search_lexical_artifacts
from app.retrieval.query_search import search_query_text
from app.retrieval.retrieval_comparison import (
    RetrievalComparisonCaseReport,
    build_retrieval_comparison_case_report,
    build_retrieval_comparison_report,
    write_retrieval_comparison_report_to_json,
)
from app.vectorstore.qdrant_schema import create_persistent_qdrant_client


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare dense Qdrant retrieval against lexical artifact retrieval."
    )
    parser.add_argument(
        "--eval-file",
        default="data/eval/retrieval_eval.json",
        help="Path to retrieval evaluation JSON file.",
    )
    parser.add_argument("--limit", type=int, default=5, help="Top-k retrieval limit.")
    parser.add_argument(
        "--lexical-artifact-dir",
        default="data/processed",
        help="Directory containing persisted .retrieval_ready.json artifacts.",
    )
    parser.add_argument(
        "--output-file",
        default="data/eval/generated/retrieval_comparison_report.json",
        help="Path where the dense-vs-lexical comparison report should be written.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = get_settings()
    configure_logging(settings.log_level)
    logger = get_logger("retrieval_comparison")

    cases = load_retrieval_eval_cases(args.eval_file)
    client = create_persistent_qdrant_client(settings.qdrant_local_path)
    try:
        if not client.collection_exists(settings.qdrant_collection_name):
            raise RuntimeError(
                "Qdrant collection does not exist. Run scripts/run_qdrant_indexing.py first."
            )

        case_reports: list[RetrievalComparisonCaseReport] = []
        for case in cases:
            dense_results = search_query_text(
                qdrant_client=client,
                collection_name=settings.qdrant_collection_name,
                query_text=case.query,
                embedding_model=settings.openai_embedding_model,
                limit=args.limit,
                document_family=case.filters.document_family,
                release_label=case.filters.release_label,
                source_kind=case.filters.source_kind,
            )
            lexical_results = search_lexical_artifacts(
                artifact_directory=args.lexical_artifact_dir,
                query_text=case.query,
                limit=args.limit,
                document_family=case.filters.document_family,
                release_label=case.filters.release_label,
                source_kind=case.filters.source_kind,
            )

            case_report = build_retrieval_comparison_case_report(
                case=case,
                dense_results=dense_results,
                lexical_results=lexical_results,
            )
            case_reports.append(case_report)

            logger.info(
                "Case %s | outcome=%s | dense_passed=%s | lexical_passed=%s | dense_results=%s | lexical_results=%s",
                case.case_id,
                case_report.comparison_outcome,
                case_report.dense_evaluation.passed,
                case_report.lexical_evaluation.passed,
                case_report.dense_evaluation.result_count,
                case_report.lexical_evaluation.result_count,
            )
            _log_top_result(logger, case.case_id, "dense", dense_results)
            _log_top_result(logger, case.case_id, "lexical", lexical_results)

        report = build_retrieval_comparison_report(case_reports)
        output_path = write_retrieval_comparison_report_to_json(report, args.output_file)
        logger.info(
            "Retrieval comparison complete | dense_passed=%s | lexical_passed=%s | both_pass=%s | dense_only=%s | lexical_only=%s | both_fail=%s | total=%s",
            report.dense_passed_count,
            report.lexical_passed_count,
            report.both_pass_count,
            report.dense_only_count,
            report.lexical_only_count,
            report.both_fail_count,
            report.total_cases,
        )
        logger.info("Wrote retrieval comparison report: %s", output_path)
    finally:
        client.close()


def _log_top_result(logger, case_id: str, retrieval_method: str, results: list[object]) -> None:
    if not results:
        logger.info("Case %s %s top result | none", case_id, retrieval_method)
        return

    top = results[0]
    payload = top.payload  # type: ignore[attr-defined]
    text = str(payload.get("text", ""))
    logger.info(
        "Case %s %s top result | score=%.4f | release=%s | source=%s | unit=%s | text=%s",
        case_id,
        retrieval_method,
        top.score,  # type: ignore[attr-defined]
        payload.get("release_label"),
        payload.get("source_kind"),
        payload.get("unit_id"),
        text[:250].replace("\n", " "),
    )


if __name__ == "__main__":
    main()
