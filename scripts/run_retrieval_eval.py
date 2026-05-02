from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.core.config import get_settings
from app.core.logging import configure_logging, get_logger
from app.retrieval.evaluation import (
    RetrievalEvalCaseReport,
    build_retrieval_eval_report,
    evaluate_retrieval_results,
    load_retrieval_eval_cases,
    serialize_search_result,
    write_retrieval_eval_report_to_json,
)
from app.retrieval.query_search import search_query_text
from app.vectorstore.qdrant_schema import create_persistent_qdrant_client


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a small retrieval evaluation set.")
    parser.add_argument(
        "--eval-file",
        default="data/eval/retrieval_eval.json",
        help="Path to retrieval evaluation JSON file.",
    )
    parser.add_argument("--limit", type=int, default=5, help="Top-k retrieval limit.")
    parser.add_argument(
        "--output-file",
        default="data/eval/generated/retrieval_eval_report.json",
        help="Path where the retrieval evaluation report JSON should be written.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = get_settings()
    configure_logging(settings.log_level)
    logger = get_logger("retrieval_eval")

    cases = load_retrieval_eval_cases(args.eval_file)
    client = create_persistent_qdrant_client(settings.qdrant_local_path)

    passed_count = 0
    expected_count = 0
    case_reports: list[RetrievalEvalCaseReport] = []
    for case in cases:
        results = search_query_text(
            qdrant_client=client,
            collection_name=settings.qdrant_collection_name,
            query_text=case.query,
            embedding_model=settings.openai_embedding_model,
            limit=args.limit,
            document_family=case.filters.document_family,
            release_label=case.filters.release_label,
            source_kind=case.filters.source_kind,
        )
        evaluation = evaluate_retrieval_results(case, results)
        passed_count += int(evaluation.passed)
        expected_count += int(evaluation.outcome_as_expected)
        case_reports.append(
            RetrievalEvalCaseReport(
                case=case,
                evaluation=evaluation,
                top_results=[serialize_search_result(result) for result in results],
            )
        )

        logger.info(
            "Case %s | passed=%s | expected_to_pass=%s | outcome_as_expected=%s | results=%s",
            evaluation.case_id,
            evaluation.passed,
            evaluation.expected_to_pass,
            evaluation.outcome_as_expected,
            evaluation.result_count,
        )
        for failure in evaluation.failures:
            logger.warning("Case %s failure: %s", evaluation.case_id, failure)
        if results:
            top = results[0]
            logger.info(
                "Case %s top result | score=%.4f | release=%s | source=%s | text=%s",
                evaluation.case_id,
                top.score,
                top.payload.get("release_label"),
                top.payload.get("source_kind"),
                str(top.payload.get("text", ""))[:250].replace("\n", " "),
            )

    logger.info(
        "Retrieval evaluation complete | passed=%s | expected_outcomes=%s | total=%s",
        passed_count,
        expected_count,
        len(cases),
    )
    report = build_retrieval_eval_report(case_reports)
    output_path = write_retrieval_eval_report_to_json(report, args.output_file)
    logger.info("Wrote retrieval evaluation report: %s", output_path)
    client.close()


if __name__ == "__main__":
    main()
