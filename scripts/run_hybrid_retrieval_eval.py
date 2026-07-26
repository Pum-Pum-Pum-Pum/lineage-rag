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
from app.retrieval.hybrid_evaluation import (
    HybridRetrievalEvalCaseReport,
    build_hybrid_retrieval_eval_case_report,
    build_hybrid_retrieval_eval_report,
    write_hybrid_retrieval_eval_report_to_json,
)
from app.retrieval.hybrid_search import fuse_dense_and_lexical_results
from app.retrieval.lexical_search import search_lexical_artifacts
from app.retrieval.query_search import search_query_text
from app.vectorstore.qdrant_schema import create_persistent_qdrant_client


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a simple dense+lexical hybrid retrieval baseline."
    )
    parser.add_argument(
        "--eval-file",
        default="data/eval/retrieval_eval.json",
        help="Path to retrieval evaluation JSON file.",
    )
    parser.add_argument("--limit", type=int, default=5, help="Final top-k retrieval limit.")
    parser.add_argument(
        "--candidate-limit",
        type=int,
        default=10,
        help="Candidate count to retrieve from each base retriever before fusion.",
    )
    parser.add_argument(
        "--dense-weight",
        type=float,
        default=0.5,
        help="Weight applied to normalized dense scores.",
    )
    parser.add_argument(
        "--lexical-weight",
        type=float,
        default=0.5,
        help="Weight applied to normalized lexical scores.",
    )
    parser.add_argument(
        "--lexical-artifact-dir",
        default="data/processed",
        help="Directory containing persisted .retrieval_ready.json artifacts.",
    )
    parser.add_argument(
        "--output-file",
        default="data/eval/generated/hybrid_retrieval_eval_report.json",
        help="Path where hybrid retrieval evaluation report should be written.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = get_settings()
    configure_logging(settings.log_level)
    logger = get_logger("hybrid_retrieval_eval")

    cases = load_retrieval_eval_cases(args.eval_file)
    client = create_persistent_qdrant_client(settings.qdrant_local_path)
    try:
        if not client.collection_exists(settings.qdrant_collection_name):
            raise RuntimeError(
                "Qdrant collection does not exist. Run scripts/run_qdrant_indexing.py first."
            )

        case_reports: list[HybridRetrievalEvalCaseReport] = []
        for case in cases:
            dense_results = search_query_text(
                qdrant_client=client,
                collection_name=settings.qdrant_collection_name,
                query_text=case.query,
                embedding_model=settings.openai_embedding_model,
                limit=args.candidate_limit,
                document_family=case.filters.document_family,
                release_label=case.filters.release_label,
                source_kind=case.filters.source_kind,
            )
            lexical_results = search_lexical_artifacts(
                artifact_directory=args.lexical_artifact_dir,
                query_text=case.query,
                limit=args.candidate_limit,
                document_family=case.filters.document_family,
                release_label=case.filters.release_label,
                source_kind=case.filters.source_kind,
            )
            hybrid_results = fuse_dense_and_lexical_results(
                dense_results=dense_results,
                lexical_results=lexical_results,
                limit=args.limit,
                dense_weight=args.dense_weight,
                lexical_weight=args.lexical_weight,
            )

            case_report = build_hybrid_retrieval_eval_case_report(
                case=case,
                dense_results=dense_results[: args.limit],
                lexical_results=lexical_results[: args.limit],
                hybrid_results=hybrid_results,
            )
            case_reports.append(case_report)
            logger.info(
                "Case %s | outcome=%s | dense=%s | lexical=%s | hybrid=%s | hybrid_results=%s",
                case.case_id,
                case_report.hybrid_outcome,
                case_report.dense_evaluation.passed,
                case_report.lexical_evaluation.passed,
                case_report.hybrid_evaluation.passed,
                case_report.hybrid_evaluation.result_count,
            )
            _log_top_result(logger, case.case_id, hybrid_results)

        report = build_hybrid_retrieval_eval_report(case_reports)
        output_path = write_hybrid_retrieval_eval_report_to_json(report, args.output_file)
        logger.info(
            "Hybrid retrieval evaluation complete | dense=%s | lexical=%s | hybrid=%s | total=%s",
            report.dense_passed_count,
            report.lexical_passed_count,
            report.hybrid_passed_count,
            report.total_cases,
        )
        logger.info("Wrote hybrid retrieval evaluation report: %s", output_path)
    finally:
        client.close()


def _log_top_result(logger, case_id: str, results: list[object]) -> None:
    if not results:
        logger.info("Case %s hybrid top result | none", case_id)
        return

    top = results[0]
    payload = top.payload  # type: ignore[attr-defined]
    text = str(payload.get("text", ""))
    logger.info(
        "Case %s hybrid top result | score=%.4f | dense_score=%s | lexical_score=%s | release=%s | source=%s | unit=%s | text=%s",
        case_id,
        top.score,  # type: ignore[attr-defined]
        payload.get("dense_score"),
        payload.get("lexical_score"),
        payload.get("release_label"),
        payload.get("source_kind"),
        payload.get("unit_id"),
        text[:250].replace("\n", " "),
    )


if __name__ == "__main__":
    main()
