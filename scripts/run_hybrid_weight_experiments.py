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
from app.retrieval.hybrid_weight_experiment import (
    DEFAULT_WEIGHT_PAIRS,
    build_hybrid_weight_experiment_report,
    build_weight_settings,
    parse_weight_pairs,
    write_hybrid_weight_experiment_report_to_json,
)
from app.retrieval.lexical_search import search_lexical_artifacts
from app.retrieval.query_search import search_query_text
from app.vectorstore.qdrant_schema import create_persistent_qdrant_client


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run hybrid dense/lexical weight experiments over the retrieval eval set."
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
        "--weights",
        default=",".join(f"{dense}:{lexical}" for dense, lexical in DEFAULT_WEIGHT_PAIRS),
        help="Comma-separated dense:lexical weight pairs, e.g. 0.8:0.2,0.5:0.5.",
    )
    parser.add_argument(
        "--lexical-artifact-dir",
        default="data/processed",
        help="Directory containing persisted .retrieval_ready.json artifacts.",
    )
    parser.add_argument(
        "--output-file",
        default="data/eval/generated/hybrid_weight_experiment_report.json",
        help="Path where hybrid weight experiment report should be written.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = get_settings()
    configure_logging(settings.log_level)
    logger = get_logger("hybrid_weight_experiments")

    cases = load_retrieval_eval_cases(args.eval_file)
    weight_settings = build_weight_settings(parse_weight_pairs(args.weights))
    client = create_persistent_qdrant_client(settings.qdrant_local_path)
    if not client.collection_exists(settings.qdrant_collection_name):
        client.close()
        raise RuntimeError(
            "Qdrant collection does not exist. Run scripts/run_qdrant_indexing.py first."
        )

    dense_results_by_case_id = {}
    lexical_results_by_case_id = {}
    for case in cases:
        dense_results_by_case_id[case.case_id] = search_query_text(
            qdrant_client=client,
            collection_name=settings.qdrant_collection_name,
            query_text=case.query,
            embedding_model=settings.openai_embedding_model,
            limit=args.candidate_limit,
            document_family=case.filters.document_family,
            release_label=case.filters.release_label,
            source_kind=case.filters.source_kind,
        )
        lexical_results_by_case_id[case.case_id] = search_lexical_artifacts(
            artifact_directory=args.lexical_artifact_dir,
            query_text=case.query,
            limit=args.candidate_limit,
            document_family=case.filters.document_family,
            release_label=case.filters.release_label,
            source_kind=case.filters.source_kind,
        )

    report = build_hybrid_weight_experiment_report(
        cases=cases,
        dense_results_by_case_id=dense_results_by_case_id,
        lexical_results_by_case_id=lexical_results_by_case_id,
        weight_settings=weight_settings,
        limit=args.limit,
    )
    output_path = write_hybrid_weight_experiment_report_to_json(report, args.output_file)

    logger.info(
        "Hybrid weight experiment complete | settings=%s | best=%s",
        report.total_settings,
        report.best_setting_labels,
    )
    for setting_report in report.settings:
        logger.info(
            "Setting %s | expected_outcomes=%s/%s | hybrid_passed=%s | unsafe_expected_failure_pass=%s | unexpected_failure=%s | outcomes=%s",
            setting_report.setting.label,
            setting_report.expected_outcome_count,
            setting_report.total_cases,
            setting_report.hybrid_passed_count,
            setting_report.unsafe_expected_failure_pass_count,
            setting_report.unexpected_failure_count,
            setting_report.outcome_counts,
        )
    logger.info("Wrote hybrid weight experiment report: %s", output_path)
    client.close()


if __name__ == "__main__":
    main()