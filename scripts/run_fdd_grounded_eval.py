from __future__ import annotations

import argparse
import sys
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.core.config import get_settings
from app.core.logging import configure_logging, get_logger
from app.llm.fdd_grounded_evaluation import (
    evaluate_fdd_grounded_response,
    load_fdd_grounded_eval_cases,
    require_reviewed_cases,
    write_fdd_grounded_eval_report,
)
from app.retrieval.retrieval_config import build_retrieval_runtime_config
from app.services.answer_orchestration import run_grounded_answer_query
from app.vectorstore.qdrant_schema import create_persistent_qdrant_client


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the JSONL FDD grounded-answer evaluation with deterministic citation checks."
    )
    parser.add_argument(
        "--eval-file",
        default="data/evaluations/fdd_grounded_eval_v1.jsonl",
        help="Path to the FDD grounded evaluation JSONL file.",
    )
    parser.add_argument(
        "--output-file",
        default=None,
        help="Report path. Defaults to a timestamped file under data/exports/evaluations/.",
    )
    parser.add_argument("--limit", type=_positive_int, default=10, help="Retrieval evidence limit. Default: 10.")
    parser.add_argument("--max-cases", type=_positive_int, default=None, help="Optional bounded case count.")
    parser.add_argument(
        "--allow-unreviewed",
        action="store_true",
        help="Allow an explicitly labelled draft baseline; never treat it as a release-quality gate.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Validate and list scope without calling OpenAI or Qdrant.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    settings = get_settings()
    configure_logging(settings.log_level)
    logger = get_logger("fdd_grounded_eval")
    cases = load_fdd_grounded_eval_cases(args.eval_file)
    if args.max_cases is not None:
        cases = cases[: args.max_cases]
    require_reviewed_cases(cases, allow_unreviewed=args.allow_unreviewed)

    reviewed_count = sum(case.sme_reviewed for case in cases)
    logger.info(
        "FDD grounded evaluation planned | cases=%s | reviewed=%s | draft=%s | collection=%s | limit=%s",
        len(cases),
        reviewed_count,
        args.allow_unreviewed,
        settings.qdrant_collection_name,
        args.limit,
    )
    if args.dry_run:
        for case in cases:
            logger.info("DRY RUN case=%s | abstain=%s | question=%s", case.case_id, case.should_abstain, case.question)
        return

    retrieval_config = build_retrieval_runtime_config(settings)
    client = None
    run_id = datetime.now(UTC).strftime("fdd-grounded-eval-%Y%m%dT%H%M%SZ")
    trace_directory = settings.exports_dir / "evaluations" / run_id / "answer_traces"
    results = []
    total_estimated_cost = 0.0
    try:
        if retrieval_config.retrieval_mode in {"dense", "hybrid"}:
            client = create_persistent_qdrant_client(settings.qdrant_local_path)
            if not client.collection_exists(settings.qdrant_collection_name):
                raise RuntimeError(
                    f"Configured Qdrant collection does not exist: {settings.qdrant_collection_name}"
                )

        for case in cases:
            orchestration = run_grounded_answer_query(
                qdrant_client=client,
                collection_name=settings.qdrant_collection_name,
                query_text=case.question,
                embedding_model=settings.openai_embedding_model,
                retrieval_config=retrieval_config,
                lexical_artifact_directory=settings.processed_dir,
                trace_output_directory=trace_directory,
                limit=args.limit,
                min_results=1,
                min_top_score=settings.retrieval_min_top_score,
                request_id=f"{run_id}-{case.case_id}",
            )
            result = evaluate_fdd_grounded_response(case, orchestration.answer_response)
            results.append(result)
            if orchestration.answer_response.cost is not None:
                total_estimated_cost += orchestration.answer_response.cost.total_cost
            logger.info(
                "Case %s | structural_passed=%s | answered=%s | trace=%s",
                case.case_id,
                result.structural_passed,
                result.is_answered,
                orchestration.trace_output_path,
            )
            for failure in result.failures:
                logger.warning("Case %s failure: %s", case.case_id, failure)
    finally:
        if client is not None:
            client.close()

    output_path = (
        Path(args.output_file)
        if args.output_file
        else settings.exports_dir / "evaluations" / f"{run_id}.json"
    )
    report_path = write_fdd_grounded_eval_report(
        output_path=output_path,
        report_metadata={
            "run_id": run_id,
            "eval_file": str(Path(args.eval_file)),
            "qdrant_collection_name": settings.qdrant_collection_name,
            "retrieval_mode": retrieval_config.retrieval_mode,
            "retrieval_limit": args.limit,
            "draft_baseline": bool(not reviewed_count == len(cases)),
            "allow_unreviewed": args.allow_unreviewed,
            "estimated_llm_cost": total_estimated_cost,
        },
        results=results,
    )
    structural_passes = sum(result.structural_passed for result in results)
    logger.info(
        "FDD grounded evaluation complete | structural_passed=%s | total=%s | claim_reviews_required=%s | report=%s",
        structural_passes,
        len(results),
        sum(result.claim_review_required for result in results),
        report_path,
    )


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than 0")
    return parsed


if __name__ == "__main__":
    main()
