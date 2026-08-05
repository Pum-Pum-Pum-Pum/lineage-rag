from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from dataclasses import dataclass


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.core.config import get_settings
from app.core.logging import configure_logging, get_logger
from app.llm.fdd_grounded_evaluation import (
    FddGroundedEvalCase,
    FddGroundedEvalResult,
    evaluate_fdd_grounded_response,
    load_fdd_grounded_eval_cases,
    require_reviewed_cases,
    write_fdd_grounded_eval_report,
)
from app.llm.answer_contract import Citation, GroundedAnswerResponse
from app.retrieval.retrieval_config import build_retrieval_runtime_config
from app.services.answer_orchestration import run_grounded_answer_query
from app.vectorstore.qdrant_schema import create_persistent_qdrant_client


@dataclass(frozen=True)
class EvaluationTarget:
    collection_name: str
    lexical_artifact_directory: Path


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
        "--case-id",
        action="append",
        default=[],
        help="Repeat to run exact named cases instead of the full manifest.",
    )
    parser.add_argument(
        "--allow-unreviewed",
        action="store_true",
        help="Allow an explicitly labelled draft baseline; never treat it as a release-quality gate.",
    )
    parser.add_argument(
        "--collection-name",
        default=None,
        help=(
            "Optional isolated Qdrant collection override. It must be provided together "
            "with --lexical-artifact-directory to prevent mixed-generation hybrid evaluation."
        ),
    )
    parser.add_argument(
        "--lexical-artifact-directory",
        type=Path,
        default=None,
        help=(
            "Optional retrieval-ready artifact directory paired with --collection-name. "
            "For v4 use data/indexes/functional_specs_v4/processed."
        ),
    )
    parser.add_argument(
        "--resume-trace-directory",
        type=Path,
        default=None,
        help=(
            "Optional directory of prior answer traces from an interrupted run. Valid traces "
            "are scored and reused; only missing case IDs make new model calls."
        ),
    )
    parser.add_argument("--dry-run", action="store_true", help="Validate and list scope without calling OpenAI or Qdrant.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    settings = get_settings()
    configure_logging(settings.log_level)
    logger = get_logger("fdd_grounded_eval")
    cases = load_fdd_grounded_eval_cases(args.eval_file)
    if args.case_id:
        cases_by_id = {case.case_id: case for case in cases}
        missing_case_ids = [case_id for case_id in args.case_id if case_id not in cases_by_id]
        if missing_case_ids:
            raise ValueError(f"Unknown evaluation case IDs: {missing_case_ids}")
        cases = [cases_by_id[case_id] for case_id in args.case_id]
    elif args.max_cases is not None:
        cases = cases[: args.max_cases]
    require_reviewed_cases(cases, allow_unreviewed=args.allow_unreviewed)
    target = resolve_evaluation_target(args=args, settings=settings)
    if not target.lexical_artifact_directory.is_dir():
        raise FileNotFoundError(
            "Lexical artifact directory does not exist: "
            f"{target.lexical_artifact_directory}"
        )
    resumed_results = (
        load_resumed_evaluation_results(cases, args.resume_trace_directory)
        if args.resume_trace_directory is not None
        else {}
    )
    cases_to_run = [case for case in cases if case.case_id not in resumed_results]

    reviewed_count = sum(case.sme_reviewed for case in cases)
    logger.info(
        "FDD grounded evaluation planned | cases=%s | resumed=%s | remaining=%s | reviewed=%s | "
        "draft=%s | collection=%s | limit=%s",
        len(cases),
        len(resumed_results),
        len(cases_to_run),
        reviewed_count,
        args.allow_unreviewed,
        target.collection_name,
        args.limit,
    )
    if args.dry_run:
        for case in cases_to_run:
            logger.info("DRY RUN case=%s | abstain=%s | question=%s", case.case_id, case.should_abstain, case.question)
        return

    retrieval_config = build_retrieval_runtime_config(settings)
    client = None
    run_id = datetime.now(UTC).strftime("fdd-grounded-eval-%Y%m%dT%H%M%SZ")
    trace_directory = settings.exports_dir / "evaluations" / run_id / "answer_traces"
    results_by_case_id = dict(resumed_results)
    total_estimated_cost = 0.0
    try:
        if retrieval_config.retrieval_mode in {"dense", "hybrid"}:
            client = create_persistent_qdrant_client(settings.qdrant_local_path)
            if not client.collection_exists(target.collection_name):
                raise RuntimeError(
                    f"Configured Qdrant collection does not exist: {target.collection_name}"
                )

        for case in cases_to_run:
            orchestration = run_grounded_answer_query(
                qdrant_client=client,
                collection_name=target.collection_name,
                query_text=case.question,
                embedding_model=settings.openai_embedding_model,
                retrieval_config=retrieval_config,
                lexical_artifact_directory=target.lexical_artifact_directory,
                trace_output_directory=trace_directory,
                limit=args.limit,
                min_results=1,
                min_top_score=settings.retrieval_min_top_score,
                request_id=f"{run_id}-{case.case_id}",
            )
            result = evaluate_fdd_grounded_response(case, orchestration.answer_response)
            results_by_case_id[case.case_id] = result
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

    results = [results_by_case_id[case.case_id] for case in cases]
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
            "qdrant_collection_name": target.collection_name,
            "lexical_artifact_directory": str(target.lexical_artifact_directory),
            "retrieval_mode": retrieval_config.retrieval_mode,
            "retrieval_limit": args.limit,
            "draft_baseline": bool(not reviewed_count == len(cases)),
            "allow_unreviewed": args.allow_unreviewed,
            "resumed_case_count": len(resumed_results),
            "resume_trace_directory": (
                str(args.resume_trace_directory) if args.resume_trace_directory is not None else None
            ),
            "estimated_llm_cost": total_estimated_cost,
        },
        results=results,
    )


def resolve_evaluation_target(*, args: argparse.Namespace, settings) -> EvaluationTarget:
    """Resolve one coherent dense/lexical generation for evaluation.

    The live defaults remain supported. A collection override is deliberately
    inseparable from an artifact-directory override because hybrid evaluation
    with dense evidence from one generation and lexical evidence from another
    produces invalid quality evidence.
    """

    has_collection_override = args.collection_name is not None
    has_lexical_override = args.lexical_artifact_directory is not None
    if has_collection_override != has_lexical_override:
        raise ValueError(
            "--collection-name and --lexical-artifact-directory must be supplied together "
            "for an isolated staged evaluation."
        )
    return EvaluationTarget(
        collection_name=args.collection_name or settings.qdrant_collection_name,
        lexical_artifact_directory=(
            args.lexical_artifact_directory
            if args.lexical_artifact_directory is not None
            else settings.processed_dir
        ),
    )


def load_resumed_evaluation_results(
    cases: list[FddGroundedEvalCase],
    trace_directory: Path,
) -> dict[str, FddGroundedEvalResult]:
    """Load exactly one validated prior trace per case from an interrupted run."""

    if not trace_directory.is_dir():
        raise FileNotFoundError(f"Resume trace directory does not exist: {trace_directory}")

    results: dict[str, FddGroundedEvalResult] = {}
    for case in cases:
        matching_paths = sorted(trace_directory.glob(f"*-{case.case_id}.json"))
        if not matching_paths:
            continue
        if len(matching_paths) != 1:
            raise RuntimeError(
                f"Expected at most one resume trace for case {case.case_id}, found {len(matching_paths)}."
            )
        payload = json.loads(matching_paths[0].read_text(encoding="utf-8"))
        if payload.get("query") != case.question:
            raise RuntimeError(
                f"Resume trace query does not match evaluation case {case.case_id}: {matching_paths[0]}"
            )
        answer_payload = payload.get("answer_response")
        if not isinstance(answer_payload, dict) or answer_payload.get("query") != case.question:
            raise RuntimeError(
                f"Resume trace has an invalid answer response for case {case.case_id}: {matching_paths[0]}"
            )
        citations_payload = answer_payload.get("citations")
        if not isinstance(citations_payload, list):
            raise RuntimeError(
                f"Resume trace citations are invalid for case {case.case_id}: {matching_paths[0]}"
            )
        try:
            response = GroundedAnswerResponse(
                query=answer_payload["query"],
                answer=answer_payload["answer"],
                is_answered=answer_payload["is_answered"],
                refusal_reason=answer_payload.get("refusal_reason"),
                citations=[Citation(**citation) for citation in citations_payload],
            )
        except (KeyError, TypeError) as error:
            raise RuntimeError(
                f"Resume trace has an invalid answer contract for case {case.case_id}: {matching_paths[0]}"
            ) from error
        results[case.case_id] = evaluate_fdd_grounded_response(case, response)
    return results
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
