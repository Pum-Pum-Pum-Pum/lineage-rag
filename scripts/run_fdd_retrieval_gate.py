from __future__ import annotations

import argparse
import sys
from datetime import UTC, datetime
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.core.config import get_settings
from app.llm.fdd_grounded_evaluation import (
    load_fdd_grounded_eval_cases,
    require_reviewed_cases,
)
from app.retrieval.fdd_retrieval_gate import (
    build_fdd_retrieval_gate_case_report,
    build_fdd_retrieval_gate_report,
    write_fdd_retrieval_gate_report,
)
from app.retrieval.retrieval_config import build_retrieval_runtime_config
from app.services.query_retrieval import retrieve_planned_query_evidence
from app.vectorstore.qdrant_schema import create_persistent_qdrant_client


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a production-parity retrieval-only gate for FDD evaluation cases."
    )
    parser.add_argument("--eval-file", type=Path, default=Path("data/evaluations/fdd_grounded_eval_v2.jsonl"))
    parser.add_argument("--collection-name", default=None)
    parser.add_argument("--lexical-artifact-directory", type=Path, default=None)
    parser.add_argument("--output-file", type=Path, default=None)
    parser.add_argument("--limit", type=_positive_int, default=10)
    parser.add_argument("--minimum-document-recall", type=_rate, default=0.90)
    parser.add_argument("--case-id", action="append", default=[])
    parser.add_argument("--allow-unreviewed", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    settings = get_settings()
    cases = load_fdd_grounded_eval_cases(args.eval_file)
    if args.case_id:
        by_id = {case.case_id: case for case in cases}
        missing = sorted(set(args.case_id).difference(by_id))
        if missing:
            raise ValueError(f"Unknown evaluation case IDs: {missing}")
        cases = [by_id[case_id] for case_id in args.case_id]
    require_reviewed_cases(cases, allow_unreviewed=args.allow_unreviewed)
    collection_name, lexical_directory = _resolve_target(args, settings)
    if not lexical_directory.is_dir():
        raise FileNotFoundError(f"Lexical artifact directory does not exist: {lexical_directory}")

    reviewed_manifest = all(case.sme_reviewed for case in cases)
    print(f"cases={len(cases)}")
    print(f"reviewed_manifest={str(reviewed_manifest).lower()}")
    print(f"collection={collection_name}")
    print(f"lexical_artifact_directory={lexical_directory}")
    if args.dry_run:
        return 0

    retrieval_config = build_retrieval_runtime_config(settings)
    client = None
    case_reports = []
    try:
        if retrieval_config.retrieval_mode in {"dense", "hybrid"}:
            client = create_persistent_qdrant_client(settings.qdrant_local_path)
            if not client.collection_exists(collection_name):
                raise RuntimeError(f"Configured Qdrant collection does not exist: {collection_name}")
        for case in cases:
            planned = retrieve_planned_query_evidence(
                qdrant_client=client,
                collection_name=collection_name,
                query_text=case.question,
                embedding_model=settings.openai_embedding_model,
                retrieval_config=retrieval_config,
                lexical_artifact_directory=lexical_directory,
                limit=args.limit,
            )
            case_report = build_fdd_retrieval_gate_case_report(case=case, planned=planned)
            case_reports.append(case_report)
            print(
                f"case={case.case_id} positive_passed={case_report.positive_gate_passed} "
                f"recall={case_report.document_recall_at_k}"
            )
    finally:
        if client is not None:
            client.close()

    run_id = datetime.now(UTC).strftime("fdd-retrieval-gate-%Y%m%dT%H%M%SZ")
    report = build_fdd_retrieval_gate_report(
        metadata={
            "run_id": run_id,
            "eval_file": str(args.eval_file),
            "collection_name": collection_name,
            "lexical_artifact_directory": str(lexical_directory),
            "retrieval_mode": retrieval_config.retrieval_mode,
            "hybrid_dense_weight": retrieval_config.hybrid_dense_weight,
            "hybrid_lexical_weight": retrieval_config.hybrid_lexical_weight,
            "hybrid_candidate_limit": retrieval_config.hybrid_candidate_limit,
            "fusion_method": "weighted_rrf",
            "retrieval_limit": args.limit,
            "reviewed_manifest": reviewed_manifest,
            "draft_baseline": not reviewed_manifest,
            "embedding_request_count": len(cases) if retrieval_config.retrieval_mode in {"dense", "hybrid"} else 0,
        },
        cases=case_reports,
        minimum_document_recall=args.minimum_document_recall,
    )
    output_path = args.output_file or Path("data/exports/evaluations") / f"{run_id}.json"
    write_fdd_retrieval_gate_report(report, output_path)
    summary = report["summary"]
    print(f"document_recall_at_k={summary['document_recall_at_k']:.4f}")
    print(f"retrieval_threshold_passed={str(summary['retrieval_threshold_passed']).lower()}")
    print(f"release_gate_eligible={str(summary['release_gate_eligible']).lower()}")
    print(f"report={output_path}")
    return 0 if summary["retrieval_threshold_passed"] else 4


def _resolve_target(args: argparse.Namespace, settings) -> tuple[str, Path]:
    has_collection = args.collection_name is not None
    has_lexical = args.lexical_artifact_directory is not None
    if has_collection != has_lexical:
        raise ValueError(
            "--collection-name and --lexical-artifact-directory must be supplied together"
        )
    return (
        args.collection_name or settings.qdrant_collection_name,
        args.lexical_artifact_directory or settings.processed_dir,
    )


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than 0")
    return parsed


def _rate(value: str) -> float:
    parsed = float(value)
    if not 0.0 <= parsed <= 1.0:
        raise argparse.ArgumentTypeError("must be between 0 and 1")
    return parsed


if __name__ == "__main__":
    raise SystemExit(main())
