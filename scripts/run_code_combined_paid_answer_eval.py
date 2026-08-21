from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from qdrant_client import QdrantClient

from app.code_indexing.contract import load_code_index_artifact
from app.code_retrieval.service import retrieve_code_evidence
from app.core.config import get_settings
from app.fdd_code_lineage.combined_retrieval import retrieve_combined_evidence
from app.fdd_code_lineage.evaluation import (
    load_code_combined_eval_cases,
    require_reviewed_code_combined_cases,
)
from app.fdd_code_lineage.models import FddCodeLineageArtifact, validate_lineage_artifact
from app.fdd_code_lineage.paid_evaluation import (
    create_no_retry_client,
    embed_one_query,
    evaluate_answer_structure,
    generate_grounded_answer,
)
from app.retrieval.lexical_search import load_retrieval_ready_documents
from app.retrieval.retrieval_config import RetrievalRuntimeConfig
from app.services.query_retrieval import retrieve_planned_query_evidence


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the explicitly authorized 10-case code/combined paid evaluation. "
            "Automatic OpenAI retries are disabled and partial runs fail closed."
        )
    )
    parser.add_argument("--eval-file", type=Path, action="append", required=True)
    parser.add_argument("--authorization-plan", type=Path, required=True)
    parser.add_argument("--code-artifact", type=Path, required=True)
    parser.add_argument("--analysis-directory", type=Path, required=True)
    parser.add_argument("--fdd-generation", required=True)
    parser.add_argument("--fdd-directory", type=Path, required=True)
    parser.add_argument("--lineage-artifact", type=Path, required=True)
    parser.add_argument("--fdd-qdrant-path", type=Path, required=True)
    parser.add_argument("--code-qdrant-path", type=Path, required=True)
    parser.add_argument("--fdd-collection", required=True)
    parser.add_argument("--code-collection", required=True)
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--candidate-limit", type=int, default=30)
    parser.add_argument("--max-units-per-parent", type=int, default=2)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument(
        "--resume-from",
        type=Path,
        help="Prior failed-closed run whose completed case traces must be reused.",
    )
    parser.add_argument("--confirm-authorized-disclosure", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cases = [
        case
        for path in args.eval_file
        for case in load_code_combined_eval_cases(path)
    ]
    _validate_preflight(args, cases)
    settings = get_settings()
    artifact = load_code_index_artifact(args.code_artifact)
    fdd_documents = load_retrieval_ready_documents(args.fdd_directory)
    known_fdd_ids = {item.document_id for item in fdd_documents}
    lineage = FddCodeLineageArtifact.model_validate_json(
        args.lineage_artifact.read_text(encoding="utf-8")
    )
    validate_lineage_artifact(
        lineage,
        fdd_document_ids=known_fdd_ids,
        code_artifact=artifact,
        analysis_directory=args.analysis_directory,
    )
    if lineage.status != "reviewed" or lineage.fdd_generation != args.fdd_generation:
        raise ValueError("Paid combined evaluation requires matching reviewed lineage")
    if artifact.vector_dimension != settings.qdrant_vector_size:
        raise ValueError("Code artifact and configured query-vector dimensions differ")

    prior_cases, prior_embedding_calls, prior_answer_calls = _load_resume(
        args.resume_from, cases, args.authorization_plan
    )
    completed_ids = {item["case_id"] for item in prior_cases}
    remaining_cases = [case for case in cases if case.case_id not in completed_ids]
    print(f"cases={len(cases)}")
    print(f"cases_already_completed={len(prior_cases)}")
    print(f"embedding_requests_planned={len(remaining_cases)}")
    print(f"answer_requests_planned={len(remaining_cases)}")
    print("automatic_openai_retries=0")
    _verify_collection(args.fdd_qdrant_path, args.fdd_collection)
    _verify_collection(args.code_qdrant_path, args.code_collection)
    if args.dry_run:
        print("external_api_calls=0")
        return 0
    if not args.confirm_authorized_disclosure:
        raise ValueError("Explicit disclosure confirmation flag is required")
    if args.output_directory.exists():
        raise FileExistsError(f"Output directory already exists: {args.output_directory}")
    args.output_directory.mkdir(parents=True)

    run = {
        "schema_version": "code_combined_paid_answer_eval_v1",
        "status": "running",
        "run_id": args.output_directory.name,
        "started_at": datetime.now(UTC).isoformat(),
        "authorization_plan": str(args.authorization_plan),
        "authorization_plan_sha256": _sha256(args.authorization_plan),
        "eval_files": {str(path): _sha256(path) for path in args.eval_file},
        "fdd_generation": args.fdd_generation,
        "fdd_collection": args.fdd_collection,
        "code_collection": args.code_collection,
        "code_snapshot_id": artifact.snapshot_id,
        "code_artifact_identity_sha256": artifact.artifact_identity_sha256,
        "lineage_artifact_identity_sha256": lineage.artifact_identity_sha256,
        "embedding_model": settings.openai_embedding_model,
        "answer_model": settings.openai_chat_model,
        "retrieval": {
            "mode": "hybrid",
            "fusion": "weighted_rrf",
            "dense_weight": 0.40,
            "lexical_weight": 0.60,
            "limit": args.limit,
            "candidate_limit": args.candidate_limit,
            "max_units_per_parent": args.max_units_per_parent,
        },
        "automatic_openai_retries": 0,
        "resumed_from": str(args.resume_from) if args.resume_from else None,
        "prior_embedding_requests_completed": prior_embedding_calls,
        "prior_answer_requests_completed": prior_answer_calls,
        "embedding_requests_completed": 0,
        "answer_requests_completed": 0,
        "cases": prior_cases,
    }
    _write(args.output_directory / "run-state.json", run)
    api_client = create_no_retry_client(
        api_key=settings.openai_api_key,
        base_url=settings.openai_base_url,
    )
    fdd_qdrant = QdrantClient(path=str(args.fdd_qdrant_path))
    code_qdrant = QdrantClient(path=str(args.code_qdrant_path))
    config = RetrievalRuntimeConfig(
        retrieval_mode="hybrid",
        hybrid_dense_weight=0.40,
        hybrid_lexical_weight=0.60,
        hybrid_candidate_limit=args.candidate_limit,
    )
    try:
        for case in remaining_cases:
            case_record = {"case": case.model_dump(mode="json"), "status": "started"}
            try:
                vector, embedding_call = embed_one_query(
                    client=api_client,
                    model=settings.openai_embedding_model,
                    question=case.question,
                    expected_dimension=artifact.vector_dimension,
                )
                run["embedding_requests_completed"] += 1
                case_record["embedding_call"] = embedding_call
                if case.mode == "code":
                    retrieval = retrieve_code_evidence(
                        artifact=artifact,
                        query=case.question,
                        mode="hybrid",
                        limit=args.limit,
                        candidate_limit=args.candidate_limit,
                        client=code_qdrant,
                        collection_name=args.code_collection,
                        query_vector=vector,
                        max_units_per_parent=args.max_units_per_parent,
                    )
                else:
                    planned = retrieve_planned_query_evidence(
                        qdrant_client=fdd_qdrant,
                        collection_name=args.fdd_collection,
                        query_text=case.question,
                        embedding_model=settings.openai_embedding_model,
                        query_vector=vector,
                        retrieval_config=config,
                        lexical_artifact_directory=args.fdd_directory,
                        limit=args.limit,
                    )
                    retrieval = retrieve_combined_evidence(
                        query=case.question,
                        fdd_results=planned.results,
                        fdd_generation=args.fdd_generation,
                        known_fdd_document_ids=known_fdd_ids,
                        code_artifact=artifact,
                        lineage_artifact=lineage,
                        analysis_directory=args.analysis_directory,
                        code_mode="hybrid",
                        code_limit=args.limit,
                        code_candidate_limit=args.candidate_limit,
                        client=code_qdrant,
                        collection_name=args.code_collection,
                        query_vector=vector,
                        code_max_units_per_parent=args.max_units_per_parent,
                    )
                    case_record["fdd_temporal_plan"] = asdict(planned.temporal_plan)
                case_record["retrieval"] = retrieval.model_dump(mode="json")
                answer, answer_call = generate_grounded_answer(
                    client=api_client,
                    model=settings.openai_chat_model,
                    case=case,
                    retrieval=retrieval,
                )
                run["answer_requests_completed"] += 1
                case_record["answer_call"] = answer_call
                case_record["answer"] = answer.model_dump(mode="json")
                case_record["structural_evaluation"] = evaluate_answer_structure(
                    case=case, answer=answer
                )
                case_record["status"] = "completed"
                _write(
                    args.output_directory / f"{case.case_id}.json", case_record
                )
                run["cases"].append(
                    {
                        "case_id": case.case_id,
                        "status": "completed",
                        "structural_passed": case_record["structural_evaluation"]["passed"],
                        "trace": f"{case.case_id}.json",
                    }
                )
                _replace(args.output_directory / "run-state.json", run)
                print(
                    f"case={case.case_id} "
                    f"structural_passed={str(case_record['structural_evaluation']['passed']).lower()}"
                )
            except Exception as error:
                case_record["status"] = "failed"
                case_record["error_type"] = type(error).__name__
                case_record["error"] = str(error)
                _write(args.output_directory / f"{case.case_id}-failed.json", case_record)
                run["status"] = "failed_closed"
                run["failed_case_id"] = case.case_id
                run["failure"] = {"type": type(error).__name__, "message": str(error)}
                run["completed_at"] = datetime.now(UTC).isoformat()
                _replace(args.output_directory / "run-state.json", run)
                raise
    finally:
        fdd_qdrant.close()
        code_qdrant.close()

    structural_passes = sum(item["structural_passed"] for item in run["cases"])
    run["status"] = "completed_pending_sme_review"
    run["completed_at"] = datetime.now(UTC).isoformat()
    run["summary"] = {
        "total_cases": len(cases),
        "structural_passes": structural_passes,
        "structural_pass_rate": structural_passes / len(cases),
        "semantic_sme_review_required": True,
        "activation_authorized": False,
        "cumulative_embedding_requests_completed": (
            prior_embedding_calls + run["embedding_requests_completed"]
        ),
        "cumulative_answer_requests_completed": (
            prior_answer_calls + run["answer_requests_completed"]
        ),
    }
    _replace(args.output_directory / "run-state.json", run)
    review_path = args.output_directory / "sme-review.md"
    _write_text(review_path, _render_review(run, args.output_directory))
    print(f"report={args.output_directory / 'run-state.json'}")
    print(f"sme_review={review_path}")
    print("activation_authorized=false")
    return 0


def _validate_preflight(args: argparse.Namespace, cases: list) -> None:
    require_reviewed_code_combined_cases(cases, allow_unreviewed=False)
    case_ids = [case.case_id for case in cases]
    if len(cases) != 10 or len(set(case_ids)) != 10:
        raise ValueError("Authorization is scoped to exactly 10 unique reviewed cases")
    plan = json.loads(args.authorization_plan.read_text(encoding="utf-8"))
    if plan.get("case_count") != 10:
        raise ValueError("Authorization plan is not scoped to 10 cases")
    planned_hashes = plan.get("eval_files", {})
    observed_hashes = {str(path): _sha256(path) for path in args.eval_file}
    if planned_hashes != observed_hashes:
        raise ValueError("Evaluation manifests differ from the authorized plan")
    retrieval_report = Path(plan["retrieval_report"])
    if _sha256(retrieval_report) != plan.get("retrieval_report_sha256"):
        raise ValueError("Authorized retrieval report hash mismatch")
    retrieval = json.loads(retrieval_report.read_text(encoding="utf-8"))
    if not retrieval.get("summary", {}).get("release_gate_eligible"):
        raise ValueError("Reviewed deterministic retrieval gate is not eligible")
    if args.limit <= 0 or args.candidate_limit < args.limit:
        raise ValueError("Candidate limit must be >= positive evidence limit")


def _render_review(run: dict, directory: Path) -> str:
    lines = [
        "# Paid code/combined grounded-answer SME review",
        "",
        f"- Run ID: `{run['run_id']}`",
        f"- FDD generation: `{run['fdd_generation']}`",
        f"- Code snapshot: `{run['code_snapshot_id']}`",
        "- Embedding requests in completed run chain: "
        f"{run['summary']['cumulative_embedding_requests_completed']}",
        "- Answer requests in completed run chain: "
        f"{run['summary']['cumulative_answer_requests_completed']}",
        f"- Structural passes: {run['summary']['structural_passes']}/10",
        "- Status: pending SME semantic review",
        "",
        "Review each answer against its cited original excerpts. Structural pass does not prove entailment.",
        "",
    ]
    for index, summary in enumerate(run["cases"], start=1):
        trace = json.loads((directory / summary["trace"]).read_text(encoding="utf-8"))
        case = trace["case"]
        answer = trace["answer"]
        lines.extend(
            [
                f"## {index}. {case['case_id']}",
                "",
                f"**Question:** {case['question']}",
                "",
                "```json",
                json.dumps(answer, indent=2, ensure_ascii=False),
                "```",
                "",
                f"Structural result: **{'pass' if summary['structural_passed'] else 'fail'}**",
                "",
                "SME verdict: accepted | corrected | needs_more_context",
                "SME rationale:",
                "Required follow-up:",
                "",
            ]
        )
    return "\n".join(lines)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_resume(
    directory: Path | None, cases: list, authorization_plan: Path
) -> tuple[list[dict], int, int]:
    if directory is None:
        return [], 0, 0
    state_path = directory / "run-state.json"
    state = json.loads(state_path.read_text(encoding="utf-8"))
    if state.get("status") != "failed_closed":
        raise ValueError("Resume source must be a failed-closed run")
    if state.get("authorization_plan_sha256") != _sha256(authorization_plan):
        raise ValueError("Resume source authorization-plan hash mismatch")
    known_ids = {case.case_id for case in cases}
    completed = list(state.get("cases", []))
    completed_ids = [str(item.get("case_id", "")) for item in completed]
    if len(completed_ids) != len(set(completed_ids)) or not set(completed_ids).issubset(
        known_ids
    ):
        raise ValueError("Resume source contains invalid completed case identities")
    normalized = []
    for item in completed:
        trace_path = directory / str(item["trace"])
        if not trace_path.is_file():
            raise ValueError(f"Resume trace is missing: {trace_path}")
        normalized.append({**item, "trace": str(trace_path.resolve())})
    return (
        normalized,
        int(state.get("embedding_requests_completed", 0)),
        int(state.get("answer_requests_completed", 0)),
    )


def _verify_collection(path: Path, collection_name: str) -> None:
    client = QdrantClient(path=str(path))
    try:
        if not client.collection_exists(collection_name):
            raise RuntimeError(
                f"Required collection does not exist at {path}: {collection_name}"
            )
    finally:
        client.close()


def _write(path: Path, value: dict) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite evaluation artifact: {path}")
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def _replace(path: Path, value: dict) -> None:
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    temp.replace(path)


def _write_text(path: Path, value: str) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite evaluation artifact: {path}")
    path.write_text(value, encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
