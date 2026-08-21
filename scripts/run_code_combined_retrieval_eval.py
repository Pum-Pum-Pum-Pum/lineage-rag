from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from qdrant_client import QdrantClient

from app.code_indexing.contract import load_code_index_artifact
from app.code_retrieval.service import retrieve_code_evidence
from app.fdd_code_lineage.combined_retrieval import retrieve_combined_evidence
from app.fdd_code_lineage.evaluation import (
    build_code_combined_retrieval_case_report,
    build_code_combined_retrieval_report,
    load_code_combined_eval_cases,
    require_reviewed_code_combined_cases,
    write_json_report_no_overwrite,
)
from app.fdd_code_lineage.models import FddCodeLineageArtifact, validate_lineage_artifact
from app.retrieval.lexical_search import (
    load_retrieval_ready_documents,
    search_lexical_artifacts,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run a local code/combined retrieval gate. This script never creates "
            "query embeddings or calls an LLM."
        )
    )
    parser.add_argument("--eval-file", type=Path, action="append", required=True)
    parser.add_argument("--code-artifact", type=Path, required=True)
    parser.add_argument("--analysis-directory", type=Path, required=True)
    parser.add_argument("--fdd-generation")
    parser.add_argument("--fdd-directory", type=Path)
    parser.add_argument("--lineage-artifact", type=Path)
    parser.add_argument(
        "--code-mode", choices=("lexical", "dense", "hybrid"), default="lexical"
    )
    parser.add_argument("--qdrant-path", type=Path)
    parser.add_argument("--collection-name")
    parser.add_argument(
        "--query-vectors-json",
        type=Path,
        help="Local JSON object mapping case_id to a precomputed query vector.",
    )
    parser.add_argument("--limit", type=_positive_int, default=10)
    parser.add_argument("--candidate-limit", type=_positive_int, default=30)
    parser.add_argument("--max-units-per-parent", type=_positive_int, default=2)
    parser.add_argument("--minimum-positive-pass-rate", type=_rate, default=0.90)
    parser.add_argument("--case-id", action="append", default=[])
    parser.add_argument("--allow-unreviewed", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output-file", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cases = []
    for path in args.eval_file:
        cases.extend(load_code_combined_eval_cases(path))
    _require_unique_case_ids(cases)
    if args.case_id:
        by_id = {case.case_id: case for case in cases}
        unknown = sorted(set(args.case_id).difference(by_id))
        if unknown:
            raise ValueError(f"Unknown evaluation case IDs: {unknown}")
        cases = [by_id[item] for item in args.case_id]
    require_reviewed_code_combined_cases(
        cases, allow_unreviewed=args.allow_unreviewed
    )

    code_artifact = load_code_index_artifact(args.code_artifact)
    if code_artifact.status != "embedded":
        raise ValueError("Evaluation requires an embedded code artifact")
    combined_cases = [case for case in cases if case.mode == "combined"]
    fdd_documents = ()
    lineage = None
    if combined_cases:
        if not all(
            (args.fdd_generation, args.fdd_directory, args.lineage_artifact)
        ):
            raise ValueError(
                "Combined cases require FDD generation/directory and lineage artifact"
            )
        fdd_documents = load_retrieval_ready_documents(args.fdd_directory)
        lineage = FddCodeLineageArtifact.model_validate_json(
            args.lineage_artifact.read_text(encoding="utf-8")
        )
        if lineage.status != "reviewed":
            raise ValueError("Combined evaluation requires reviewed lineage")
        if lineage.fdd_generation != args.fdd_generation:
            raise ValueError("Lineage and requested FDD generations do not match")
        validate_lineage_artifact(
            lineage,
            fdd_document_ids={item.document_id for item in fdd_documents},
            code_artifact=code_artifact,
            analysis_directory=args.analysis_directory,
        )

    vectors = _load_query_vectors(args.query_vectors_json)
    needs_dense = args.code_mode in {"dense", "hybrid"}
    if needs_dense and not all((args.qdrant_path, args.collection_name, vectors)):
        raise ValueError(
            "Dense/hybrid evaluation requires Qdrant path, collection, and local "
            "precomputed query vectors"
        )
    missing_vectors = [case.case_id for case in cases if needs_dense and case.case_id not in vectors]
    if missing_vectors:
        raise ValueError(f"Missing precomputed query vectors: {missing_vectors}")

    reviewed_manifest = all(case.sme_reviewed for case in cases)
    print(f"cases={len(cases)}")
    print(f"reviewed_manifest={str(reviewed_manifest).lower()}")
    print(f"code_mode={args.code_mode}")
    print("external_api_calls=0")
    if args.dry_run:
        return 0

    client = QdrantClient(path=str(args.qdrant_path)) if needs_dense else None
    reports = []
    try:
        for case in cases:
            vector = vectors.get(case.case_id)
            if case.mode == "code":
                retrieval = retrieve_code_evidence(
                    artifact=code_artifact,
                    query=case.question,
                    mode=args.code_mode,
                    limit=args.limit,
                    candidate_limit=args.candidate_limit,
                    client=client,
                    collection_name=args.collection_name,
                    query_vector=vector,
                    max_units_per_parent=args.max_units_per_parent,
                )
            else:
                assert lineage is not None
                fdd_results = search_lexical_artifacts(
                    args.fdd_directory, case.question, limit=args.limit
                )
                retrieval = retrieve_combined_evidence(
                    query=case.question,
                    fdd_results=fdd_results,
                    fdd_generation=args.fdd_generation,
                    known_fdd_document_ids={item.document_id for item in fdd_documents},
                    code_artifact=code_artifact,
                    lineage_artifact=lineage,
                    analysis_directory=args.analysis_directory,
                    code_mode=args.code_mode,
                    code_limit=args.limit,
                    code_candidate_limit=args.candidate_limit,
                    client=client,
                    collection_name=args.collection_name,
                    query_vector=vector,
                    code_max_units_per_parent=args.max_units_per_parent,
                )
            report = build_code_combined_retrieval_case_report(
                case=case, retrieval=retrieval
            )
            reports.append(report)
            print(
                f"case={case.case_id} positive_passed={report.positive_gate_passed} "
                f"code_recall={report.code_recall_at_k} fdd_recall={report.fdd_recall_at_k}"
            )
    finally:
        if client is not None:
            client.close()

    run_id = datetime.now(UTC).strftime("code-combined-retrieval-%Y%m%dT%H%M%SZ")
    report = build_code_combined_retrieval_report(
        metadata={
            "run_id": run_id,
            "eval_files": [str(path) for path in args.eval_file],
            "eval_file_sha256": {
                str(path): _sha256(path) for path in args.eval_file
            },
            "code_artifact": str(args.code_artifact),
            "code_artifact_identity_sha256": code_artifact.artifact_identity_sha256,
            "code_snapshot_id": code_artifact.snapshot_id,
            "code_mode": args.code_mode,
            "fusion_method": "weighted_rrf" if args.code_mode == "hybrid" else None,
            "hybrid_dense_weight": 0.40 if args.code_mode == "hybrid" else None,
            "hybrid_lexical_weight": 0.60 if args.code_mode == "hybrid" else None,
            "collection_name": args.collection_name,
            "fdd_generation": args.fdd_generation,
            "fdd_directory": str(args.fdd_directory) if args.fdd_directory else None,
            "lineage_artifact_identity_sha256": (
                lineage.artifact_identity_sha256 if lineage else None
            ),
            "retrieval_limit": args.limit,
            "candidate_limit": args.candidate_limit,
            "max_units_per_parent": args.max_units_per_parent,
            "reviewed_manifest": reviewed_manifest,
            "draft_baseline": not reviewed_manifest,
            "external_api_calls": 0,
        },
        cases=reports,
        minimum_positive_pass_rate=args.minimum_positive_pass_rate,
    )
    output = args.output_file or Path("data/exports/evaluations") / f"{run_id}.json"
    write_json_report_no_overwrite(report, output)
    print(f"report={output}")
    print(
        "release_gate_eligible="
        f"{str(report['summary']['release_gate_eligible']).lower()}"
    )
    return 0 if report["summary"]["retrieval_threshold_passed"] else 1


def _load_query_vectors(path: Path | None) -> dict[str, list[float]]:
    if path is None:
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Query-vector file must contain a JSON object")
    vectors: dict[str, list[float]] = {}
    for case_id, vector in payload.items():
        if not isinstance(vector, list) or not vector:
            raise ValueError(f"Invalid query vector for case {case_id}")
        vectors[str(case_id)] = [float(value) for value in vector]
    return vectors


def _require_unique_case_ids(cases: list) -> None:
    ids = [case.case_id for case in cases]
    duplicates = sorted({case_id for case_id in ids if ids.count(case_id) > 1})
    if duplicates:
        raise ValueError(f"Duplicate case IDs across evaluation files: {duplicates}")


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be greater than zero")
    return parsed


def _rate(value: str) -> float:
    parsed = float(value)
    if not 0.0 <= parsed <= 1.0:
        raise argparse.ArgumentTypeError("value must be between 0 and 1")
    return parsed


if __name__ == "__main__":
    raise SystemExit(main())
