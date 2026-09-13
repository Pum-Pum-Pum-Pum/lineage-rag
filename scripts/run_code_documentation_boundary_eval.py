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

from app.code_indexing.contract import load_code_index_artifact
from app.fdd_code_lineage.combined_retrieval import retrieve_combined_evidence
from app.fdd_code_lineage.documentation_boundary import (
    build_documentation_boundary_case_report,
    load_documentation_boundary_cases,
    require_reviewed_documentation_boundary_cases,
)
from app.fdd_code_lineage.models import validate_lineage_artifact
from app.fdd_code_lineage.reviewed_bundle import load_evaluation_lineage
from app.retrieval.lexical_search import (
    load_retrieval_ready_documents,
    search_lexical_artifacts,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run local lexical diagnostics for code evidence with no approved FDD/code "
            "lineage. This script never calls OpenAI."
        )
    )
    parser.add_argument("--eval-file", type=Path, required=True)
    parser.add_argument("--code-artifact", type=Path, required=True)
    parser.add_argument("--analysis-directory", type=Path, required=True)
    parser.add_argument("--fdd-generation", required=True)
    parser.add_argument("--fdd-directory", type=Path, required=True)
    parser.add_argument("--lineage-artifact", type=Path, required=True)
    parser.add_argument("--limit", type=_positive_int, default=10)
    parser.add_argument("--fdd-candidate-limit", type=_positive_int, default=30)
    parser.add_argument("--candidate-limit", type=_positive_int, default=30)
    parser.add_argument("--max-units-per-parent", type=_positive_int, default=2)
    parser.add_argument("--allow-unreviewed", action="store_true")
    parser.add_argument("--output-file", type=Path)
    args = parser.parse_args(argv)

    cases = load_documentation_boundary_cases(args.eval_file)
    require_reviewed_documentation_boundary_cases(
        cases, allow_unreviewed=args.allow_unreviewed
    )
    artifact = load_code_index_artifact(args.code_artifact)
    if artifact.status != "embedded":
        raise ValueError("Documentation-boundary evaluation requires an embedded code artifact")
    documents = load_retrieval_ready_documents(args.fdd_directory)
    lineage = load_evaluation_lineage(args.lineage_artifact)
    if lineage.status != "reviewed" or lineage.fdd_generation != args.fdd_generation:
        raise ValueError("Documentation-boundary evaluation requires matching reviewed lineage")
    validate_lineage_artifact(
        lineage,
        fdd_document_ids={item.document_id for item in documents},
        code_artifact=artifact,
        analysis_directory=args.analysis_directory,
    )
    if args.fdd_candidate_limit < args.limit:
        raise ValueError("fdd-candidate-limit must be greater than or equal to limit")

    reports = []
    for case in cases:
        retrieval = retrieve_combined_evidence(
            query=case.question,
            fdd_results=search_lexical_artifacts(
                args.fdd_directory, case.question, limit=args.fdd_candidate_limit
            ),
            fdd_generation=args.fdd_generation,
            known_fdd_document_ids={item.document_id for item in documents},
            code_artifact=artifact,
            lineage_artifact=lineage,
            analysis_directory=args.analysis_directory,
            code_mode="lexical",
            code_limit=args.limit,
            code_candidate_limit=args.candidate_limit,
            code_max_units_per_parent=args.max_units_per_parent,
            fdd_limit=args.limit,
        )
        report = build_documentation_boundary_case_report(case=case, retrieval=retrieval)
        reports.append(report)
        print(f"case={case.case_id} passed={str(report.passed).lower()}")

    passed = sum(item.passed for item in reports)
    payload = {
        "schema_version": "code_documentation_boundary_eval_v1",
        "metadata": {
            "run_id": datetime.now(UTC).strftime("code-documentation-boundary-%Y%m%dT%H%M%SZ"),
            "eval_file": str(args.eval_file),
            "eval_file_sha256": _sha256(args.eval_file),
            "code_artifact_identity_sha256": artifact.artifact_identity_sha256,
            "code_snapshot_id": artifact.snapshot_id,
            "fdd_generation": args.fdd_generation,
            "lineage_artifact_identity_sha256": lineage.artifact_identity_sha256,
            "code_mode": "lexical",
            "external_api_calls": 0,
        },
        "summary": {
            "total_cases": len(reports),
            "passed_cases": passed,
            "passed": passed == len(reports),
            "reviewed_manifest": all(item.sme_reviewed for item in cases),
        },
        "cases": [item.model_dump(mode="json") for item in reports],
    }
    output = args.output_file or (
        Path("data/exports/evaluations") / f"{payload['metadata']['run_id']}.json"
    )
    if output.exists():
        raise FileExistsError(f"Refusing to overwrite evaluation report: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")
    print(f"report={output}")
    print(f"passed={str(payload['summary']['passed']).lower()}")
    print("external_api_calls=0")
    return 0 if payload["summary"]["passed"] else 1


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be greater than zero")
    return parsed


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
