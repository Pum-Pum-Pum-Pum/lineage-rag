"""Create reviewable, no-overwrite R3 code-evaluation draft manifests.

The R3 benchmark identifies the package pairs and the one intentionally
undocumented pair.  This script turns that reviewed selection into two
*draft* contracts: code-only retrieval cases and a documentation-boundary
case.  It deliberately does not create combined FDD/code cases: those require
the later, separately SME-reviewed R3 lineage artifact.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.code_ingestion.r3_benchmark import load_r3_benchmark_manifest
from app.fdd_code_lineage.documentation_boundary import load_documentation_boundary_cases
from app.fdd_code_lineage.evaluation import load_code_combined_eval_cases


def _code_question(routine: str, body_path: str) -> str:
    return (
        f"Where is {routine} implemented in {body_path}, and what behavior is "
        "visible in the approved custom source?"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Create no-overwrite R3 code and documentation-boundary eval drafts."
    )
    parser.add_argument("--benchmark-manifest", type=Path, required=True)
    parser.add_argument("--code-output", type=Path, required=True)
    parser.add_argument("--boundary-output", type=Path, required=True)
    args = parser.parse_args(argv)

    benchmark = load_r3_benchmark_manifest(args.benchmark_manifest)
    if not benchmark.sme_reviewed or benchmark.review_status != "reviewed":
        raise ValueError("R3 evaluation drafts require a reviewed R3 benchmark manifest")
    existing = [str(path) for path in (args.code_output, args.boundary_output) if path.exists()]
    if existing:
        raise FileExistsError(f"Refusing to overwrite R3 evaluation drafts: {existing}")

    code_cases: list[dict[str, object]] = []
    boundary_cases: list[dict[str, object]] = []
    case_number = 0
    for pair in benchmark.package_pairs:
        primary = pair.key_routines[0]
        if pair.category == "no_approved_fdd_mapping":
            boundary_cases.append(
                {
                    "schema_version": "code_documentation_boundary_case_v1",
                    "case_id": f"r3-{pair.pair_id}-boundary-001",
                    "question": (
                        f"What visible custom behavior is implemented by {primary} "
                        f"in {pair.body_path}?"
                    ),
                    "expected_code_paths": [pair.body_path],
                    "expected_code_symbols": [primary],
                    "expected_code_symbol_policy": "all",
                    "expected_documentation_state": "no_reviewed_fdd_lineage",
                    "sme_reviewed": False,
                    "review_status": "draft",
                    "rationale": pair.rationale,
                }
            )
            continue
        for routine in pair.key_routines:
            case_number += 1
            code_cases.append(
                {
                    "schema_version": "code_combined_eval_case_v2",
                    "case_id": f"r3-{pair.pair_id}-{case_number:03d}",
                    "mode": "code",
                    "question": _code_question(routine, pair.body_path),
                    "analysis_kind": "impact_analysis"
                    if pair.category == "cross_package_dependency"
                    else "explanation",
                    "expected_claims": [
                        "Describe only behavior visible in the approved custom source and "
                        "qualify unavailable dependencies or database semantics."
                    ],
                    "expected_code_paths": [pair.body_path],
                    "expected_code_symbols": [routine],
                    "expected_code_symbol_policy": "all",
                    "expected_fdd_document_ids": [],
                    "require_reviewed_lineage": False,
                    "should_abstain": False,
                    "expected_unknown_kinds": [],
                    "sme_reviewed": False,
                    "review_status": "draft",
                    "rationale": pair.rationale,
                }
            )

    code_content = "".join(
        json.dumps(case, separators=(",", ":")) + "\n"
        for case in code_cases
    )
    boundary_content = "".join(
        json.dumps(case, separators=(",", ":")) + "\n"
        for case in boundary_cases
    )
    args.code_output.parent.mkdir(parents=True, exist_ok=True)
    args.boundary_output.parent.mkdir(parents=True, exist_ok=True)
    args.code_output.write_text(code_content, encoding="utf-8")
    args.boundary_output.write_text(boundary_content, encoding="utf-8")

    # Validate the exact bytes that will be reviewed.
    load_code_combined_eval_cases(args.code_output)
    load_documentation_boundary_cases(args.boundary_output)
    print("status=draft")
    print(f"code_cases={len(code_cases)}")
    print(f"documentation_boundary_cases={len(boundary_cases)}")
    print(f"code_output={args.code_output}")
    print(f"boundary_output={args.boundary_output}")
    print("external_calls_performed=false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
