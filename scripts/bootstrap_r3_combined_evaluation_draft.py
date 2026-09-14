"""Create a reviewable R3 combined-evaluation draft from reviewed lineage.

The draft is intentionally derived only from mappings that have already passed
the separate FDD-to-code SME review.  It creates one narrowly-scoped combined
case per approved implementation target, so a multi-routine mapping cannot
hide a missing target behind a package-level result.
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
from app.fdd_code_lineage.evaluation import load_code_combined_eval_cases
from app.fdd_code_lineage.reviewed_bundle import load_reviewed_lineage


def _routine_name(qualified_name: str | None) -> str:
    if not qualified_name:
        raise ValueError("R3 combined evaluation requires routine-level lineage targets")
    return qualified_name.rsplit(".", 1)[-1]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Create a no-overwrite R3 combined FDD/code evaluation draft from "
            "a reviewed R3 lineage artifact."
        )
    )
    parser.add_argument("--benchmark-manifest", type=Path, required=True)
    parser.add_argument("--lineage-artifact", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite R3 combined evaluation draft: {args.output}")
    benchmark = load_r3_benchmark_manifest(args.benchmark_manifest)
    if not benchmark.sme_reviewed or benchmark.review_status != "reviewed":
        raise ValueError("R3 combined evaluation draft requires a reviewed R3 benchmark")
    lineage = load_reviewed_lineage(args.lineage_artifact)
    if lineage.fdd_generation != benchmark.fdd_generation:
        raise ValueError("Reviewed lineage FDD generation does not match the R3 benchmark")

    pairs_by_document = {
        pair.fdd_document_ids[0]: pair
        for pair in benchmark.package_pairs
        if pair.category == "fdd_enhancement"
    }
    mappings = [
        mapping
        for mapping in lineage.mappings
        if mapping.fdd_document_id in pairs_by_document
    ]
    found_documents = {mapping.fdd_document_id for mapping in mappings}
    expected_documents = set(pairs_by_document)
    if found_documents != expected_documents:
        raise ValueError(
            "Reviewed lineage does not cover exactly the R3 enhancement FDDs: "
            f"missing={sorted(expected_documents - found_documents)}, "
            f"unexpected={sorted(found_documents - expected_documents)}"
        )

    cases: list[dict[str, object]] = []
    seen_identity: set[tuple[str, str, str]] = set()
    for mapping in sorted(mappings, key=lambda item: item.fdd_document_id):
        pair = pairs_by_document[mapping.fdd_document_id]
        for target in mapping.targets:
            if target.path != pair.body_path:
                raise ValueError(
                    "Reviewed R3 lineage target is outside its benchmark package body: "
                    f"{target.path}"
                )
            routine = _routine_name(target.qualified_name)
            identity = (mapping.fdd_document_id, target.path, routine.casefold())
            if identity in seen_identity:
                continue
            seen_identity.add(identity)
            cases.append(
                {
                    "schema_version": "code_combined_eval_case_v2",
                    "case_id": f"r3-combined-{len(cases) + 1:03d}",
                    "mode": "combined",
                    "question": (
                        f"How does the documented Zakat Khultah enhancement relate to "
                        f"the visible custom routine {routine} in {target.path}?"
                    ),
                    "analysis_kind": "explanation",
                    "expected_claims": [
                        "Separate the documented requirement from visible custom "
                        "implementation and cite only the reviewed FDD-to-code lineage."
                    ],
                    "expected_code_paths": [target.path],
                    "expected_code_symbols": [routine],
                    "expected_code_symbol_policy": "all",
                    "expected_fdd_document_ids": [mapping.fdd_document_id],
                    "require_reviewed_lineage": True,
                    "should_abstain": False,
                    "expected_unknown_kinds": [],
                    "sme_reviewed": False,
                    "review_status": "draft",
                    "rationale": (
                        "Derived from the reviewed R3 lineage mapping "
                        f"{mapping.mapping_id} for the benchmark pair {pair.pair_id}; "
                        "the exact routine remains an independently reviewed "
                        "retrieval expectation."
                    ),
                }
            )
    if not cases:
        raise ValueError("No routine-level R3 combined evaluation cases were derived")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "".join(json.dumps(item, separators=(",", ":")) + "\n" for item in cases),
        encoding="utf-8",
    )
    load_code_combined_eval_cases(args.output)
    print("status=draft")
    print(f"reviewed_lineage_artifact_identity_sha256={lineage.artifact_identity_sha256}")
    print(f"combined_cases={len(cases)}")
    print(f"output={args.output}")
    print("external_calls_performed=false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
