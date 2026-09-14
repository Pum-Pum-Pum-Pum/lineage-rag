from __future__ import annotations

import json

from app.fdd_code_lineage.documentation_boundary import load_documentation_boundary_cases
from app.fdd_code_lineage.evaluation import load_code_combined_eval_cases
from scripts.bootstrap_code_r3_evaluation_drafts import main


def _pair(index: int, category: str, outcome: str) -> dict[str, object]:
    pair: dict[str, object] = {
        "pair_id": f"pair-{index:02d}",
        "category": category,
        "spec_path": f"pkg{index}.spc",
        "body_path": f"pkg{index}.sql",
        "key_routines": [f"spRoutine{index}", f"spOtherRoutine{index}"],
        "expected_outcome": outcome,
        "rationale": f"R3 test rationale for pair number {index}.",
    }
    if category == "fdd_enhancement":
        pair.update({"fdd_document_ids": [f"FDD-{index}"], "enhancement_markers": [f"REQ-{index}"]})
    if category == "cross_package_dependency":
        pair.update({"expected_callers": ["pkg.spCaller"], "expected_callees": ["pkgp.spCallee"]})
    return pair


def test_bootstrap_writes_separate_reviewable_code_and_boundary_drafts(tmp_path) -> None:
    benchmark = tmp_path / "benchmark.json"
    benchmark.write_text(
        json.dumps(
            {
                "schema_version": "code_r3_benchmark_manifest_v1",
                "snapshot_request": "fci-custom-r3",
                "base_snapshot_id": "fci-custom-r2-ffd9732906d4",
                "fdd_generation": "functional_specs_v9",
                "modified_base_source_path": "utpks_utduh_custom.sql",
                "package_pairs": [
                    _pair(1, "fdd_enhancement", "reviewed_lineage_candidate"),
                    _pair(2, "fdd_enhancement", "reviewed_lineage_candidate"),
                    _pair(3, "multi_routine_validation", "code_workflow_evidence"),
                    _pair(4, "multi_routine_validation", "code_workflow_evidence"),
                    _pair(5, "table_heavy", "table_operation_evidence"),
                    _pair(6, "no_approved_fdd_mapping", "no_reviewed_fdd_lineage"),
                    _pair(7, "cross_package_dependency", "cross_package_workflow_evidence"),
                ],
                "sme_reviewed": True,
                "review_status": "reviewed",
                "reviewer": "Pum",
            }
        ),
        encoding="utf-8",
    )
    code_output = tmp_path / "code_r3_grounded_eval_draft.jsonl"
    boundary_output = tmp_path / "code_r3_documentation_boundary_draft.jsonl"

    assert main(
        [
            "--benchmark-manifest", str(benchmark),
            "--code-output", str(code_output),
            "--boundary-output", str(boundary_output),
        ]
    ) == 0

    code_cases = load_code_combined_eval_cases(code_output)
    boundary_cases = load_documentation_boundary_cases(boundary_output)
    assert len(code_cases) == 12
    assert {case.mode for case in code_cases} == {"code"}
    assert all(case.review_status == "draft" for case in code_cases)
    assert all(len(case.expected_code_symbols) == 1 for case in code_cases)
    assert len(boundary_cases) == 1
    assert boundary_cases[0].expected_documentation_state == "no_reviewed_fdd_lineage"
