from __future__ import annotations

import pytest

from app.code_ingestion.r3_benchmark import R3BenchmarkManifest, R3PackagePair


def _pair(index: int, category: str) -> R3PackagePair:
    values = {
        "pair_id": f"pair-{index}", "category": category,
        "spec_path": f"new/pkg{index}.spc", "body_path": f"new/pkg{index}.sql",
        "key_routines": (f"spRoutine{index}",), "rationale": "A reviewed R3 benchmark selection.",
        "expected_outcome": "code_workflow_evidence",
    }
    if category == "fdd_enhancement":
        values.update(fdd_document_ids=(f"FDD-{index}",), enhancement_markers=(f"REQ{index}",), expected_outcome="reviewed_lineage_candidate")
    elif category == "table_heavy":
        values["expected_outcome"] = "table_operation_evidence"
    elif category == "no_approved_fdd_mapping":
        values["expected_outcome"] = "no_reviewed_fdd_lineage"
    elif category == "cross_package_dependency":
        values.update(expected_callers=("spCaller",), expected_callees=("spCallee",), expected_outcome="cross_package_workflow_evidence")
    return R3PackagePair(**values)


def test_r3_benchmark_requires_exactly_the_agreed_category_mix() -> None:
    pairs = (
        _pair(1, "fdd_enhancement"), _pair(2, "fdd_enhancement"),
        _pair(3, "multi_routine_validation"), _pair(4, "multi_routine_validation"),
        _pair(5, "table_heavy"), _pair(6, "no_approved_fdd_mapping"),
        _pair(7, "cross_package_dependency"),
    )
    manifest = R3BenchmarkManifest(
        modified_base_source_path="baseline/pkg_old.sql", package_pairs=pairs
    )
    assert len(manifest.package_pairs) == 7


def test_r3_benchmark_rejects_modified_base_source_inside_new_pair_set() -> None:
    pairs = (
        _pair(1, "fdd_enhancement"), _pair(2, "fdd_enhancement"),
        _pair(3, "multi_routine_validation"), _pair(4, "multi_routine_validation"),
        _pair(5, "table_heavy"), _pair(6, "no_approved_fdd_mapping"),
        _pair(7, "cross_package_dependency"),
    )
    with pytest.raises(ValueError, match="modified base source"):
        R3BenchmarkManifest(modified_base_source_path="new/pkg1.sql", package_pairs=pairs)
