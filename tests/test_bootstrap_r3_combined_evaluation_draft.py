from __future__ import annotations

import hashlib
import json

from app.fdd_code_lineage.evaluation import load_code_combined_eval_cases
from app.fdd_code_lineage.models import FddCodeTarget, create_mapping
from scripts.bootstrap_r3_combined_evaluation_draft import main


def _pair(index: int, category: str, outcome: str) -> dict[str, object]:
    pair: dict[str, object] = {
        "pair_id": f"pair-{index:02d}",
        "category": category,
        "spec_path": f"pkg{index}.spc",
        "body_path": f"pkg{index}.sql",
        "key_routines": [f"spRoutine{index}"],
        "expected_outcome": outcome,
        "rationale": f"R3 test rationale for pair number {index}.",
    }
    if category == "fdd_enhancement":
        pair.update({"fdd_document_ids": [f"FDD-{index}"], "enhancement_markers": [f"REQ-{index}"]})
    if category == "cross_package_dependency":
        pair.update({"expected_callers": ["pkg.spCaller"], "expected_callees": ["pkgp.spCallee"]})
    return pair


def _lineage_payload() -> dict[str, object]:
    targets = (
        FddCodeTarget(
            module_id="fci-custom",
            path="pkg1.sql",
            qualified_name="PKG1.SPONE",
            symbol_kind="procedure",
            selector_scope="all_overloads",
            rationale="A reviewed R3 implementation target for the first FDD.",
        ),
        FddCodeTarget(
            module_id="fci-custom",
            path="pkg2.sql",
            qualified_name="PKG2.SPTWO",
            symbol_kind="procedure",
            selector_scope="all_overloads",
            rationale="A reviewed R3 implementation target for the second FDD.",
        ),
    )
    mappings = [
        create_mapping(
            fdd_document_id=f"FDD-{index}",
            fdd_release_label="R25",
            code_snapshot_id="fci-custom-r3-test",
            targets=(targets[index - 1],),
            rationale="Reviewed R3 mapping for combined-evaluation bootstrap testing.",
            mapping_status="reviewed",
            reviewer="Pum",
        ).model_dump(mode="json")
        for index in (1, 2)
    ]
    values = {
        "status": "reviewed",
        "fdd_generation": "functional_specs_v9",
        "code_snapshot_id": "fci-custom-r3-test",
        "code_artifact_identity_sha256": "a" * 64,
        "mappings": mappings,
        "source_candidate_artifact_identity_sha256": "b" * 64,
        "review_packet_sha256": "c" * 64,
        "reviewer": "Pum",
    }
    values["artifact_identity_sha256"] = hashlib.sha256(
        json.dumps(values, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    ).hexdigest()
    return values


def test_bootstrap_creates_one_combined_case_per_reviewed_target(tmp_path) -> None:
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
    lineage = tmp_path / "reviewed_lineage.json"
    lineage.write_text(json.dumps(_lineage_payload()), encoding="utf-8")
    output = tmp_path / "r3_combined_draft.jsonl"

    assert main(
        [
            "--benchmark-manifest", str(benchmark),
            "--lineage-artifact", str(lineage),
            "--output", str(output),
        ]
    ) == 0

    cases = load_code_combined_eval_cases(output)
    assert len(cases) == 2
    assert all(case.mode == "combined" for case in cases)
    assert all(case.require_reviewed_lineage for case in cases)
    assert {case.expected_code_symbols[0] for case in cases} == {"SPONE", "SPTWO"}
    assert {case.expected_fdd_document_ids[0] for case in cases} == {"FDD-1", "FDD-2"}
