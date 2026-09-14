from __future__ import annotations

import hashlib
import json

from scripts.promote_code_r3_benchmark_review import main


def test_promotion_ledger_binds_exact_reviewed_file_bytes(tmp_path) -> None:
    draft = tmp_path / "draft.json"
    reviewed = tmp_path / "reviewed.json"
    ledger_path = tmp_path / "review-ledger.json"
    draft.write_text(
        json.dumps(
            {
                "schema_version": "code_r3_benchmark_manifest_v1",
                "snapshot_request": "fci-custom-r3",
                "base_snapshot_id": "fci-custom-r2-ffd9732906d4",
                "fdd_generation": "functional_specs_v9",
                "modified_base_source_path": "utpks_utduh_custom.sql",
                "package_pairs": [
                    {
                        "pair_id": "zakat-wrapper",
                        "category": "fdd_enhancement",
                        "spec_path": "a.spc",
                        "body_path": "a.sql",
                        "key_routines": ["spA"],
                        "fdd_document_ids": ["FDD-A"],
                        "enhancement_markers": ["REQ-A"],
                        "expected_outcome": "reviewed_lineage_candidate",
                        "rationale": "Tests an exact enhancement marker.",
                    },
                    {
                        "pair_id": "zakat-process",
                        "category": "fdd_enhancement",
                        "spec_path": "b.spc",
                        "body_path": "b.sql",
                        "key_routines": ["spB"],
                        "fdd_document_ids": ["FDD-B"],
                        "enhancement_markers": ["REQ-B"],
                        "expected_outcome": "reviewed_lineage_candidate",
                        "rationale": "Tests another enhancement marker.",
                    },
                    {
                        "pair_id": "multi-one",
                        "category": "multi_routine_validation",
                        "spec_path": "c.spc",
                        "body_path": "c.sql",
                        "key_routines": ["spC"],
                        "expected_outcome": "code_workflow_evidence",
                        "rationale": "Tests a multi-routine workflow.",
                    },
                    {
                        "pair_id": "multi-two",
                        "category": "multi_routine_validation",
                        "spec_path": "d.spc",
                        "body_path": "d.sql",
                        "key_routines": ["spD"],
                        "expected_outcome": "code_workflow_evidence",
                        "rationale": "Tests a second multi-routine workflow.",
                    },
                    {
                        "pair_id": "table-heavy",
                        "category": "table_heavy",
                        "spec_path": "e.spc",
                        "body_path": "e.sql",
                        "key_routines": ["spE"],
                        "expected_outcome": "table_operation_evidence",
                        "rationale": "Tests a table-heavy workflow.",
                    },
                    {
                        "pair_id": "no-lineage",
                        "category": "no_approved_fdd_mapping",
                        "spec_path": "f.spc",
                        "body_path": "f.sql",
                        "key_routines": ["spF"],
                        "expected_outcome": "no_reviewed_fdd_lineage",
                        "rationale": "Tests the no-reviewed-lineage boundary.",
                    },
                    {
                        "pair_id": "cross-package",
                        "category": "cross_package_dependency",
                        "spec_path": "g.spc",
                        "body_path": "g.sql",
                        "key_routines": ["spG"],
                        "expected_callers": ["spCaller"],
                        "expected_callees": ["spCallee"],
                        "expected_outcome": "cross_package_workflow_evidence",
                        "rationale": "Tests a cross-package workflow.",
                    },
                ],
                "sme_reviewed": False,
                "review_status": "draft",
                "reviewer": None,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    assert main(
        [
            "--draft-manifest", str(draft),
            "--reviewer", "Pum",
            "--approval-note", "Reviewed package selection.",
            "--output", str(reviewed),
            "--ledger", str(ledger_path),
        ]
    ) == 0

    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    assert ledger["reviewed_manifest_sha256"] == hashlib.sha256(reviewed.read_bytes()).hexdigest()
