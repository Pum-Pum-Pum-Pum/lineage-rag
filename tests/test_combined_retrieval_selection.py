from __future__ import annotations

from app.code_retrieval.models import CodeEvidence
from app.fdd_code_lineage.combined_retrieval import (
    FddEvidence,
    _reserve_identifier_affinity_slot,
    _select_lineage_anchored_fdd_evidence,
)
from app.fdd_code_lineage.models import (
    FddCodeLineageArtifact,
    FddCodeTarget,
    create_mapping,
)


def _code(name: str, *, score: float, unit_id: str | None = None) -> CodeEvidence:
    return CodeEvidence(
        unit_id=unit_id or name,
        point_id=f"point-{unit_id or name}",
        score=score,
        retrieval_method="lexical",
        snapshot_id="snapshot-r2",
        module_id="module",
        source_path="pkg_aml.sql",
        source_kind="procedure",
        display_name=name,
        parent_unit_id=None,
        package_name="PKG_AML",
        start_line=1,
        end_line=2,
        parser_state="full_parse",
        conditional_state="unconditional",
        text=f"PROCEDURE {name} IS BEGIN NULL; END;",
    )


def _fdd(document_id: str, *, score: float) -> FddEvidence:
    return FddEvidence(
        unit_id=f"unit-{document_id}",
        document_id=document_id,
        document_family="family",
        release_label="R24",
        source_kind="paragraph",
        score=score,
        text=document_id,
    )


def _lineage(*, selector_scope: str) -> FddCodeLineageArtifact:
    target_values = {
        "module_id": "module",
        "path": "pkg_aml.sql",
        "selector_scope": selector_scope,
        "rationale": "The reviewed target is relevant to the documented AML flow.",
    }
    if selector_scope != "file":
        target_values.update(
            {
                "qualified_name": "PKG_AML.SPSENDBATCHTXNENDDATA",
                "symbol_kind": "procedure",
            }
        )
    mapping = create_mapping(
        fdd_document_id="expected-fdd",
        fdd_release_label="R24",
        code_snapshot_id="snapshot-r2",
        targets=[FddCodeTarget(**target_values)],
        rationale="The SME reviewed this FDD-to-code relationship.",
        mapping_status="reviewed",
        reviewer="SME",
    )
    # This selector does not consume the code artifact, so a small valid lineage
    # model keeps the test focused on the bounded selection policy.
    return FddCodeLineageArtifact(
        status="reviewed",
        fdd_generation="functional_specs_v9",
        code_snapshot_id="snapshot-r2",
        code_artifact_identity_sha256="a" * 64,
        mappings=(mapping,),
        source_candidate_artifact_identity_sha256="b" * 64,
        review_packet_sha256="c" * 64,
        reviewer="SME",
        artifact_identity_sha256="d" * 64,
    )


def test_exact_reviewed_symbol_can_reserve_one_fdd_slot() -> None:
    selected = _select_lineage_anchored_fdd_evidence(
        query="How is batch transaction data sent?",
        candidates=(_fdd("other-fdd", score=2.0), _fdd("expected-fdd", score=1.0)),
        direct_code_candidates=(_code("spSendBatchTxnEndData", score=1.0),),
        lineage_artifact=_lineage(selector_scope="all_overloads"),
        limit=1,
    )

    assert [item.document_id for item in selected] == ["expected-fdd"]


def test_file_scoped_lineage_cannot_override_fdd_ranking() -> None:
    selected = _select_lineage_anchored_fdd_evidence(
        query="How is batch transaction data sent?",
        candidates=(_fdd("other-fdd", score=2.0), _fdd("expected-fdd", score=1.0)),
        direct_code_candidates=(_code("spSendBatchTxnEndData", score=1.0),),
        lineage_artifact=_lineage(selector_scope="file"),
        limit=1,
    )

    assert [item.document_id for item in selected] == ["other-fdd"]


def test_identifier_affinity_replaces_only_one_code_slot() -> None:
    selected = _reserve_identifier_affinity_slot(
        query="How is batch transaction data sent?",
        candidates=(
            _code("spRealtimeSubsTransaction", score=5.0),
            _code("spUHEndPoint", score=4.0),
            _code("spSendBatchTxnEndData", score=1.0),
        ),
        selected=(
            _code("spRealtimeSubsTransaction", score=5.0),
            _code("spUHEndPoint", score=4.0),
        ),
        limit=2,
        max_units_per_parent=2,
    )

    assert len(selected) == 2
    assert {item.display_name for item in selected} == {
        "spRealtimeSubsTransaction",
        "spSendBatchTxnEndData",
    }
