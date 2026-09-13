from __future__ import annotations

from app.code_retrieval.models import CodeEvidence
from app.fdd_code_lineage.combined_retrieval import CombinedRetrievalResult, FddEvidence, ReviewedLineageUse
from app.fdd_code_lineage.documentation_boundary import (
    DocumentationBoundaryCase,
    build_documentation_boundary_case_report,
)


def _retrieval(*, mappings=()) -> CombinedRetrievalResult:
    return CombinedRetrievalResult(
        query="What does spUndocumented do?", fdd_generation="functional_specs_v9",
        code_snapshot_id="fci-custom-r3-x",
        fdd_evidence=(FddEvidence(unit_id="f1", document_id="generic-fdd", document_family="Generic", release_label="R1", source_kind="paragraph", score=1.0, text="generic"),),
        code_evidence=(CodeEvidence(unit_id="c1", point_id="p1", score=1.0, retrieval_method="lexical", snapshot_id="fci-custom-r3-x", module_id="fci-custom", source_path="pkg_undocumented.sql", source_kind="procedure", display_name="spUndocumented", start_line=1, end_line=2, parser_state="full_parse", conditional_state="known", text="BEGIN NULL; END;"),),
        direct_code_evidence=(), mapped_code_evidence=(), reviewed_lineage=mappings,
    )


def _case() -> DocumentationBoundaryCase:
    return DocumentationBoundaryCase(
        case_id="r3-undocumented-001", question="What does spUndocumented do?",
        expected_code_paths=("pkg_undocumented.sql",), expected_code_symbols=("spUndocumented",),
        rationale="Validates an approved no-lineage documentation boundary.",
    )


def test_boundary_case_keeps_fdd_hits_as_unreviewed_candidates() -> None:
    report = build_documentation_boundary_case_report(case=_case(), retrieval=_retrieval())
    assert report.passed is True
    assert report.fdd_evidence_status == "unreviewed_candidate"
    assert report.reviewed_mapping_ids == ()


def test_boundary_case_fails_if_reviewed_mapping_is_present() -> None:
    report = build_documentation_boundary_case_report(
        case=_case(),
        retrieval=_retrieval(mappings=(ReviewedLineageUse(mapping_id="m1", fdd_document_id="generic-fdd", code_unit_ids=("c1",)),)),
    )
    assert report.passed is False
    assert "reviewed FDD-to-code mapping" in report.failures[0]
