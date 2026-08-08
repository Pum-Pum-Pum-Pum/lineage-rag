from types import SimpleNamespace

from app.llm.fdd_grounded_evaluation import FddGroundedEvalCase
from app.retrieval.fdd_retrieval_gate import (
    build_fdd_retrieval_gate_case_report,
    build_fdd_retrieval_gate_report,
)
from app.retrieval.retrieval_router import RoutedRetrievalResult
from app.retrieval.temporal_query import TemporalQueryPlan
from app.services.query_retrieval import PlannedRetrievalResult
from app.vectorstore.qdrant_search import QdrantSearchResult


def test_positive_case_requires_expected_document_and_release() -> None:
    case_report = build_fdd_retrieval_gate_case_report(
        case=_case(),
        planned=_planned(document_id="expected-doc", release_label="R21"),
    )

    assert case_report.positive_gate_passed is True
    assert case_report.document_recall_at_k == 1.0
    assert case_report.missing_document_ids == []


def test_nearby_wrong_document_fails_positive_gate() -> None:
    case_report = build_fdd_retrieval_gate_case_report(
        case=_case(),
        planned=_planned(document_id="nearby-doc", release_label="R21"),
    )

    assert case_report.positive_gate_passed is False
    assert case_report.document_recall_at_k == 0.0
    assert case_report.missing_document_ids == ["expected-doc"]


def test_unreviewed_retrieval_can_pass_threshold_but_not_release_gate() -> None:
    case_report = build_fdd_retrieval_gate_case_report(
        case=_case(),
        planned=_planned(document_id="expected-doc", release_label="R21"),
    )

    report = build_fdd_retrieval_gate_report(
        metadata={"reviewed_manifest": False},
        cases=[case_report],
        minimum_document_recall=0.9,
    )

    assert report["summary"]["retrieval_threshold_passed"] is True
    assert report["summary"]["release_gate_eligible"] is False


def _case() -> FddGroundedEvalCase:
    return FddGroundedEvalCase(
        case_id="case-1",
        question="What behavior is supported?",
        expected_claims=["Expected behavior."],
        expected_evidence=[{"document_id": "expected-doc", "evidence": "Expected behavior."}],
        expected_document_ids=["expected-doc"],
        expected_release_labels=["R21"],
        required_citation_document_ids=["expected-doc"],
        should_abstain=False,
        sme_reviewed=False,
        review_status="pending_sme_approval",
    )


def _planned(*, document_id: str, release_label: str) -> PlannedRetrievalResult:
    result = QdrantSearchResult(
        point_id="point-1",
        score=0.8,
        payload={
            "unit_id": "unit-1",
            "document_id": document_id,
            "release_label": release_label,
            "source_kind": "paragraph",
        },
    )
    return PlannedRetrievalResult(
        routed=RoutedRetrievalResult(
            retrieval_mode="hybrid",
            results=[result],
            dense_candidates=[result],
            lexical_candidates=[result],
        ),
        results=[result],
        temporal_plan=TemporalQueryPlan(
            original_query="What behavior is supported?",
            retrieval_query="What behavior is supported?",
            is_current_state=False,
            effective_release_label=None,
            release_source=None,
        ),
        retrieval_candidate_limit=10,
    )
