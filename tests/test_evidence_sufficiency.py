import pytest

from app.retrieval.evidence_sufficiency import assess_evidence_sufficiency
from app.vectorstore.qdrant_search import QdrantSearchResult


def _result(score: float, text: str) -> QdrantSearchResult:
    return QdrantSearchResult(
        point_id="point-1",
        score=score,
        payload={"text": text, "release_label": "R24", "source_kind": "paragraph"},
    )


def test_assess_evidence_sufficiency_passes_for_good_evidence() -> None:
    decision = assess_evidence_sufficiency(
        [_result(0.75, "The enhancements scoped multiple Teller reports")],
        min_results=1,
        min_top_score=0.30,
        required_text_markers=["multiple Teller reports"],
    )

    assert decision.is_sufficient is True
    assert decision.top_score == 0.75


def test_assess_evidence_sufficiency_fails_for_no_results() -> None:
    decision = assess_evidence_sufficiency([], min_results=1)

    assert decision.is_sufficient is False
    assert "Expected at least" in decision.reason


def test_assess_evidence_sufficiency_fails_for_low_top_score() -> None:
    decision = assess_evidence_sufficiency(
        [_result(0.20, "Some weakly related content")],
        min_top_score=0.30,
    )

    assert decision.is_sufficient is False
    assert "below required threshold" in decision.reason


def test_assess_evidence_sufficiency_fails_when_required_markers_missing() -> None:
    decision = assess_evidence_sufficiency(
        [_result(0.80, "Traceability Matrix")],
        required_text_markers=["B-01", "Branch End of Day", "layout"],
    )

    assert decision.is_sufficient is False
    assert "required text markers" in decision.reason


def test_assess_evidence_sufficiency_rejects_invalid_thresholds() -> None:
    with pytest.raises(ValueError, match="min_results"):
        assess_evidence_sufficiency([], min_results=0)

    with pytest.raises(ValueError, match="min_top_score"):
        assess_evidence_sufficiency([], min_top_score=-0.1)
