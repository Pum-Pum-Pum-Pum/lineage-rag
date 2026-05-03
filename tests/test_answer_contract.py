import pytest

from app.llm.answer_contract import (
    GroundedAnswerRequest,
    build_citations_from_results,
    build_insufficient_evidence_response,
)
from app.retrieval.evidence_sufficiency import EvidenceSufficiencyDecision
from app.vectorstore.qdrant_search import QdrantSearchResult


def _result(score: float = 0.42) -> QdrantSearchResult:
    return QdrantSearchResult(
        point_id="point-1",
        score=score,
        payload={
            "unit_id": "doc::chunk_1",
            "document_family": "FS_FCIS_14.7.0.0.0$ASNB",
            "release_label": "R24",
            "source_kind": "paragraph",
            "text": "The enhancements scoped in the document are relating to multiple Teller reports.",
        },
    )


def test_build_citations_from_results() -> None:
    citations = build_citations_from_results([_result()], max_citations=1, preview_chars=20)

    assert len(citations) == 1
    assert citations[0].unit_id == "doc::chunk_1"
    assert citations[0].release_label == "R24"
    assert citations[0].source_kind == "paragraph"
    assert citations[0].text_preview == "The enhancements sco"


def test_build_citations_from_results_rejects_invalid_limits() -> None:
    with pytest.raises(ValueError, match="max_citations"):
        build_citations_from_results([_result()], max_citations=0)

    with pytest.raises(ValueError, match="preview_chars"):
        build_citations_from_results([_result()], preview_chars=0)


def test_build_insufficient_evidence_response() -> None:
    request = GroundedAnswerRequest(
        query="What is the mobile app login flow?",
        retrieved_results=[_result(score=0.20)],
        sufficiency=EvidenceSufficiencyDecision(
            is_sufficient=False,
            reason="Top score is below threshold.",
            result_count=1,
            top_score=0.20,
        ),
    )

    response = build_insufficient_evidence_response(request)

    assert response.query == "What is the mobile app login flow?"
    assert response.is_answered is False
    assert response.refusal_reason == "Top score is below threshold."
    assert "could not find sufficient evidence" in response.answer
    assert len(response.citations) == 1
    assert response.usage is None
    assert response.cost is None
