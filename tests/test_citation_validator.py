from app.llm.answer_contract import Citation, GroundedAnswerResponse
from app.llm.citation_validator import (
    build_available_citation_ids,
    extract_citation_ids,
    validate_answer_citations,
)


def _citation(score: float = 0.5) -> Citation:
    return Citation(
        unit_id="doc::chunk_1",
        document_family="family",
        release_label="R24",
        source_kind="paragraph",
        score=score,
        text_preview="Evidence",
    )


def test_extract_citation_ids() -> None:
    assert extract_citation_ids("Answer [C1] and [C23].") == ["C1", "C23"]


def test_build_available_citation_ids() -> None:
    assert build_available_citation_ids([_citation(), _citation()]) == ["C1", "C2"]


def test_validate_answer_citations_passes_for_valid_citations() -> None:
    response = GroundedAnswerResponse(
        query="query",
        answer="Supported answer [C1].",
        is_answered=True,
        refusal_reason=None,
        citations=[_citation()],
    )

    validation = validate_answer_citations(response)

    assert validation.is_valid is True
    assert validation.cited_ids == ["C1"]
    assert validation.invalid_ids == []
    assert validation.missing_citation is False


def test_validate_answer_citations_fails_for_invalid_citation_id() -> None:
    response = GroundedAnswerResponse(
        query="query",
        answer="Unsupported citation [C99].",
        is_answered=True,
        refusal_reason=None,
        citations=[_citation()],
    )

    validation = validate_answer_citations(response)

    assert validation.is_valid is False
    assert validation.invalid_ids == ["C99"]


def test_validate_answer_citations_fails_when_answer_has_no_citations() -> None:
    response = GroundedAnswerResponse(
        query="query",
        answer="Answer without citations.",
        is_answered=True,
        refusal_reason=None,
        citations=[_citation()],
    )

    validation = validate_answer_citations(response)
    assert validation.is_valid is False
    assert validation.missing_citation is True


def test_validate_answer_citations_allows_refusal_without_citation() -> None:
    response = GroundedAnswerResponse(
        query="query",
        answer="I could not find sufficient evidence.",
        is_answered=False,
        refusal_reason="Insufficient evidence",
        citations=[_citation()],
    )

    validation = validate_answer_citations(response)
    assert validation.is_valid is True
    assert validation.missing_citation is False
