from __future__ import annotations

import re
from dataclasses import dataclass

from app.llm.answer_contract import Citation, GroundedAnswerResponse


CITATION_PATTERN = re.compile(r"\[C(?P<number>\d+)\]")


@dataclass(frozen=True)
class CitationValidationResult:
    is_valid: bool
    cited_ids: list[str]
    available_ids: list[str]
    invalid_ids: list[str]
    missing_citation: bool


def extract_citation_ids(answer: str) -> list[str]:
    """Extract citation IDs like C1, C2 from an answer string."""

    return [f"C{match.group('number')}" for match in CITATION_PATTERN.finditer(answer)]


def build_available_citation_ids(citations: list[Citation]) -> list[str]:
    """Build available citation IDs based on citation order."""

    return [f"C{index}" for index, _ in enumerate(citations, start=1)]


def validate_answer_citations(response: GroundedAnswerResponse) -> CitationValidationResult:
    """Validate that answer citation IDs refer to available citations."""

    available_ids = build_available_citation_ids(response.citations)
    cited_ids = extract_citation_ids(response.answer)
    invalid_ids = [citation_id for citation_id in cited_ids if citation_id not in available_ids]
    missing_citation = response.is_answered and bool(response.citations) and not cited_ids

    return CitationValidationResult(
        is_valid=not invalid_ids and not missing_citation,
        cited_ids=cited_ids,
        available_ids=available_ids,
        invalid_ids=invalid_ids,
        missing_citation=missing_citation,
    )
