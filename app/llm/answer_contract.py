from __future__ import annotations

from dataclasses import dataclass

from app.llm.usage import LLMCostEstimate, LLMUsage
from app.retrieval.evidence_sufficiency import EvidenceSufficiencyDecision
from app.vectorstore.qdrant_search import QdrantSearchResult


@dataclass(frozen=True)
class Citation:
    unit_id: str
    document_family: str | None
    release_label: str | None
    source_kind: str | None
    score: float
    text_preview: str


@dataclass(frozen=True)
class GroundedAnswerRequest:
    query: str
    retrieved_results: list[QdrantSearchResult]
    sufficiency: EvidenceSufficiencyDecision


@dataclass(frozen=True)
class GroundedAnswerResponse:
    query: str
    answer: str
    is_answered: bool
    refusal_reason: str | None
    citations: list[Citation]
    usage: LLMUsage | None = None
    cost: LLMCostEstimate | None = None


def build_citations_from_results(
    results: list[QdrantSearchResult],
    max_citations: int = 5,
    preview_chars: int = 240,
) -> list[Citation]:
    """Build citation objects from retrieved Qdrant payloads."""

    if max_citations <= 0:
        raise ValueError("max_citations must be greater than 0")
    if preview_chars <= 0:
        raise ValueError("preview_chars must be greater than 0")

    citations: list[Citation] = []

    for result in results[:max_citations]:
        payload = result.payload
        text = str(payload.get("text", ""))
        citations.append(
            Citation(
                unit_id=str(payload.get("unit_id", "")),
                document_family=payload.get("document_family"),
                release_label=payload.get("release_label"),
                source_kind=payload.get("source_kind"),
                score=result.score,
                text_preview=text[:preview_chars],
            )
        )

    return citations


def build_insufficient_evidence_response(
    request: GroundedAnswerRequest,
) -> GroundedAnswerResponse:
    """Build a safe refusal response when retrieved evidence is insufficient."""

    citations = build_citations_from_results(request.retrieved_results)
    answer = (
        "I could not find sufficient evidence in the indexed documents to answer this confidently. "
        "The information may be missing from the indexed text/tables, or it may exist in unsupported content such as embedded attachments, screenshots, or files not currently extracted."
    )

    return GroundedAnswerResponse(
        query=request.query,
        answer=answer,
        is_answered=False,
        refusal_reason=request.sufficiency.reason,
        citations=citations,
    )
