from __future__ import annotations

from typing import Any

from app.llm.answer_contract import (
    GroundedAnswerRequest,
    GroundedAnswerResponse,
    build_citations_from_results,
    build_insufficient_evidence_response,
)
from app.llm.client import generate_chat_completion
from app.llm.prompt_template import build_grounded_prompt
from app.retrieval.evidence_sufficiency import EvidenceSufficiencyDecision
from app.vectorstore.qdrant_search import QdrantSearchResult


def generate_grounded_answer(
    query: str,
    retrieved_results: list[QdrantSearchResult],
    sufficiency: EvidenceSufficiencyDecision,
    llm_client: Any | None = None,
    model: str | None = None,
) -> GroundedAnswerResponse:
    """Generate a grounded answer or safe refusal from retrieved evidence."""

    request = GroundedAnswerRequest(
        query=query,
        retrieved_results=retrieved_results,
        sufficiency=sufficiency,
    )

    if not sufficiency.is_sufficient:
        return build_insufficient_evidence_response(request)

    prompt = build_grounded_prompt(request)
    answer_text = generate_chat_completion(
        prompt=prompt,
        model=model,
        client=llm_client,
    )
    citations = build_citations_from_results(retrieved_results)

    return GroundedAnswerResponse(
        query=query,
        answer=answer_text,
        is_answered=True,
        refusal_reason=None,
        citations=citations,
    )
