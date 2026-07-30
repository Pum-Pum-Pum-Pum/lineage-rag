from __future__ import annotations

from typing import Any

from app.core.config import get_settings
from app.llm.answer_contract import (
    GroundedAnswerRequest,
    GroundedAnswerResponse,
    build_citations_from_results,
    build_insufficient_evidence_response,
)
from app.llm.citation_validator import validate_answer_citations
from app.llm.client import generate_chat_completion_with_usage
from app.llm.prompt_template import EvidenceBudgetExceededError, build_grounded_prompt
from app.llm.usage import estimate_llm_cost
from app.retrieval.evidence_sufficiency import EvidenceSufficiencyDecision
from app.vectorstore.qdrant_search import QdrantSearchResult


def generate_grounded_answer(
    query: str,
    retrieved_results: list[QdrantSearchResult],
    sufficiency: EvidenceSufficiencyDecision,
    llm_client: Any | None = None,
    model: str | None = None,
    conversation_context: str | None = None,
    current_state_requested: bool = False,
    effective_release_label: str | None = None,
) -> GroundedAnswerResponse:
    """Generate a grounded answer or safe refusal from retrieved evidence."""

    request = GroundedAnswerRequest(
        query=query,
        retrieved_results=retrieved_results,
        sufficiency=sufficiency,
        conversation_context=conversation_context,
        current_state_requested=current_state_requested,
        effective_release_label=effective_release_label,
    )

    if not sufficiency.is_sufficient:
        return build_insufficient_evidence_response(request)

    settings = get_settings()
    try:
        prompt = build_grounded_prompt(
            request,
            max_evidence_tokens=settings.conversation_reserved_evidence_tokens,
        )
    except EvidenceBudgetExceededError:
        return GroundedAnswerResponse(
            query=query,
            answer=(
                "I found relevant indexed evidence, but the highest-ranked "
                "evidence unit is too large to fit the configured grounded "
                "prompt budget. I am refusing to answer rather than silently "
                "truncate evidence."
            ),
            is_answered=False,
            refusal_reason=(
                "Highest-ranked evidence unit exceeds the grounded evidence "
                "token budget."
            ),
            citations=build_citations_from_results(retrieved_results),
        )
    completion = generate_chat_completion_with_usage(
        prompt=prompt,
        model=model,
        client=llm_client,
    )
    cost = estimate_llm_cost(
        completion.usage,
        input_cost_per_1k_tokens=settings.llm_input_cost_per_1k_tokens,
        output_cost_per_1k_tokens=settings.llm_output_cost_per_1k_tokens,
    )
    citations = prompt.citations

    response = GroundedAnswerResponse(
        query=query,
        answer=completion.content,
        is_answered=True,
        refusal_reason=None,
        citations=citations,
        usage=completion.usage,
        cost=cost,
    )

    citation_validation = validate_answer_citations(response)
    if not citation_validation.is_valid:
        return GroundedAnswerResponse(
            query=query,
            answer=(
                "I generated an answer, but citation validation failed. "
                "To avoid presenting unsupported citations, I am refusing to return the generated answer."
            ),
            is_answered=False,
            refusal_reason=f"Citation validation failed: {citation_validation}",
            citations=citations,
            usage=completion.usage,
            cost=cost,
        )

    return response
