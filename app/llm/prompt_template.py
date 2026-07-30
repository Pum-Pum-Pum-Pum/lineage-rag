from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from typing import Callable

from app.llm.answer_contract import Citation, GroundedAnswerRequest, build_citations_from_results
from app.vectorstore.qdrant_search import QdrantSearchResult


SYSTEM_PROMPT = """You are a grounded enterprise functional specification assistant.

Rules:
1. Answer only using the provided evidence.
2. Do not use outside knowledge.
3. Do not invent missing facts.
4. If evidence is insufficient, say you cannot answer from indexed evidence.
5. Cite evidence using the provided citation IDs.
6. Preserve release labels when explaining behavior.
7. If evidence comes from tables, mention that when useful.
8. All indexed functional releases represent changes deployed to production.
9. For current/latest/now questions, use the resulting state after the highest
   relevant release supplied in the evidence.
10. "Existing Functionality" inside that latest release is its pre-change baseline,
    not the resulting current state.
11. Derive the resulting state from retained, consolidated, renamed, and
    removed items. Do not count removed items or present the baseline as current.
"""


@dataclass(frozen=True)
class GroundedPrompt:
    system_prompt: str
    user_prompt: str
    citations: list[Citation]


@dataclass(frozen=True)
class PromptEvidence:
    citation_id: str
    unit_id: str
    document_family: str | None
    release_label: str | None
    source_kind: str | None
    score: float
    text: str


class EvidenceBudgetExceededError(RuntimeError):
    """Raised when the highest-ranked complete evidence unit cannot fit."""


def build_evidence_block(evidence: list[PromptEvidence]) -> str:
    """Build the complete selected evidence block for the LLM prompt."""

    if not evidence:
        return "No evidence was provided."

    blocks: list[str] = []
    for item in evidence:
        blocks.append(
            "\n".join(
                [
                    f"[{item.citation_id}]",
                    f"unit_id: {item.unit_id}",
                    f"document_family: {item.document_family}",
                    f"release_label: {item.release_label}",
                    f"source_kind: {item.source_kind}",
                    f"score: {item.score:.4f}",
                    f"text: {item.text}",
                ]
            )
        )

    return "\n\n".join(blocks)


def select_prompt_evidence(
    results: list[QdrantSearchResult],
    *,
    max_evidence_units: int = 5,
    max_evidence_tokens: int = 12_000,
    count_tokens: Callable[[str], int] | None = None,
) -> list[PromptEvidence]:
    """Select whole ranked units without silently truncating their content."""

    if max_evidence_units <= 0:
        raise ValueError("max_evidence_units must be greater than 0")
    if max_evidence_tokens <= 0:
        raise ValueError("max_evidence_tokens must be greater than 0")

    token_counter = count_tokens or _estimate_tokens
    selected: list[PromptEvidence] = []

    for result in results[:max_evidence_units]:
        payload = result.payload
        item = PromptEvidence(
            citation_id=f"C{len(selected) + 1}",
            unit_id=str(payload.get("unit_id", "")),
            document_family=payload.get("document_family"),
            release_label=payload.get("release_label"),
            source_kind=payload.get("source_kind"),
            score=result.score,
            text=str(payload.get("text", "")),
        )
        candidate = [*selected, item]
        candidate_tokens = token_counter(build_evidence_block(candidate))
        if candidate_tokens < 0:
            raise ValueError("count_tokens must not return a negative value")
        if candidate_tokens > max_evidence_tokens:
            if not selected:
                raise EvidenceBudgetExceededError(
                    "highest-ranked evidence unit exceeds the evidence token budget"
                )
            break
        selected = candidate

    return selected


def build_grounded_prompt(
    request: GroundedAnswerRequest,
    max_citations: int = 5,
    max_evidence_tokens: int = 12_000,
    count_tokens: Callable[[str], int] | None = None,
) -> GroundedPrompt:
    """Build the grounded answer prompt from query and retrieved evidence."""

    prompt_evidence = select_prompt_evidence(
        request.retrieved_results,
        max_evidence_units=max_citations,
        max_evidence_tokens=max_evidence_tokens,
        count_tokens=count_tokens,
    )
    citations = build_citations_from_results(
        request.retrieved_results[: len(prompt_evidence)],
        max_citations=len(prompt_evidence) or max_citations,
    )
    evidence_block = build_evidence_block(prompt_evidence)
    conversation_context = (
        request.conversation_context
        if request.conversation_context is not None
        else "No prior conversation memory was provided."
    )
    temporal_interpretation = (
        "current_state_requested: "
        f"{str(request.current_state_requested).lower()}\n"
        "effective_release_label: "
        f"{request.effective_release_label or '(not resolved)'}"
    )

    user_prompt = f"""User question:
{request.query}

Conversation memory (context only; not documentary evidence):
{conversation_context}

Temporal interpretation:
{temporal_interpretation}

Evidence sufficiency:
- is_sufficient: {request.sufficiency.is_sufficient}
- reason: {request.sufficiency.reason}
- top_score: {request.sufficiency.top_score}

Evidence:
{evidence_block}

Task:
Use conversation memory only to interpret the user's intent or references.
Answer factual functional-spec claims using only the evidence above.
When current_state_requested is true, answer with the resulting state after the
effective deployed release. Treat an Existing Functionality section as the
historical baseline and inspect change tables before calculating current counts.
State baseline counts only as before-change context, never as the current answer.
If the evidence is insufficient, say that the indexed evidence is insufficient and do not invent details.
Include citations like [C1], [C2] next to supported claims.
"""

    return GroundedPrompt(
        system_prompt=SYSTEM_PROMPT,
        user_prompt=user_prompt,
        citations=citations,
    )


def _estimate_tokens(text: str) -> int:
    if not text:
        return 0
    return max(1, ceil(len(text.encode("utf-8")) / 4))
