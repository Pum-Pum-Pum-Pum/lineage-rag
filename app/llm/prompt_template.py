from __future__ import annotations

from dataclasses import dataclass

from app.llm.answer_contract import Citation, GroundedAnswerRequest, build_citations_from_results


SYSTEM_PROMPT = """You are a grounded enterprise functional specification assistant.

Rules:
1. Answer only using the provided evidence.
2. Do not use outside knowledge.
3. Do not invent missing facts.
4. If evidence is insufficient, say you cannot answer from indexed evidence.
5. Cite evidence using the provided citation IDs.
6. Preserve release labels when explaining behavior.
7. If evidence comes from tables, mention that when useful.
"""


@dataclass(frozen=True)
class GroundedPrompt:
    system_prompt: str
    user_prompt: str
    citations: list[Citation]


def build_evidence_block(citations: list[Citation]) -> str:
    """Build a compact evidence block for the LLM prompt."""

    if not citations:
        return "No evidence was provided."

    blocks: list[str] = []
    for index, citation in enumerate(citations, start=1):
        blocks.append(
            "\n".join(
                [
                    f"[C{index}]",
                    f"unit_id: {citation.unit_id}",
                    f"document_family: {citation.document_family}",
                    f"release_label: {citation.release_label}",
                    f"source_kind: {citation.source_kind}",
                    f"score: {citation.score:.4f}",
                    f"text: {citation.text_preview}",
                ]
            )
        )

    return "\n\n".join(blocks)


def build_grounded_prompt(
    request: GroundedAnswerRequest,
    max_citations: int = 5,
) -> GroundedPrompt:
    """Build the grounded answer prompt from query and retrieved evidence."""

    citations = build_citations_from_results(
        request.retrieved_results,
        max_citations=max_citations,
    )
    evidence_block = build_evidence_block(citations)

    user_prompt = f"""User question:
{request.query}

Evidence sufficiency:
- is_sufficient: {request.sufficiency.is_sufficient}
- reason: {request.sufficiency.reason}
- top_score: {request.sufficiency.top_score}

Evidence:
{evidence_block}

Task:
Answer the user question using only the evidence above.
If the evidence is insufficient, say that the indexed evidence is insufficient and do not invent details.
Include citations like [C1], [C2] next to supported claims.
"""

    return GroundedPrompt(
        system_prompt=SYSTEM_PROMPT,
        user_prompt=user_prompt,
        citations=citations,
    )
