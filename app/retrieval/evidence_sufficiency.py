from __future__ import annotations

from dataclasses import dataclass

from app.vectorstore.qdrant_search import QdrantSearchResult


@dataclass(frozen=True)
class EvidenceSufficiencyDecision:
    is_sufficient: bool
    reason: str
    result_count: int
    top_score: float | None


def assess_evidence_sufficiency(
    results: list[QdrantSearchResult],
    min_results: int = 1,
    min_top_score: float = 0.30,
    required_text_markers: list[str] | None = None,
) -> EvidenceSufficiencyDecision:
    """Assess whether retrieved evidence is strong enough to answer.

    This is a simple baseline before LLM generation. It intentionally uses
    transparent rules instead of pretending vector search always returns
    answerable evidence.
    """

    if min_results <= 0:
        raise ValueError("min_results must be greater than 0")
    if min_top_score < 0:
        raise ValueError("min_top_score must be non-negative")

    result_count = len(results)
    top_score = results[0].score if results else None

    if result_count < min_results:
        return EvidenceSufficiencyDecision(
            is_sufficient=False,
            reason=f"Expected at least {min_results} results, got {result_count}.",
            result_count=result_count,
            top_score=top_score,
        )

    if top_score is None or top_score < min_top_score:
        return EvidenceSufficiencyDecision(
            is_sufficient=False,
            reason=f"Top score {top_score} is below required threshold {min_top_score}.",
            result_count=result_count,
            top_score=top_score,
        )

    if required_text_markers:
        combined_text = "\n".join(str(result.payload.get("text", "")) for result in results)
        if not any(marker in combined_text for marker in required_text_markers):
            return EvidenceSufficiencyDecision(
                is_sufficient=False,
                reason="Retrieved evidence does not contain required text markers.",
                result_count=result_count,
                top_score=top_score,
            )

    return EvidenceSufficiencyDecision(
        is_sufficient=True,
        reason="Retrieved evidence passed baseline sufficiency checks.",
        result_count=result_count,
        top_score=top_score,
    )
