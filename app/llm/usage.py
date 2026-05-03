from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class LLMUsage:
    model: str
    prompt_tokens: int | None
    completion_tokens: int | None
    total_tokens: int | None


@dataclass(frozen=True)
class LLMCostEstimate:
    model: str
    input_cost: float | None
    output_cost: float | None
    total_cost: float | None
    currency: str = "USD"


def estimate_llm_cost(
    usage: LLMUsage,
    input_cost_per_1k_tokens: float,
    output_cost_per_1k_tokens: float,
) -> LLMCostEstimate:
    """Estimate LLM cost from token usage and configured prices."""

    if usage.prompt_tokens is None or usage.completion_tokens is None:
        return LLMCostEstimate(
            model=usage.model,
            input_cost=None,
            output_cost=None,
            total_cost=None,
        )

    input_cost = (usage.prompt_tokens / 1000) * input_cost_per_1k_tokens
    output_cost = (usage.completion_tokens / 1000) * output_cost_per_1k_tokens

    return LLMCostEstimate(
        model=usage.model,
        input_cost=input_cost,
        output_cost=output_cost,
        total_cost=input_cost + output_cost,
    )
