from app.llm.usage import LLMUsage, estimate_llm_cost


def test_estimate_llm_cost() -> None:
    usage = LLMUsage(
        model="test-model",
        prompt_tokens=2000,
        completion_tokens=500,
        total_tokens=2500,
    )

    cost = estimate_llm_cost(
        usage,
        input_cost_per_1k_tokens=0.10,
        output_cost_per_1k_tokens=0.30,
    )

    assert cost.model == "test-model"
    assert cost.input_cost == 0.20
    assert cost.output_cost == 0.15
    assert cost.total_cost == 0.35
    assert cost.currency == "USD"


def test_estimate_llm_cost_returns_none_when_usage_is_missing() -> None:
    usage = LLMUsage(
        model="test-model",
        prompt_tokens=None,
        completion_tokens=None,
        total_tokens=None,
    )

    cost = estimate_llm_cost(
        usage,
        input_cost_per_1k_tokens=0.10,
        output_cost_per_1k_tokens=0.30,
    )

    assert cost.input_cost is None
    assert cost.output_cost is None
    assert cost.total_cost is None