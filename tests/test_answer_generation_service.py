from types import SimpleNamespace

from app.retrieval.evidence_sufficiency import EvidenceSufficiencyDecision
from app.services import answer_generation
from app.services.answer_generation import generate_grounded_answer
from app.vectorstore.qdrant_search import QdrantSearchResult


class FakeMessage:
    def __init__(self, content: str | None) -> None:
        self.content = content


class FakeChoice:
    def __init__(self, content: str | None) -> None:
        self.message = FakeMessage(content)


class FakeChatResponse:
    def __init__(self, content: str) -> None:
        self.choices = [FakeChoice(content)]
        self.usage = FakeUsage()


class FakeUsage:
    prompt_tokens = 20
    completion_tokens = 7
    total_tokens = 27


class FakeCompletionsAPI:
    def __init__(self, content: str = "DECISION: ANSWER\nThe reports were consolidated [C1].") -> None:
        self.content = content
        self.calls: list[dict] = []

    def create(self, model: str, messages: list[dict]) -> FakeChatResponse:
        self.calls.append({"model": model, "messages": messages})
        return FakeChatResponse(self.content)


class FakeChatAPI:
    def __init__(self, content: str = "DECISION: ANSWER\nThe reports were consolidated [C1].") -> None:
        self.completions = FakeCompletionsAPI(content)


class FakeOpenAIClient:
    def __init__(self, content: str = "DECISION: ANSWER\nThe reports were consolidated [C1].") -> None:
        self.chat = FakeChatAPI(content)


def _result() -> QdrantSearchResult:
    return QdrantSearchResult(
        point_id="point-1",
        score=0.75,
        payload={
            "unit_id": "doc::chunk_1",
            "document_family": "FS_FCIS_14.7.0.0.0$ASNB",
            "release_label": "R24",
            "source_kind": "paragraph",
            "text": "The enhancements scoped in the document relate to multiple Teller reports.",
        },
    )


def test_generate_grounded_answer_refuses_when_evidence_is_insufficient() -> None:
    fake_client = FakeOpenAIClient()
    response = generate_grounded_answer(
        query="What is the mobile app login flow?",
        retrieved_results=[_result()],
        sufficiency=EvidenceSufficiencyDecision(
            is_sufficient=False,
            reason="Top score is below threshold.",
            result_count=1,
            top_score=0.20,
        ),
        llm_client=fake_client,
        model="test-model",
    )

    assert response.is_answered is False
    assert response.refusal_reason == "Top score is below threshold."
    assert "could not find sufficient evidence" in response.answer
    assert "Suggested next question:" in response.answer
    assert fake_client.chat.completions.calls == []


def test_generate_grounded_answer_calls_llm_when_evidence_is_sufficient() -> None:
    fake_client = FakeOpenAIClient()
    response = generate_grounded_answer(
        query="What changed in branch reports?",
        retrieved_results=[_result()],
        sufficiency=EvidenceSufficiencyDecision(
            is_sufficient=True,
            reason="Retrieved evidence passed baseline sufficiency checks.",
            result_count=1,
            top_score=0.75,
        ),
        llm_client=fake_client,
        model="test-model",
    )

    assert response.is_answered is True
    assert response.answer == "The reports were consolidated [C1]."
    assert response.refusal_reason is None
    assert len(response.citations) == 1
    assert response.usage is not None
    assert response.usage.total_tokens == 27
    assert response.cost is not None
    assert response.cost.total_cost == 0.0
    assert fake_client.chat.completions.calls[0]["model"] == "test-model"


def test_generate_grounded_answer_refuses_when_llm_returns_invalid_citation() -> None:
    fake_client = FakeOpenAIClient(content="DECISION: ANSWER\nThe reports were consolidated [C99].")
    response = generate_grounded_answer(
        query="What changed in branch reports?",
        retrieved_results=[_result()],
        sufficiency=EvidenceSufficiencyDecision(
            is_sufficient=True,
            reason="Retrieved evidence passed baseline sufficiency checks.",
            result_count=1,
            top_score=0.75,
        ),
        llm_client=fake_client,
        model="test-model",
    )

    assert response.is_answered is False
    assert response.refusal_reason is not None
    assert "Citation validation failed" in response.refusal_reason


def test_generate_grounded_answer_returns_redirecting_abstention_when_model_refuses() -> None:
    fake_client = FakeOpenAIClient(
        content=(
            "DECISION: REFUSE\n"
            "I could not find a direct interest-rate rule in the indexed evidence. "
            "Related evidence discusses investment limits [C1]. Try asking about the Minor Program investment limit."
        )
    )
    response = generate_grounded_answer(
        query="What interest rate applies?",
        retrieved_results=[_result()],
        sufficiency=EvidenceSufficiencyDecision(
            is_sufficient=True,
            reason="Retrieved evidence passed baseline sufficiency checks.",
            result_count=1,
            top_score=0.75,
        ),
        llm_client=fake_client,
        model="test-model",
    )

    assert response.is_answered is False
    assert response.refusal_reason == "No direct evidence supports every material part of the requested answer."
    assert "Related evidence" in response.answer
    assert "Suggested next question:" in response.answer
    assert response.citations[0].unit_id == "doc::chunk_1"


def test_generate_grounded_answer_preserves_model_follow_up_question() -> None:
    fake_client = FakeOpenAIClient(
        content=(
            "DECISION: REFUSE\n"
            "No direct evidence supports this request [C1].\n"
            "Suggested next question: What changes are documented for the cited release?"
        )
    )
    response = generate_grounded_answer(
        query="Unsupported request",
        retrieved_results=[_result()],
        sufficiency=EvidenceSufficiencyDecision(
            is_sufficient=True,
            reason="Retrieved evidence passed baseline sufficiency checks.",
            result_count=1,
            top_score=0.75,
        ),
        llm_client=fake_client,
        model="test-model",
    )

    assert response.is_answered is False
    assert response.answer.count("Suggested next question:") == 1
    assert "What changes are documented" in response.answer


def test_generate_grounded_answer_refuses_when_model_omits_decision() -> None:
    fake_client = FakeOpenAIClient(content="The reports were consolidated [C1].")
    response = generate_grounded_answer(
        query="What changed in branch reports?",
        retrieved_results=[_result()],
        sufficiency=EvidenceSufficiencyDecision(
            is_sufficient=True,
            reason="Retrieved evidence passed baseline sufficiency checks.",
            result_count=1,
            top_score=0.75,
        ),
        llm_client=fake_client,
        model="test-model",
    )

    assert response.is_answered is False
    assert response.refusal_reason == "Grounded answerability decision was missing or invalid."
    assert "could not validate" in response.answer


def test_generate_grounded_answer_refuses_without_llm_when_evidence_unit_exceeds_budget(
    monkeypatch,
) -> None:
    fake_client = FakeOpenAIClient()
    monkeypatch.setattr(
        answer_generation,
        "get_settings",
        lambda: SimpleNamespace(
            conversation_reserved_evidence_tokens=1,
            llm_input_cost_per_1k_tokens=0.0,
            llm_output_cost_per_1k_tokens=0.0,
        ),
    )

    response = generate_grounded_answer(
        query="What changed in branch reports?",
        retrieved_results=[_result()],
        sufficiency=EvidenceSufficiencyDecision(
            is_sufficient=True,
            reason="Evidence passed.",
            result_count=1,
            top_score=0.75,
        ),
        llm_client=fake_client,
        model="test-model",
    )

    assert response.is_answered is False
    assert "rather than silently truncate evidence" in response.answer
    assert "token budget" in response.refusal_reason
    assert fake_client.chat.completions.calls == []
