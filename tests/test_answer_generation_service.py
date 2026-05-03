from app.retrieval.evidence_sufficiency import EvidenceSufficiencyDecision
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
    def __init__(self) -> None:
        self.calls: list[dict] = []

    def create(self, model: str, messages: list[dict]) -> FakeChatResponse:
        self.calls.append({"model": model, "messages": messages})
        return FakeChatResponse("The reports were consolidated [C1].")


class FakeChatAPI:
    def __init__(self) -> None:
        self.completions = FakeCompletionsAPI()


class FakeOpenAIClient:
    def __init__(self) -> None:
        self.chat = FakeChatAPI()


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
