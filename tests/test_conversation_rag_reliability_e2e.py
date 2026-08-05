from pathlib import Path
from types import SimpleNamespace

from fastapi.testclient import TestClient

from app.api.main import create_app
from app.api.routes import conversations as conversation_route
from app.conversation.models import MessageRole
from app.conversation.store import SqliteConversationStore
from app.llm.answer_contract import GroundedAnswerResponse
from app.retrieval.evidence_sufficiency import EvidenceSufficiencyDecision
from app.schemas.query_api import (
    CitationResponse,
    EvidenceSufficiencyResponse,
    QueryResponse,
)
from app.services.answer_generation import generate_grounded_answer
from app.vectorstore.qdrant_search import QdrantSearchResult


class _FakeCompletionAPI:
    def __init__(self, content: str) -> None:
        self._content = content

    def create(self, **kwargs):
        del kwargs
        return SimpleNamespace(
            choices=[
                SimpleNamespace(
                    message=SimpleNamespace(content=self._content)
                )
            ],
            usage=SimpleNamespace(
                prompt_tokens=20,
                completion_tokens=8,
                total_tokens=28,
            ),
        )


class _FakeLLMClient:
    def __init__(self, content: str) -> None:
        self.chat = SimpleNamespace(
            completions=_FakeCompletionAPI(content)
        )


def _client(store: SqliteConversationStore) -> TestClient:
    app = create_app()
    app.dependency_overrides[
        conversation_route.get_conversation_store
    ] = lambda: store
    return TestClient(app)


def _query_response(
    *,
    query: str,
    answer: str = "Grounded R24 answer [C1].",
    is_answered: bool = True,
    refusal_reason: str | None = None,
) -> QueryResponse:
    return QueryResponse(
        query=query,
        answer=answer,
        is_answered=is_answered,
        refusal_reason=refusal_reason,
        retrieval_mode="hybrid",
        citations=[],
        sufficiency=EvidenceSufficiencyResponse(
            is_sufficient=is_answered,
            reason=refusal_reason or "Evidence passed.",
            result_count=1,
            top_score=0.8,
        ),
        trace_id=f"trace-{query[:8]}",
        trace_output_path="test-trace.json",
    )


def _as_query_response(response: GroundedAnswerResponse) -> QueryResponse:
    return QueryResponse(
        query=response.query,
        answer=response.answer,
        is_answered=response.is_answered,
        refusal_reason=response.refusal_reason,
        retrieval_mode="hybrid",
        citations=[
            CitationResponse(
                unit_id=item.unit_id,
                document_family=item.document_family,
                release_label=item.release_label,
                source_kind=item.source_kind,
                score=item.score,
                text_preview=item.text_preview,
            )
            for item in response.citations
        ],
        sufficiency=EvidenceSufficiencyResponse(
            is_sufficient=response.is_answered,
            reason=response.refusal_reason or "Evidence passed.",
            result_count=len(response.citations),
            top_score=(
                response.citations[0].score
                if response.citations
                else None
            ),
        ),
        trace_id="trace-grounding",
        trace_output_path="test-trace.json",
    )


def _evidence() -> QdrantSearchResult:
    return QdrantSearchResult(
        point_id="r24-branch",
        score=0.82,
        payload={
            "unit_id": "R24::table_chunk_7",
            "document_family": "ASNB",
            "release_label": "R24",
            "source_kind": "table",
            "text": "R24 retains B-01, B-02, B-03, and B-04.",
        },
    )


def test_follow_up_context_is_durable_and_isolated_by_conversation(
    monkeypatch,
    tmp_path: Path,
) -> None:
    captured_contexts: list[str] = []

    def fake_execute(request, *, conversation_context=None):
        captured_contexts.append(conversation_context or "")
        return _query_response(query=request.query)

    monkeypatch.setattr(
        conversation_route,
        "execute_query_request",
        fake_execute,
    )
    with SqliteConversationStore(tmp_path / "chat.sqlite3") as store:
        client = _client(store)
        first_id = client.post(
            "/conversations", json={"title": "R24"}
        ).json()["conversation_id"]
        second_id = client.post(
            "/conversations", json={"title": "Other"}
        ).json()["conversation_id"]

        first_turn = client.post(
            f"/conversations/{first_id}/messages",
            json={"content": "Explain R24 branch reports."},
        )
        follow_up = client.post(
            f"/conversations/{first_id}/messages",
            json={"content": "How many of them remain currently?"},
        )
        isolated = client.post(
            f"/conversations/{second_id}/messages",
            json={"content": "What did I ask before?"},
        )

        assert first_turn.status_code == 200
        assert follow_up.status_code == 200
        assert isolated.status_code == 200
        assert "Explain R24 branch reports." in captured_contexts[1]
        assert "Grounded R24 answer [C1]." in captured_contexts[1]
        assert "How many of them remain currently?" in captured_contexts[1]
        assert "R24 branch reports" not in captured_contexts[2]
        assert "Grounded R24 answer" not in captured_contexts[2]
        assert [
            message.role
            for message in store.list_messages(first_id)
        ] == [
            MessageRole.USER,
            MessageRole.ASSISTANT,
            MessageRole.USER,
            MessageRole.ASSISTANT,
        ]


def test_abstention_and_invalid_citation_are_persisted_as_safe_outcomes(
    monkeypatch,
    tmp_path: Path,
) -> None:
    insufficient = generate_grounded_answer(
        query="What is the mobile login flow?",
        retrieved_results=[_evidence()],
        sufficiency=EvidenceSufficiencyDecision(
            is_sufficient=False,
            reason="Top score is below the required threshold.",
            result_count=1,
            top_score=0.1,
        ),
        llm_client=_FakeLLMClient("must not be called"),
    )
    invalid_citation = generate_grounded_answer(
        query="What changed in R24?",
        retrieved_results=[_evidence()],
        sufficiency=EvidenceSufficiencyDecision(
            is_sufficient=True,
            reason="Evidence passed.",
            result_count=1,
            top_score=0.82,
        ),
        llm_client=_FakeLLMClient(
            "DECISION: ANSWER\nUnsupported claim [C99]."
        ),
        model="test-model",
    )
    responses = iter(
        [
            _as_query_response(insufficient),
            _as_query_response(invalid_citation),
        ]
    )
    monkeypatch.setattr(
        conversation_route,
        "execute_query_request",
        lambda *args, **kwargs: next(responses),
    )

    with SqliteConversationStore(tmp_path / "chat.sqlite3") as store:
        client = _client(store)
        conversation_id = client.post(
            "/conversations", json={"title": "Safety"}
        ).json()["conversation_id"]
        first = client.post(
            f"/conversations/{conversation_id}/messages",
            json={"content": "What is the mobile login flow?"},
        )
        second = client.post(
            f"/conversations/{conversation_id}/messages",
            json={"content": "What changed in R24?"},
        )

        assert first.status_code == 200
        assert first.json()["answer"]["is_answered"] is False
        assert "sufficient evidence" in first.json()["answer"]["answer"]
        assert second.status_code == 200
        assert second.json()["answer"]["is_answered"] is False
        assert "citation validation failed" in (
            second.json()["answer"]["answer"].lower()
        )
        stored = store.list_messages(conversation_id)
        assert [message.role for message in stored] == [
            MessageRole.USER,
            MessageRole.ASSISTANT,
            MessageRole.USER,
            MessageRole.ASSISTANT,
        ]
        assert "Unsupported claim" not in stored[-1].content
