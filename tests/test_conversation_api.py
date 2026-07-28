from pathlib import Path

from fastapi import HTTPException
from fastapi.testclient import TestClient

from app.api.main import create_app
from app.api.routes import conversations as conversation_route
from app.conversation.context import ContextBudgetExceededError
from app.conversation.models import MessageRole
from app.conversation.store import SqliteConversationStore
from app.schemas.query_api import (
    EvidenceSufficiencyResponse,
    QueryResponse,
)


def fake_answer(query: str = "What changed?") -> QueryResponse:
    return QueryResponse(
        query=query,
        answer="Grounded answer [C1].",
        is_answered=True,
        refusal_reason=None,
        retrieval_mode="lexical",
        citations=[],
        sufficiency=EvidenceSufficiencyResponse(
            is_sufficient=True,
            reason="Evidence passed.",
            result_count=1,
            top_score=0.8,
        ),
        trace_id="trace-123",
        trace_output_path="safe-test-path.json",
    )


def client_for_store(store: SqliteConversationStore) -> TestClient:
    app = create_app()
    app.dependency_overrides[
        conversation_route.get_conversation_store
    ] = lambda: store
    return TestClient(app)


def test_conversation_lifecycle_returns_history_and_hides_archive(
    tmp_path: Path,
) -> None:
    with SqliteConversationStore(tmp_path / "chat.sqlite3") as store:
        client = client_for_store(store)

        created_response = client.post(
            "/conversations",
            json={"title": "  Release investigation  "},
        )
        assert created_response.status_code == 201
        created = created_response.json()
        conversation_id = created["conversation_id"]
        assert created["title"] == "Release investigation"
        assert client.get("/conversations").json() == [created]

        store.add_message(
            conversation_id,
            MessageRole.USER,
            "What changed?",
        )
        store.save_summary(
            conversation_id,
            "The user asks about changes.",
            summarized_through_sequence=1,
        )

        detail_response = client.get(f"/conversations/{conversation_id}")
        assert detail_response.status_code == 200
        detail = detail_response.json()
        assert detail["messages"][0]["sequence_number"] == 1
        assert detail["summary"]["summarized_through_sequence"] == 1
        assert detail["summary"]["version"] == 1

        archive_response = client.post(
            f"/conversations/{conversation_id}/archive"
        )
        assert archive_response.status_code == 200
        assert archive_response.json()["is_archived"] is True
        assert client.get("/conversations").json() == []
        archived_list = client.get(
            "/conversations?include_archived=true"
        ).json()
        assert len(archived_list) == 1
        assert archived_list[0]["conversation_id"] == conversation_id


def test_submit_message_persists_grounded_turn_and_bounded_context(
    monkeypatch,
    tmp_path: Path,
) -> None:
    captured: dict[str, object] = {}
    with SqliteConversationStore(tmp_path / "chat.sqlite3") as store:
        conversation = store.create_conversation()
        client = client_for_store(store)

        def fake_execute(request, *, conversation_context=None):
            captured["request"] = request
            captured["context"] = conversation_context
            return fake_answer(request.query)

        monkeypatch.setattr(
            conversation_route,
            "execute_query_request",
            fake_execute,
        )

        response = client.post(
            f"/conversations/{conversation.conversation_id}/messages",
            json={
                "content": "What changed in R24?",
                "limit": 3,
                "release_label": "R24",
            },
        )

        assert response.status_code == 200
        payload = response.json()
        assert payload["user_message"]["sequence_number"] == 1
        assert payload["assistant_message"]["sequence_number"] == 2
        assert payload["assistant_message"]["trace_id"] == "trace-123"
        assert payload["answer"]["answer"] == "Grounded answer [C1]."
        assert payload["context_estimated_tokens"] <= payload[
            "context_budget_tokens"
        ]

        request = captured["request"]
        assert request.query == "What changed in R24?"
        assert request.limit == 3
        assert request.release_label == "R24"
        context = captured["context"]
        assert "<conversation_memory>" in context
        assert "What changed in R24?" in context

        stored = store.list_messages(conversation.conversation_id)
        assert [message.role for message in stored] == [
            MessageRole.USER,
            MessageRole.ASSISTANT,
        ]
        assert stored[1].trace_id == "trace-123"


def test_conversation_history_is_isolated_and_unknown_id_returns_404(
    tmp_path: Path,
) -> None:
    with SqliteConversationStore(tmp_path / "chat.sqlite3") as store:
        first = store.create_conversation("First")
        second = store.create_conversation("Second")
        store.add_message(first.conversation_id, MessageRole.USER, "Private")
        client = client_for_store(store)

        second_detail = client.get(
            f"/conversations/{second.conversation_id}"
        )
        missing = client.get("/conversations/missing")

        assert second_detail.status_code == 200
        assert second_detail.json()["messages"] == []
        assert missing.status_code == 404
        assert missing.json()["detail"] == "Conversation not found."
        assert "Private" not in second_detail.text


def test_archived_conversation_rejects_message_without_query(
    monkeypatch,
    tmp_path: Path,
) -> None:
    with SqliteConversationStore(tmp_path / "chat.sqlite3") as store:
        conversation = store.create_conversation()
        store.archive_conversation(conversation.conversation_id)
        client = client_for_store(store)

        def fail_if_called(*args, **kwargs):
            raise AssertionError("Archived turn must not run a query")

        monkeypatch.setattr(
            conversation_route,
            "execute_query_request",
            fail_if_called,
        )

        response = client.post(
            f"/conversations/{conversation.conversation_id}/messages",
            json={"content": "Late question"},
        )

        assert response.status_code == 409
        assert response.json()["detail"] == (
            "Archived conversations are read-only."
        )
        assert store.list_messages(conversation.conversation_id) == []


def test_query_failure_preserves_user_message_but_not_assistant_message(
    monkeypatch,
    tmp_path: Path,
) -> None:
    with SqliteConversationStore(tmp_path / "chat.sqlite3") as store:
        conversation = store.create_conversation()
        client = client_for_store(store)

        def fail_query(*args, **kwargs):
            raise HTTPException(
                status_code=503,
                detail="Retrieval dependency unavailable.",
            )

        monkeypatch.setattr(
            conversation_route,
            "execute_query_request",
            fail_query,
        )

        response = client.post(
            f"/conversations/{conversation.conversation_id}/messages",
            json={"content": "Persist my failed attempt"},
        )

        assert response.status_code == 503
        assert response.json()["detail"] == "Retrieval dependency unavailable."
        messages = store.list_messages(conversation.conversation_id)
        assert [message.role for message in messages] == [MessageRole.USER]


def test_context_overflow_is_safe_and_does_not_run_grounded_query(
    monkeypatch,
    tmp_path: Path,
) -> None:
    with SqliteConversationStore(tmp_path / "chat.sqlite3") as store:
        conversation = store.create_conversation()
        client = client_for_store(store)

        def overflow(*args, **kwargs):
            raise ContextBudgetExceededError("secret budget details")

        def fail_query(*args, **kwargs):
            raise AssertionError("Overflow must stop before grounded query")

        monkeypatch.setattr(
            conversation_route,
            "build_conversation_context",
            overflow,
        )
        monkeypatch.setattr(
            conversation_route,
            "execute_query_request",
            fail_query,
        )

        response = client.post(
            f"/conversations/{conversation.conversation_id}/messages",
            json={"content": "Large context trigger"},
        )

        assert response.status_code == 413
        assert "configured token budget" in response.json()["detail"]
        assert "secret budget details" not in response.text
        messages = store.list_messages(conversation.conversation_id)
        assert [message.role for message in messages] == [MessageRole.USER]


def test_message_contract_rejects_blank_or_unsupported_input(
    tmp_path: Path,
) -> None:
    with SqliteConversationStore(tmp_path / "chat.sqlite3") as store:
        conversation = store.create_conversation()
        client = client_for_store(store)
        url = f"/conversations/{conversation.conversation_id}/messages"

        blank = client.post(url, json={"content": " "})
        unsupported = client.post(
            url,
            json={"content": "Question", "source_kind": "image"},
        )

        assert blank.status_code == 422
        assert unsupported.status_code == 422
        assert store.list_messages(conversation.conversation_id) == []
