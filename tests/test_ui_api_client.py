import json

import httpx
import pytest

from app.schemas.query_api import QueryRequest
from app.schemas.conversation_api import (
    ConversationMessageRequest,
    CreateConversationRequest,
)
from app.ui.api_client import RagApiClient, UiApiError


def test_ui_api_client_returns_typed_health_response() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert request.method == "GET"
        assert str(request.url) == "http://rag.test/health"
        return httpx.Response(200, json=_health_payload())

    with _client(handler) as api:
        health = api.get_health()

    assert health.status == "ok"
    assert health.retrieval_mode == "hybrid"
    assert health.qdrant_required_for_current_mode is True


def test_ui_api_client_returns_typed_readiness_response() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_readiness_payload())

    with _client(handler) as api:
        readiness = api.get_readiness()

    assert readiness.is_ready is True
    assert readiness.checks[0].name == "qdrant_collection"


def test_ui_api_client_posts_validated_query_payload() -> None:
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["payload"] = json.loads(request.content)
        return httpx.Response(200, json=_query_payload())

    with _client(handler) as api:
        result = api.query(
            QueryRequest(
                query="  What changed in branch reports?  ",
                limit=3,
                release_label="R24",
            )
        )

    assert captured["payload"] == {
        "query": "What changed in branch reports?",
        "limit": 3,
        "release_label": "R24",
    }
    assert result.is_answered is True
    assert result.citations[0].unit_id == "unit-1"


def test_ui_api_client_supports_conversation_lifecycle_and_turns() -> None:
    calls: list[tuple[str, str, dict | None]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content) if request.content else None
        calls.append((request.method, request.url.path, payload))
        if request.method == "POST" and request.url.path == "/conversations":
            return httpx.Response(201, json=_conversation_payload())
        if request.method == "GET" and request.url.path == "/conversations":
            assert request.url.params["include_archived"] == "false"
            return httpx.Response(200, json=[_conversation_payload()])
        if request.method == "GET":
            return httpx.Response(200, json=_conversation_detail_payload())
        if request.url.path.endswith("/messages"):
            return httpx.Response(200, json=_conversation_turn_payload())
        if request.url.path.endswith("/archive"):
            archived = _conversation_payload()
            archived["is_archived"] = True
            return httpx.Response(200, json=archived)
        raise AssertionError(f"Unexpected request: {request.method} {request.url}")

    with _client(handler) as api:
        created = api.create_conversation(
            CreateConversationRequest(title="Release chat")
        )
        listed = api.list_conversations()
        detail = api.get_conversation("conversation-1")
        turn = api.submit_conversation_message(
            "conversation-1",
            ConversationMessageRequest(content="What changed?"),
        )
        archived = api.archive_conversation("conversation-1")

    assert created.title == "Release chat"
    assert listed[0].conversation_id == "conversation-1"
    assert detail.messages == []
    assert turn.assistant_message.trace_id == "trace-1"
    assert turn.answer.is_answered is True
    assert archived.is_archived is True
    assert calls[0][2] == {"title": "Release chat"}
    assert calls[3][2] == {"content": "What changed?", "limit": 5}


@pytest.mark.parametrize(
    ("status_code", "expected_code"),
    [
        (404, "not_found"),
        (409, "archived"),
        (413, "context_too_large"),
        (503, "not_ready"),
        (500, "http_error"),
    ],
)
def test_ui_api_client_maps_conversation_failures_without_leaking_body(
    status_code: int,
    expected_code: str,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            status_code,
            json={"detail": "secret-server-detail"},
        )

    with _client(handler) as api:
        with pytest.raises(UiApiError) as exc_info:
            api.get_conversation("conversation-1")

    assert exc_info.value.code == expected_code
    assert exc_info.value.status_code == status_code
    assert "secret-server-detail" not in str(exc_info.value)


def test_ui_api_client_maps_readiness_503_without_leaking_body() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            503,
            json={"detail": "secret-local-path-and-backend-state"},
        )

    with _client(handler) as api:
        with pytest.raises(UiApiError) as exc_info:
            api.get_readiness()

    assert exc_info.value.code == "not_ready"
    assert exc_info.value.status_code == 503
    assert "secret-local-path" not in str(exc_info.value)


def test_ui_api_client_maps_timeout_to_safe_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ReadTimeout("secret-timeout-detail", request=request)

    with _client(handler) as api:
        with pytest.raises(UiApiError) as exc_info:
            api.get_health()

    assert exc_info.value.code == "timeout"
    assert "secret-timeout-detail" not in str(exc_info.value)


def test_ui_api_client_maps_connection_failure_to_safe_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("secret-host-detail", request=request)

    with _client(handler) as api:
        with pytest.raises(UiApiError) as exc_info:
            api.get_health()

    assert exc_info.value.code == "unavailable"
    assert "secret-host-detail" not in str(exc_info.value)


@pytest.mark.parametrize(
    "response",
    [
        httpx.Response(200, content=b"not-json"),
        httpx.Response(200, json={"status": "ok"}),
    ],
)
def test_ui_api_client_rejects_malformed_or_schema_invalid_response(
    response: httpx.Response,
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return response

    with _client(handler) as api:
        with pytest.raises(UiApiError) as exc_info:
            api.get_health()

    assert exc_info.value.code == "invalid_response"
    assert str(exc_info.value) == "The RAG API returned an invalid response."


def _client(handler) -> RagApiClient:
    transport = httpx.MockTransport(handler)
    http_client = httpx.Client(transport=transport)
    return RagApiClient(
        "http://rag.test/",
        timeout=2.0,
        client=http_client,
    )


def _health_payload() -> dict[str, object]:
    return {
        "status": "ok",
        "app_name": "Culling Blade Lineage RAG",
        "environment": "test",
        "retrieval_mode": "hybrid",
        "hybrid_dense_weight": 0.5,
        "hybrid_lexical_weight": 0.5,
        "hybrid_candidate_limit": 10,
        "retrieval_min_top_score": 0.25,
        "qdrant_collection_name": "lineage_chunks",
        "qdrant_required_for_current_mode": True,
    }


def _readiness_payload() -> dict[str, object]:
    return {
        "status": "ready",
        "is_ready": True,
        "app_name": "Culling Blade Lineage RAG",
        "environment": "test",
        "retrieval_mode": "hybrid",
        "qdrant_required_for_current_mode": True,
        "lexical_artifacts_required_for_current_mode": True,
        "checks": [
            {
                "name": "qdrant_collection",
                "required": True,
                "is_ready": True,
                "detail": "Collection exists.",
            }
        ],
    }


def _query_payload() -> dict[str, object]:
    return {
        "query": "What changed in branch reports?",
        "answer": "Branch report evidence changed [C1].",
        "is_answered": True,
        "refusal_reason": None,
        "retrieval_mode": "hybrid",
        "citations": [
            {
                "unit_id": "unit-1",
                "document_family": "ASNB",
                "release_label": "R24",
                "source_kind": "paragraph",
                "score": 0.82,
                "text_preview": "Branch report evidence.",
            }
        ],
        "sufficiency": {
            "is_sufficient": True,
            "reason": "Enough grounded evidence.",
            "result_count": 1,
            "top_score": 0.82,
        },
        "trace_id": "trace-1",
        "trace_output_path": "data/traces/trace-1.json",
        "retrieval_metadata": None,
        "usage": None,
        "cost": None,
    }


def _conversation_payload() -> dict[str, object]:
    return {
        "conversation_id": "conversation-1",
        "title": "Release chat",
        "created_at_utc": "2026-07-29T00:00:00Z",
        "updated_at_utc": "2026-07-29T00:00:00Z",
        "is_archived": False,
    }


def _conversation_detail_payload() -> dict[str, object]:
    return {
        "conversation": _conversation_payload(),
        "messages": [],
        "summary": None,
    }


def _conversation_turn_payload() -> dict[str, object]:
    return {
        "user_message": {
            "message_id": "message-1",
            "conversation_id": "conversation-1",
            "sequence_number": 1,
            "role": "user",
            "content": "What changed?",
            "created_at_utc": "2026-07-29T00:00:00Z",
            "trace_id": None,
        },
        "assistant_message": {
            "message_id": "message-2",
            "conversation_id": "conversation-1",
            "sequence_number": 2,
            "role": "assistant",
            "content": "Grounded answer [C1].",
            "created_at_utc": "2026-07-29T00:00:01Z",
            "trace_id": "trace-1",
        },
        "answer": _query_payload(),
        "context_estimated_tokens": 20,
        "context_budget_tokens": 100,
        "summarized_through_sequence": 0,
    }
