import json

import httpx
import pytest

from app.schemas.query_api import QueryRequest
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
