import subprocess
import sys

import httpx

from scripts.run_api_smoke_test import (
    _build_query_payload,
    _extract_success_json,
    run_api_smoke_test,
)


class FakeHttpClient:
    def __init__(self, readiness_status_code: int = 200) -> None:
        self.get_calls: list[str] = []
        self.post_calls: list[tuple[str, dict]] = []
        self.readiness_status_code = readiness_status_code

    def get(self, url: str) -> httpx.Response:
        self.get_calls.append(url)
        if url.endswith("/ready"):
            return httpx.Response(
                status_code=self.readiness_status_code,
                json={
                    "status": "ready" if self.readiness_status_code < 400 else "not_ready",
                    "is_ready": self.readiness_status_code < 400,
                    "retrieval_mode": "hybrid",
                    "qdrant_required_for_current_mode": True,
                    "lexical_artifacts_required_for_current_mode": True,
                    "checks": [
                        {
                            "name": "qdrant_collection",
                            "required": True,
                            "is_ready": self.readiness_status_code < 400,
                            "detail": "secret-readiness-detail-should-not-leak",
                        }
                    ],
                },
            )
        return httpx.Response(
            status_code=200,
            json={
                "status": "ok",
                "retrieval_mode": "hybrid",
                "qdrant_required_for_current_mode": True,
            },
        )

    def post(self, url: str, json: dict) -> httpx.Response:
        self.post_calls.append((url, json))
        return httpx.Response(
            status_code=200,
            json={
                "query": json["query"],
                "answer": "Grounded answer [C1].",
                "is_answered": True,
                "retrieval_mode": "hybrid",
                "trace_id": "trace-1",
                "sufficiency": {"is_sufficient": True, "reason": "ok"},
                "citations": [],
            },
        )


def test_api_smoke_script_help_runs() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/run_api_smoke_test.py", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert "--base-url" in result.stdout
    assert "--query" in result.stdout
    assert "--check-ready" in result.stdout


def test_run_api_smoke_test_calls_health_only_when_query_is_omitted() -> None:
    client = FakeHttpClient()

    result = run_api_smoke_test(
        client=client,
        base_url="http://localhost:8000/",
    )

    assert client.get_calls == ["http://localhost:8000/health"]
    assert client.post_calls == []
    assert result.health_payload["status"] == "ok"
    assert result.readiness_payload is None
    assert result.query_payload is None


def test_run_api_smoke_test_optionally_calls_readiness_before_query() -> None:
    client = FakeHttpClient()

    result = run_api_smoke_test(
        client=client,
        base_url="http://localhost:8000",
        query="What changed in branch reports?",
        check_ready=True,
        limit=3,
    )

    assert client.get_calls == ["http://localhost:8000/health", "http://localhost:8000/ready"]
    assert client.post_calls == [
        (
            "http://localhost:8000/query",
            {
                "query": "What changed in branch reports?",
                "limit": 3,
            },
        )
    ]
    assert result.readiness_payload is not None
    assert result.readiness_payload["is_ready"] is True
    assert result.query_payload is not None


def test_run_api_smoke_test_readiness_failure_blocks_query_without_leaking_body() -> None:
    client = FakeHttpClient(readiness_status_code=503)

    try:
        run_api_smoke_test(
            client=client,
            base_url="http://localhost:8000",
            query="What changed in branch reports?",
            check_ready=True,
        )
    except RuntimeError as exc:
        assert str(exc) == "GET /ready failed with HTTP 503"
        assert "secret-readiness-detail" not in str(exc)
    else:
        raise AssertionError("Expected RuntimeError for failed readiness check")

    assert client.get_calls == ["http://localhost:8000/health", "http://localhost:8000/ready"]
    assert client.post_calls == []


def test_run_api_smoke_test_calls_query_when_query_is_supplied() -> None:
    client = FakeHttpClient()

    result = run_api_smoke_test(
        client=client,
        base_url="http://localhost:8000",
        query="What changed in branch reports?",
        check_ready=False,
        limit=3,
        document_family="ASNB",
        release_label="R24",
        source_kind="paragraph",
        min_top_score=0.25,
    )

    assert client.get_calls == ["http://localhost:8000/health"]
    assert client.post_calls == [
        (
            "http://localhost:8000/query",
            {
                "query": "What changed in branch reports?",
                "limit": 3,
                "document_family": "ASNB",
                "release_label": "R24",
                "source_kind": "paragraph",
                "min_top_score": 0.25,
            },
        )
    ]
    assert result.query_payload is not None
    assert result.query_payload["is_answered"] is True


def test_build_query_payload_omits_none_filters() -> None:
    payload = _build_query_payload(
        query="branch report",
        limit=5,
        document_family=None,
        release_label="R24",
        source_kind=None,
        min_top_score=None,
    )

    assert payload == {
        "query": "branch report",
        "limit": 5,
        "release_label": "R24",
    }


def test_extract_success_json_raises_safe_error_for_http_error() -> None:
    response = httpx.Response(
        status_code=500,
        json={"detail": "secret-token-should-not-leak"},
    )

    try:
        _extract_success_json(response, label="GET /health")
    except RuntimeError as exc:
        assert str(exc) == "GET /health failed with HTTP 500"
        assert "secret-token" not in str(exc)
    else:
        raise AssertionError("Expected RuntimeError for HTTP 500 response")