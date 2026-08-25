from __future__ import annotations

import asyncio
import json
from types import SimpleNamespace
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.api.main import create_app
from app.api import main as api_main
from app.api.routes import query as query_route
from app.mcp.adapter import (
    DISCLOSURE_DISABLED_MESSAGE,
    MCPRetrievalAdapter,
    MCPToolFailure,
    encode_mcp_result,
)
from app.mcp.server import create_mcp_server
from app.retrieval.knowledge_service import KnowledgeFetchResponse, KnowledgeSearchHit, KnowledgeSearchResponse
from app.retrieval.retrieval_config import RetrievalRuntimeConfig
from app.ui import streamlit_app


def _response(mode: str) -> KnowledgeSearchResponse:
    source_type = "fdd" if mode == "fdd" else "code"
    prefix = "fdd" if source_type == "fdd" else "code"
    return KnowledgeSearchResponse(
        query="How is the AML batch processed?",
        mode=mode,
        retrieval_mode="hybrid",
        ranking_scope="per_source_type" if mode == "combined" else "global",
        results=(
            KnowledgeSearchHit(
                id=f"{prefix}_{'a' * 64}",
                title="Approved source",
                source_type=source_type,
                short_excerpt="Approved bounded excerpt.",
                score=0.9,
                metadata={"generation": "test"},
                source_reference=f"{source_type}:approved#source=test",
            ),
        ),
    )


class FakeService:
    def __init__(self) -> None:
        self.search_calls: list[tuple[str, str, int]] = []
        self.fetch_calls: list[str] = []

    def search(self, *, query: str, mode: str, limit: int) -> KnowledgeSearchResponse:
        self.search_calls.append((query, mode, limit))
        return _response(mode)

    def fetch(self, public_id: str) -> KnowledgeFetchResponse:
        self.fetch_calls.append(public_id)
        return KnowledgeFetchResponse(
            id=public_id,
            title="Approved source",
            text="Exact approved source text.",
            source_type="fdd",
            metadata={"generation": "test"},
            source_reference="fdd:approved#source=test",
        )


def _settings(*, disclosure_enabled: bool, interface_mode: str = "both"):
    return SimpleNamespace(
        mcp_evidence_disclosure_enabled=disclosure_enabled,
        interface_mode=interface_mode,
        log_level="INFO",
        retrieval_mode="lexical",
        hybrid_dense_weight=0.4,
        hybrid_lexical_weight=0.6,
        hybrid_candidate_limit=10,
        fdd_retrieval_artifact_dir=Path("data/processed"),
        code_modes_enabled=False,
    )


@pytest.mark.parametrize("mode", ["fdd", "code", "combined"])
def test_fastapi_search_and_mcp_adapter_return_equivalent_shared_results(
    monkeypatch,
    mode: str,
) -> None:
    service = FakeService()
    settings = _settings(disclosure_enabled=True)
    retrieval_config = RetrievalRuntimeConfig(
        retrieval_mode="hybrid",
        hybrid_dense_weight=0.4,
        hybrid_lexical_weight=0.6,
        hybrid_candidate_limit=10,
    )
    monkeypatch.setattr(query_route, "get_settings", lambda: settings)
    monkeypatch.setattr(query_route, "build_retrieval_runtime_config", lambda _: retrieval_config)
    monkeypatch.setattr(query_route, "build_knowledge_retrieval_service", lambda **_: service)

    api_response = TestClient(create_app()).post(
        "/search",
        json={"query": "How is the AML batch processed?", "mode": mode},
    )
    adapter = MCPRetrievalAdapter(
        settings_provider=lambda: settings,
        service_factory=lambda _: service,
    )

    assert api_response.status_code == 200
    assert api_response.json() == adapter.search(
        query="How is the AML batch processed?", mode=mode
    ).model_dump(mode="json")
    assert service.search_calls == [
        ("How is the AML batch processed?", mode, 5),
        ("How is the AML batch processed?", mode, 5),
    ]


def test_disclosure_disabled_performs_no_service_work_and_leaks_no_evidence() -> None:
    service = FakeService()
    adapter = MCPRetrievalAdapter(
        settings_provider=lambda: _settings(disclosure_enabled=False),
        service_factory=lambda _: service,
    )

    with pytest.raises(MCPToolFailure, match="^Evidence disclosure is disabled\\.$"):
        adapter.search(query="internal query", mode="combined")
    with pytest.raises(MCPToolFailure, match="^Evidence disclosure is disabled\\.$"):
        adapter.fetch(public_id=f"fdd_{'a' * 64}")

    assert service.search_calls == []
    assert service.fetch_calls == []


def test_mcp_structured_output_and_text_share_one_canonical_payload() -> None:
    encoded = encode_mcp_result(_response("fdd"))

    assert encoded.is_error is False
    assert encoded.structured_content == json.loads(encoded.content[0].text)
    assert encoded.structured_content["results"][0]["short_excerpt"] == "Approved bounded excerpt."


def test_mcp_server_disabled_disclosure_returns_only_generic_tool_error() -> None:
    service = FakeService()
    settings = _settings(disclosure_enabled=False)
    adapter = MCPRetrievalAdapter(
        settings_provider=lambda: settings,
        service_factory=lambda _: service,
    )
    server = create_mcp_server(settings=settings, adapter=adapter)

    result = asyncio.run(
        server.call_tool(
            "search",
            {"query": "internal query", "mode": "combined"},
        )
    )

    assert result.is_error is True
    assert result.structured_content is None
    assert [item.text for item in result.content] == [DISCLOSURE_DISABLED_MESSAGE]
    assert service.search_calls == []


def test_mcp_server_publishes_only_read_only_closed_world_tools() -> None:
    settings = _settings(disclosure_enabled=True)
    server = create_mcp_server(settings=settings, adapter=MCPRetrievalAdapter(
        settings_provider=lambda: settings,
        service_factory=lambda _: FakeService(),
    ))

    tools = asyncio.run(server.list_tools())

    assert [tool.name for tool in tools] == ["search", "fetch"]
    for tool in tools:
        assert tool.annotations.read_only_hint is True
        assert tool.annotations.destructive_hint is False
        assert tool.annotations.open_world_hint is False
        assert tool.output_schema is not None


def test_mcp_server_refuses_startup_in_fastapi_only_mode() -> None:
    with pytest.raises(RuntimeError, match="MCP is disabled"):
        create_mcp_server(settings=_settings(disclosure_enabled=False, interface_mode="fastapi"))


def test_fastapi_refuses_startup_in_mcp_only_mode(monkeypatch) -> None:
    settings = _settings(disclosure_enabled=False, interface_mode="mcp")
    monkeypatch.setattr(api_main, "get_settings", lambda: settings)

    with pytest.raises(RuntimeError, match="FastAPI is disabled"):
        api_main.create_app()


def test_streamlit_refuses_startup_in_mcp_only_mode(monkeypatch) -> None:
    monkeypatch.setattr(
        streamlit_app,
        "get_settings",
        lambda: _settings(disclosure_enabled=False, interface_mode="mcp"),
    )

    with pytest.raises(RuntimeError, match="Streamlit is disabled"):
        streamlit_app.main()
