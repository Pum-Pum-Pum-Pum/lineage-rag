from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from app.api.main import create_app
from app.api.routes import query as query_route
from app.llm.answer_contract import Citation, GroundedAnswerResponse
from app.code_retrieval.answer_contract import CodeAnswerResponse, CodeCitation
from app.fdd_code_lineage.combined_answer import (
    CombinedAnswerResponse,
    CombinedSectionResponse,
)
from app.retrieval.evidence_sufficiency import EvidenceSufficiencyDecision
from app.retrieval.retrieval_config import RetrievalRuntimeConfig
from app.services.answer_orchestration import AnswerOrchestrationResult
from app.services.answer_trace import build_answer_trace
from app.vectorstore.qdrant_search import QdrantSearchResult


def _settings(tmp_path: Path) -> SimpleNamespace:
    return SimpleNamespace(
        app_name="Test RAG API",
        retrieval_min_top_score=0.25,
        qdrant_local_path=tmp_path / "qdrant",
        qdrant_collection_name="lineage_chunks",
        openai_embedding_model="test-embedding-model",
        processed_dir=tmp_path / "processed",
        exports_dir=tmp_path / "exports",
    )


def _retrieval_config(mode: str = "hybrid") -> RetrievalRuntimeConfig:
    return RetrievalRuntimeConfig(
        retrieval_mode=mode,
        hybrid_dense_weight=0.6,
        hybrid_lexical_weight=0.4,
        hybrid_candidate_limit=10,
    )


class FakeQdrantClient:
    def __init__(self, collection_exists: bool = True) -> None:
        self.collection_exists_value = collection_exists
        self.collection_exists_calls: list[str] = []
        self.closed = False

    def collection_exists(self, collection_name: str) -> bool:
        self.collection_exists_calls.append(collection_name)
        return self.collection_exists_value

    def close(self) -> None:
        self.closed = True


class FailingCollectionCheckQdrantClient(FakeQdrantClient):
    def collection_exists(self, collection_name: str) -> bool:
        self.collection_exists_calls.append(collection_name)
        raise RuntimeError("secret-qdrant-check-detail")


def _orchestration_result(tmp_path: Path, retrieval_mode: str = "hybrid") -> AnswerOrchestrationResult:
    retrieved_result = QdrantSearchResult(
        point_id="point-1",
        score=0.82,
        payload={
            "unit_id": "FS_ASNB_R24::chunk_1",
            "text": "Branch report evidence",
            "document_family": "ASNB",
            "release_label": "R24",
            "source_kind": "paragraph",
        },
    )
    sufficiency = EvidenceSufficiencyDecision(
        is_sufficient=True,
        reason="Retrieved evidence passed baseline sufficiency checks.",
        result_count=1,
        top_score=0.82,
    )
    response = GroundedAnswerResponse(
        query="What changed in branch reports?",
        answer="Grounded answer [C1].",
        is_answered=True,
        refusal_reason=None,
        citations=[
            Citation(
                unit_id="FS_ASNB_R24::chunk_1",
                document_family="ASNB",
                release_label="R24",
                source_kind="paragraph",
                score=0.82,
                text_preview="Branch report evidence",
            )
        ],
    )
    trace = build_answer_trace(
        query="What changed in branch reports?",
        filters={"document_family": "ASNB", "release_label": "R24", "source_kind": "paragraph"},
        sufficiency=sufficiency,
        answer_response=response,
        retrieval_results=[retrieved_result],
        request_id="api-test-request",
        retrieval_metadata={"retrieval_mode": retrieval_mode, "limit": 3},
    )
    return AnswerOrchestrationResult(
        retrieval_mode=retrieval_mode,
        retrieval_results=[retrieved_result],
        sufficiency=sufficiency,
        answer_response=response,
        trace=trace,
        trace_output_path=tmp_path / "exports" / "answer_runs" / "api-test-request.json",
    )


def test_query_endpoint_calls_orchestration_and_formats_response(monkeypatch, tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    retrieval_config = _retrieval_config("hybrid")
    fake_client = FakeQdrantClient(collection_exists=True)
    captured: dict[str, object] = {}

    def fake_run_grounded_answer_query(**kwargs):
        captured["orchestration_kwargs"] = kwargs
        return _orchestration_result(tmp_path, retrieval_mode="hybrid")

    monkeypatch.setattr(query_route, "get_settings", lambda: settings)
    monkeypatch.setattr(query_route, "build_retrieval_runtime_config", lambda loaded_settings: retrieval_config)
    monkeypatch.setattr(query_route, "create_persistent_qdrant_client", lambda path: fake_client)
    monkeypatch.setattr(query_route, "run_grounded_answer_query", fake_run_grounded_answer_query)

    response = TestClient(create_app()).post(
        "/query",
        headers={"X-Request-ID": "query-correlation-123"},
        json={
            "query": "What changed in branch reports?",
            "limit": 3,
            "document_family": "ASNB",
            "release_label": "R24",
            "source_kind": "paragraph",
            "min_top_score": 0.25,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["query"] == "What changed in branch reports?"
    assert payload["answer"] == "Grounded answer [C1]."
    assert payload["is_answered"] is True
    assert payload["retrieval_mode"] == "hybrid"
    assert payload["trace_id"] == "api-test-request"
    assert payload["citations"][0]["unit_id"] == "FS_ASNB_R24::chunk_1"
    assert payload["sufficiency"]["is_sufficient"] is True
    assert payload["retrieval_metadata"]["limit"] == 3

    orchestration_kwargs = captured["orchestration_kwargs"]
    assert orchestration_kwargs["qdrant_client"] is fake_client
    assert orchestration_kwargs["collection_name"] == "lineage_chunks"
    assert orchestration_kwargs["query_text"] == "What changed in branch reports?"
    assert orchestration_kwargs["embedding_model"] == "test-embedding-model"
    assert orchestration_kwargs["retrieval_config"] == retrieval_config
    assert orchestration_kwargs["lexical_artifact_directory"] == settings.processed_dir
    assert orchestration_kwargs["trace_output_directory"] == settings.exports_dir / "answer_runs"
    assert orchestration_kwargs["limit"] == 3
    assert orchestration_kwargs["min_top_score"] == 0.25
    assert orchestration_kwargs["document_family"] == "ASNB"
    assert orchestration_kwargs["release_label"] == "R24"
    assert orchestration_kwargs["source_kind"] == "paragraph"
    assert (
        orchestration_kwargs["correlation_id"]
        == "query-correlation-123"
    )
    assert response.headers["X-Request-ID"] == "query-correlation-123"
    assert fake_client.collection_exists_calls == ["lineage_chunks"]
    assert fake_client.closed is True


def test_query_endpoint_rejects_blank_query() -> None:
    response = TestClient(create_app()).post("/query", json={"query": "   "})

    assert response.status_code == 422


def test_code_mode_is_fail_closed_until_explicitly_enabled(monkeypatch, tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    settings.code_modes_enabled = False
    monkeypatch.setattr(query_route, "get_settings", lambda: settings)
    monkeypatch.setattr(
        query_route,
        "build_retrieval_runtime_config",
        lambda loaded_settings: _retrieval_config("hybrid"),
    )

    response = TestClient(create_app()).post(
        "/query", json={"query": "Explain the routine", "knowledge_mode": "code"}
    )

    assert response.status_code == 503
    assert response.json()["detail"] == "Code and combined knowledge modes are not activated."


def test_enabled_code_mode_uses_explicit_runtime_contract(monkeypatch, tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    settings.code_modes_enabled = True
    citation = CodeCitation(
        citation_id="C1",
        unit_id="unit-1",
        snapshot_id="snapshot-1",
        source_path="pkg_custom.sql",
        display_name="PROCESS_AML",
        source_kind="procedure",
        start_line=10,
        end_line=20,
        score=0.9,
        text_preview="PROCEDURE PROCESS_AML",
    )
    result = SimpleNamespace(
        mode="code",
        answer=CodeAnswerResponse(
            query="Explain the routine",
            analysis_kind="explanation",
            answer="Visible behavior [C1].",
            is_answered=True,
            citations=(citation,),
        ),
        answer_call={
            "model": "test-chat-model",
            "prompt_tokens": 100,
            "completion_tokens": 20,
            "total_tokens": 120,
        },
        trace_id="code-trace",
        trace_output_path=tmp_path / "code-trace.json",
    )
    monkeypatch.setattr(query_route, "get_settings", lambda: settings)
    monkeypatch.setattr(
        query_route,
        "build_retrieval_runtime_config",
        lambda loaded_settings: _retrieval_config("hybrid"),
    )
    monkeypatch.setattr(query_route, "run_code_or_combined_query", lambda **kwargs: result)

    response = TestClient(create_app()).post(
        "/query", json={"query": "Explain the routine", "knowledge_mode": "code"}
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["knowledge_mode"] == "code"
    assert payload["requested_claim_supported"] is True
    assert payload["code_citations"][0]["source_path"] == "pkg_custom.sql"
    assert payload["citations"] == []


def test_combined_contract_refusal_returns_200_with_explicit_unsupported_state(
    monkeypatch, tmp_path: Path,
) -> None:
    settings = _settings(tmp_path)
    settings.code_modes_enabled = True
    refused = CombinedSectionResponse(
        status="refused",
        text="The generated response did not satisfy the grounded-answer contract.",
    )
    answer = CombinedAnswerResponse(
        query="Explain the integration",
        requested_claim_supported=False,
        related_grounded_context_provided=False,
        documented_functionality=refused,
        visible_custom_implementation=refused,
        impact_and_likely_change_locations=refused,
        unknown_or_unavailable_behavior=CombinedSectionResponse(
            status="answered",
            text="No functional claim is returned because validation failed.",
        ),
    )
    result = SimpleNamespace(
        mode="combined",
        answer=answer,
        answer_call={
            "model": "test-chat", "prompt_tokens": 10,
            "completion_tokens": 3, "total_tokens": 13,
        },
        trace_id="combined-contract-refusal-trace",
        trace_output_path=tmp_path / "combined-contract-refusal.json",
    )
    monkeypatch.setattr(query_route, "get_settings", lambda: settings)
    monkeypatch.setattr(
        query_route, "build_retrieval_runtime_config",
        lambda loaded_settings: _retrieval_config("hybrid"),
    )
    monkeypatch.setattr(
        query_route, "run_code_or_combined_query", lambda **kwargs: result,
    )

    response = TestClient(create_app()).post(
        "/query",
        json={"query": "Explain the integration", "knowledge_mode": "combined"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["requested_claim_supported"] is False
    assert payload["is_answered"] is False
    assert payload["refusal_reason"] == "requested_claim_unsupported"
    assert payload["combined_sections"]["documented_functionality"]["status"] == "refused"


def test_code_mode_feature_flag_supports_atomic_enable_and_rollback(
    monkeypatch, tmp_path: Path
) -> None:
    settings = _settings(tmp_path)
    settings.code_modes_enabled = False
    calls = {"runtime": 0}
    answer = CodeAnswerResponse(
        query="Explain the routine",
        analysis_kind="explanation",
        answer="Visible behavior [C1].",
        is_answered=True,
        citations=(
            CodeCitation(
                citation_id="C1",
                unit_id="unit-1",
                snapshot_id="snapshot-1",
                source_path="pkg_custom.sql",
                display_name="PROCESS_AML",
                source_kind="procedure",
                start_line=1,
                end_line=2,
                score=0.9,
                text_preview="PROCESS_AML",
            ),
        ),
    )

    def fake_runtime(**kwargs):
        calls["runtime"] += 1
        return SimpleNamespace(
            mode="code",
            answer=answer,
            answer_call={
                "model": "test-chat",
                "prompt_tokens": 10,
                "completion_tokens": 5,
                "total_tokens": 15,
            },
            trace_id="rollback-trace",
            trace_output_path=tmp_path / "rollback-trace.json",
        )

    monkeypatch.setattr(query_route, "get_settings", lambda: settings)
    monkeypatch.setattr(
        query_route,
        "build_retrieval_runtime_config",
        lambda loaded_settings: _retrieval_config("hybrid"),
    )
    monkeypatch.setattr(query_route, "run_code_or_combined_query", fake_runtime)
    client = TestClient(create_app())
    body = {"query": "Explain the routine", "knowledge_mode": "code"}

    assert client.post("/query", json=body).status_code == 503
    settings.code_modes_enabled = True
    assert client.post("/query", json=body).status_code == 200
    settings.code_modes_enabled = False
    assert client.post("/query", json=body).status_code == 503
    assert calls["runtime"] == 1


def test_query_endpoint_skips_qdrant_collection_check_for_lexical(monkeypatch, tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    retrieval_config = _retrieval_config("lexical")
    captured: dict[str, object] = {}

    def fail_if_qdrant_created(path):
        raise AssertionError("Lexical query should not create a Qdrant client")

    def fake_run_grounded_answer_query(**kwargs):
        captured["orchestration_kwargs"] = kwargs
        return _orchestration_result(tmp_path, retrieval_mode="lexical")

    monkeypatch.setattr(query_route, "get_settings", lambda: settings)
    monkeypatch.setattr(query_route, "build_retrieval_runtime_config", lambda loaded_settings: retrieval_config)
    monkeypatch.setattr(query_route, "create_persistent_qdrant_client", fail_if_qdrant_created)
    monkeypatch.setattr(query_route, "run_grounded_answer_query", fake_run_grounded_answer_query)

    response = TestClient(create_app()).post("/query", json={"query": "exact branch report"})

    assert response.status_code == 200
    assert response.json()["retrieval_mode"] == "lexical"
    orchestration_kwargs = captured["orchestration_kwargs"]
    assert orchestration_kwargs["qdrant_client"] is None
    assert orchestration_kwargs["collection_name"] == "lineage_chunks"


@pytest.mark.parametrize("retrieval_mode", ["dense", "hybrid"])
def test_query_endpoint_returns_service_unavailable_when_required_qdrant_collection_is_missing(
    monkeypatch, tmp_path: Path, retrieval_mode: str
) -> None:
    settings = _settings(tmp_path)
    fake_client = FakeQdrantClient(collection_exists=False)

    monkeypatch.setattr(query_route, "get_settings", lambda: settings)
    monkeypatch.setattr(
        query_route,
        "build_retrieval_runtime_config",
        lambda loaded_settings: _retrieval_config(retrieval_mode),
    )
    monkeypatch.setattr(query_route, "create_persistent_qdrant_client", lambda path: fake_client)

    response = TestClient(create_app()).post("/query", json={"query": "branch report"})

    assert response.status_code == 503
    assert "Qdrant collection does not exist" in response.json()["detail"]
    assert fake_client.collection_exists_calls == ["lineage_chunks"]
    assert fake_client.closed is True


@pytest.mark.parametrize("retrieval_mode", ["dense", "hybrid"])
def test_query_endpoint_returns_safe_503_when_required_qdrant_client_creation_fails(
    monkeypatch, tmp_path: Path, retrieval_mode: str
) -> None:
    settings = _settings(tmp_path)

    def fail_client_creation(path):
        raise RuntimeError("secret-qdrant-client-path")

    def fail_if_orchestration_runs(**kwargs):
        raise AssertionError("Query orchestration should not run when Qdrant dependency check fails")

    monkeypatch.setattr(query_route, "get_settings", lambda: settings)
    monkeypatch.setattr(
        query_route,
        "build_retrieval_runtime_config",
        lambda loaded_settings: _retrieval_config(retrieval_mode),
    )
    monkeypatch.setattr(query_route, "create_persistent_qdrant_client", fail_client_creation)
    monkeypatch.setattr(query_route, "run_grounded_answer_query", fail_if_orchestration_runs)

    response = TestClient(create_app()).post("/query", json={"query": "branch report"})

    assert response.status_code == 503
    assert response.json()["detail"] == "Qdrant dependency check failed. Verify vector-store availability before querying."
    assert "secret-qdrant-client-path" not in response.text


@pytest.mark.parametrize("retrieval_mode", ["dense", "hybrid"])
def test_query_endpoint_returns_safe_503_when_required_qdrant_collection_check_fails(
    monkeypatch, tmp_path: Path, retrieval_mode: str
) -> None:
    settings = _settings(tmp_path)
    fake_client = FailingCollectionCheckQdrantClient(collection_exists=True)

    def fail_if_orchestration_runs(**kwargs):
        raise AssertionError("Query orchestration should not run when Qdrant dependency check fails")

    monkeypatch.setattr(query_route, "get_settings", lambda: settings)
    monkeypatch.setattr(
        query_route,
        "build_retrieval_runtime_config",
        lambda loaded_settings: _retrieval_config(retrieval_mode),
    )
    monkeypatch.setattr(query_route, "create_persistent_qdrant_client", lambda path: fake_client)
    monkeypatch.setattr(query_route, "run_grounded_answer_query", fail_if_orchestration_runs)

    response = TestClient(create_app()).post("/query", json={"query": "branch report"})

    assert response.status_code == 503
    assert response.json()["detail"] == "Qdrant dependency check failed. Verify vector-store availability before querying."
    assert "secret-qdrant-check-detail" not in response.text
    assert fake_client.collection_exists_calls == ["lineage_chunks"]
    assert fake_client.closed is True


def test_query_endpoint_returns_safe_error_for_unexpected_failure(monkeypatch, tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    fake_client = FakeQdrantClient(collection_exists=True)

    def failing_orchestration(**kwargs):
        raise RuntimeError("secret-token-should-not-leak")

    monkeypatch.setattr(query_route, "get_settings", lambda: settings)
    monkeypatch.setattr(query_route, "build_retrieval_runtime_config", lambda loaded_settings: _retrieval_config("hybrid"))
    monkeypatch.setattr(query_route, "create_persistent_qdrant_client", lambda path: fake_client)
    monkeypatch.setattr(query_route, "run_grounded_answer_query", failing_orchestration)

    response = TestClient(create_app()).post("/query", json={"query": "branch report"})

    assert response.status_code == 500
    assert response.json()["detail"] == "Internal query processing error."
    assert "secret-token" not in response.text
    assert fake_client.closed is True
