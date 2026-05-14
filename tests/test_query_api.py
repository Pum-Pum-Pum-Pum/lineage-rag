from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from app.api.main import create_app
from app.api.routes import query as query_route
from app.llm.answer_contract import Citation, GroundedAnswerResponse
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
    assert fake_client.collection_exists_calls == ["lineage_chunks"]
    assert fake_client.closed is True


def test_query_endpoint_rejects_blank_query() -> None:
    response = TestClient(create_app()).post("/query", json={"query": "   "})

    assert response.status_code == 422


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