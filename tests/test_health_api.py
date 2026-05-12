from types import SimpleNamespace

from fastapi.testclient import TestClient

from app.api import main as api_main
from app.api.main import create_app
from app.api.routes import health as health_route
from app.retrieval.retrieval_config import RetrievalRuntimeConfig


def _settings() -> SimpleNamespace:
    return SimpleNamespace(
        app_name="Test RAG API",
        environment="test",
        retrieval_mode="hybrid",
        retrieval_min_top_score=0.25,
        hybrid_dense_weight=0.6,
        hybrid_lexical_weight=0.4,
        hybrid_candidate_limit=10,
        qdrant_collection_name="lineage_chunks",
    )


def _retrieval_config(mode: str = "hybrid") -> RetrievalRuntimeConfig:
    return RetrievalRuntimeConfig(
        retrieval_mode=mode,
        hybrid_dense_weight=0.6,
        hybrid_lexical_weight=0.4,
        hybrid_candidate_limit=10,
    )


def test_health_endpoint_returns_liveness_and_retrieval_config(monkeypatch) -> None:
    settings = _settings()

    monkeypatch.setattr(api_main, "get_settings", lambda: settings)
    monkeypatch.setattr(health_route, "get_settings", lambda: settings)
    monkeypatch.setattr(
        health_route,
        "build_retrieval_runtime_config",
        lambda loaded_settings: _retrieval_config("hybrid"),
    )

    response = TestClient(create_app()).get("/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload == {
        "status": "ok",
        "app_name": "Test RAG API",
        "environment": "test",
        "retrieval_mode": "hybrid",
        "hybrid_dense_weight": 0.6,
        "hybrid_lexical_weight": 0.4,
        "hybrid_candidate_limit": 10,
        "retrieval_min_top_score": 0.25,
        "qdrant_collection_name": "lineage_chunks",
        "qdrant_required_for_current_mode": True,
    }


def test_health_endpoint_reports_lexical_mode_does_not_require_qdrant(monkeypatch) -> None:
    settings = _settings()

    monkeypatch.setattr(api_main, "get_settings", lambda: settings)
    monkeypatch.setattr(health_route, "get_settings", lambda: settings)
    monkeypatch.setattr(
        health_route,
        "build_retrieval_runtime_config",
        lambda loaded_settings: _retrieval_config("lexical"),
    )

    response = TestClient(create_app()).get("/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["retrieval_mode"] == "lexical"
    assert payload["qdrant_required_for_current_mode"] is False


def test_health_endpoint_returns_safe_error_for_invalid_retrieval_config(monkeypatch) -> None:
    settings = _settings()

    def fail_config(loaded_settings):
        raise ValueError("Unsupported retrieval mode: secret-mode")

    monkeypatch.setattr(api_main, "get_settings", lambda: settings)
    monkeypatch.setattr(health_route, "get_settings", lambda: settings)
    monkeypatch.setattr(health_route, "build_retrieval_runtime_config", fail_config)

    response = TestClient(create_app()).get("/health")

    assert response.status_code == 500
    assert response.json()["detail"] == "Invalid retrieval runtime configuration."
    assert "secret-mode" not in response.text