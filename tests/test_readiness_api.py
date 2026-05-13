from pathlib import Path
from types import SimpleNamespace

from fastapi.testclient import TestClient

from app.api import main as api_main
from app.api.main import create_app
from app.api.routes import readiness as readiness_route
from app.retrieval.retrieval_config import RetrievalRuntimeConfig


def _settings(tmp_path: Path) -> SimpleNamespace:
    return SimpleNamespace(
        app_name="Test RAG API",
        environment="test",
        qdrant_local_path=tmp_path / "qdrant",
        qdrant_collection_name="lineage_chunks",
        processed_dir=tmp_path / "processed",
        openai_api_key="test-api-key",
        openai_embedding_model="test-embedding-model",
        openai_chat_model="test-chat-model",
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


def _write_retrieval_ready_artifact(processed_dir: Path) -> None:
    processed_dir.mkdir(parents=True, exist_ok=True)
    (processed_dir / "example.retrieval_ready.json").write_text(
        '{"document_name":"example.docx","units":[]}',
        encoding="utf-8",
    )


def _check_by_name(payload: dict, name: str) -> dict:
    return next(check for check in payload["checks"] if check["name"] == name)


def test_ready_endpoint_returns_ready_for_hybrid_when_dependencies_exist(monkeypatch, tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    _write_retrieval_ready_artifact(settings.processed_dir)
    fake_client = FakeQdrantClient(collection_exists=True)

    monkeypatch.setattr(api_main, "get_settings", lambda: settings)
    monkeypatch.setattr(readiness_route, "get_settings", lambda: settings)
    monkeypatch.setattr(
        readiness_route,
        "build_retrieval_runtime_config",
        lambda loaded_settings: _retrieval_config("hybrid"),
    )
    monkeypatch.setattr(readiness_route, "create_persistent_qdrant_client", lambda path: fake_client)

    response = TestClient(create_app()).get("/ready")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ready"
    assert payload["is_ready"] is True
    assert payload["retrieval_mode"] == "hybrid"
    assert payload["qdrant_required_for_current_mode"] is True
    assert payload["lexical_artifacts_required_for_current_mode"] is True
    assert _check_by_name(payload, "model_configuration")["is_ready"] is True
    assert _check_by_name(payload, "retrieval_ready_artifacts")["is_ready"] is True
    assert _check_by_name(payload, "qdrant_collection")["is_ready"] is True
    assert fake_client.collection_exists_calls == ["lineage_chunks"]
    assert fake_client.closed is True


def test_ready_endpoint_returns_503_when_required_qdrant_collection_is_missing(
    monkeypatch, tmp_path: Path
) -> None:
    settings = _settings(tmp_path)
    _write_retrieval_ready_artifact(settings.processed_dir)
    fake_client = FakeQdrantClient(collection_exists=False)

    monkeypatch.setattr(api_main, "get_settings", lambda: settings)
    monkeypatch.setattr(readiness_route, "get_settings", lambda: settings)
    monkeypatch.setattr(
        readiness_route,
        "build_retrieval_runtime_config",
        lambda loaded_settings: _retrieval_config("dense"),
    )
    monkeypatch.setattr(readiness_route, "create_persistent_qdrant_client", lambda path: fake_client)

    response = TestClient(create_app()).get("/ready")

    assert response.status_code == 503
    payload = response.json()
    assert payload["status"] == "not_ready"
    assert payload["is_ready"] is False
    qdrant_check = _check_by_name(response.json(), "qdrant_collection")
    assert qdrant_check["required"] is True
    assert qdrant_check["is_ready"] is False
    assert "Run indexing" in qdrant_check["detail"]
    assert fake_client.collection_exists_calls == ["lineage_chunks"]
    assert fake_client.closed is True


def test_ready_endpoint_skips_qdrant_for_lexical_but_requires_artifacts(monkeypatch, tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    _write_retrieval_ready_artifact(settings.processed_dir)

    def fail_if_qdrant_created(path):
        raise AssertionError("Lexical readiness should not create a Qdrant client")

    monkeypatch.setattr(api_main, "get_settings", lambda: settings)
    monkeypatch.setattr(readiness_route, "get_settings", lambda: settings)
    monkeypatch.setattr(
        readiness_route,
        "build_retrieval_runtime_config",
        lambda loaded_settings: _retrieval_config("lexical"),
    )
    monkeypatch.setattr(readiness_route, "create_persistent_qdrant_client", fail_if_qdrant_created)

    response = TestClient(create_app()).get("/ready")

    assert response.status_code == 200
    payload = response.json()
    assert payload["retrieval_mode"] == "lexical"
    assert payload["qdrant_required_for_current_mode"] is False
    assert payload["lexical_artifacts_required_for_current_mode"] is True
    assert _check_by_name(payload, "qdrant_collection")["required"] is False
    assert _check_by_name(payload, "retrieval_ready_artifacts")["required"] is True


def test_ready_endpoint_returns_503_when_required_retrieval_ready_artifacts_are_missing(
    monkeypatch, tmp_path: Path
) -> None:
    settings = _settings(tmp_path)

    monkeypatch.setattr(api_main, "get_settings", lambda: settings)
    monkeypatch.setattr(readiness_route, "get_settings", lambda: settings)
    monkeypatch.setattr(
        readiness_route,
        "build_retrieval_runtime_config",
        lambda loaded_settings: _retrieval_config("lexical"),
    )

    response = TestClient(create_app()).get("/ready")

    assert response.status_code == 503
    artifact_check = _check_by_name(response.json(), "retrieval_ready_artifacts")
    assert artifact_check["required"] is True
    assert artifact_check["is_ready"] is False
    assert "Run ingestion" in artifact_check["detail"]


def test_ready_endpoint_returns_safe_error_for_invalid_retrieval_config(monkeypatch, tmp_path: Path) -> None:
    settings = _settings(tmp_path)

    def fail_config(loaded_settings):
        raise ValueError("Unsupported retrieval mode: secret-mode")

    monkeypatch.setattr(api_main, "get_settings", lambda: settings)
    monkeypatch.setattr(readiness_route, "get_settings", lambda: settings)
    monkeypatch.setattr(readiness_route, "build_retrieval_runtime_config", fail_config)

    response = TestClient(create_app()).get("/ready")

    assert response.status_code == 500
    assert response.json()["detail"] == "Invalid retrieval runtime configuration."
    assert "secret-mode" not in response.text


def test_ready_endpoint_flags_missing_model_configuration(monkeypatch, tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    settings.openai_api_key = ""
    _write_retrieval_ready_artifact(settings.processed_dir)

    monkeypatch.setattr(api_main, "get_settings", lambda: settings)
    monkeypatch.setattr(readiness_route, "get_settings", lambda: settings)
    monkeypatch.setattr(
        readiness_route,
        "build_retrieval_runtime_config",
        lambda loaded_settings: _retrieval_config("lexical"),
    )

    response = TestClient(create_app()).get("/ready")

    assert response.status_code == 503
    model_check = _check_by_name(response.json(), "model_configuration")
    assert model_check["is_ready"] is False
    assert "OPENAI_API_KEY" in model_check["detail"]
    assert "test-api-key" not in response.text