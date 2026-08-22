from pathlib import Path
from types import SimpleNamespace

from app.retrieval.retrieval_config import RetrievalRuntimeConfig
from app.services import knowledge_mode_orchestration as orchestration


class FakeQdrant:
    def __init__(self, path: str) -> None:
        self.path = path
        self.closed = False

    def collection_exists(self, name: str) -> bool:
        return name == "code_custom_test_v1"

    def close(self) -> None:
        self.closed = True


class Dumpable(SimpleNamespace):
    def model_dump(self, mode: str = "json") -> dict:
        return dict(self.__dict__)


def test_code_runtime_embeds_query_once_and_persists_trace(monkeypatch, tmp_path: Path) -> None:
    calls = {"embedding": 0, "retrieval": 0, "generation": 0}
    captured = {}
    artifact = SimpleNamespace(vector_dimension=3)
    retrieval = Dumpable(evidence=(), query="Explain custom behavior")
    answer = Dumpable(
        query="Explain custom behavior",
        answer="Grounded visible behavior.",
        is_answered=True,
    )

    def fake_embed(**kwargs):
        calls["embedding"] += 1
        captured["embedding_question"] = kwargs["question"]
        return [0.1, 0.2, 0.3], {"request_id": "embed-1", "total_tokens": 4}

    def fake_retrieve(**kwargs):
        calls["retrieval"] += 1
        captured["retrieval_query"] = kwargs["query"]
        assert kwargs["query_vector"] == [0.1, 0.2, 0.3]
        return retrieval

    def fake_generate(**kwargs):
        calls["generation"] += 1
        captured["generation_context"] = kwargs["conversation_context"]
        return answer, {
            "request_id": "answer-1",
            "model": "test-chat",
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15,
        }

    monkeypatch.setattr(orchestration, "load_code_index_artifact", lambda path: artifact)
    monkeypatch.setattr(orchestration, "embed_one_query", fake_embed)
    monkeypatch.setattr(orchestration, "retrieve_code_evidence", fake_retrieve)
    monkeypatch.setattr(orchestration, "generate_grounded_answer", fake_generate)
    monkeypatch.setattr(orchestration, "QdrantClient", FakeQdrant)
    settings = SimpleNamespace(
        code_index_artifact_path=tmp_path / "artifact.json",
        openai_embedding_model="test-embedding",
        openai_chat_model="test-chat",
        code_qdrant_local_path=tmp_path / "qdrant-code",
        code_qdrant_collection_name="code_custom_test_v1",
        exports_dir=tmp_path / "exports",
    )
    config = RetrievalRuntimeConfig("hybrid", 0.4, 0.6, 10)

    result = orchestration.run_code_or_combined_query(
        mode="code",
        query="Explain custom behavior",
        analysis_kind="explanation",
        settings=settings,
        retrieval_config=config,
        limit=5,
        correlation_id="trace-1",
        conversation_context="Earlier package: PKG_AML_CUSTOM",
        openai_client=object(),
    )

    assert calls == {"embedding": 1, "retrieval": 1, "generation": 1}
    assert "PKG_AML_CUSTOM" in captured["embedding_question"]
    assert captured["embedding_question"] == captured["retrieval_query"]
    assert captured["generation_context"] == "Earlier package: PKG_AML_CUSTOM"
    assert result.trace_id == "trace-1"
    assert result.trace_output_path.is_file()
    assert '"knowledge_mode": "code"' in result.trace_output_path.read_text(encoding="utf-8")
