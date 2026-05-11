import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

from app.llm.answer_contract import GroundedAnswerResponse
from app.retrieval.retrieval_config import RetrievalRuntimeConfig
from app.retrieval.retrieval_router import RoutedRetrievalResult
from app.vectorstore.qdrant_search import QdrantSearchResult
from scripts import run_answer_smoke_test


def test_answer_smoke_script_help_runs() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/run_answer_smoke_test.py", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert "--query" in result.stdout
    assert "--min-top-score" in result.stdout
    assert "configured retrieval mode" in result.stdout


def test_answer_smoke_collection_requirement_depends_on_retrieval_mode() -> None:
    assert run_answer_smoke_test._requires_qdrant_collection("dense") is True
    assert run_answer_smoke_test._requires_qdrant_collection("hybrid") is True
    assert run_answer_smoke_test._requires_qdrant_collection("lexical") is False


def test_answer_smoke_script_uses_retrieval_service(monkeypatch, tmp_path: Path) -> None:
    settings = SimpleNamespace(
        log_level="INFO",
        retrieval_min_top_score=0.25,
        qdrant_local_path=tmp_path / "qdrant",
        qdrant_collection_name="lineage_chunks",
        openai_embedding_model="test-embedding-model",
        processed_dir=tmp_path / "processed",
        exports_dir=tmp_path / "exports",
    )
    retrieval_config = RetrievalRuntimeConfig(
        retrieval_mode="hybrid",
        hybrid_dense_weight=0.6,
        hybrid_lexical_weight=0.4,
        hybrid_candidate_limit=10,
    )
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
    captured: dict[str, object] = {}

    class FakeQdrantClient:
        def __init__(self) -> None:
            self.collection_exists_calls: list[str] = []
            self.closed = False

        def collection_exists(self, collection_name: str) -> bool:
            self.collection_exists_calls.append(collection_name)
            return True

        def close(self) -> None:
            self.closed = True

    fake_client = FakeQdrantClient()

    def fake_retrieve_query_evidence(**kwargs):
        captured["retrieve_kwargs"] = kwargs
        return RoutedRetrievalResult(
            retrieval_mode="hybrid",
            results=[retrieved_result],
        )

    def fake_generate_grounded_answer(query, retrieved_results, sufficiency):
        captured["answer_query"] = query
        captured["answer_results"] = retrieved_results
        captured["sufficiency"] = sufficiency
        return GroundedAnswerResponse(
            query=query,
            answer="Grounded answer [C1].",
            is_answered=True,
            refusal_reason=None,
            citations=[],
        )

    monkeypatch.setattr(run_answer_smoke_test, "get_settings", lambda: settings)
    monkeypatch.setattr(run_answer_smoke_test, "build_retrieval_runtime_config", lambda loaded_settings: retrieval_config)
    monkeypatch.setattr(run_answer_smoke_test, "create_persistent_qdrant_client", lambda path: fake_client)
    monkeypatch.setattr(run_answer_smoke_test, "retrieve_query_evidence", fake_retrieve_query_evidence)
    monkeypatch.setattr(run_answer_smoke_test, "generate_grounded_answer", fake_generate_grounded_answer)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_answer_smoke_test.py",
            "--query",
            "What changed in branch reports?",
            "--limit",
            "3",
            "--document-family",
            "ASNB",
            "--release-label",
            "R24",
            "--source-kind",
            "paragraph",
        ],
    )

    run_answer_smoke_test.main()

    retrieve_kwargs = captured["retrieve_kwargs"]
    assert retrieve_kwargs["qdrant_client"] is fake_client
    assert retrieve_kwargs["collection_name"] == "lineage_chunks"
    assert retrieve_kwargs["query_text"] == "What changed in branch reports?"
    assert retrieve_kwargs["embedding_model"] == "test-embedding-model"
    assert retrieve_kwargs["retrieval_config"] == retrieval_config
    assert retrieve_kwargs["lexical_artifact_directory"] == settings.processed_dir
    assert retrieve_kwargs["limit"] == 3
    assert retrieve_kwargs["document_family"] == "ASNB"
    assert retrieve_kwargs["release_label"] == "R24"
    assert retrieve_kwargs["source_kind"] == "paragraph"
    assert captured["answer_results"] == [retrieved_result]
    assert captured["sufficiency"].is_sufficient is True
    assert fake_client.collection_exists_calls == ["lineage_chunks"]
    assert fake_client.closed is True
