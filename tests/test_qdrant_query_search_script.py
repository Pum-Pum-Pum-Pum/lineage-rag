import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

from app.retrieval.retrieval_config import RetrievalRuntimeConfig
from app.retrieval.retrieval_router import RoutedRetrievalResult
from app.vectorstore.qdrant_search import QdrantSearchResult
from scripts import run_qdrant_query_search


def test_qdrant_query_search_script_help_runs() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/run_qdrant_query_search.py", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert "--min-top-score" in result.stdout
    assert "configured retrieval mode" in result.stdout


def test_qdrant_query_search_collection_requirement_depends_on_retrieval_mode() -> None:
    assert run_qdrant_query_search._requires_qdrant_collection("dense") is True
    assert run_qdrant_query_search._requires_qdrant_collection("hybrid") is True
    assert run_qdrant_query_search._requires_qdrant_collection("lexical") is False


def test_qdrant_query_search_uses_qdrant_for_hybrid_mode(monkeypatch, tmp_path: Path) -> None:
    settings = SimpleNamespace(
        log_level="INFO",
        retrieval_min_top_score=0.25,
        qdrant_local_path=tmp_path / "qdrant",
        qdrant_collection_name="lineage_chunks",
        openai_embedding_model="test-embedding-model",
        processed_dir=tmp_path / "processed",
    )
    retrieval_config = RetrievalRuntimeConfig(
        retrieval_mode="hybrid",
        hybrid_dense_weight=0.6,
        hybrid_lexical_weight=0.4,
        hybrid_candidate_limit=10,
    )
    result = QdrantSearchResult(
        point_id="hybrid-point-1",
        score=0.82,
        payload={
            "unit_id": "FS_ASNB_R24::chunk_1",
            "text": "Hybrid branch report evidence",
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
        captured["retrieval_kwargs"] = kwargs
        return RoutedRetrievalResult(retrieval_mode="hybrid", results=[result])

    monkeypatch.setattr(run_qdrant_query_search, "get_settings", lambda: settings)
    monkeypatch.setattr(run_qdrant_query_search, "build_retrieval_runtime_config", lambda loaded_settings: retrieval_config)
    monkeypatch.setattr(run_qdrant_query_search, "create_persistent_qdrant_client", lambda path: fake_client)
    monkeypatch.setattr(run_qdrant_query_search, "retrieve_query_evidence", fake_retrieve_query_evidence)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_qdrant_query_search.py",
            "--query",
            "What changed in branch reports?",
            "--limit",
            "3",
            "--release-label",
            "R24",
        ],
    )

    run_qdrant_query_search.main()

    retrieval_kwargs = captured["retrieval_kwargs"]
    assert retrieval_kwargs["qdrant_client"] is fake_client
    assert retrieval_kwargs["collection_name"] == "lineage_chunks"
    assert retrieval_kwargs["query_text"] == "What changed in branch reports?"
    assert retrieval_kwargs["embedding_model"] == "test-embedding-model"
    assert retrieval_kwargs["retrieval_config"] == retrieval_config
    assert retrieval_kwargs["lexical_artifact_directory"] == settings.processed_dir
    assert retrieval_kwargs["limit"] == 3
    assert retrieval_kwargs["release_label"] == "R24"
    assert fake_client.collection_exists_calls == ["lineage_chunks"]
    assert fake_client.closed is True


def test_qdrant_query_search_skips_qdrant_for_lexical_mode(monkeypatch, tmp_path: Path) -> None:
    settings = SimpleNamespace(
        log_level="INFO",
        retrieval_min_top_score=0.25,
        qdrant_local_path=tmp_path / "qdrant",
        qdrant_collection_name="lineage_chunks",
        openai_embedding_model="test-embedding-model",
        processed_dir=tmp_path / "processed",
    )
    retrieval_config = RetrievalRuntimeConfig(
        retrieval_mode="lexical",
        hybrid_dense_weight=0.6,
        hybrid_lexical_weight=0.4,
        hybrid_candidate_limit=10,
    )
    result = QdrantSearchResult(
        point_id="lexical-point-1",
        score=4.2,
        payload={
            "unit_id": "FS_ASNB_R24::chunk_1",
            "text": "Lexical branch report evidence",
            "document_family": "ASNB",
            "release_label": "R24",
            "source_kind": "paragraph",
        },
    )
    captured: dict[str, object] = {}

    def fail_if_qdrant_created(path):
        raise AssertionError("Lexical query search should not create a Qdrant client")

    def fake_retrieve_query_evidence(**kwargs):
        captured["retrieval_kwargs"] = kwargs
        return RoutedRetrievalResult(retrieval_mode="lexical", results=[result])

    monkeypatch.setattr(run_qdrant_query_search, "get_settings", lambda: settings)
    monkeypatch.setattr(run_qdrant_query_search, "build_retrieval_runtime_config", lambda loaded_settings: retrieval_config)
    monkeypatch.setattr(run_qdrant_query_search, "create_persistent_qdrant_client", fail_if_qdrant_created)
    monkeypatch.setattr(run_qdrant_query_search, "retrieve_query_evidence", fake_retrieve_query_evidence)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_qdrant_query_search.py",
            "--query",
            "What changed in branch reports?",
            "--limit",
            "3",
        ],
    )

    run_qdrant_query_search.main()

    retrieval_kwargs = captured["retrieval_kwargs"]
    assert retrieval_kwargs["qdrant_client"] is None
    assert retrieval_kwargs["collection_name"] == "lineage_chunks"
    assert retrieval_kwargs["query_text"] == "What changed in branch reports?"
    assert retrieval_kwargs["retrieval_config"] == retrieval_config
    assert retrieval_kwargs["lexical_artifact_directory"] == settings.processed_dir
    assert retrieval_kwargs["limit"] == 3
