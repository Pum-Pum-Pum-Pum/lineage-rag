import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from app.retrieval.evaluation import (
    RetrievalEvalCase,
    RetrievalEvalExpectation,
    RetrievalEvalFilters,
)
from app.vectorstore.qdrant_search import QdrantSearchResult
from scripts import run_hybrid_retrieval_eval


def test_hybrid_retrieval_eval_script_help_runs() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/run_hybrid_retrieval_eval.py", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert "--dense-weight" in result.stdout
    assert "--lexical-weight" in result.stdout


def test_hybrid_retrieval_eval_closes_qdrant_client_on_success(
    monkeypatch,
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path)
    fake_client = _FakeQdrantClient()
    captured: dict[str, object] = {}

    monkeypatch.setattr(run_hybrid_retrieval_eval, "get_settings", lambda: settings)
    monkeypatch.setattr(run_hybrid_retrieval_eval, "load_retrieval_eval_cases", lambda path: [_case()])
    monkeypatch.setattr(
        run_hybrid_retrieval_eval,
        "create_persistent_qdrant_client",
        lambda path: fake_client,
    )
    monkeypatch.setattr(
        run_hybrid_retrieval_eval,
        "search_query_text",
        lambda **kwargs: [_result("dense", 0.8)],
    )
    monkeypatch.setattr(
        run_hybrid_retrieval_eval,
        "search_lexical_artifacts",
        lambda **kwargs: [_result("lexical", 3.0)],
    )
    monkeypatch.setattr(
        run_hybrid_retrieval_eval,
        "write_hybrid_retrieval_eval_report_to_json",
        lambda report, output_file: captured.setdefault("report", report) or Path(output_file),
    )
    monkeypatch.setattr(
        sys,
        "argv",
        ["run_hybrid_retrieval_eval.py", "--output-file", str(tmp_path / "report.json")],
    )

    run_hybrid_retrieval_eval.main()

    assert captured["report"].total_cases == 1
    assert fake_client.collection_exists_calls == ["lineage_chunks"]
    assert fake_client.closed is True


def test_hybrid_retrieval_eval_closes_qdrant_client_when_search_fails(
    monkeypatch,
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path)
    fake_client = _FakeQdrantClient()

    def fail_search(**kwargs):
        raise RuntimeError("dense retrieval failed")

    monkeypatch.setattr(run_hybrid_retrieval_eval, "get_settings", lambda: settings)
    monkeypatch.setattr(run_hybrid_retrieval_eval, "load_retrieval_eval_cases", lambda path: [_case()])
    monkeypatch.setattr(
        run_hybrid_retrieval_eval,
        "create_persistent_qdrant_client",
        lambda path: fake_client,
    )
    monkeypatch.setattr(run_hybrid_retrieval_eval, "search_query_text", fail_search)
    monkeypatch.setattr(sys, "argv", ["run_hybrid_retrieval_eval.py"])

    with pytest.raises(RuntimeError, match="dense retrieval failed"):
        run_hybrid_retrieval_eval.main()

    assert fake_client.collection_exists_calls == ["lineage_chunks"]
    assert fake_client.closed is True


def test_hybrid_retrieval_eval_fails_before_search_when_collection_is_missing(
    monkeypatch,
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path)
    fake_client = _FakeQdrantClient(collection_exists=False)
    search_called = False

    def unexpected_search(**kwargs):
        nonlocal search_called
        search_called = True
        raise AssertionError("dense search should not run without the collection")

    monkeypatch.setattr(run_hybrid_retrieval_eval, "get_settings", lambda: settings)
    monkeypatch.setattr(
        run_hybrid_retrieval_eval,
        "load_retrieval_eval_cases",
        lambda path: [_case()],
    )
    monkeypatch.setattr(
        run_hybrid_retrieval_eval,
        "create_persistent_qdrant_client",
        lambda path: fake_client,
    )
    monkeypatch.setattr(run_hybrid_retrieval_eval, "search_query_text", unexpected_search)
    monkeypatch.setattr(sys, "argv", ["run_hybrid_retrieval_eval.py"])

    with pytest.raises(
        RuntimeError,
        match="Qdrant collection does not exist.*run_qdrant_indexing.py",
    ):
        run_hybrid_retrieval_eval.main()

    assert search_called is False
    assert fake_client.collection_exists_calls == ["lineage_chunks"]
    assert fake_client.closed is True


class _FakeQdrantClient:
    def __init__(self, *, collection_exists: bool = True) -> None:
        self._collection_exists = collection_exists
        self.collection_exists_calls: list[str] = []
        self.closed = False

    def collection_exists(self, collection_name: str) -> bool:
        self.collection_exists_calls.append(collection_name)
        return self._collection_exists

    def close(self) -> None:
        self.closed = True


def _settings(tmp_path: Path) -> SimpleNamespace:
    return SimpleNamespace(
        log_level="INFO",
        qdrant_local_path=tmp_path / "qdrant",
        qdrant_collection_name="lineage_chunks",
        openai_embedding_model="test-embedding-model",
    )


def _case() -> RetrievalEvalCase:
    return RetrievalEvalCase(
        case_id="branch_reports",
        query="branch reports",
        filters=RetrievalEvalFilters(),
        expectation=RetrievalEvalExpectation(
            expected_to_pass=True,
            min_results=1,
            expected_text_contains_any=["branch report"],
        ),
    )


def _result(point_id: str, score: float) -> QdrantSearchResult:
    return QdrantSearchResult(
        point_id=point_id,
        score=score,
        payload={"unit_id": point_id, "text": "branch report evidence"},
    )
