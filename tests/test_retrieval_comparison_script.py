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
from scripts import run_retrieval_comparison

def test_retrieval_comparison_script_help_runs() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/run_retrieval_comparison.py", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert "--lexical-artifact-dir" in result.stdout
    assert "--output-file" in result.stdout


def test_retrieval_comparison_script_closes_qdrant_client_on_success(monkeypatch, tmp_path: Path) -> None:
    settings = SimpleNamespace(
        log_level="INFO",
        qdrant_local_path=tmp_path / "qdrant",
        qdrant_collection_name="lineage_chunks",
        openai_embedding_model="test-embedding-model",
    )
    case = _case()
    dense_result = _result("dense-point-1", 0.82, "Dense branch report evidence")
    lexical_result = _result("lexical-point-1", 4.2, "Lexical branch report evidence")
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

    def fake_search_query_text(**kwargs):
        captured["dense_kwargs"] = kwargs
        return [dense_result]

    def fake_search_lexical_artifacts(**kwargs):
        captured["lexical_kwargs"] = kwargs
        return [lexical_result]

    def fake_write_report(report, output_file):
        captured["report"] = report
        captured["output_file"] = output_file
        return Path(output_file)

    monkeypatch.setattr(run_retrieval_comparison, "get_settings", lambda: settings)
    monkeypatch.setattr(run_retrieval_comparison, "load_retrieval_eval_cases", lambda path: [case])
    monkeypatch.setattr(run_retrieval_comparison, "create_persistent_qdrant_client", lambda path: fake_client)
    monkeypatch.setattr(run_retrieval_comparison, "search_query_text", fake_search_query_text)
    monkeypatch.setattr(run_retrieval_comparison, "search_lexical_artifacts", fake_search_lexical_artifacts)
    monkeypatch.setattr(run_retrieval_comparison, "write_retrieval_comparison_report_to_json", fake_write_report)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_retrieval_comparison.py",
            "--eval-file",
            "eval.json",
            "--limit",
            "3",
            "--lexical-artifact-dir",
            str(tmp_path / "processed"),
            "--output-file",
            str(tmp_path / "comparison.json"),
        ],
    )

    run_retrieval_comparison.main()

    dense_kwargs = captured["dense_kwargs"]
    lexical_kwargs = captured["lexical_kwargs"]
    report = captured["report"]
    assert dense_kwargs["qdrant_client"] is fake_client
    assert dense_kwargs["collection_name"] == "lineage_chunks"
    assert dense_kwargs["embedding_model"] == "test-embedding-model"
    assert dense_kwargs["limit"] == 3
    assert lexical_kwargs["artifact_directory"] == str(tmp_path / "processed")
    assert lexical_kwargs["limit"] == 3
    assert report.total_cases == 1
    assert fake_client.collection_exists_calls == ["lineage_chunks"]
    assert fake_client.closed is True


def test_retrieval_comparison_script_closes_qdrant_client_when_search_fails(
    monkeypatch,
    tmp_path: Path,
) -> None:
    settings = SimpleNamespace(
        log_level="INFO",
        qdrant_local_path=tmp_path / "qdrant",
        qdrant_collection_name="lineage_chunks",
        openai_embedding_model="test-embedding-model",
    )

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

    def fail_search_query_text(**kwargs):
        raise RuntimeError("dense retrieval failed")

    monkeypatch.setattr(run_retrieval_comparison, "get_settings", lambda: settings)
    monkeypatch.setattr(run_retrieval_comparison, "load_retrieval_eval_cases", lambda path: [_case()])
    monkeypatch.setattr(run_retrieval_comparison, "create_persistent_qdrant_client", lambda path: fake_client)
    monkeypatch.setattr(run_retrieval_comparison, "search_query_text", fail_search_query_text)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_retrieval_comparison.py",
            "--eval-file",
            "eval.json",
            "--output-file",
            str(tmp_path / "comparison.json"),
        ],
    )

    with pytest.raises(RuntimeError, match="dense retrieval failed"):
        run_retrieval_comparison.main()

    assert fake_client.collection_exists_calls == ["lineage_chunks"]
    assert fake_client.closed is True


def test_retrieval_comparison_fails_before_search_when_collection_is_missing(
    monkeypatch,
    tmp_path: Path,
) -> None:
    settings = SimpleNamespace(
        log_level="INFO",
        qdrant_local_path=tmp_path / "qdrant",
        qdrant_collection_name="lineage_chunks",
        openai_embedding_model="test-embedding-model",
    )
    search_called = False

    class MissingCollectionQdrantClient:
        def __init__(self) -> None:
            self.collection_exists_calls: list[str] = []
            self.closed = False

        def collection_exists(self, collection_name: str) -> bool:
            self.collection_exists_calls.append(collection_name)
            return False

        def close(self) -> None:
            self.closed = True

    fake_client = MissingCollectionQdrantClient()

    def unexpected_search(**kwargs):
        nonlocal search_called
        search_called = True
        raise AssertionError("dense search should not run without the collection")

    monkeypatch.setattr(run_retrieval_comparison, "get_settings", lambda: settings)
    monkeypatch.setattr(
        run_retrieval_comparison,
        "load_retrieval_eval_cases",
        lambda path: [_case()],
    )
    monkeypatch.setattr(
        run_retrieval_comparison,
        "create_persistent_qdrant_client",
        lambda path: fake_client,
    )
    monkeypatch.setattr(run_retrieval_comparison, "search_query_text", unexpected_search)
    monkeypatch.setattr(sys, "argv", ["run_retrieval_comparison.py"])

    with pytest.raises(
        RuntimeError,
        match="Qdrant collection does not exist.*run_qdrant_indexing.py",
    ):
        run_retrieval_comparison.main()

    assert search_called is False
    assert fake_client.collection_exists_calls == ["lineage_chunks"]
    assert fake_client.closed is True


def _case() -> RetrievalEvalCase:
    return RetrievalEvalCase(
        case_id="branch_reports",
        query="branch reports",
        filters=RetrievalEvalFilters(release_label="R24", source_kind="paragraph"),
        expectation=RetrievalEvalExpectation(
            expected_to_pass=True,
            min_results=1,
            expected_release_label="R24",
            expected_source_kind="paragraph",
            expected_text_contains_any=["branch report"],
        ),
    )


def _result(point_id: str, score: float, text: str) -> QdrantSearchResult:
    return QdrantSearchResult(
        point_id=point_id,
        score=score,
        payload={
            "unit_id": point_id,
            "text": text,
            "document_family": "ASNB",
            "release_label": "R24",
            "source_kind": "paragraph",
        },
    )
