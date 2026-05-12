import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

from app.llm.answer_contract import Citation
from app.llm.answer_contract import GroundedAnswerResponse
from app.retrieval.evidence_sufficiency import EvidenceSufficiencyDecision
from app.retrieval.retrieval_config import RetrievalRuntimeConfig
from app.services.answer_orchestration import AnswerOrchestrationResult
from app.services.answer_trace import build_answer_trace
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


def test_answer_smoke_script_uses_answer_orchestration_service(monkeypatch, tmp_path: Path) -> None:
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
        request_id="answer-smoke-test-request",
        retrieval_metadata={"retrieval_mode": "hybrid"},
    )
    trace_output_path = tmp_path / "exports" / "answer_runs" / "answer-smoke-test-request.json"
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

    def fake_run_grounded_answer_query(**kwargs):
        captured["orchestration_kwargs"] = kwargs
        return AnswerOrchestrationResult(
            retrieval_mode="hybrid",
            retrieval_results=[retrieved_result],
            sufficiency=sufficiency,
            answer_response=response,
            trace=trace,
            trace_output_path=trace_output_path,
        )

    monkeypatch.setattr(run_answer_smoke_test, "get_settings", lambda: settings)
    monkeypatch.setattr(run_answer_smoke_test, "build_retrieval_runtime_config", lambda loaded_settings: retrieval_config)
    monkeypatch.setattr(run_answer_smoke_test, "create_persistent_qdrant_client", lambda path: fake_client)
    monkeypatch.setattr(run_answer_smoke_test, "run_grounded_answer_query", fake_run_grounded_answer_query)
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

    orchestration_kwargs = captured["orchestration_kwargs"]
    assert orchestration_kwargs["qdrant_client"] is fake_client
    assert orchestration_kwargs["collection_name"] == "lineage_chunks"
    assert orchestration_kwargs["query_text"] == "What changed in branch reports?"
    assert orchestration_kwargs["embedding_model"] == "test-embedding-model"
    assert orchestration_kwargs["retrieval_config"] == retrieval_config
    assert orchestration_kwargs["lexical_artifact_directory"] == settings.processed_dir
    assert orchestration_kwargs["trace_output_directory"] == settings.exports_dir / "answer_runs"
    assert orchestration_kwargs["limit"] == 3
    assert orchestration_kwargs["min_results"] == 1
    assert orchestration_kwargs["min_top_score"] == 0.25
    assert orchestration_kwargs["document_family"] == "ASNB"
    assert orchestration_kwargs["release_label"] == "R24"
    assert orchestration_kwargs["source_kind"] == "paragraph"
    assert fake_client.collection_exists_calls == ["lineage_chunks"]
    assert fake_client.closed is True
