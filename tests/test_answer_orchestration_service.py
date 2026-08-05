import json
from pathlib import Path

from app.llm.answer_contract import GroundedAnswerResponse
from app.retrieval.retrieval_config import RetrievalRuntimeConfig
from app.retrieval.retrieval_router import RoutedRetrievalResult
from app.services import answer_orchestration
from app.services.answer_orchestration import run_grounded_answer_query
from app.vectorstore.qdrant_search import QdrantSearchResult


def _retrieval_config(mode: str = "hybrid") -> RetrievalRuntimeConfig:
    return RetrievalRuntimeConfig(
        retrieval_mode=mode,
        hybrid_dense_weight=0.6,
        hybrid_lexical_weight=0.4,
        hybrid_candidate_limit=10,
    )


def _retrieved_result(score: float = 0.82) -> QdrantSearchResult:
    return QdrantSearchResult(
        point_id="point-1",
        score=score,
        payload={
            "unit_id": "FS_ASNB_R24::chunk_1",
            "text": "Branch report evidence",
            "document_family": "ASNB",
            "release_label": "R24",
            "source_kind": "paragraph",
        },
    )


def test_run_grounded_answer_query_orchestrates_answer_flow(monkeypatch, tmp_path: Path) -> None:
    retrieved_result = _retrieved_result(score=0.82)
    retrieval_config = _retrieval_config("hybrid")
    captured: dict[str, object] = {}
    fake_qdrant_client = object()
    fake_embedding_client = object()
    fake_llm_client = object()

    def fake_retrieve_query_evidence(**kwargs):
        captured["retrieve_kwargs"] = kwargs
        return RoutedRetrievalResult(
            retrieval_mode="hybrid",
            results=[retrieved_result],
        )

    def fake_generate_grounded_answer(**kwargs):
        captured["answer_query"] = kwargs["query"]
        captured["answer_results"] = kwargs["retrieved_results"]
        captured["answer_sufficiency"] = kwargs["sufficiency"]
        captured["llm_client"] = kwargs["llm_client"]
        captured["llm_model"] = kwargs["model"]
        return GroundedAnswerResponse(
            query=kwargs["query"],
            answer="Grounded answer [C1].",
            is_answered=True,
            refusal_reason=None,
            citations=[],
        )

    monkeypatch.setattr(answer_orchestration, "retrieve_query_evidence", fake_retrieve_query_evidence)
    monkeypatch.setattr(answer_orchestration, "generate_grounded_answer", fake_generate_grounded_answer)

    result = run_grounded_answer_query(
        qdrant_client=fake_qdrant_client,
        collection_name="lineage_chunks",
        query_text="What changed in branch reports?",
        embedding_model="test-embedding-model",
        embedding_client=fake_embedding_client,
        retrieval_config=retrieval_config,
        lexical_artifact_directory=tmp_path / "processed",
        trace_output_directory=tmp_path / "answer_runs",
        llm_client=fake_llm_client,
        llm_model="test-llm-model",
        limit=3,
        min_top_score=0.25,
        document_family="ASNB",
        release_label="R24",
        source_kind="paragraph",
        request_id="request-123",
    )

    retrieve_kwargs = captured["retrieve_kwargs"]
    assert retrieve_kwargs["qdrant_client"] is fake_qdrant_client
    assert retrieve_kwargs["collection_name"] == "lineage_chunks"
    assert retrieve_kwargs["query_text"] == "What changed in branch reports?"
    assert retrieve_kwargs["embedding_model"] == "test-embedding-model"
    assert retrieve_kwargs["embedding_client"] is fake_embedding_client
    assert retrieve_kwargs["retrieval_config"] == retrieval_config
    assert retrieve_kwargs["lexical_artifact_directory"] == tmp_path / "processed"
    assert retrieve_kwargs["limit"] == 3
    assert retrieve_kwargs["document_family"] == "ASNB"
    assert retrieve_kwargs["release_label"] == "R24"
    assert retrieve_kwargs["source_kind"] == "paragraph"

    assert result.retrieval_mode == "hybrid"
    assert result.retrieval_results == [retrieved_result]
    assert result.sufficiency.is_sufficient is True
    assert captured["answer_results"] == [retrieved_result]
    assert captured["answer_sufficiency"] == result.sufficiency
    assert captured["llm_client"] is fake_llm_client
    assert captured["llm_model"] == "test-llm-model"
    assert result.trace_output_path.exists()

    trace_payload = json.loads(result.trace_output_path.read_text(encoding="utf-8"))
    assert trace_payload["request_id"] == "request-123"
    assert trace_payload["filters"]["release_label"] == "R24"
    assert trace_payload["retrieval_metadata"]["retrieval_mode"] == "hybrid"
    assert trace_payload["retrieval_metadata"]["hybrid_dense_weight"] == 0.6
    assert trace_payload["retrieval_metadata"]["limit"] == 3
    assert trace_payload["answer_response"]["is_answered"] is True
    assert trace_payload["retrieval_results"][0]["payload"]["unit_id"] == "FS_ASNB_R24::chunk_1"


def test_run_grounded_answer_query_refuses_when_evidence_is_insufficient(monkeypatch, tmp_path: Path) -> None:
    retrieved_result = _retrieved_result(score=0.10)

    def fake_retrieve_query_evidence(**kwargs):
        return RoutedRetrievalResult(
            retrieval_mode="lexical",
            results=[retrieved_result],
        )

    monkeypatch.setattr(answer_orchestration, "retrieve_query_evidence", fake_retrieve_query_evidence)

    result = run_grounded_answer_query(
        qdrant_client=object(),
        collection_name="lineage_chunks",
        query_text="What is the mobile app login flow?",
        embedding_model="test-embedding-model",
        retrieval_config=_retrieval_config("lexical"),
        lexical_artifact_directory=tmp_path / "processed",
        trace_output_directory=tmp_path / "answer_runs",
        llm_client=object(),
        limit=1,
        min_top_score=0.90,
        request_id="insufficient-request",
    )

    assert result.retrieval_mode == "lexical"
    assert result.sufficiency.is_sufficient is False
    assert "below required threshold" in result.sufficiency.reason
    assert result.answer_response.is_answered is False
    assert result.answer_response.refusal_reason == result.sufficiency.reason
    assert result.trace_output_path.exists()

    trace_payload = json.loads(result.trace_output_path.read_text(encoding="utf-8"))
    assert trace_payload["request_id"] == "insufficient-request"
    assert trace_payload["retrieval_metadata"]["retrieval_mode"] == "lexical"
    assert trace_payload["answer_response"]["is_answered"] is False


def test_current_query_expands_candidates_scopes_latest_release_and_marks_answer(
    monkeypatch,
    tmp_path: Path,
) -> None:
    captured: dict[str, object] = {}

    def fake_retrieve_query_evidence(**kwargs):
        captured["retrieve_kwargs"] = kwargs
        return RoutedRetrievalResult(
            retrieval_mode="hybrid",
            results=[
                QdrantSearchResult(
                    point_id="r2",
                    score=0.9,
                    payload={
                        "unit_id": "r2",
                        "text": "Older R2 evidence",
                        "release_label": "R2",
                        "source_kind": "paragraph",
                    },
                ),
                QdrantSearchResult(
                    point_id="r24-teller",
                    score=0.8,
                    payload={
                        "unit_id": "r24-teller",
                        "text": "R24 teller realignment",
                        "release_label": "R24",
                        "source_kind": "table",
                    },
                ),
                QdrantSearchResult(
                    point_id="r24-branch",
                    score=0.7,
                    payload={
                        "unit_id": "r24-branch",
                        "text": "R24 branch realignment",
                        "release_label": "R24",
                        "source_kind": "table",
                    },
                ),
            ],
        )

    def fake_generate_grounded_answer(**kwargs):
        captured["answer_kwargs"] = kwargs
        return GroundedAnswerResponse(
            query=kwargs["query"],
            answer="There are currently 2 teller and 4 branch reports [C1][C2].",
            is_answered=True,
            refusal_reason=None,
            citations=[],
        )

    monkeypatch.setattr(
        answer_orchestration,
        "retrieve_query_evidence",
        fake_retrieve_query_evidence,
    )
    monkeypatch.setattr(
        answer_orchestration,
        "generate_grounded_answer",
        fake_generate_grounded_answer,
    )

    result = run_grounded_answer_query(
        qdrant_client=object(),
        collection_name="lineage_chunks",
        query_text=(
            "Give me a summary: how many teller and branch reports are "
            "there currently?"
        ),
        embedding_model="test-embedding-model",
        retrieval_config=RetrievalRuntimeConfig(
            retrieval_mode="hybrid",
            hybrid_dense_weight=0.4,
            hybrid_lexical_weight=0.6,
            hybrid_candidate_limit=10,
        ),
        lexical_artifact_directory=tmp_path / "processed",
        trace_output_directory=tmp_path / "answer_runs",
        limit=5,
        min_top_score=0.3,
        request_id="current-state-request",
    )

    retrieve_kwargs = captured["retrieve_kwargs"]
    assert retrieve_kwargs["limit"] == 10
    assert "resulting production state" in retrieve_kwargs["query_text"]
    assert [item.point_id for item in result.retrieval_results] == [
        "r24-teller",
        "r24-branch",
    ]
    answer_kwargs = captured["answer_kwargs"]
    assert answer_kwargs["current_state_requested"] is True
    assert answer_kwargs["effective_release_label"] == "R24"
    trace_payload = json.loads(
        result.trace_output_path.read_text(encoding="utf-8")
    )
    assert trace_payload["filters"]["release_label"] == "R24"
    assert (
        trace_payload["retrieval_metadata"]["release_source"]
        == "retrieved_candidates"
    )


def test_current_query_keeps_historical_release_mentions_out_of_retrieval_filter(
    monkeypatch,
    tmp_path: Path,
) -> None:
    captured: dict[str, object] = {}

    def fake_retrieve_query_evidence(**kwargs):
        captured["retrieve_kwargs"] = kwargs
        return RoutedRetrievalResult(
            retrieval_mode="hybrid",
            results=[
                QdrantSearchResult(
                    point_id="r2",
                    score=0.9,
                    payload={"unit_id": "r2", "text": "R2", "release_label": "R2"},
                ),
                QdrantSearchResult(
                    point_id="r24",
                    score=0.8,
                    payload={"unit_id": "r24", "text": "R24", "release_label": "R24"},
                ),
            ],
        )

    monkeypatch.setattr(answer_orchestration, "retrieve_query_evidence", fake_retrieve_query_evidence)
    monkeypatch.setattr(
        answer_orchestration,
        "generate_grounded_answer",
        lambda **kwargs: GroundedAnswerResponse(
            query=kwargs["query"], answer="R24 [C1].", is_answered=True, refusal_reason=None, citations=[]
        ),
    )

    result = run_grounded_answer_query(
        qdrant_client=object(),
        collection_name="lineage_chunks",
        query_text=(
            "Which current release contains the Teller and Branch Reports Re-alignment "
            "change, rather than the original R2 report specifications?"
        ),
        embedding_model="test-embedding-model",
        retrieval_config=_retrieval_config("hybrid"),
        lexical_artifact_directory=tmp_path / "processed",
        trace_output_directory=tmp_path / "answer_runs",
    )

    retrieve_kwargs = captured["retrieve_kwargs"]
    assert retrieve_kwargs["release_label"] is None
    assert [item.point_id for item in result.retrieval_results] == ["r2", "r24"]
    trace_payload = json.loads(result.trace_output_path.read_text(encoding="utf-8"))
    assert trace_payload["retrieval_metadata"]["effective_release_label"] == "R24"
    assert trace_payload["retrieval_metadata"]["referenced_release_labels"] == ["R2"]
