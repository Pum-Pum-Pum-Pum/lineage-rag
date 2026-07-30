import json
from pathlib import Path

from app.llm.answer_contract import GroundedAnswerResponse
from app.retrieval.evidence_sufficiency import EvidenceSufficiencyDecision
from app.services.answer_trace import build_answer_trace, write_answer_trace
from app.vectorstore.qdrant_search import QdrantSearchResult


def test_build_and_write_answer_trace(tmp_path: Path) -> None:
    sufficiency = EvidenceSufficiencyDecision(
        is_sufficient=True,
        reason="ok",
        result_count=1,
        top_score=0.8,
    )
    response = GroundedAnswerResponse(
        query="branch reports realignment",
        answer="The reports were consolidated [C1].",
        is_answered=True,
        refusal_reason=None,
        citations=[],
    )
    retrieval_results = [
        QdrantSearchResult(
            point_id="point-1",
            score=0.8,
            payload={"unit_id": "doc::chunk_1", "text": "evidence"},
        )
    ]

    trace = build_answer_trace(
        query="branch reports realignment",
        filters={"release_label": "R24", "source_kind": "paragraph"},
        sufficiency=sufficiency,
        answer_response=response,
        retrieval_results=retrieval_results,
        request_id="test-request-id",
    )
    output_file = write_answer_trace(trace, tmp_path / "answer_runs")

    assert output_file.name == "test-request-id.json"
    payload = json.loads(output_file.read_text(encoding="utf-8"))
    assert payload["request_id"] == "test-request-id"
    assert payload["query"] == "branch reports realignment"
    assert payload["filters"]["release_label"] == "R24"
    assert payload["answer_response"]["is_answered"] is True
    assert (
        payload["retrieval_results"][0]["payload"]["unit_id"]
        == "doc::chunk_1"
    )


def test_answer_trace_keeps_correlation_separate_from_unique_trace_id(
    tmp_path: Path,
) -> None:
    trace = build_answer_trace(
        query="What changed?",
        filters={},
        sufficiency=EvidenceSufficiencyDecision(
            is_sufficient=False,
            reason="Insufficient evidence.",
            result_count=0,
            top_score=None,
        ),
        answer_response=GroundedAnswerResponse(
            query="What changed?",
            answer="I cannot answer.",
            is_answered=False,
            refusal_reason="Insufficient evidence.",
            citations=[],
        ),
        retrieval_results=[],
        correlation_id="api-request-123",
    )

    output = write_answer_trace(trace, tmp_path / "answer_runs")
    payload = json.loads(output.read_text(encoding="utf-8"))

    assert payload["correlation_id"] == "api-request-123"
    assert payload["request_id"] != payload["correlation_id"]
