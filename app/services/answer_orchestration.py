from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from qdrant_client import QdrantClient

from app.llm.answer_contract import GroundedAnswerResponse
from app.retrieval.evidence_sufficiency import (
    EvidenceSufficiencyDecision,
    assess_evidence_sufficiency,
)
from app.retrieval.retrieval_config import RetrievalRuntimeConfig
from app.services.answer_generation import generate_grounded_answer
from app.services.answer_trace import AnswerTrace, build_answer_trace, write_answer_trace
from app.services.query_retrieval import retrieve_query_evidence
from app.vectorstore.qdrant_search import QdrantSearchResult


@dataclass(frozen=True)
class AnswerOrchestrationResult:
    """Structured output for one retrieval-to-answer query run."""

    retrieval_mode: str
    retrieval_results: list[QdrantSearchResult]
    sufficiency: EvidenceSufficiencyDecision
    answer_response: GroundedAnswerResponse
    trace: AnswerTrace
    trace_output_path: Path


def run_grounded_answer_query(
    qdrant_client: QdrantClient,
    collection_name: str,
    query_text: str,
    embedding_model: str,
    retrieval_config: RetrievalRuntimeConfig,
    lexical_artifact_directory: str | Path,
    trace_output_directory: str | Path,
    embedding_client: Any | None = None,
    llm_client: Any | None = None,
    llm_model: str | None = None,
    limit: int = 5,
    min_results: int = 1,
    min_top_score: float = 0.30,
    document_family: str | None = None,
    release_label: str | None = None,
    source_kind: str | None = None,
    request_id: str | None = None,
) -> AnswerOrchestrationResult:
    """Run the reusable retrieval -> sufficiency -> answer -> trace flow.

    The caller owns infrastructure lifecycle such as creating and closing the
    Qdrant client. This service focuses on query orchestration so API, CLI, and
    future UI layers can share one tested answer path.
    """

    routed = retrieve_query_evidence(
        qdrant_client=qdrant_client,
        collection_name=collection_name,
        query_text=query_text,
        embedding_model=embedding_model,
        embedding_client=embedding_client,
        retrieval_config=retrieval_config,
        lexical_artifact_directory=lexical_artifact_directory,
        limit=limit,
        document_family=document_family,
        release_label=release_label,
        source_kind=source_kind,
    )
    retrieval_results = routed.results

    sufficiency = assess_evidence_sufficiency(
        retrieval_results,
        min_results=min_results,
        min_top_score=min_top_score,
    )

    answer_response = generate_grounded_answer(
        query=query_text,
        retrieved_results=retrieval_results,
        sufficiency=sufficiency,
        llm_client=llm_client,
        model=llm_model,
    )

    trace = build_answer_trace(
        query=query_text,
        filters={
            "document_family": document_family,
            "release_label": release_label,
            "source_kind": source_kind,
        },
        sufficiency=sufficiency,
        answer_response=answer_response,
        retrieval_results=retrieval_results,
        request_id=request_id,
        retrieval_metadata={
            "retrieval_mode": routed.retrieval_mode,
            "hybrid_dense_weight": retrieval_config.hybrid_dense_weight,
            "hybrid_lexical_weight": retrieval_config.hybrid_lexical_weight,
            "hybrid_candidate_limit": retrieval_config.hybrid_candidate_limit,
            "limit": limit,
            "min_results": min_results,
            "min_top_score": min_top_score,
        },
    )
    trace_output_path = write_answer_trace(trace, trace_output_directory)

    return AnswerOrchestrationResult(
        retrieval_mode=routed.retrieval_mode,
        retrieval_results=retrieval_results,
        sufficiency=sufficiency,
        answer_response=answer_response,
        trace=trace,
        trace_output_path=trace_output_path,
    )