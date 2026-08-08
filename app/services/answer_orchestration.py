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
from app.retrieval.hybrid_search import DEFAULT_RRF_RANK_CONSTANT
from app.retrieval.retrieval_config import RetrievalRuntimeConfig
from app.services.answer_generation import generate_grounded_answer
from app.services.answer_trace import AnswerTrace, build_answer_trace, write_answer_trace
from app.services.query_retrieval import (
    retrieve_planned_query_evidence,
    retrieve_query_evidence,
)
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
    qdrant_client: QdrantClient | None,
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
    conversation_context: str | None = None,
    correlation_id: str | None = None,
) -> AnswerOrchestrationResult:
    """Run the reusable retrieval -> sufficiency -> answer -> trace flow.

    The caller owns infrastructure lifecycle such as creating and closing the
    Qdrant client. Lexical-only callers may pass ``None`` because lexical
    retrieval reads local artifacts instead of vector-store state. This service
    focuses on query orchestration so API, CLI, and future UI layers can share
    one tested answer path.
    """

    planned_retrieval = retrieve_planned_query_evidence(
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
        conversation_context=conversation_context,
        retrieval_callable=retrieve_query_evidence,
    )
    routed = planned_retrieval.routed
    retrieval_results = planned_retrieval.results
    temporal_plan = planned_retrieval.temporal_plan

    sufficiency = assess_evidence_sufficiency(
        retrieval_results,
        min_results=min_results,
        min_top_score=min_top_score,
    )

    answer_kwargs: dict[str, Any] = {
        "query": query_text,
        "retrieved_results": retrieval_results,
        "sufficiency": sufficiency,
        "llm_client": llm_client,
        "model": llm_model,
    }
    if conversation_context is not None:
        answer_kwargs["conversation_context"] = conversation_context
    if temporal_plan.is_current_state:
        answer_kwargs["current_state_requested"] = True
    if temporal_plan.effective_release_label is not None:
        answer_kwargs["effective_release_label"] = (
            temporal_plan.effective_release_label
        )
    answer_response = generate_grounded_answer(
        **answer_kwargs,
    )

    trace = build_answer_trace(
        query=query_text,
        filters={
            "document_family": document_family,
            "release_label": temporal_plan.effective_release_label,
            "source_kind": source_kind,
        },
        sufficiency=sufficiency,
        answer_response=answer_response,
        retrieval_results=retrieval_results,
        request_id=request_id,
        correlation_id=correlation_id,
        retrieval_metadata={
            "retrieval_mode": routed.retrieval_mode,
            "hybrid_dense_weight": retrieval_config.hybrid_dense_weight,
            "hybrid_lexical_weight": retrieval_config.hybrid_lexical_weight,
            "hybrid_candidate_limit": retrieval_config.hybrid_candidate_limit,
            "hybrid_fusion_method": "weighted_rrf",
            "hybrid_rrf_rank_constant": DEFAULT_RRF_RANK_CONSTANT,
            "limit": limit,
            "retrieval_candidate_limit": planned_retrieval.retrieval_candidate_limit,
            "original_query": temporal_plan.original_query,
            "retrieval_query": temporal_plan.retrieval_query,
            "current_state_requested": temporal_plan.is_current_state,
            "historical_context_requested": temporal_plan.historical_context_requested,
            "effective_release_label": temporal_plan.effective_release_label,
            "release_source": temporal_plan.release_source,
            "referenced_release_labels": list(temporal_plan.referenced_release_labels),
            "min_results": min_results,
            "min_top_score": min_top_score,
            "candidate_lanes": {
                "dense": _summarize_retrieval_candidates(routed.dense_candidates),
                "lexical": _summarize_retrieval_candidates(routed.lexical_candidates),
            },
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


def _summarize_retrieval_candidates(
    results: list[QdrantSearchResult],
) -> list[dict[str, object]]:
    """Keep retrieval-lane diagnostics without duplicating source text in traces."""

    return [
        {
            "rank": rank,
            "point_id": result.point_id,
            "unit_id": str(result.payload.get("unit_id", "")),
            "document_id": result.payload.get("document_id"),
            "release_label": result.payload.get("release_label"),
            "source_kind": result.payload.get("source_kind"),
            "score": result.score,
        }
        for rank, result in enumerate(results, start=1)
    ]
