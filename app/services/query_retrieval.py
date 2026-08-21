from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from qdrant_client import QdrantClient

from app.retrieval.lexical_search import search_lexical_artifacts
from app.retrieval.query_search import search_query_text
from app.retrieval.retrieval_config import RetrievalRuntimeConfig
from app.retrieval.retrieval_router import RoutedRetrievalResult, route_retrieval
from app.retrieval.temporal_query import (
    TemporalQueryPlan,
    build_temporal_query_plan,
    scope_results_to_temporal_plan,
)
from app.vectorstore.qdrant_search import QdrantSearchResult, search_vectors


@dataclass(frozen=True)
class PlannedRetrievalResult:
    routed: RoutedRetrievalResult
    results: list[QdrantSearchResult]
    temporal_plan: TemporalQueryPlan
    retrieval_candidate_limit: int


def retrieve_query_evidence(
    qdrant_client: QdrantClient | None,
    collection_name: str,
    query_text: str,
    embedding_model: str,
    retrieval_config: RetrievalRuntimeConfig,
    lexical_artifact_directory: str | Path,
    embedding_client: Any | None = None,
    query_vector: list[float] | None = None,
    limit: int = 5,
    document_family: str | None = None,
    release_label: str | None = None,
    source_kind: str | None = None,
) -> RoutedRetrievalResult:
    """Retrieve query evidence using the configured retrieval mode.

    This service is the integration point between real retrieval dependencies
    and the mode router. It builds dense and lexical callables with the same
    query, filters, and runtime dependencies, then delegates mode selection to
    `route_retrieval`.

    It intentionally does not call the LLM or generate answers. Lexical-only
    retrieval can run without a Qdrant client; dense and hybrid retrieval cannot.
    """

    if limit <= 0:
        raise ValueError("Retrieval limit must be greater than 0")

    if retrieval_config.retrieval_mode in {"dense", "hybrid"} and qdrant_client is None:
        raise ValueError("qdrant_client is required for dense or hybrid retrieval")

    def dense_search(search_limit: int):
        if query_vector is not None:
            return search_vectors(
                client=qdrant_client,
                collection_name=collection_name,
                query_vector=query_vector,
                limit=search_limit,
                document_family=document_family,
                release_label=release_label,
                source_kind=source_kind,
            )
        return search_query_text(
            qdrant_client=qdrant_client,
            collection_name=collection_name,
            query_text=query_text,
            embedding_model=embedding_model,
            embedding_client=embedding_client,
            limit=search_limit,
            document_family=document_family,
            release_label=release_label,
            source_kind=source_kind,
        )

    def lexical_search(search_limit: int):
        return search_lexical_artifacts(
            artifact_directory=lexical_artifact_directory,
            query_text=query_text,
            limit=search_limit,
            document_family=document_family,
            release_label=release_label,
            source_kind=source_kind,
        )

    return route_retrieval(
        config=retrieval_config,
        dense_search=dense_search,
        lexical_search=lexical_search,
        limit=limit,
    )


def retrieve_planned_query_evidence(
    *,
    qdrant_client: QdrantClient | None,
    collection_name: str,
    query_text: str,
    embedding_model: str,
    retrieval_config: RetrievalRuntimeConfig,
    lexical_artifact_directory: str | Path,
    embedding_client: Any | None = None,
    query_vector: list[float] | None = None,
    limit: int = 5,
    document_family: str | None = None,
    release_label: str | None = None,
    source_kind: str | None = None,
    conversation_context: str | None = None,
    retrieval_callable: Callable[..., RoutedRetrievalResult] | None = None,
) -> PlannedRetrievalResult:
    """Run the production temporal plan and retrieval path without generation."""

    temporal_plan = build_temporal_query_plan(
        query_text,
        requested_release_label=release_label,
        conversation_context=conversation_context,
    )
    retrieval_candidate_limit = (
        max(limit, retrieval_config.hybrid_candidate_limit)
        if temporal_plan.is_current_state
        else limit
    )
    retrieval_service = retrieval_callable or retrieve_query_evidence
    routed = retrieval_service(
        qdrant_client=qdrant_client,
        collection_name=collection_name,
        query_text=temporal_plan.retrieval_query,
        embedding_model=embedding_model,
        embedding_client=embedding_client,
        query_vector=query_vector,
        retrieval_config=retrieval_config,
        lexical_artifact_directory=lexical_artifact_directory,
        limit=retrieval_candidate_limit,
        document_family=document_family,
        release_label=temporal_plan.effective_release_label,
        source_kind=source_kind,
    )
    results, temporal_plan = scope_results_to_temporal_plan(
        routed.results,
        temporal_plan,
        limit=limit,
    )
    return PlannedRetrievalResult(
        routed=routed,
        results=results,
        temporal_plan=temporal_plan,
        retrieval_candidate_limit=retrieval_candidate_limit,
    )
