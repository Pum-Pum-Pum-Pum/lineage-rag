from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from app.retrieval.hybrid_search import fuse_dense_and_lexical_results
from app.retrieval.retrieval_config import RetrievalRuntimeConfig
from app.vectorstore.qdrant_search import QdrantSearchResult


SearchCallable = Callable[[int], list[Any]]


@dataclass(frozen=True)
class RoutedRetrievalResult:
    retrieval_mode: str
    results: list[QdrantSearchResult]
    dense_candidates: list[QdrantSearchResult] = field(default_factory=list)
    lexical_candidates: list[QdrantSearchResult] = field(default_factory=list)


def route_retrieval(
    config: RetrievalRuntimeConfig,
    dense_search: SearchCallable,
    lexical_search: SearchCallable,
    limit: int = 5,
) -> RoutedRetrievalResult:
    """Route one query to dense, lexical, or hybrid retrieval.

    The router is intentionally dependency-injected: callers provide dense and
    lexical search callables, which makes the routing behavior easy to test and
    keeps this layer independent from API clients and vector-store setup.
    """

    if limit <= 0:
        raise ValueError("Retrieval limit must be greater than 0")

    if config.retrieval_mode == "dense":
        dense_results = normalize_routed_results(dense_search(limit))
        return RoutedRetrievalResult(
            retrieval_mode="dense",
            results=dense_results,
            dense_candidates=dense_results,
        )

    if config.retrieval_mode == "lexical":
        lexical_results = normalize_routed_results(lexical_search(limit))
        return RoutedRetrievalResult(
            retrieval_mode="lexical",
            results=lexical_results,
            lexical_candidates=lexical_results,
        )

    if config.retrieval_mode == "hybrid":
        candidate_limit = max(limit, config.hybrid_candidate_limit)
        dense_results = normalize_routed_results(dense_search(candidate_limit))
        lexical_results = normalize_routed_results(lexical_search(candidate_limit))
        hybrid_results = fuse_dense_and_lexical_results(
            dense_results=dense_results,
            lexical_results=lexical_results,
            limit=limit,
            dense_weight=config.hybrid_dense_weight,
            lexical_weight=config.hybrid_lexical_weight,
        )
        return RoutedRetrievalResult(
            retrieval_mode="hybrid",
            results=normalize_routed_results(hybrid_results),
            dense_candidates=dense_results,
            lexical_candidates=lexical_results,
        )

    raise ValueError(f"Unsupported retrieval mode: {config.retrieval_mode}")


def normalize_routed_results(results: list[Any]) -> list[QdrantSearchResult]:
    """Normalize dense, lexical, or hybrid result objects to one shape."""

    return [
        QdrantSearchResult(
            point_id=str(result.point_id),
            score=float(result.score),
            payload=dict(result.payload),
        )
        for result in results
    ]
