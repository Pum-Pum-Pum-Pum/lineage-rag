from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from qdrant_client import QdrantClient


@dataclass(frozen=True)
class QdrantSearchResult:
    point_id: str
    score: float
    payload: dict[str, Any]


def search_vectors(
    client: QdrantClient,
    collection_name: str,
    query_vector: list[float],
    limit: int = 5,
) -> list[QdrantSearchResult]:
    """Run a basic vector similarity search against a Qdrant collection."""

    if limit <= 0:
        raise ValueError("Search limit must be greater than 0")
    if not query_vector:
        raise ValueError("Query vector must not be empty")

    results = client.query_points(
        collection_name=collection_name,
        query=query_vector,
        limit=limit,
        with_payload=True,
    )

    return [
        QdrantSearchResult(
            point_id=str(point.id),
            score=point.score,
            payload=point.payload or {},
        )
        for point in results.points
    ]
