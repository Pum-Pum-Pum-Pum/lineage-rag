from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchValue


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
    document_family: str | None = None,
    release_label: str | None = None,
    source_kind: str | None = None,
) -> list[QdrantSearchResult]:
    """Run a basic vector similarity search against a Qdrant collection."""

    if limit <= 0:
        raise ValueError("Search limit must be greater than 0")
    if not query_vector:
        raise ValueError("Query vector must not be empty")

    query_filter = build_metadata_filter(
        document_family=document_family,
        release_label=release_label,
        source_kind=source_kind,
    )

    results = client.query_points(
        collection_name=collection_name,
        query=query_vector,
        limit=limit,
        query_filter=query_filter,
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


def build_metadata_filter(
    document_family: str | None = None,
    release_label: str | None = None,
    source_kind: str | None = None,
) -> Filter | None:
    """Build a Qdrant metadata filter from optional payload fields."""

    conditions: list[FieldCondition] = []

    if document_family is not None:
        conditions.append(
            FieldCondition(
                key="document_family",
                match=MatchValue(value=document_family),
            )
        )
    if release_label is not None:
        conditions.append(
            FieldCondition(
                key="release_label",
                match=MatchValue(value=release_label),
            )
        )
    if source_kind is not None:
        conditions.append(
            FieldCondition(
                key="source_kind",
                match=MatchValue(value=source_kind),
            )
        )

    if not conditions:
        return None

    return Filter(must=conditions)
