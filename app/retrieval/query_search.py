from __future__ import annotations

from typing import Any

from qdrant_client import QdrantClient

from app.embeddings.client import get_embedding_client
from app.vectorstore.qdrant_search import QdrantSearchResult, search_vectors


def embed_query_text(
    query_text: str,
    embedding_model: str,
    embedding_client: Any | None = None,
) -> list[float]:
    """Embed a user query into one query vector."""

    cleaned_query = query_text.strip()
    if not cleaned_query:
        raise ValueError("Query text must not be empty")

    client = embedding_client or get_embedding_client()
    response = client.embeddings.create(
        model=embedding_model,
        input=[cleaned_query],
    )

    if len(response.data) != 1:
        raise RuntimeError(
            "Expected exactly one query embedding, "
            f"received={len(response.data)}"
        )

    return response.data[0].embedding


def search_query_text(
    qdrant_client: QdrantClient,
    collection_name: str,
    query_text: str,
    embedding_model: str,
    embedding_client: Any | None = None,
    limit: int = 5,
    document_family: str | None = None,
    release_label: str | None = None,
    source_kind: str | None = None,
) -> list[QdrantSearchResult]:
    """Embed a text query and search Qdrant for matching payloads."""

    query_vector = embed_query_text(
        query_text=query_text,
        embedding_model=embedding_model,
        embedding_client=embedding_client,
    )
    return search_vectors(
        client=qdrant_client,
        collection_name=collection_name,
        query_vector=query_vector,
        limit=limit,
        document_family=document_family,
        release_label=release_label,
        source_kind=source_kind,
    )
