from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams


@dataclass(frozen=True)
class QdrantCollectionConfig:
    collection_name: str
    vector_size: int
    distance: Distance = Distance.COSINE


def create_local_qdrant_client() -> QdrantClient:
    """Create an in-memory Qdrant client for local tests and learning."""

    return QdrantClient(":memory:")


def create_persistent_qdrant_client(path: str | Path) -> QdrantClient:
    """Create a local persistent Qdrant client without Docker or a server."""

    storage_path = Path(path)
    storage_path.mkdir(parents=True, exist_ok=True)
    return QdrantClient(path=str(storage_path))


def ensure_collection(
    client: QdrantClient,
    config: QdrantCollectionConfig,
) -> None:
    """Create the collection if it does not already exist."""

    if client.collection_exists(config.collection_name):
        return

    client.create_collection(
        collection_name=config.collection_name,
        vectors_config=VectorParams(
            size=config.vector_size,
            distance=config.distance,
        ),
    )
