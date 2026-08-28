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


class LocalQdrantLockError(RuntimeError):
    """Safe runtime error for an embedded Qdrant store held by another process."""


def create_local_qdrant_client() -> QdrantClient:
    """Create an in-memory Qdrant client for local tests and learning."""

    return QdrantClient(":memory:")


def create_persistent_qdrant_client(path: str | Path) -> QdrantClient:
    """Create a local persistent Qdrant client without Docker or a server."""

    storage_path = Path(path)
    storage_path.mkdir(parents=True, exist_ok=True)
    try:
        return QdrantClient(path=str(storage_path))
    except RuntimeError as exc:
        if _looks_like_local_storage_lock(exc):
            raise LocalQdrantLockError(
                "Local Qdrant storage is in use by another process."
            ) from exc
        raise


def _looks_like_local_storage_lock(error: RuntimeError) -> bool:
    """Recognize local-client exclusivity failures without exposing a path."""

    detail = str(error).casefold()
    return (
        "already accessed by another instance" in detail
        or ("qdrant" in detail and ".lock" in detail)
        or ("storage" in detail and "locked" in detail)
    )


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
