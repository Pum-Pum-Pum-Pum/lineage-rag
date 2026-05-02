from __future__ import annotations

import json
from pathlib import Path

from app.embeddings.embedding_contract import EmbeddingRecord


USABLE_EMBEDDING_STATUSES = {"embedded", "cached"}


def load_embedding_cache(cache_directory: str | Path) -> dict[str, EmbeddingRecord]:
    """Load embedded records from persisted embedding JSON artifacts.

    This is a minimal cache lookup layer. It only loads records that already
    have usable vectors. Pending or incomplete records are ignored.
    """

    directory = Path(cache_directory)
    if not directory.exists():
        return {}

    cache: dict[str, EmbeddingRecord] = {}

    for embedding_file in sorted(directory.glob("*.embeddings.json")):
        payload = json.loads(embedding_file.read_text(encoding="utf-8"))

        for record_payload in payload.get("records", []):
            if record_payload.get("embedding_status") not in USABLE_EMBEDDING_STATUSES:
                continue
            if record_payload.get("vector") is None:
                continue

            record = EmbeddingRecord(**record_payload)
            existing = cache.get(record.cache_key)
            if existing is not None and existing.vector != record.vector:
                raise RuntimeError(
                    "Conflicting cached embeddings found for cache_key="
                    f"{record.cache_key}"
                )

            cache[record.cache_key] = record

    return cache


def find_cached_embedding(
    cache_key: str,
    cache_directory: str | Path,
) -> EmbeddingRecord | None:
    """Find one embedded record by cache key, if available."""

    return load_embedding_cache(cache_directory).get(cache_key)
