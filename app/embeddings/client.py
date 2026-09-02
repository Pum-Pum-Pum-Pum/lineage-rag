from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Sequence

from openai import OpenAI

from app.embeddings.embedding_cache import load_embedding_cache
from app.core.config import get_settings
from app.embeddings.embedding_contract import EmbeddingBatch, EmbeddingRecord, validate_embedding_batch_inputs


def get_embedding_client() -> OpenAI:
    settings = get_settings()
    return OpenAI(
        api_key=settings.openai_api_key,
        base_url=settings.openai_base_url or None,
    )


def embed_batch(
    batch: EmbeddingBatch,
    client: OpenAI | None = None,
    cache_directory: str | Path | None = None,
    additional_cache_directories: Sequence[str | Path] | None = None,
    request_batch_size: int | None = None,
) -> EmbeddingBatch:
    """Embed all records in an embedding batch.

    Cached records are reused. Uncached records are sent in bounded API
    requests when ``request_batch_size`` is supplied, then returned in their
    original order with vectors filled in and status updated.
    """

    if request_batch_size is not None and request_batch_size <= 0:
        raise ValueError("request_batch_size must be greater than 0 when provided")

    validate_embedding_batch_inputs(batch)

    if not batch.records:
        return EmbeddingBatch(
            document_name=batch.document_name,
            total_records=0,
            records=[],
            cached_count=0,
            embedded_count=0,
            cache_miss_count=0,
        )

    cache = _load_compatible_embedding_cache(
        cache_directory=cache_directory,
        additional_cache_directories=additional_cache_directories or (),
    )
    updated_records: list[EmbeddingRecord | None] = [None] * len(batch.records)
    uncached_records: list[tuple[int, EmbeddingRecord]] = []
    cached_count = 0

    for index, record in enumerate(batch.records):
        cached_record = cache.get(record.cache_key)
        if cached_record is not None:
            cached_count += 1
            updated_records[index] = replace(
                record,
                embedding_status="cached",
                vector=cached_record.vector,
            )
        else:
            uncached_records.append((index, record))

    if uncached_records:
        settings = get_settings()
        embedding_client = client or get_embedding_client()
        grouped_records = _group_uncached_records_by_cache_key(uncached_records)
        unique_request_records = [
            (cache_key, records[0][1], records)
            for cache_key, records in grouped_records.items()
        ]
        resolved_batch_size = request_batch_size or len(unique_request_records)

        for start in range(0, len(unique_request_records), resolved_batch_size):
            request_records = unique_request_records[start : start + resolved_batch_size]
            texts = [representative.text for _, representative, _ in request_records]
            response = embedding_client.embeddings.create(
                model=settings.openai_embedding_model,
                input=texts,
            )

            if len(response.data) != len(request_records):
                raise RuntimeError(
                    "Embedding response count does not match input record count: "
                    f"expected={len(request_records)}, received={len(response.data)}"
                )

            response_items = _response_items_in_input_order(
                response.data,
                expected_count=len(request_records),
            )
            for (_, _, matching_records), response_item in zip(request_records, response_items):
                for index, record in matching_records:
                    updated_records[index] = replace(
                        record,
                        embedding_status="embedded",
                        vector=response_item.embedding,
                    )

    finalized_records = [record for record in updated_records if record is not None]

    return EmbeddingBatch(
        document_name=batch.document_name,
        total_records=len(finalized_records),
        records=finalized_records,
        cached_count=cached_count,
        embedded_count=len(uncached_records),
        cache_miss_count=len(uncached_records),
    )


def _group_uncached_records_by_cache_key(
    uncached_records: list[tuple[int, EmbeddingRecord]],
) -> dict[str, list[tuple[int, EmbeddingRecord]]]:
    """Group identical embedding inputs so one vector is reused consistently.

    The cache key represents normalized text, embedding model, and artifact
    version. If two records have the same key but disagree on those fields, the
    cache identity itself is invalid and embedding must stop rather than merge
    unrelated evidence.
    """

    grouped: dict[str, list[tuple[int, EmbeddingRecord]]] = {}
    for index, record in uncached_records:
        records = grouped.setdefault(record.cache_key, [])
        if records:
            representative = records[0][1]
            if (
                record.content_hash != representative.content_hash
                or record.embedding_model != representative.embedding_model
                or record.artifact_version != representative.artifact_version
                or record.text != representative.text
            ):
                raise RuntimeError(
                    "Embedding cache-key collision across non-identical retrieval units: "
                    f"{record.cache_key}"
                )
        records.append((index, record))
    return grouped


def _load_compatible_embedding_cache(
    *,
    cache_directory: str | Path | None,
    additional_cache_directories: Sequence[str | Path],
) -> dict[str, EmbeddingRecord]:
    """Merge read-only cache sources without accepting conflicting vectors."""

    directories: list[str | Path] = []
    if cache_directory is not None:
        directories.append(cache_directory)
    directories.extend(additional_cache_directories)

    cache: dict[str, EmbeddingRecord] = {}
    for directory in directories:
        for cache_key, record in load_embedding_cache(directory).items():
            existing = cache.get(cache_key)
            if existing is not None and existing.vector != record.vector:
                raise RuntimeError(
                    "Conflicting cached embeddings found across cache directories for cache_key="
                    f"{cache_key}"
                )
            cache.setdefault(cache_key, record)
    return cache


def _response_items_in_input_order(response_items, *, expected_count: int):
    """Use the provider's response indexes instead of assuming response order.

    The OpenAI embedding response carries an input ``index``. Mapping by that
    index prevents a reordered provider response from attaching a valid vector
    to the wrong retrieval unit. Older local fakes without an index retain
    positional behavior solely for backward-compatible deterministic tests.
    """

    if len(response_items) != expected_count:
        raise RuntimeError(
            "Embedding response count does not match input record count: "
            f"expected={expected_count}, received={len(response_items)}"
        )

    indexed_items = {}
    for fallback_index, response_item in enumerate(response_items):
        response_index = getattr(response_item, "index", fallback_index)
        if not isinstance(response_index, int) or response_index < 0 or response_index >= expected_count:
            raise RuntimeError(f"Embedding response contains an invalid input index: {response_index!r}")
        if response_index in indexed_items:
            raise RuntimeError(f"Embedding response contains a duplicate input index: {response_index}")
        indexed_items[response_index] = response_item

    missing_indexes = set(range(expected_count)) - set(indexed_items)
    if missing_indexes:
        raise RuntimeError(
            "Embedding response is missing input indexes: "
            + ", ".join(str(index) for index in sorted(missing_indexes))
        )
    return [indexed_items[index] for index in range(expected_count)]
