from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from openai import OpenAI

from app.code_indexing.contract import load_code_index_artifact
from app.code_indexing.models import CodeIndexArtifact, CodeIndexRecord


@dataclass(frozen=True)
class CodeEmbeddingSummary:
    total_records: int
    unique_embedding_inputs: int
    cached_records: int
    embedded_records: int
    request_count: int
    vector_dimension: int


def embed_code_index_artifact(
    prepared: CodeIndexArtifact,
    *,
    client: OpenAI,
    cache_artifact_paths: Iterable[Path] = (),
    request_batch_size: int = 32,
) -> tuple[CodeIndexArtifact, CodeEmbeddingSummary]:
    if prepared.status != "prepared":
        raise ValueError("Only a prepared code index artifact may be embedded")
    if request_batch_size <= 0:
        raise ValueError("request_batch_size must be greater than zero")
    cache = _load_cache(cache_artifact_paths, embedding_model=prepared.embedding_model)
    grouped: dict[str, list[CodeIndexRecord]] = {}
    for record in prepared.records:
        same = grouped.setdefault(record.cache_key, [])
        if same and (
            same[0].content_sha256 != record.content_sha256
            or same[0].embedding_text != record.embedding_text
            or same[0].embedding_model != record.embedding_model
        ):
            raise RuntimeError(f"Conflicting code embedding cache identity: {record.cache_key}")
        same.append(record)

    vectors: dict[str, tuple[float, ...]] = {}
    cached_keys = set()
    missing = []
    for cache_key, records in grouped.items():
        cached = cache.get(cache_key)
        if cached is not None:
            vectors[cache_key] = cached.vector or ()
            cached_keys.add(cache_key)
        else:
            missing.append((cache_key, records[0].embedding_text))

    request_count = 0
    for start in range(0, len(missing), request_batch_size):
        request = missing[start : start + request_batch_size]
        response = client.embeddings.create(
            model=prepared.embedding_model,
            input=[text for _, text in request],
        )
        request_count += 1
        items = _ordered_response_items(response.data, expected_count=len(request))
        for (cache_key, _), item in zip(request, items, strict=True):
            vector = tuple(float(value) for value in item.embedding)
            if not vector:
                raise RuntimeError("Embedding provider returned an empty vector")
            vectors[cache_key] = vector

    dimensions = {len(vector) for vector in vectors.values()}
    if len(dimensions) != 1:
        raise RuntimeError(f"Code embedding vectors have inconsistent dimensions: {sorted(dimensions)}")
    vector_dimension = next(iter(dimensions), 0)
    if prepared.records and vector_dimension == 0:
        raise RuntimeError("No usable code embedding vectors were produced")

    embedded_records = tuple(
        record.model_copy(
            update={
                "embedding_status": "cached" if record.cache_key in cached_keys else "embedded",
                "vector": vectors[record.cache_key],
            }
        )
        for record in prepared.records
    )
    artifact = prepared.model_copy(
        update={
            "status": "embedded",
            "vector_dimension": vector_dimension,
            "records": embedded_records,
        }
    )
    cached_record_count = sum(record.cache_key in cached_keys for record in prepared.records)
    return artifact, CodeEmbeddingSummary(
        total_records=len(prepared.records),
        unique_embedding_inputs=len(grouped),
        cached_records=cached_record_count,
        embedded_records=len(prepared.records) - cached_record_count,
        request_count=request_count,
        vector_dimension=vector_dimension,
    )


def _load_cache(
    artifact_paths: Iterable[Path],
    *,
    embedding_model: str,
) -> dict[str, CodeIndexRecord]:
    cache: dict[str, CodeIndexRecord] = {}
    for path in artifact_paths:
        artifact = load_code_index_artifact(path)
        if artifact.status != "embedded" or artifact.embedding_model != embedding_model:
            continue
        for record in artifact.records:
            existing = cache.get(record.cache_key)
            if existing is not None and existing.vector != record.vector:
                raise RuntimeError(
                    f"Conflicting cached code embeddings for cache_key={record.cache_key}"
                )
            cache.setdefault(record.cache_key, record)
    return cache


def _ordered_response_items(items, *, expected_count: int):
    if len(items) != expected_count:
        raise RuntimeError(
            "Embedding response count mismatch: "
            f"expected={expected_count}, received={len(items)}"
        )
    indexed = {}
    for fallback_index, item in enumerate(items):
        response_index = getattr(item, "index", fallback_index)
        if (
            not isinstance(response_index, int)
            or response_index < 0
            or response_index >= expected_count
            or response_index in indexed
        ):
            raise RuntimeError(f"Invalid embedding response index: {response_index!r}")
        indexed[response_index] = item
    if len(indexed) != expected_count:
        raise RuntimeError("Embedding response omitted one or more input indexes")
    return [indexed[index] for index in range(expected_count)]
