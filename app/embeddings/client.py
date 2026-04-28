from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from openai import OpenAI

from app.embeddings.embedding_cache import load_embedding_cache
from app.core.config import get_settings
from app.embeddings.embedding_contract import EmbeddingBatch, EmbeddingRecord


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
) -> EmbeddingBatch:
    """Embed all records in an embedding batch.

    This baseline implementation embeds each retrieval-ready unit and returns a
    new batch with vectors filled in and status updated.
    """

    if not batch.records:
        return EmbeddingBatch(
            document_name=batch.document_name,
            total_records=0,
            records=[],
            cached_count=0,
            embedded_count=0,
            cache_miss_count=0,
        )

    cache = load_embedding_cache(cache_directory) if cache_directory is not None else {}
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
        texts = [record.text for _, record in uncached_records]

        response = embedding_client.embeddings.create(
            model=settings.openai_embedding_model,
            input=texts,
        )

        if len(response.data) != len(uncached_records):
            raise RuntimeError(
                "Embedding response count does not match input record count: "
                f"expected={len(uncached_records)}, received={len(response.data)}"
            )

        for (index, record), response_item in zip(uncached_records, response.data):
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
