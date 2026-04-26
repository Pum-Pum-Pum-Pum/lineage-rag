from __future__ import annotations

from dataclasses import replace

from openai import OpenAI

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
) -> EmbeddingBatch:
    """Embed all records in an embedding batch.

    This baseline implementation embeds each retrieval-ready unit and returns a
    new batch with vectors filled in and status updated.
    """

    settings = get_settings()
    embedding_client = client or get_embedding_client()
    texts = [record.text for record in batch.records]

    response = embedding_client.embeddings.create(
        model=settings.openai_embedding_model,
        input=texts,
    )

    updated_records: list[EmbeddingRecord] = []
    for record, response_item in zip(batch.records, response.data):
        updated_records.append(
            replace(
                record,
                embedding_status="embedded",
                vector=response_item.embedding,
            )
        )

    return EmbeddingBatch(
        document_name=batch.document_name,
        total_records=len(updated_records),
        records=updated_records,
    )
