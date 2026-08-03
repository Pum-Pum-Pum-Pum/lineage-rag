from __future__ import annotations

from dataclasses import dataclass
import json
from uuid import NAMESPACE_URL, uuid5

from qdrant_client import QdrantClient
from qdrant_client.models import PointStruct

from app.embeddings.embedding_contract import EmbeddingBatch, EmbeddingRecord


@dataclass(frozen=True)
class QdrantUpsertSummary:
    collection_name: str
    attempted_records: int
    upserted_points: int
    skipped_records: int


def build_qdrant_point_id(record: EmbeddingRecord) -> str:
    """Build a deterministic point ID for one citeable retrieval unit.

    ``cache_key`` identifies reusable vector content. It is not sufficient as a
    vector-store identity because identical text can legitimately occur in two
    chunks or releases that need distinct payload metadata and citations.
    """

    identity = json.dumps(
        {
            "cache_key": record.cache_key,
            "unit_id": record.unit_id,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return str(uuid5(NAMESPACE_URL, identity))


def embedding_record_to_qdrant_point(record: EmbeddingRecord) -> PointStruct:
    """Convert one embedded record into a Qdrant point."""

    if record.vector is None:
        raise ValueError(f"Cannot build Qdrant point without vector: {record.unit_id}")

    payload = {
        "unit_id": record.unit_id,
        "unit_index": record.unit_index,
        "source_kind": record.source_kind,
        "document_family": record.document_family,
        "release_label": record.release_label,
        "content_hash": record.content_hash,
        "artifact_version": record.artifact_version,
        "cache_key": record.cache_key,
        "document_id": record.document_id,
        "embedding_model": record.embedding_model,
        "embedding_status": record.embedding_status,
        "text": record.source_text or record.text,
        "retrieval_text": record.text,
        "parent_unit_id": record.parent_unit_id,
    }

    return PointStruct(
        id=build_qdrant_point_id(record),
        vector=record.vector,
        payload=payload,
    )


def upsert_embedding_batch(
    client: QdrantClient,
    collection_name: str,
    batch: EmbeddingBatch,
) -> QdrantUpsertSummary:
    """Upsert embedded records into a Qdrant collection.

    Records without vectors are skipped because they are not indexable.
    """

    points: list[PointStruct] = []
    skipped_records = 0

    for record in batch.records:
        if record.vector is None:
            skipped_records += 1
            continue
        points.append(embedding_record_to_qdrant_point(record))

    if points:
        client.upsert(
            collection_name=collection_name,
            points=points,
        )

    return QdrantUpsertSummary(
        collection_name=collection_name,
        attempted_records=len(batch.records),
        upserted_points=len(points),
        skipped_records=skipped_records,
    )
