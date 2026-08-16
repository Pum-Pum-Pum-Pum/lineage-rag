from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from app.code_indexing.models import CodeIndexArtifact, CodeIndexRecord


@dataclass(frozen=True)
class CodeQdrantVerification:
    collection_name: str
    expected_points: int
    verified_points: int
    vector_dimension: int
    artifact_identity_sha256: str


def index_code_artifact_new_collection(
    client: QdrantClient,
    *,
    collection_name: str,
    artifact: CodeIndexArtifact,
    batch_size: int = 64,
) -> CodeQdrantVerification:
    if artifact.status != "embedded" or artifact.vector_dimension is None:
        raise ValueError("Only a complete embedded code artifact may be indexed")
    if not collection_name.startswith("code_custom_"):
        raise ValueError("Code collection names must start with 'code_custom_'")
    if batch_size <= 0:
        raise ValueError("batch_size must be greater than zero")
    if client.collection_exists(collection_name):
        raise FileExistsError(
            f"Code collection already exists and will not be modified: {collection_name}"
        )
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(size=artifact.vector_dimension, distance=Distance.COSINE),
    )
    for start in range(0, len(artifact.records), batch_size):
        records = artifact.records[start : start + batch_size]
        client.upsert(
            collection_name=collection_name,
            points=[_point(record) for record in records],
            wait=True,
        )
    return verify_code_collection(
        client,
        collection_name=collection_name,
        artifact=artifact,
    )


def verify_code_collection(
    client: QdrantClient,
    *,
    collection_name: str,
    artifact: CodeIndexArtifact,
) -> CodeQdrantVerification:
    if artifact.status != "embedded" or artifact.vector_dimension is None:
        raise ValueError("Verification requires a complete embedded code artifact")
    if not client.collection_exists(collection_name):
        raise RuntimeError(f"Code collection does not exist: {collection_name}")
    count = client.count(collection_name=collection_name, exact=True).count
    if count != artifact.total_records:
        raise RuntimeError(
            f"Code collection point count mismatch: expected={artifact.total_records}, actual={count}"
        )
    expected = {record.point_id: record for record in artifact.records}
    points = client.retrieve(
        collection_name=collection_name,
        ids=list(expected),
        with_payload=True,
        with_vectors=True,
    )
    observed = {str(point.id): point for point in points}
    missing = sorted(set(expected) - set(observed))
    if missing:
        raise RuntimeError(f"Code collection is missing expected points: {missing[:5]}")
    for point_id, record in expected.items():
        point = observed[point_id]
        payload = point.payload or {}
        expected_payload = _payload(record)
        for key, value in expected_payload.items():
            if payload.get(key) != value:
                raise RuntimeError(
                    f"Code point payload mismatch for {point_id}: {key}"
                )
        vector = point.vector
        if not isinstance(vector, list) or len(vector) != artifact.vector_dimension:
            raise RuntimeError(f"Code point vector dimension mismatch for {point_id}")
    return CodeQdrantVerification(
        collection_name=collection_name,
        expected_points=artifact.total_records,
        verified_points=len(observed),
        vector_dimension=artifact.vector_dimension,
        artifact_identity_sha256=artifact.artifact_identity_sha256,
    )


def _point(record: CodeIndexRecord) -> PointStruct:
    if record.vector is None:
        raise ValueError(f"Code record has no vector: {record.unit_id}")
    return PointStruct(id=record.point_id, vector=list(record.vector), payload=_payload(record))


def _payload(record: CodeIndexRecord) -> dict:
    return {
        "knowledge_lane": "code",
        "unit_id": record.unit_id,
        "unit_index": record.unit_index,
        "snapshot_id": record.snapshot_id,
        "module_id": record.module_id,
        "source_path": record.source_path,
        "source_kind": record.source_kind,
        "display_name": record.display_name,
        "package_name": record.package_name,
        "start_line": record.source_map.start_line,
        "end_line": record.source_map.end_line,
        "start_offset": record.source_map.start_offset,
        "end_offset": record.source_map.end_offset,
        "parent_unit_id": record.parent_unit_id,
        "chunk_index": record.chunk_index,
        "chunk_count": record.chunk_count,
        "parser_state": record.parser_state,
        "conditional_state": record.conditional_state,
        "content_sha256": record.content_sha256,
        "cache_key": record.cache_key,
        "embedding_model": record.embedding_model,
        "text": record.citation_text,
        "retrieval_text": record.embedding_text,
    }
