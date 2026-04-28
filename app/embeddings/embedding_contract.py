from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json

from app.core.config import get_settings
from app.ingestion.retrieval_ready_artifact import RetrievalReadyArtifact, RetrievalReadyUnit


@dataclass(frozen=True)
class EmbeddingRecord:
    unit_id: str
    unit_index: int
    source_kind: str
    document_family: str
    release_label: str
    content_hash: str
    artifact_version: str
    cache_key: str
    text: str
    embedding_model: str
    embedding_status: str
    vector: list[float] | None = None


@dataclass(frozen=True)
class EmbeddingBatch:
    document_name: str
    total_records: int
    records: list[EmbeddingRecord]
    cached_count: int = 0
    embedded_count: int = 0
    cache_miss_count: int = 0


def compute_content_hash(text: str) -> str:
    """Compute a stable hash for retrieval-unit text.

    This is a lightweight cache-key building block. A full cache key later should
    combine this content hash with embedding model and preprocessing version.
    """

    normalized_text = text.replace("\r\n", "\n").replace("\r", "\n")
    return hashlib.sha256(normalized_text.encode("utf-8")).hexdigest()


def compute_embedding_cache_key(
    content_hash: str,
    embedding_model: str,
    artifact_version: str,
) -> str:
    """Compute a deterministic cache key for an embedding record.

    The same text should not blindly reuse an embedding if the embedding model
    or artifact/preprocessing version changes.
    """

    payload = {
        "artifact_version": artifact_version,
        "content_hash": content_hash,
        "embedding_model": embedding_model,
    }
    stable_payload = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(stable_payload.encode("utf-8")).hexdigest()


def build_embedding_batch_contract(
    artifact: RetrievalReadyArtifact,
    embedding_model: str,
    artifact_version: str | None = None,
) -> EmbeddingBatch:
    """Create a local embedding contract from retrieval-ready units.

    This does not call any embedding API yet. It only prepares the exact unit
    structure that the later embedding stage should consume.
    """

    resolved_artifact_version = artifact_version or get_settings().artifact_version
    records = []

    for unit in artifact.units:
        content_hash = compute_content_hash(unit.text)
        cache_key = compute_embedding_cache_key(
            content_hash=content_hash,
            embedding_model=embedding_model,
            artifact_version=resolved_artifact_version,
        )
        records.append(
            EmbeddingRecord(
                unit_id=unit.unit_id,
                unit_index=unit.unit_index,
                source_kind=unit.source_kind,
                document_family=unit.document_family,
                release_label=unit.release_label,
                content_hash=content_hash,
                artifact_version=resolved_artifact_version,
                cache_key=cache_key,
                text=unit.text,
                embedding_model=embedding_model,
                embedding_status="pending",
                vector=None,
            )
        )

    return EmbeddingBatch(
        document_name=artifact.document_name,
        total_records=len(records),
        records=records,
    )
