from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json

from app.core.config import get_settings
from app.ingestion.embedding_input_limits import MAX_EMBEDDING_INPUT_BYTES, utf8_byte_length
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
    document_id: str = ""
    source_text: str = ""
    parent_unit_id: str | None = None
    document_lineage_key: str = ""
    document_revision: str | None = None
    attachment_path: str | None = None
    attachment_sha256: str | None = None
    sheet_name: str | None = None
    sheet_role: str | None = None
    source_range: str | None = None


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
        retrieval_text = unit.retrieval_text or unit.text
        content_hash = compute_content_hash(retrieval_text)
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
                text=retrieval_text,
                embedding_model=embedding_model,
                embedding_status="pending",
                vector=None,
                document_id=unit.document_id or artifact.document_name.rsplit(".", 1)[0],
                source_text=unit.text,
                parent_unit_id=unit.parent_unit_id,
                document_lineage_key=unit.document_lineage_key,
                document_revision=unit.document_revision,
                attachment_path=unit.attachment_path,
                attachment_sha256=unit.attachment_sha256,
                sheet_name=unit.sheet_name,
                sheet_role=unit.sheet_role,
                source_range=unit.source_range,
            )
        )

    return EmbeddingBatch(
        document_name=artifact.document_name,
        total_records=len(records),
        records=records,
    )


def validate_embedding_batch_inputs(batch: EmbeddingBatch) -> None:
    """Fail before a provider call when any exact embedding input is oversized."""

    oversized = [
        record
        for record in batch.records
        if utf8_byte_length(record.text) > MAX_EMBEDDING_INPUT_BYTES
    ]
    if oversized:
        samples = ", ".join(
            (
                f"{record.unit_id}"
                f" (source_kind={record.source_kind}, bytes={utf8_byte_length(record.text)})"
            )
            for record in oversized[:5]
        )
        raise ValueError(
            "Embedding input preflight failed: each unit must be at most "
            f"{MAX_EMBEDDING_INPUT_BYTES} UTF-8 bytes. Oversized units: {samples}"
        )
