from __future__ import annotations

from dataclasses import dataclass

from app.ingestion.retrieval_ready_artifact import RetrievalReadyArtifact, RetrievalReadyUnit


@dataclass(frozen=True)
class EmbeddingRecord:
    unit_id: str
    unit_index: int
    source_kind: str
    document_family: str
    release_label: str
    text: str
    embedding_model: str
    embedding_status: str
    vector: list[float] | None = None


@dataclass(frozen=True)
class EmbeddingBatch:
    document_name: str
    total_records: int
    records: list[EmbeddingRecord]


def build_embedding_batch_contract(
    artifact: RetrievalReadyArtifact,
    embedding_model: str,
) -> EmbeddingBatch:
    """Create a local embedding contract from retrieval-ready units.

    This does not call any embedding API yet. It only prepares the exact unit
    structure that the later embedding stage should consume.
    """

    records = [
        EmbeddingRecord(
            unit_id=unit.unit_id,
            unit_index=unit.unit_index,
            source_kind=unit.source_kind,
            document_family=unit.document_family,
            release_label=unit.release_label,
            text=unit.text,
            embedding_model=embedding_model,
            embedding_status="pending",
            vector=None,
        )
        for unit in artifact.units
    ]

    return EmbeddingBatch(
        document_name=artifact.document_name,
        total_records=len(records),
        records=records,
    )
