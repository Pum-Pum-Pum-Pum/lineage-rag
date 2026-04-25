from __future__ import annotations

from dataclasses import dataclass

from app.ingestion.chunker import ChunkedDocument, TextChunk
from app.ingestion.normalized_artifact import NormalizedDocxArtifact
from app.ingestion.table_chunker import ChunkedTableDocument, TableChunk


@dataclass(frozen=True)
class RetrievalReadyUnit:
    unit_id: str
    unit_index: int
    source_kind: str
    text: str
    document_family: str
    release_label: str


@dataclass(frozen=True)
class RetrievalReadyArtifact:
    document_name: str
    document_family: str
    release_label: str
    total_units: int
    units: list[RetrievalReadyUnit]


def build_retrieval_ready_artifact(
    normalized_artifact: NormalizedDocxArtifact,
    paragraph_chunks: ChunkedDocument,
    table_chunks: ChunkedTableDocument,
) -> RetrievalReadyArtifact:
    """Combine paragraph chunks and table chunks into one retrieval-ready artifact.

    This keeps both content streams available for later embeddings and retrieval
    while preserving source type and release-aware metadata.
    """

    units: list[RetrievalReadyUnit] = []

    for chunk in paragraph_chunks.chunks:
        units.append(
            RetrievalReadyUnit(
                unit_id=chunk.chunk_id,
                unit_index=len(units),
                source_kind="paragraph",
                text=chunk.text,
                document_family=paragraph_chunks.document_family,
                release_label=paragraph_chunks.release_label,
            )
        )

    for table_chunk in table_chunks.table_chunks:
        units.append(
            RetrievalReadyUnit(
                unit_id=table_chunk.chunk_id,
                unit_index=len(units),
                source_kind="table",
                text=table_chunk.text,
                document_family=table_chunks.document_family,
                release_label=table_chunks.release_label,
            )
        )

    return RetrievalReadyArtifact(
        document_name=normalized_artifact.raw_artifact.document_name,
        document_family=normalized_artifact.raw_artifact.parsed_name.document_family,
        release_label=normalized_artifact.raw_artifact.parsed_name.release_label,
        total_units=len(units),
        units=units,
    )
