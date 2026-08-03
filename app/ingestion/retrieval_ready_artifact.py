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
    document_id: str = ""
    retrieval_text: str = ""
    parent_unit_id: str | None = None


@dataclass(frozen=True)
class RetrievalReadyArtifact:
    document_name: str
    document_family: str
    release_label: str
    total_units: int
    units: list[RetrievalReadyUnit]
    document_id: str = ""


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
                document_id=normalized_artifact.raw_artifact.parsed_name.document_id,
                retrieval_text=chunk.text,
            )
        )

    for table_chunk in table_chunks.table_chunks:
        parent_unit_id = _find_parent_unit_id(
            paragraph_chunks,
            table_chunk.preceding_paragraph_index,
        )
        units.append(
            RetrievalReadyUnit(
                unit_id=table_chunk.chunk_id,
                unit_index=len(units),
                source_kind="table",
                text=table_chunk.text,
                document_family=table_chunks.document_family,
                release_label=table_chunks.release_label,
                document_id=normalized_artifact.raw_artifact.parsed_name.document_id,
                retrieval_text=_build_table_retrieval_text(table_chunk),
                parent_unit_id=parent_unit_id,
            )
        )

    return RetrievalReadyArtifact(
        document_name=normalized_artifact.raw_artifact.document_name,
        document_family=normalized_artifact.raw_artifact.parsed_name.document_family,
        release_label=normalized_artifact.raw_artifact.parsed_name.release_label,
        total_units=len(units),
        units=units,
        document_id=normalized_artifact.raw_artifact.parsed_name.document_id,
    )


def _find_parent_unit_id(
    paragraph_chunks: ChunkedDocument,
    preceding_paragraph_index: int | None,
) -> str | None:
    if preceding_paragraph_index is None:
        return None
    for chunk in paragraph_chunks.chunks:
        if (
            chunk.original_paragraph_start_index
            <= preceding_paragraph_index
            <= chunk.original_paragraph_end_index
        ):
            return chunk.chunk_id
    return None


def _build_table_retrieval_text(table_chunk: TableChunk) -> str:
    """Enrich table search text without changing its citeable source text."""

    context = (table_chunk.preceding_paragraph_text or "").strip()
    if not context:
        return table_chunk.text
    return f"Parent context: {context}\n\nTable:\n{table_chunk.text}"
