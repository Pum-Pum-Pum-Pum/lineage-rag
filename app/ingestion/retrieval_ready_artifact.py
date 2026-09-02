from __future__ import annotations

from dataclasses import dataclass

from app.ingestion.chunker import ChunkedDocument
from app.ingestion.embedding_input_limits import MAX_DERIVED_CONTEXT_BYTES, truncate_utf8
from app.ingestion.embedded_workbook_chunker import EmbeddedWorkbookChunk, chunk_embedded_workbook
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
    document_lineage_key: str = ""
    document_revision: str | None = None
    attachment_path: str | None = None
    attachment_sha256: str | None = None
    sheet_name: str | None = None
    sheet_role: str | None = None
    source_range: str | None = None


@dataclass(frozen=True)
class RetrievalReadyArtifact:
    document_name: str
    document_family: str
    release_label: str
    total_units: int
    units: list[RetrievalReadyUnit]
    document_id: str = ""
    document_lineage_key: str = ""
    document_revision: str | None = None


def build_retrieval_ready_artifact(
    normalized_artifact: NormalizedDocxArtifact,
    paragraph_chunks: ChunkedDocument,
    table_chunks: ChunkedTableDocument,
) -> RetrievalReadyArtifact:
    """Combine Word and embedded-workbook evidence without mixing provenance."""

    parsed = normalized_artifact.raw_artifact.parsed_name
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
                document_id=parsed.document_id,
                retrieval_text=chunk.text,
                document_lineage_key=parsed.document_lineage_key,
                document_revision=parsed.document_revision,
            )
        )
    for table_chunk in table_chunks.table_chunks:
        parent_unit_id = _find_parent_unit_id(paragraph_chunks, table_chunk.preceding_paragraph_index)
        units.append(
            RetrievalReadyUnit(
                unit_id=table_chunk.chunk_id,
                unit_index=len(units),
                source_kind="table",
                text=table_chunk.text,
                document_family=table_chunks.document_family,
                release_label=table_chunks.release_label,
                document_id=parsed.document_id,
                retrieval_text=_build_table_retrieval_text(table_chunk),
                parent_unit_id=parent_unit_id,
                document_lineage_key=parsed.document_lineage_key,
                document_revision=parsed.document_revision,
                source_range=f"table_{table_chunk.table_index};part={table_chunk.chunk_index}",
            )
        )
    for workbook in normalized_artifact.raw_artifact.embedded_workbooks.workbooks:
        for workbook_chunk in chunk_embedded_workbook(
            document_name=normalized_artifact.raw_artifact.document_name,
            workbook=workbook,
        ):
            parent_unit_id = _find_parent_unit_id(
                paragraph_chunks,
                workbook_chunk.preceding_paragraph_index,
            )
            units.append(
                RetrievalReadyUnit(
                    unit_id=workbook_chunk.chunk_id,
                    unit_index=len(units),
                    source_kind="embedded_workbook",
                    text=workbook_chunk.text,
                    document_family=parsed.document_family,
                    release_label=parsed.release_label,
                    document_id=parsed.document_id,
                    retrieval_text=_build_workbook_retrieval_text(workbook_chunk),
                    parent_unit_id=parent_unit_id,
                    document_lineage_key=parsed.document_lineage_key,
                    document_revision=parsed.document_revision,
                    attachment_path=workbook_chunk.workbook_path,
                    attachment_sha256=workbook_chunk.workbook_sha256,
                    sheet_name=workbook_chunk.sheet_name,
                    sheet_role=workbook_chunk.sheet_role,
                    source_range=(
                        f"{workbook_chunk.sheet_name}!"
                        f"{workbook_chunk.row_start}:{workbook_chunk.row_end}"
                    ),
                )
            )
    return RetrievalReadyArtifact(
        document_name=normalized_artifact.raw_artifact.document_name,
        document_family=parsed.document_family,
        release_label=parsed.release_label,
        total_units=len(units),
        units=units,
        document_id=parsed.document_id,
        document_lineage_key=parsed.document_lineage_key,
        document_revision=parsed.document_revision,
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
    context = _bounded_derived_context(table_chunk.preceding_paragraph_text)
    return table_chunk.text if not context else f"Parent context: {context}\n\nTable:\n{table_chunk.text}"


def _build_workbook_retrieval_text(workbook_chunk: EmbeddedWorkbookChunk) -> str:
    """Add derived context while leaving original cell text citeable."""

    context = _bounded_derived_context(workbook_chunk.preceding_paragraph_text)
    header = (
        "DERIVED RETRIEVAL CONTEXT - NOT A CITATION SOURCE\n"
        f"Embedded workbook: {workbook_chunk.workbook_path}\n"
        f"Sheet: {workbook_chunk.sheet_name} ({workbook_chunk.sheet_role})\n"
        f"Source range: {workbook_chunk.sheet_name}!"
        f"{workbook_chunk.row_start}:{workbook_chunk.row_end}"
    )
    if context:
        header += f"\nParent context: {context}"
    return f"{header}\n\nORIGINAL WORKBOOK CELLS:\n{workbook_chunk.text}"


def _bounded_derived_context(text: str | None) -> str:
    context = (text or "").strip()
    if not context:
        return ""
    bounded = truncate_utf8(context, MAX_DERIVED_CONTEXT_BYTES)
    return bounded if bounded == context else f"{bounded}\n[context truncated for embedding bound]"
