from __future__ import annotations

from dataclasses import dataclass

from app.ingestion.embedding_input_limits import MAX_SOURCE_UNIT_BYTES, split_text_by_utf8_bytes
from app.ingestion.normalized_artifact import NormalizedDocxArtifact


@dataclass(frozen=True)
class TableChunk:
    chunk_id: str
    chunk_index: int
    table_index: int
    row_count: int
    column_count: int
    text: str
    row_start: int | None
    row_end: int | None
    preceding_paragraph_index: int | None
    preceding_paragraph_text: str | None


@dataclass(frozen=True)
class ChunkedTableDocument:
    document_name: str
    document_family: str
    release_label: str
    total_table_chunks: int
    table_chunks: list[TableChunk]


def chunk_tables_from_artifact(artifact: NormalizedDocxArtifact) -> ChunkedTableDocument:
    """Create baseline table chunks from extracted table text.

    This keeps table-derived retrieval units separate from paragraph chunks so we
    can inspect and evaluate them independently before later fusion.
    """

    table_chunks: list[TableChunk] = []

    for table in artifact.raw_artifact.extracted_tables.tables:
        if not table.text_representation.strip():
            continue

        for part_index, part_text in enumerate(
            split_text_by_utf8_bytes(table.text_representation, MAX_SOURCE_UNIT_BYTES)
        ):
            chunk_index = len(table_chunks)
            table_chunks.append(
                TableChunk(
                    chunk_id=(
                        f"{artifact.raw_artifact.document_name}::table_{table.table_index}"
                        f"::part_{part_index}"
                    ),
                    chunk_index=chunk_index,
                    table_index=table.table_index,
                    row_count=table.row_count,
                    column_count=table.column_count,
                    text=part_text,
                    row_start=None,
                    row_end=None,
                    preceding_paragraph_index=table.preceding_paragraph_index,
                    preceding_paragraph_text=table.preceding_paragraph_text,
                )
            )

    return ChunkedTableDocument(
        document_name=artifact.raw_artifact.document_name,
        document_family=artifact.raw_artifact.parsed_name.document_family,
        release_label=artifact.raw_artifact.parsed_name.release_label,
        total_table_chunks=len(table_chunks),
        table_chunks=table_chunks,
    )
