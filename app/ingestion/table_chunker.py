from __future__ import annotations

from dataclasses import dataclass

from app.ingestion.normalized_artifact import NormalizedDocxArtifact


@dataclass(frozen=True)
class TableChunk:
    chunk_id: str
    chunk_index: int
    table_index: int
    row_count: int
    column_count: int
    text: str


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

    for chunk_index, table in enumerate(artifact.raw_artifact.extracted_tables.tables):
        if not table.text_representation.strip():
            continue

        chunk_id = f"{artifact.raw_artifact.document_name}::table_chunk_{chunk_index}"
        table_chunks.append(
            TableChunk(
                chunk_id=chunk_id,
                chunk_index=chunk_index,
                table_index=table.table_index,
                row_count=table.row_count,
                column_count=table.column_count,
                text=table.text_representation,
            )
        )

    return ChunkedTableDocument(
        document_name=artifact.raw_artifact.document_name,
        document_family=artifact.raw_artifact.parsed_name.document_family,
        release_label=artifact.raw_artifact.parsed_name.release_label,
        total_table_chunks=len(table_chunks),
        table_chunks=table_chunks,
    )
