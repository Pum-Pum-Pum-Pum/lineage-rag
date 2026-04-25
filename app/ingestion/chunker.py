from __future__ import annotations

from dataclasses import dataclass

from app.ingestion.normalized_artifact import NormalizedDocxArtifact


@dataclass(frozen=True)
class TextChunk:
    chunk_id: str
    chunk_index: int
    paragraph_start_index: int
    paragraph_end_index: int
    paragraph_count: int
    text: str


@dataclass(frozen=True)
class ChunkedDocument:
    document_name: str
    document_family: str
    release_label: str
    total_chunks: int
    chunks: list[TextChunk]


def chunk_normalized_artifact(
    artifact: NormalizedDocxArtifact,
    max_paragraphs_per_chunk: int = 5,
) -> ChunkedDocument:
    """Create baseline paragraph-aware chunks from normalized text.

    This first chunking step is intentionally simple: fixed-size chunking by
    cleaned paragraphs before introducing section-aware chunking.
    """

    if max_paragraphs_per_chunk <= 0:
        raise ValueError("max_paragraphs_per_chunk must be greater than 0")

    paragraphs = artifact.normalized_text.cleaned_paragraphs
    chunks: list[TextChunk] = []

    for chunk_index, start in enumerate(range(0, len(paragraphs), max_paragraphs_per_chunk)):
        end = min(start + max_paragraphs_per_chunk, len(paragraphs))
        chunk_paragraphs = paragraphs[start:end]
        chunk_text = "\n".join(chunk_paragraphs)
        chunk_id = f"{artifact.raw_artifact.document_name}::chunk_{chunk_index}"

        chunks.append(
            TextChunk(
                chunk_id=chunk_id,
                chunk_index=chunk_index,
                paragraph_start_index=start,
                paragraph_end_index=end - 1,
                paragraph_count=len(chunk_paragraphs),
                text=chunk_text,
            )
        )

    return ChunkedDocument(
        document_name=artifact.raw_artifact.document_name,
        document_family=artifact.raw_artifact.parsed_name.document_family,
        release_label=artifact.raw_artifact.parsed_name.release_label,
        total_chunks=len(chunks),
        chunks=chunks,
    )
