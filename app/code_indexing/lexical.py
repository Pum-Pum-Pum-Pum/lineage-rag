from __future__ import annotations

from app.code_indexing.models import CodeIndexArtifact
from app.retrieval.lexical_search import (
    LexicalSearchDocument,
    LexicalSearchResult,
    search_lexical_documents,
)


def search_code_lexical_artifact(
    artifact: CodeIndexArtifact,
    query: str,
    *,
    limit: int = 10,
    source_kind: str | None = None,
) -> list[LexicalSearchResult]:
    documents = [
        LexicalSearchDocument(
            document_name=record.source_path,
            document_id=record.snapshot_id,
            unit_id=record.unit_id,
            unit_index=record.unit_index,
            source_kind=record.source_kind,
            document_family=record.module_id,
            release_label=record.snapshot_id,
            text=record.citation_text,
            retrieval_text=record.embedding_text,
            parent_unit_id=record.parent_unit_id,
        )
        for record in artifact.records
    ]
    return search_lexical_documents(
        documents,
        query,
        limit=limit,
        source_kind=source_kind,
    )
