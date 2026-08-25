"""Framework-independent retrieval and bounded source-resolution service.

This module intentionally owns retrieval only.  API, Streamlit, and future MCP
adapters may format or generate answers, but they must not recreate embedding,
retrieval, lineage, source-identity, or Qdrant lifecycle logic.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Literal

from pydantic import BaseModel, ConfigDict, Field
from qdrant_client import QdrantClient

from app.code_indexing.contract import load_code_index_artifact
from app.code_indexing.models import CodeIndexArtifact
from app.code_retrieval.models import CodeRetrievalResult
from app.code_retrieval.service import retrieve_code_evidence
from app.embeddings.client import get_embedding_client
from app.fdd_code_lineage.combined_retrieval import CombinedRetrievalResult, retrieve_combined_evidence
from app.fdd_code_lineage.models import FddCodeLineageArtifact, validate_lineage_artifact
from app.fdd_code_lineage.paid_evaluation import embed_one_query
from app.retrieval.lexical_search import LexicalSearchDocument, load_retrieval_ready_documents
from app.retrieval.retrieval_config import RetrievalRuntimeConfig, build_retrieval_runtime_config
from app.services.query_retrieval import PlannedRetrievalResult, retrieve_planned_query_evidence


KnowledgeMode = Literal["fdd", "code", "combined"]
SourceType = Literal["fdd", "code"]


class FrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class KnowledgeSearchHit(FrozenModel):
    """Safe, source-aware retrieval item for transport adapters."""

    id: str = Field(pattern=r"^(fdd|code)_[0-9a-f]{64}$")
    title: str
    source_type: SourceType
    short_excerpt: str = Field(max_length=240)
    score: float | None = None
    metadata: dict[str, Any]
    source_reference: str


class KnowledgeSearchResponse(FrozenModel):
    """Transport-neutral bounded retrieval response."""

    schema_version: Literal["knowledge_search_v1"] = "knowledge_search_v1"
    query: str
    mode: KnowledgeMode
    retrieval_mode: Literal["dense", "lexical", "hybrid"]
    ranking_scope: Literal["global", "per_source_type"]
    results: tuple[KnowledgeSearchHit, ...]


class KnowledgeFetchResponse(FrozenModel):
    """Exact original source unit permitted by an opaque lookup ID."""

    schema_version: Literal["knowledge_fetch_v1"] = "knowledge_fetch_v1"
    id: str = Field(pattern=r"^(fdd|code)_[0-9a-f]{64}$")
    title: str
    text: str
    source_type: SourceType
    metadata: dict[str, Any]
    source_reference: str


class _CatalogSource(FrozenModel):
    """One active source occurrence; never expose its internal ID publicly."""

    public_id: str = Field(pattern=r"^(fdd|code)_[0-9a-f]{64}$")
    internal_unit_id: str
    title: str
    source_type: SourceType
    text: str
    metadata: dict[str, Any]
    source_reference: str


@dataclass(frozen=True)
class SourceCatalog:
    """Immutable active-source lookup used for safe result and fetch identity."""

    by_public_id: dict[str, _CatalogSource]
    fdd_by_internal_id: dict[str, _CatalogSource]
    code_by_internal_id: dict[str, _CatalogSource]

    @classmethod
    def build(
        cls,
        *,
        fdd_documents: list[LexicalSearchDocument],
        code_artifact: CodeIndexArtifact | None,
        fdd_generation: str,
    ) -> "SourceCatalog":
        by_public: dict[str, _CatalogSource] = {}
        fdd_by_internal: dict[str, _CatalogSource] = {}
        code_by_internal: dict[str, _CatalogSource] = {}

        for document in fdd_documents:
            public_id = _public_id(
                "fdd",
                document.unit_id,
                _content_sha256(document.text),
            )
            source = _CatalogSource(
                public_id=public_id,
                internal_unit_id=document.unit_id,
                title=document.document_id or document.document_name,
                source_type="fdd",
                text=document.text,
                metadata={
                    "document_id": document.document_id,
                    "document_family": document.document_family,
                    "release_label": document.release_label,
                    "source_kind": document.source_kind,
                    "unit_index": document.unit_index,
                    "fdd_generation": fdd_generation,
                },
                source_reference=(
                    f"document:{document.document_id or document.document_name}"
                    f"#source={public_id}"
                ),
            )
            _register_source(by_public, fdd_by_internal, source)

        if code_artifact is not None:
            for record in code_artifact.records:
                source = _CatalogSource(
                    public_id=_public_id("code", record.unit_id, record.content_sha256),
                    internal_unit_id=record.unit_id,
                    title=record.display_name,
                    source_type="code",
                    text=record.citation_text,
                    metadata={
                        "snapshot_id": record.snapshot_id,
                        "module_id": record.module_id,
                        "source_path": record.source_path,
                        "source_kind": record.source_kind,
                        "display_name": record.display_name,
                        "package_name": record.package_name,
                        "start_line": record.source_map.start_line,
                        "end_line": record.source_map.end_line,
                        "parser_state": record.parser_state,
                        "conditional_state": record.conditional_state,
                    },
                    source_reference=(
                        f"code:{record.snapshot_id}/{_safe_code_path(record.source_path)}"
                        f"#L{record.source_map.start_line}-L{record.source_map.end_line}"
                    ),
                )
                _register_source(by_public, code_by_internal, source)

        return cls(
            by_public_id=by_public,
            fdd_by_internal_id=fdd_by_internal,
            code_by_internal_id=code_by_internal,
        )

    def fetch(self, public_id: str) -> KnowledgeFetchResponse:
        source = self.by_public_id.get(public_id)
        if source is None:
            raise LookupError("Requested source is unavailable.")
        return KnowledgeFetchResponse(
            id=source.public_id,
            title=source.title,
            text=source.text,
            source_type=source.source_type,
            metadata=source.metadata,
            source_reference=source.source_reference,
        )


@dataclass(frozen=True)
class KnowledgeRetrievalExecution:
    """Raw retrieval result for answer orchestration and safe result formatting."""

    mode: KnowledgeMode
    query: str
    retrieval_mode: Literal["dense", "lexical", "hybrid"]
    fdd: PlannedRetrievalResult | None = None
    code: CodeRetrievalResult | None = None
    combined: CombinedRetrievalResult | None = None
    embedding_call: dict[str, Any] | None = None


QdrantClientFactory = Callable[[Path], QdrantClient]
EmbeddingClientFactory = Callable[[], Any]
CodeArtifactLoader = Callable[[Path], CodeIndexArtifact]
FddDocumentLoader = Callable[[Path], list[LexicalSearchDocument]]


class KnowledgeRetrievalService:
    """One reusable retrieval boundary for all approved knowledge lanes."""

    def __init__(
        self,
        *,
        settings: Any,
        retrieval_config: RetrievalRuntimeConfig | None = None,
        qdrant_client_factory: QdrantClientFactory | None = None,
        embedding_client_factory: EmbeddingClientFactory | None = None,
        code_artifact_loader: CodeArtifactLoader = load_code_index_artifact,
        fdd_document_loader: FddDocumentLoader = load_retrieval_ready_documents,
    ) -> None:
        self.settings = settings
        self.retrieval_config = retrieval_config or build_retrieval_runtime_config(settings)
        self._qdrant_client_factory = qdrant_client_factory or (
            lambda path: QdrantClient(path=str(path))
        )
        self._embedding_client_factory = embedding_client_factory or get_embedding_client
        self._code_artifact_loader = code_artifact_loader
        self._fdd_document_loader = fdd_document_loader

    def retrieve(
        self,
        *,
        query: str,
        mode: KnowledgeMode,
        limit: int = 5,
        document_family: str | None = None,
        release_label: str | None = None,
        source_kind: str | None = None,
        conversation_context: str | None = None,
    ) -> KnowledgeRetrievalExecution:
        """Retrieve one lane without generating an answer or writing a trace."""

        cleaned_query = query.strip()
        if not cleaned_query:
            raise ValueError("Retrieval query must not be blank")
        if mode not in {"fdd", "code", "combined"}:
            raise ValueError("Knowledge mode must be fdd, code, or combined")
        if limit <= 0:
            raise ValueError("Retrieval limit must be greater than zero")
        if mode != "fdd" and not bool(getattr(self.settings, "code_modes_enabled", False)):
            raise PermissionError("Code and combined knowledge modes are not activated.")

        retrieval_query = _with_conversation_context(cleaned_query, conversation_context)
        if mode == "fdd":
            return self._retrieve_fdd(
                query=cleaned_query,
                retrieval_query=retrieval_query,
                limit=limit,
                document_family=document_family,
                release_label=release_label,
                source_kind=source_kind,
                conversation_context=conversation_context,
            )
        return self._retrieve_code_or_combined(
            query=cleaned_query,
            retrieval_query=retrieval_query,
            mode=mode,
            limit=limit,
        )

    def search(self, *, query: str, mode: KnowledgeMode, limit: int = 5) -> KnowledgeSearchResponse:
        """Return bounded, safe source summaries for an approved retrieval query."""

        execution = self.retrieve(query=query, mode=mode, limit=limit)
        catalog = self._build_catalog(include_code=mode in {"code", "combined"})
        results: list[KnowledgeSearchHit] = []
        ranking_scope: Literal["global", "per_source_type"] = "global"

        if execution.fdd is not None:
            results.extend(
                _to_search_hits(
                    execution.fdd.results,
                    catalog.fdd_by_internal_id,
                    limit=limit,
                )
            )
        if execution.code is not None:
            results.extend(
                _to_code_search_hits(execution.code, catalog.code_by_internal_id, limit=limit)
            )
        if execution.combined is not None:
            ranking_scope = "per_source_type"
            results.extend(
                _to_fdd_evidence_hits(execution.combined, catalog.fdd_by_internal_id, limit=limit)
            )
            results.extend(
                _to_combined_code_hits(execution.combined, catalog.code_by_internal_id, limit=limit)
            )
        return KnowledgeSearchResponse(
            query=execution.query,
            mode=execution.mode,
            retrieval_mode=execution.retrieval_mode,
            ranking_scope=ranking_scope,
            results=tuple(results),
        )

    def fetch(self, public_id: str, *, include_code: bool = True) -> KnowledgeFetchResponse:
        """Resolve one active opaque source ID without accepting a filesystem path."""

        if not _is_public_id(public_id):
            raise LookupError("Requested source is unavailable.")
        catalog = self._build_catalog(include_code=include_code)
        return catalog.fetch(public_id)

    def _retrieve_fdd(
        self,
        *,
        query: str,
        retrieval_query: str,
        limit: int,
        document_family: str | None,
        release_label: str | None,
        source_kind: str | None,
        conversation_context: str | None,
    ) -> KnowledgeRetrievalExecution:
        client = None
        query_vector = None
        embedding_call = None
        try:
            if self.retrieval_config.retrieval_mode in {"dense", "hybrid"}:
                client = self._qdrant_client_factory(self.settings.qdrant_local_path)
                if not client.collection_exists(self.settings.qdrant_collection_name):
                    raise RuntimeError("Configured FDD collection is unavailable")
                query_vector, embedding_call = self._embed_query(
                    retrieval_query,
                    expected_dimension=int(self.settings.qdrant_vector_size),
                )
            planned = retrieve_planned_query_evidence(
                qdrant_client=client,
                collection_name=self.settings.qdrant_collection_name,
                query_text=retrieval_query,
                embedding_model=self.settings.openai_embedding_model,
                query_vector=query_vector,
                retrieval_config=self.retrieval_config,
                lexical_artifact_directory=self._fdd_artifact_directory(),
                limit=limit,
                document_family=document_family,
                release_label=release_label,
                source_kind=source_kind,
                conversation_context=conversation_context,
            )
        finally:
            if client is not None:
                client.close()
        return KnowledgeRetrievalExecution(
            mode="fdd",
            query=query,
            retrieval_mode=self.retrieval_config.retrieval_mode,
            fdd=planned,
            embedding_call=embedding_call,
        )

    def _retrieve_code_or_combined(
        self,
        *,
        query: str,
        retrieval_query: str,
        mode: Literal["code", "combined"],
        limit: int,
    ) -> KnowledgeRetrievalExecution:
        artifact = self._code_artifact_loader(self.settings.code_index_artifact_path)
        requires_vector = self.retrieval_config.retrieval_mode in {"dense", "hybrid"}
        query_vector = None
        embedding_call = None
        if requires_vector:
            query_vector, embedding_call = self._embed_query(
                retrieval_query,
                expected_dimension=int(artifact.vector_dimension or 0),
            )

        code_client = None
        fdd_client = None
        try:
            if requires_vector:
                code_client = self._qdrant_client_factory(self.settings.code_qdrant_local_path)
                if not code_client.collection_exists(self.settings.code_qdrant_collection_name):
                    raise RuntimeError("Configured code collection is unavailable")
            if mode == "code":
                code = retrieve_code_evidence(
                    artifact=artifact,
                    query=retrieval_query,
                    mode=self.retrieval_config.retrieval_mode,
                    limit=limit,
                    candidate_limit=max(limit, self.retrieval_config.hybrid_candidate_limit),
                    client=code_client,
                    collection_name=self.settings.code_qdrant_collection_name,
                    query_vector=query_vector,
                    dense_weight=self.retrieval_config.hybrid_dense_weight,
                    lexical_weight=self.retrieval_config.hybrid_lexical_weight,
                )
                return KnowledgeRetrievalExecution(
                    mode="code",
                    query=query,
                    retrieval_mode=self.retrieval_config.retrieval_mode,
                    code=code,
                    embedding_call=embedding_call,
                )

            if requires_vector:
                fdd_client = self._qdrant_client_factory(self.settings.qdrant_local_path)
                if not fdd_client.collection_exists(self.settings.qdrant_collection_name):
                    raise RuntimeError("Configured FDD collection is unavailable")
                if int(self.settings.qdrant_vector_size) != int(artifact.vector_dimension or 0):
                    raise RuntimeError("FDD and code query vector dimensions are incompatible")

            documents = self._fdd_document_loader(self._fdd_artifact_directory())
            lineage = FddCodeLineageArtifact.model_validate_json(
                Path(self.settings.fdd_code_lineage_artifact_path).read_text(encoding="utf-8")
            )
            validate_lineage_artifact(
                lineage,
                fdd_document_ids={item.document_id for item in documents},
                code_artifact=artifact,
                analysis_directory=self.settings.code_analysis_directory,
            )
            if lineage.status != "reviewed" or lineage.fdd_generation != self.settings.fdd_generation:
                raise RuntimeError("Configured FDD/code lineage is not reviewed or generation-compatible")
            planned = retrieve_planned_query_evidence(
                qdrant_client=fdd_client,
                collection_name=self.settings.qdrant_collection_name,
                query_text=retrieval_query,
                embedding_model=self.settings.openai_embedding_model,
                query_vector=query_vector,
                retrieval_config=self.retrieval_config,
                lexical_artifact_directory=self._fdd_artifact_directory(),
                limit=limit,
            )
            combined = retrieve_combined_evidence(
                query=retrieval_query,
                fdd_results=planned.results,
                fdd_generation=self.settings.fdd_generation,
                known_fdd_document_ids={item.document_id for item in documents},
                code_artifact=artifact,
                lineage_artifact=lineage,
                analysis_directory=self.settings.code_analysis_directory,
                code_mode=self.retrieval_config.retrieval_mode,
                code_limit=limit,
                code_candidate_limit=max(limit, self.retrieval_config.hybrid_candidate_limit),
                client=code_client,
                collection_name=self.settings.code_qdrant_collection_name,
                query_vector=query_vector,
            )
            return KnowledgeRetrievalExecution(
                mode="combined",
                query=query,
                retrieval_mode=self.retrieval_config.retrieval_mode,
                fdd=planned,
                combined=combined,
                embedding_call=embedding_call,
            )
        finally:
            if code_client is not None:
                code_client.close()
            if fdd_client is not None:
                fdd_client.close()

    def _embed_query(self, query: str, *, expected_dimension: int) -> tuple[list[float], dict[str, Any]]:
        if expected_dimension <= 0:
            raise RuntimeError("Configured query vector dimension is invalid")
        return embed_one_query(
            client=self._embedding_client_factory(),
            model=self.settings.openai_embedding_model,
            question=query,
            expected_dimension=expected_dimension,
        )

    def _build_catalog(self, *, include_code: bool) -> SourceCatalog:
        documents = self._fdd_document_loader(self._fdd_artifact_directory())
        artifact = (
            self._code_artifact_loader(self.settings.code_index_artifact_path)
            if include_code
            else None
        )
        return SourceCatalog.build(
            fdd_documents=documents,
            code_artifact=artifact,
            fdd_generation=self.settings.fdd_generation,
        )

    def _fdd_artifact_directory(self) -> Path:
        return Path(
            getattr(
                self.settings,
                "fdd_retrieval_artifact_dir",
                getattr(self.settings, "processed_dir"),
            )
        )


def _to_search_hits(results: list[Any], catalog: dict[str, _CatalogSource], *, limit: int) -> list[KnowledgeSearchHit]:
    hits: list[KnowledgeSearchHit] = []
    for result in results[:limit]:
        source = _catalog_source(catalog, str(result.payload.get("unit_id", "")))
        hits.append(_search_hit(source, float(result.score)))
    return hits


def _to_code_search_hits(result: CodeRetrievalResult, catalog: dict[str, _CatalogSource], *, limit: int) -> list[KnowledgeSearchHit]:
    return [_search_hit(_catalog_source(catalog, item.unit_id), item.score) for item in result.evidence[:limit]]


def _to_fdd_evidence_hits(result: CombinedRetrievalResult, catalog: dict[str, _CatalogSource], *, limit: int) -> list[KnowledgeSearchHit]:
    return [_search_hit(_catalog_source(catalog, item.unit_id), item.score) for item in result.fdd_evidence[:limit]]


def _to_combined_code_hits(result: CombinedRetrievalResult, catalog: dict[str, _CatalogSource], *, limit: int) -> list[KnowledgeSearchHit]:
    return [_search_hit(_catalog_source(catalog, item.unit_id), item.score) for item in result.code_evidence[:limit]]


def _catalog_source(catalog: dict[str, _CatalogSource], internal_unit_id: str) -> _CatalogSource:
    source = catalog.get(internal_unit_id)
    if source is None:
        raise RuntimeError("Retrieved source is absent from the active source catalog")
    return source


def _search_hit(source: _CatalogSource, score: float) -> KnowledgeSearchHit:
    return KnowledgeSearchHit(
        id=source.public_id,
        title=source.title,
        source_type=source.source_type,
        short_excerpt=" ".join(source.text.split())[:240],
        score=score,
        metadata=source.metadata,
        source_reference=source.source_reference,
    )


def _with_conversation_context(query: str, conversation_context: str | None) -> str:
    if not conversation_context or not conversation_context.strip():
        return query
    return (
        f"{query}\n\nConversation context for reference resolution only:\n"
        f"{conversation_context.strip()}"
    )


def _register_source(
    by_public: dict[str, _CatalogSource],
    by_internal: dict[str, _CatalogSource],
    source: _CatalogSource,
) -> None:
    if source.public_id in by_public or source.internal_unit_id in by_internal:
        raise RuntimeError("Active source catalog contains a duplicate source identity")
    by_public[source.public_id] = source
    by_internal[source.internal_unit_id] = source


def _public_id(source_type: SourceType, internal_unit_id: str, content_sha256: str) -> str:
    digest = hashlib.sha256(
        f"knowledge_source_v1\0{source_type}\0{internal_unit_id}\0{content_sha256}".encode("utf-8")
    ).hexdigest()
    return f"{source_type}_{digest}"


def _content_sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _safe_reference_component(value: str) -> str:
    return value.replace("#", "%23").replace("\\", "/")


def _safe_code_path(value: str) -> str:
    candidate = Path(value.replace("\\", "/"))
    if candidate.is_absolute() or ".." in candidate.parts:
        raise RuntimeError("Code source path escapes the approved snapshot")
    return candidate.as_posix()


def _is_public_id(value: str) -> bool:
    if not value.startswith(("fdd_", "code_")):
        return False
    prefix_length = 4 if value.startswith("fdd_") else 5
    return len(value) == prefix_length + 64 and all(
        character in "0123456789abcdef" for character in value[prefix_length:]
    )
