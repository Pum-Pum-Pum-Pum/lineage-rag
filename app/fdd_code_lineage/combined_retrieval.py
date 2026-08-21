from __future__ import annotations

from pathlib import Path
from typing import Any, Literal, Sequence

from pydantic import BaseModel, ConfigDict, Field
from qdrant_client import QdrantClient

from app.code_indexing.models import CodeIndexArtifact
from app.code_retrieval.models import CodeCandidateSummary, CodeEvidence, CodeRetrievalResult
from app.code_retrieval.service import retrieve_code_evidence
from app.fdd_code_lineage.models import (
    FddCodeLineageArtifact,
    resolve_target_unit_ids,
)


class FrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class FddEvidence(FrozenModel):
    unit_id: str
    document_id: str
    document_family: str
    release_label: str
    source_kind: str
    score: float
    text: str


class ReviewedLineageUse(FrozenModel):
    mapping_id: str
    fdd_document_id: str
    code_unit_ids: tuple[str, ...]


class CombinedRetrievalResult(FrozenModel):
    query: str
    mode: Literal["combined"] = "combined"
    fdd_generation: str
    code_snapshot_id: str
    fdd_evidence: tuple[FddEvidence, ...]
    code_evidence: tuple[CodeEvidence, ...]
    direct_code_evidence: tuple[CodeEvidence, ...]
    mapped_code_evidence: tuple[CodeEvidence, ...]
    direct_dense_candidates: tuple[CodeCandidateSummary, ...] = ()
    direct_lexical_candidates: tuple[CodeCandidateSummary, ...] = ()
    mapped_dense_candidates: tuple[CodeCandidateSummary, ...] = ()
    mapped_lexical_candidates: tuple[CodeCandidateSummary, ...] = ()
    reviewed_lineage: tuple[ReviewedLineageUse, ...] = ()
    unknowns: tuple[str, ...] = ()


def retrieve_combined_evidence(
    *,
    query: str,
    fdd_results: Sequence[Any],
    fdd_generation: str,
    known_fdd_document_ids: set[str],
    code_artifact: CodeIndexArtifact,
    lineage_artifact: FddCodeLineageArtifact,
    analysis_directory: Path,
    code_mode: Literal["dense", "lexical", "hybrid"] = "hybrid",
    code_limit: int = 5,
    code_candidate_limit: int = 20,
    client: QdrantClient | None = None,
    collection_name: str | None = None,
    query_vector: Sequence[float] | None = None,
    code_max_units_per_parent: int = 2,
) -> CombinedRetrievalResult:
    """Keep FDD and code retrieval independent, then follow reviewed links.

    ``fdd_results`` must come from the existing FDD retrieval path. This service
    never merges FDD and code scores because the lanes have different evidence
    contracts and thresholds.
    """

    fdd_evidence = tuple(_fdd_evidence(item) for item in fdd_results)
    selected_document_ids = {item.document_id for item in fdd_evidence}
    direct = retrieve_code_evidence(
        artifact=code_artifact,
        query=query,
        mode=code_mode,
        limit=code_limit,
        candidate_limit=code_candidate_limit,
        client=client,
        collection_name=collection_name,
        query_vector=query_vector,
        max_units_per_parent=code_max_units_per_parent,
    )
    mapped_unit_ids, mapping_ids = resolve_target_unit_ids(
        lineage_artifact,
        known_fdd_document_ids=known_fdd_document_ids,
        selected_fdd_document_ids=selected_document_ids,
        code_artifact=code_artifact,
        analysis_directory=analysis_directory,
    )
    mapped = retrieve_code_evidence(
        artifact=code_artifact,
        query=query,
        mode=code_mode,
        limit=code_limit,
        candidate_limit=code_candidate_limit,
        client=client,
        collection_name=collection_name,
        query_vector=query_vector,
        allowed_unit_ids=mapped_unit_ids,
        max_units_per_parent=code_max_units_per_parent,
    )
    merged = _merge_code_evidence(
        direct.evidence,
        mapped.evidence,
        mapping_ids,
        limit=code_limit,
        max_units_per_parent=code_max_units_per_parent,
    )
    lineage_uses = tuple(
        ReviewedLineageUse(
            mapping_id=mapping_id,
            fdd_document_id=next(
                item.fdd_document_id
                for item in lineage_artifact.mappings
                if item.mapping_id == mapping_id
            ),
            code_unit_ids=tuple(item.unit_id for item in mapped.evidence),
        )
        for mapping_id in mapping_ids
    )
    unknowns: list[str] = []
    if not fdd_evidence:
        unknowns.append("No FDD evidence was retrieved.")
    if not direct.evidence:
        unknowns.append("No direct custom-code evidence was retrieved.")
    if selected_document_ids and not mapping_ids:
        unknowns.append("No reviewed FDD-to-code mapping applies to the retrieved documents.")
    return CombinedRetrievalResult(
        query=query,
        fdd_generation=fdd_generation,
        code_snapshot_id=code_artifact.snapshot_id,
        fdd_evidence=fdd_evidence,
        code_evidence=merged,
        direct_code_evidence=direct.evidence,
        mapped_code_evidence=mapped.evidence,
        direct_dense_candidates=direct.dense_candidates,
        direct_lexical_candidates=direct.lexical_candidates,
        mapped_dense_candidates=mapped.dense_candidates,
        mapped_lexical_candidates=mapped.lexical_candidates,
        reviewed_lineage=lineage_uses,
        unknowns=tuple(unknowns),
    )


def _fdd_evidence(result: Any) -> FddEvidence:
    payload = dict(result.payload)
    required = (
        "unit_id",
        "document_id",
        "document_family",
        "release_label",
        "source_kind",
        "text",
    )
    missing = [key for key in required if not str(payload.get(key, "")).strip()]
    if missing:
        raise RuntimeError(f"FDD evidence is missing required identity: {missing}")
    return FddEvidence(
        unit_id=str(payload["unit_id"]),
        document_id=str(payload["document_id"]),
        document_family=str(payload["document_family"]),
        release_label=str(payload["release_label"]),
        source_kind=str(payload["source_kind"]),
        score=float(result.score),
        text=str(payload["text"]),
    )


def _merge_code_evidence(
    direct: Sequence[CodeEvidence],
    mapped: Sequence[CodeEvidence],
    mapping_ids: Sequence[str],
    *,
    limit: int,
    max_units_per_parent: int,
) -> tuple[CodeEvidence, ...]:
    by_unit = {item.unit_id: item for item in direct}
    for item in mapped:
        metadata = dict(item.retrieval_metadata)
        metadata["reviewed_mapping_ids"] = list(mapping_ids)
        mapped_item = item.model_copy(update={"retrieval_metadata": metadata})
        existing = by_unit.get(item.unit_id)
        if existing is None:
            by_unit[item.unit_id] = mapped_item
        else:
            combined = dict(existing.retrieval_metadata)
            combined["reviewed_mapping_ids"] = list(mapping_ids)
            by_unit[item.unit_id] = existing.model_copy(
                update={"retrieval_metadata": combined}
            )
    ranked = sorted(by_unit.values(), key=lambda item: (-item.score, item.unit_id))
    grouped: dict[str, list[tuple[int, CodeEvidence]]] = {}
    for rank, item in enumerate(ranked):
        parent_key = item.parent_unit_id or item.unit_id
        grouped.setdefault(parent_key, []).append((rank, item))
    selected: list[CodeEvidence] = []
    for occurrence in range(max_units_per_parent):
        layer = sorted(
            (items[occurrence] for items in grouped.values() if len(items) > occurrence),
            key=lambda ranked_item: ranked_item[0],
        )
        for _, item in layer:
            selected.append(item)
            if len(selected) == limit:
                return tuple(selected)
    return tuple(selected)
