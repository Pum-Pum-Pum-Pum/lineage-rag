from __future__ import annotations

from pathlib import Path
import re
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
from app.retrieval.identifier_affinity import identifier_affinity
from app.retrieval.lexical_search import tokenize


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
    fdd_limit: int | None = None,
) -> CombinedRetrievalResult:
    """Keep FDD and code retrieval independent, then follow reviewed links.

    ``fdd_results`` must come from the existing FDD retrieval path. This service
    never merges FDD and code scores because the lanes have different evidence
    contracts and thresholds.
    """

    if fdd_limit is None:
        fdd_limit = code_limit
    if fdd_limit <= 0:
        raise ValueError("fdd_limit must be greater than zero")
    direct = retrieve_code_evidence(
        artifact=code_artifact,
        query=query,
        mode=code_mode,
        limit=code_candidate_limit,
        candidate_limit=code_candidate_limit,
        client=client,
        collection_name=collection_name,
        query_vector=query_vector,
        max_units_per_parent=code_max_units_per_parent,
    )
    baseline_fdd = tuple(_fdd_evidence(item) for item in fdd_results)
    fdd_evidence = _select_lineage_anchored_fdd_evidence(
        query=query,
        candidates=baseline_fdd,
        direct_code_candidates=direct.evidence,
        lineage_artifact=lineage_artifact,
        limit=fdd_limit,
    )
    # Exact reviewed symbol selection takes precedence. Otherwise preserve one
    # additional explicit-topic document within the same candidate/output bounds.
    if fdd_evidence == baseline_fdd[:fdd_limit]:
        fdd_evidence = _reserve_fdd_topic_diversity_slot(
            query=query, candidates=baseline_fdd, limit=fdd_limit
        )
    selected_document_ids = {item.document_id for item in fdd_evidence}
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
        limit=code_candidate_limit,
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
        query=query,
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
        direct_code_evidence=_select_parent_diverse_evidence(
            direct.evidence,
            limit=code_limit,
            max_units_per_parent=code_max_units_per_parent,
        ),
        mapped_code_evidence=_select_parent_diverse_evidence(
            mapped.evidence,
            limit=code_limit,
            max_units_per_parent=code_max_units_per_parent,
        ),
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
    query: str,
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
    return _reserve_identifier_affinity_slot(
        query=query,
        candidates=ranked,
        selected=_select_parent_diverse_evidence(
            ranked, limit=limit, max_units_per_parent=max_units_per_parent
        ),
        limit=limit,
        max_units_per_parent=max_units_per_parent,
    )


def _select_parent_diverse_evidence(
    ranked: Sequence[CodeEvidence], *, limit: int, max_units_per_parent: int
) -> tuple[CodeEvidence, ...]:
    """Apply the existing per-parent evidence bound to already ranked items."""

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


def _reserve_identifier_affinity_slot(
    *,
    query: str,
    candidates: Sequence[CodeEvidence],
    selected: tuple[CodeEvidence, ...],
    limit: int,
    max_units_per_parent: int,
    minimum_matches: int = 3,
) -> tuple[CodeEvidence, ...]:
    """Replace one selected item only for a stronger bounded routine-name match."""

    if len(candidates) <= limit or not selected:
        return selected
    selected_ids = {item.unit_id for item in selected}
    affinity_by_id = {
        item.unit_id: identifier_affinity(query, item.display_name) for item in candidates
    }
    eligible = [
        item
        for item in candidates
        if item.unit_id not in selected_ids
        and affinity_by_id[item.unit_id] >= minimum_matches
    ]
    if not eligible:
        return selected
    candidate = max(
        eligible,
        key=lambda item: (affinity_by_id[item.unit_id], item.score, item.unit_id),
    )
    weakest_affinity = min(affinity_by_id[item.unit_id] for item in selected)
    if affinity_by_id[candidate.unit_id] <= weakest_affinity:
        return selected

    candidate_parent = candidate.parent_unit_id or candidate.unit_id
    parent_count = sum(
        1
        for item in selected
        if (item.parent_unit_id or item.unit_id) == candidate_parent
    )
    replacement_pool = list(selected)
    if parent_count >= max_units_per_parent:
        replacement_pool = [
            item
            for item in selected
            if (item.parent_unit_id or item.unit_id) == candidate_parent
        ]
    if not replacement_pool:
        return selected
    replace = min(
        replacement_pool,
        key=lambda item: (affinity_by_id[item.unit_id], item.score, item.unit_id),
    )
    updated = [item for item in selected if item.unit_id != replace.unit_id]
    updated.append(candidate)
    return tuple(updated[:limit])


def _reserve_fdd_topic_diversity_slot(
    *, query: str, candidates: tuple[FddEvidence, ...], limit: int
) -> tuple[FddEvidence, ...]:
    """Retain one additional explicit-topic document, without asserting lineage.

    Only uppercase acronyms or mixed-case names explicitly supplied by the caller
    qualify. Match whole tokens in both document identity and source text; exclude
    title tokens shared by every candidate document (e.g. application prefixes).
    This is a bounded diversity fallback, not semantic relevance or a new link.
    Existing order/scores break ties; no aliases, stemming, release inference,
    extra retrieval, or case-specific document names are used.
    """
    if limit <= 0:
        raise ValueError("limit must be greater than zero")
    selected = list(candidates[:limit])
    if len(candidates) <= limit or not selected:
        return tuple(selected)
    subjects = {
        token.casefold()
        for token in re.findall(r"\b[A-Za-z][A-Za-z0-9]*\b", query)
        if len(token) >= 2 and (
            token.isupper() or (token[0].isupper() and any(c.isupper() for c in token[1:]))
        )
    }
    titles = {
        item.document_id: set(tokenize(item.document_id.replace("_", " ").replace("$", " ")))
        for item in candidates
    }
    subjects -= set.intersection(*titles.values())
    if not subjects:
        return tuple(selected)

    def affinity(item: FddEvidence) -> int:
        return len(subjects & titles[item.document_id] & set(tokenize(item.text)))

    selected_documents = {item.document_id for item in selected}
    eligible = [item for item in candidates[limit:]
                if item.document_id not in selected_documents and affinity(item) > 0]
    if not eligible:
        return tuple(selected)
    # max retains the original candidate order on ties, for lexical/dense/hybrid.
    replacement = max(eligible, key=affinity)
    weakest = min(range(len(selected)), key=lambda i: (affinity(selected[i]), -i))
    if affinity(replacement) <= affinity(selected[weakest]):
        return tuple(selected)
    selected[weakest] = replacement
    return tuple(selected)


def _select_lineage_anchored_fdd_evidence(
    *,
    query: str,
    candidates: tuple[FddEvidence, ...],
    direct_code_candidates: Sequence[CodeEvidence],
    lineage_artifact: FddCodeLineageArtifact,
    limit: int,
) -> tuple[FddEvidence, ...]:
    """Reserve one FDD slot only for one unambiguous reviewed symbol-level link.

    File-scoped mappings deliberately cannot steer FDD ranking: a whole package
    is too broad to determine which of several reviewed FDDs answers a query.
    """

    selected = list(candidates[:limit])
    if len(candidates) <= limit or not direct_code_candidates:
        return tuple(selected)
    selected_documents = {item.document_id for item in selected}
    direct_names = {
        item.display_name.casefold() for item in direct_code_candidates
        if identifier_affinity(query, item.display_name) > 0
    }
    anchor_documents = {
        mapping.fdd_document_id
        for mapping in lineage_artifact.mappings
        if mapping.mapping_status == "reviewed"
        and any(
            target.selector_scope != "file"
            and target.qualified_name is not None
            and target.qualified_name.rsplit(".", 1)[-1].casefold() in direct_names
            for target in mapping.targets
        )
    }
    # Ambiguous links remain visible only if their FDD was already ranked.  This
    # avoids treating a broad or competing lineage edge as a ranking override.
    if len(anchor_documents) != 1:
        return tuple(selected)
    anchor_document = next(iter(anchor_documents))
    if anchor_document in selected_documents:
        return tuple(selected)
    replacement = next(
        (item for item in candidates[limit:] if item.document_id == anchor_document), None
    )
    if replacement is None:
        return tuple(selected)
    selected[-1] = replacement
    return tuple(selected)
