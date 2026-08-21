from __future__ import annotations

from typing import Any, Literal, Sequence

from qdrant_client import QdrantClient
from qdrant_client.models import FieldCondition, Filter, MatchAny, MatchValue

from app.code_indexing.lexical import search_code_lexical_artifact
from app.code_indexing.models import CodeIndexArtifact, CodeIndexRecord
from app.code_retrieval.models import (
    CodeCandidateSummary,
    CodeEvidence,
    CodeRetrievalResult,
)
from app.retrieval.hybrid_search import fuse_dense_and_lexical_results
from app.vectorstore.qdrant_search import QdrantSearchResult


RetrievalMode = Literal["dense", "lexical", "hybrid"]


def retrieve_code_evidence(
    *,
    artifact: CodeIndexArtifact,
    query: str,
    mode: RetrievalMode,
    limit: int = 5,
    candidate_limit: int = 20,
    source_kind: str | None = None,
    client: QdrantClient | None = None,
    collection_name: str | None = None,
    query_vector: Sequence[float] | None = None,
    dense_weight: float = 0.40,
    lexical_weight: float = 0.60,
    allowed_unit_ids: set[str] | None = None,
    max_units_per_parent: int = 2,
) -> CodeRetrievalResult:
    """Retrieve only evidence belonging to one reviewed code artifact generation.

    Query-vector creation is intentionally outside this boundary. That keeps a
    local retrieval call from silently making a paid external embedding call.
    """

    query = query.strip()
    if not query:
        raise ValueError("Code retrieval query must not be blank")
    if mode not in {"dense", "lexical", "hybrid"}:
        raise ValueError("Code retrieval mode must be dense, lexical, or hybrid")
    if artifact.status != "embedded" or artifact.dependency_review_status != "reviewed":
        raise ValueError("Code retrieval requires a reviewed embedded artifact")
    if limit <= 0 or candidate_limit < limit:
        raise ValueError("candidate_limit must be greater than or equal to positive limit")
    if max_units_per_parent <= 0:
        raise ValueError("max_units_per_parent must be greater than zero")
    if allowed_unit_ids is not None and not allowed_unit_ids:
        return CodeRetrievalResult(
            query=query,
            mode=mode,
            snapshot_id=artifact.snapshot_id,
            artifact_identity_sha256=artifact.artifact_identity_sha256,
            collection_name=collection_name if mode != "lexical" else None,
            evidence=(),
        )
    known_unit_ids = {record.unit_id for record in artifact.records}
    if allowed_unit_ids is not None and not allowed_unit_ids.issubset(known_unit_ids):
        raise ValueError("Code retrieval unit filter contains IDs outside the artifact")

    records = {record.unit_id: record for record in artifact.records}
    lexical_results: list[Any] = []
    dense_results: list[Any] = []

    if mode in {"lexical", "hybrid"}:
        lexical_results = search_code_lexical_artifact(
            artifact,
            query,
            limit=candidate_limit,
            source_kind=source_kind,
            allowed_unit_ids=allowed_unit_ids,
        )
        lexical_results = _canonicalize_results(
            lexical_results, records, artifact, retrieval_method="lexical"
        )

    if mode in {"dense", "hybrid"}:
        dense_results = _search_code_vectors(
            artifact=artifact,
            client=client,
            collection_name=collection_name,
            query_vector=query_vector,
            limit=candidate_limit,
            source_kind=source_kind,
            allowed_unit_ids=allowed_unit_ids,
        )
        dense_results = _canonicalize_results(
            dense_results, records, artifact, retrieval_method="dense"
        )

    if mode == "lexical":
        selected = _select_parent_diverse(
            lexical_results,
            records,
            limit=limit,
            max_units_per_parent=max_units_per_parent,
        )
    elif mode == "dense":
        selected = _select_parent_diverse(
            dense_results,
            records,
            limit=limit,
            max_units_per_parent=max_units_per_parent,
        )
    else:
        selected = fuse_dense_and_lexical_results(
            dense_results,
            lexical_results,
            limit=candidate_limit,
            dense_weight=dense_weight,
            lexical_weight=lexical_weight,
        )
        selected = _canonicalize_results(
            selected, records, artifact, retrieval_method="hybrid"
        )
        selected = _select_parent_diverse(
            selected,
            records,
            limit=limit,
            max_units_per_parent=max_units_per_parent,
        )

    return CodeRetrievalResult(
        query=query,
        mode=mode,
        snapshot_id=artifact.snapshot_id,
        artifact_identity_sha256=artifact.artifact_identity_sha256,
        collection_name=collection_name if mode != "lexical" else None,
        evidence=tuple(_to_evidence(result, records) for result in selected),
        dense_candidates=tuple(_to_summary(item, records) for item in dense_results),
        lexical_candidates=tuple(_to_summary(item, records) for item in lexical_results),
    )


def _search_code_vectors(
    *,
    artifact: CodeIndexArtifact,
    client: QdrantClient | None,
    collection_name: str | None,
    query_vector: Sequence[float] | None,
    limit: int,
    source_kind: str | None,
    allowed_unit_ids: set[str] | None,
) -> list[QdrantSearchResult]:
    if client is None or not collection_name:
        raise ValueError("Dense code retrieval requires a Qdrant client and collection")
    if not collection_name.startswith("code_custom_"):
        raise ValueError("Code collection names must start with 'code_custom_'")
    if not client.collection_exists(collection_name):
        raise RuntimeError(f"Code collection does not exist: {collection_name}")
    if query_vector is None or not query_vector:
        raise ValueError("Dense code retrieval requires an explicit query vector")
    if len(query_vector) != artifact.vector_dimension:
        raise ValueError(
            "Code query vector dimension does not match the embedded artifact"
        )

    conditions = [
        FieldCondition(key="knowledge_lane", match=MatchValue(value="code")),
        FieldCondition(
            key="snapshot_id", match=MatchValue(value=artifact.snapshot_id)
        ),
    ]
    if source_kind is not None:
        conditions.append(
            FieldCondition(key="source_kind", match=MatchValue(value=source_kind))
        )
    if allowed_unit_ids is not None:
        conditions.append(
            FieldCondition(
                key="unit_id",
                match=MatchAny(any=sorted(allowed_unit_ids)),
            )
        )
    response = client.query_points(
        collection_name=collection_name,
        query=list(query_vector),
        query_filter=Filter(must=conditions),
        limit=limit,
        with_payload=True,
    )
    return [
        QdrantSearchResult(
            point_id=str(point.id), score=float(point.score), payload=point.payload or {}
        )
        for point in response.points
    ]


def _canonicalize_results(
    results: Sequence[Any],
    records: dict[str, CodeIndexRecord],
    artifact: CodeIndexArtifact,
    *,
    retrieval_method: RetrievalMode,
) -> list[QdrantSearchResult]:
    canonical: list[QdrantSearchResult] = []
    for result in results:
        incoming = dict(result.payload)
        unit_id = str(incoming.get("unit_id", ""))
        record = records.get(unit_id)
        if record is None:
            raise RuntimeError(f"Retrieved code unit is absent from artifact: {unit_id}")
        _validate_incoming_identity(incoming, record, artifact)
        if retrieval_method == "dense" and str(result.point_id) != record.point_id:
            raise RuntimeError(
                f"Retrieved code point ID mismatch for {record.unit_id}"
            )
        payload = _record_payload(record)
        payload.update(_retrieval_diagnostics(incoming))
        payload["retrieval_method"] = retrieval_method
        canonical.append(
            QdrantSearchResult(
                point_id=record.point_id,
                score=float(result.score),
                payload=payload,
            )
        )
    return canonical


def _validate_incoming_identity(
    payload: dict[str, Any],
    record: CodeIndexRecord,
    artifact: CodeIndexArtifact,
) -> None:
    checks = {
        "unit_id": record.unit_id,
        "snapshot_id": artifact.snapshot_id,
        "module_id": record.module_id,
        "source_path": record.source_path,
        "source_kind": record.source_kind,
        "display_name": record.display_name,
        "package_name": record.package_name,
        "start_line": record.source_map.start_line,
        "end_line": record.source_map.end_line,
        "content_sha256": record.content_sha256,
        "cache_key": record.cache_key,
    }
    for key, expected in checks.items():
        observed = payload.get(key)
        if observed not in (None, "") and observed != expected:
            raise RuntimeError(
                f"Retrieved code identity mismatch for {record.unit_id}: {key}"
            )
    lane = payload.get("knowledge_lane")
    if lane not in (None, "", "code"):
        raise RuntimeError(f"Retrieved non-code evidence for {record.unit_id}")


def _record_payload(record: CodeIndexRecord) -> dict[str, Any]:
    return {
        "knowledge_lane": "code",
        "unit_id": record.unit_id,
        "unit_index": record.unit_index,
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
        "text": record.citation_text,
    }


def _retrieval_diagnostics(payload: dict[str, Any]) -> dict[str, Any]:
    keys = {
        "matched_query_terms",
        "dense_score",
        "lexical_score",
        "dense_rank",
        "lexical_rank",
        "contributing_retrievers",
        "fusion_method",
        "raw_rrf_score",
    }
    return {key: payload[key] for key in keys if key in payload}


def _to_evidence(result: Any, records: dict[str, CodeIndexRecord]) -> CodeEvidence:
    payload = dict(result.payload)
    record = records[str(payload["unit_id"])]
    return CodeEvidence(
        unit_id=record.unit_id,
        point_id=record.point_id,
        score=float(result.score),
        retrieval_method=payload["retrieval_method"],
        snapshot_id=record.snapshot_id,
        module_id=record.module_id,
        source_path=record.source_path,
        source_kind=record.source_kind,
        display_name=record.display_name,
        parent_unit_id=record.parent_unit_id,
        package_name=record.package_name,
        start_line=record.source_map.start_line,
        end_line=record.source_map.end_line,
        parser_state=record.parser_state,
        conditional_state=record.conditional_state,
        text=record.citation_text,
        retrieval_metadata=_retrieval_diagnostics(payload),
    )


def _to_summary(result: Any, records: dict[str, CodeIndexRecord]) -> CodeCandidateSummary:
    payload = dict(result.payload)
    record = records[str(payload["unit_id"])]
    return CodeCandidateSummary(
        unit_id=record.unit_id,
        point_id=record.point_id,
        score=float(result.score),
        source_path=record.source_path,
        display_name=record.display_name,
        parent_unit_id=record.parent_unit_id,
        start_line=record.source_map.start_line,
        end_line=record.source_map.end_line,
    )


def _select_parent_diverse(
    results: Sequence[QdrantSearchResult],
    records: dict[str, CodeIndexRecord],
    *,
    limit: int,
    max_units_per_parent: int,
) -> list[QdrantSearchResult]:
    """Bound repeated child chunks while preserving ranked parent diversity."""

    grouped: dict[str, list[tuple[int, QdrantSearchResult]]] = {}
    for rank, result in enumerate(results):
        record = records[str(result.payload["unit_id"])]
        parent_key = record.parent_unit_id or record.unit_id
        grouped.setdefault(parent_key, []).append((rank, result))
    selected: list[QdrantSearchResult] = []
    for occurrence in range(max_units_per_parent):
        layer = sorted(
            (items[occurrence] for items in grouped.values() if len(items) > occurrence),
            key=lambda item: item[0],
        )
        for _, result in layer:
            selected.append(result)
            if len(selected) == limit:
                return selected
    return selected
