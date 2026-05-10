from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence


@dataclass(frozen=True)
class HybridSearchResult:
    point_id: str
    score: float
    payload: dict[str, Any]


def fuse_dense_and_lexical_results(
    dense_results: Sequence[Any],
    lexical_results: Sequence[Any],
    limit: int = 5,
    dense_weight: float = 0.5,
    lexical_weight: float = 0.5,
) -> list[HybridSearchResult]:
    """Fuse dense and lexical retrieval results with simple normalized score fusion.

    This is intentionally a baseline. It normalizes each retriever's scores by
    that retriever's maximum score for the query, then combines them with fixed
    weights. It does not rerank with a model and does not claim unsupported
    attachment evidence is valid.
    """

    if limit <= 0:
        raise ValueError("Hybrid search limit must be greater than 0")
    if dense_weight < 0 or lexical_weight < 0:
        raise ValueError("Hybrid search weights must be non-negative")
    if dense_weight == 0 and lexical_weight == 0:
        raise ValueError("At least one hybrid search weight must be greater than 0")

    dense_max_score = _max_score(dense_results)
    lexical_max_score = _max_score(lexical_results)
    fused_by_key: dict[str, dict[str, Any]] = {}

    _merge_results(
        fused_by_key=fused_by_key,
        results=dense_results,
        retriever_name="dense",
        max_score=dense_max_score,
        weight=dense_weight,
    )
    _merge_results(
        fused_by_key=fused_by_key,
        results=lexical_results,
        retriever_name="lexical",
        max_score=lexical_max_score,
        weight=lexical_weight,
    )

    fused_results: list[HybridSearchResult] = []
    for key, state in fused_by_key.items():
        payload = dict(state["payload"])
        dense_score = state.get("dense_score")
        lexical_score = state.get("lexical_score")
        normalized_dense_score = state.get("normalized_dense_score", 0.0)
        normalized_lexical_score = state.get("normalized_lexical_score", 0.0)
        hybrid_score = state["hybrid_score"]
        contributing_retrievers = sorted(state["contributing_retrievers"])

        payload.update(
            {
                "retrieval_method": "hybrid",
                "hybrid_score": hybrid_score,
                "dense_score": dense_score,
                "lexical_score": lexical_score,
                "normalized_dense_score": normalized_dense_score,
                "normalized_lexical_score": normalized_lexical_score,
                "dense_rank": state.get("dense_rank"),
                "lexical_rank": state.get("lexical_rank"),
                "contributing_retrievers": contributing_retrievers,
            }
        )
        fused_results.append(
            HybridSearchResult(
                point_id=str(payload.get("unit_id", key)),
                score=hybrid_score,
                payload=payload,
            )
        )

    return sorted(
        fused_results,
        key=lambda result: (
            -result.score,
            result.payload.get("dense_rank") is None,
            result.payload.get("dense_rank") or 999999,
            result.payload.get("lexical_rank") or 999999,
            str(result.payload.get("unit_id", result.point_id)),
        ),
    )[:limit]


def _merge_results(
    fused_by_key: dict[str, dict[str, Any]],
    results: Sequence[Any],
    retriever_name: str,
    max_score: float,
    weight: float,
) -> None:
    for rank, result in enumerate(results, start=1):
        payload = dict(result.payload)
        key = str(payload.get("unit_id", result.point_id))
        raw_score = float(result.score)
        normalized_score = _normalize_score(raw_score, max_score)
        weighted_score = normalized_score * weight

        state = fused_by_key.setdefault(
            key,
            {
                "payload": payload,
                "hybrid_score": 0.0,
                "contributing_retrievers": set(),
            },
        )

        # Prefer the richer payload by retaining text/metadata already present,
        # but fill in missing keys from later retrievers.
        for payload_key, payload_value in payload.items():
            state["payload"].setdefault(payload_key, payload_value)

        state["hybrid_score"] += weighted_score
        state["contributing_retrievers"].add(retriever_name)
        state[f"{retriever_name}_score"] = raw_score
        state[f"normalized_{retriever_name}_score"] = normalized_score
        state[f"{retriever_name}_rank"] = rank


def _max_score(results: Sequence[Any]) -> float:
    if not results:
        return 0.0
    return max(float(result.score) for result in results)


def _normalize_score(score: float, max_score: float) -> float:
    if max_score <= 0:
        return 0.0
    return score / max_score