from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence


DEFAULT_RRF_RANK_CONSTANT = 1.0


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
    rrf_rank_constant: float = DEFAULT_RRF_RANK_CONSTANT,
) -> list[HybridSearchResult]:
    """Fuse dense and lexical rankings with weighted Reciprocal Rank Fusion.

    Dense similarity and lexical relevance scores are not directly comparable.
    RRF therefore combines their ordinal ranks instead of normalizing and adding
    unrelated score scales. A small rank constant preserves meaningful
    separation within the deliberately short candidate lists used here.
    """

    if limit <= 0:
        raise ValueError("Hybrid search limit must be greater than 0")
    if dense_weight < 0 or lexical_weight < 0:
        raise ValueError("Hybrid search weights must be non-negative")
    if dense_weight == 0 and lexical_weight == 0:
        raise ValueError("At least one hybrid search weight must be greater than 0")
    if rrf_rank_constant <= 0:
        raise ValueError("RRF rank constant must be greater than 0")

    fused_by_key: dict[str, dict[str, Any]] = {}

    _merge_results(
        fused_by_key=fused_by_key,
        results=dense_results,
        retriever_name="dense",
        weight=dense_weight,
        rrf_rank_constant=rrf_rank_constant,
    )
    _merge_results(
        fused_by_key=fused_by_key,
        results=lexical_results,
        retriever_name="lexical",
        weight=lexical_weight,
        rrf_rank_constant=rrf_rank_constant,
    )

    fused_results: list[HybridSearchResult] = []
    maximum_rrf_score = (dense_weight + lexical_weight) / (
        rrf_rank_constant + 1
    )
    for key, state in fused_by_key.items():
        payload = dict(state["payload"])
        dense_score = state.get("dense_score")
        lexical_score = state.get("lexical_score")
        raw_rrf_score = state["raw_rrf_score"]
        hybrid_score = raw_rrf_score / maximum_rrf_score
        contributing_retrievers = sorted(state["contributing_retrievers"])

        payload.update(
            {
                "retrieval_method": "hybrid",
                "hybrid_score": hybrid_score,
                "raw_rrf_score": raw_rrf_score,
                "dense_score": dense_score,
                "lexical_score": lexical_score,
                "dense_rank": state.get("dense_rank"),
                "lexical_rank": state.get("lexical_rank"),
                "dense_rrf_contribution": state.get("dense_rrf_contribution", 0.0),
                "lexical_rrf_contribution": state.get(
                    "lexical_rrf_contribution",
                    0.0,
                ),
                "fusion_method": "weighted_rrf",
                "rrf_rank_constant": rrf_rank_constant,
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
            -len(result.payload.get("contributing_retrievers", [])),
            min(
                result.payload.get("dense_rank") or 999999,
                result.payload.get("lexical_rank") or 999999,
            ),
            (result.payload.get("dense_rank") or 999999)
            + (result.payload.get("lexical_rank") or 999999),
            str(result.payload.get("unit_id", result.point_id)),
        ),
    )[:limit]


def _merge_results(
    fused_by_key: dict[str, dict[str, Any]],
    results: Sequence[Any],
    retriever_name: str,
    weight: float,
    rrf_rank_constant: float,
) -> None:
    for rank, result in enumerate(results, start=1):
        payload = dict(result.payload)
        key = str(payload.get("unit_id", result.point_id))
        raw_score = float(result.score)
        rrf_contribution = weight / (rrf_rank_constant + rank)

        state = fused_by_key.setdefault(
            key,
            {
                "payload": payload,
                "raw_rrf_score": 0.0,
                "contributing_retrievers": set(),
            },
        )

        # Prefer the richer payload by retaining text/metadata already present,
        # but fill in missing keys from later retrievers.
        for payload_key, payload_value in payload.items():
            state["payload"].setdefault(payload_key, payload_value)

        state["raw_rrf_score"] += rrf_contribution
        state["contributing_retrievers"].add(retriever_name)
        state[f"{retriever_name}_score"] = raw_score
        state[f"{retriever_name}_rrf_contribution"] = rrf_contribution
        state[f"{retriever_name}_rank"] = rank
