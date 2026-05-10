from __future__ import annotations

from dataclasses import dataclass
from typing import Any


SUPPORTED_RETRIEVAL_MODES = {"dense", "lexical", "hybrid"}


@dataclass(frozen=True)
class RetrievalRuntimeConfig:
    retrieval_mode: str
    hybrid_dense_weight: float
    hybrid_lexical_weight: float
    hybrid_candidate_limit: int


def build_retrieval_runtime_config(settings: Any) -> RetrievalRuntimeConfig:
    """Build and validate retrieval runtime configuration from settings.

    This keeps the provisional hybrid default explicit and reversible. It does
    not wire hybrid into answer generation by itself.
    """

    retrieval_mode = str(settings.retrieval_mode).strip().lower()
    hybrid_dense_weight = float(settings.hybrid_dense_weight)
    hybrid_lexical_weight = float(settings.hybrid_lexical_weight)
    hybrid_candidate_limit = int(settings.hybrid_candidate_limit)

    validate_retrieval_runtime_config(
        retrieval_mode=retrieval_mode,
        hybrid_dense_weight=hybrid_dense_weight,
        hybrid_lexical_weight=hybrid_lexical_weight,
        hybrid_candidate_limit=hybrid_candidate_limit,
    )

    return RetrievalRuntimeConfig(
        retrieval_mode=retrieval_mode,
        hybrid_dense_weight=hybrid_dense_weight,
        hybrid_lexical_weight=hybrid_lexical_weight,
        hybrid_candidate_limit=hybrid_candidate_limit,
    )


def validate_retrieval_runtime_config(
    retrieval_mode: str,
    hybrid_dense_weight: float,
    hybrid_lexical_weight: float,
    hybrid_candidate_limit: int,
) -> None:
    """Validate retrieval mode and hybrid tuning parameters."""

    if retrieval_mode not in SUPPORTED_RETRIEVAL_MODES:
        raise ValueError(
            "Unsupported retrieval mode: "
            f"{retrieval_mode}. Supported modes: {sorted(SUPPORTED_RETRIEVAL_MODES)}"
        )
    if hybrid_dense_weight < 0 or hybrid_lexical_weight < 0:
        raise ValueError("Hybrid retrieval weights must be non-negative")
    if hybrid_dense_weight == 0 and hybrid_lexical_weight == 0:
        raise ValueError("At least one hybrid retrieval weight must be greater than 0")
    if hybrid_candidate_limit <= 0:
        raise ValueError("Hybrid candidate limit must be greater than 0")


def is_hybrid_enabled(config: RetrievalRuntimeConfig) -> bool:
    """Return whether the configured retrieval mode is hybrid."""

    return config.retrieval_mode == "hybrid"