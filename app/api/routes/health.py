from __future__ import annotations

from fastapi import APIRouter, HTTPException, status

from app.core.config import get_settings
from app.core.logging import get_logger
from app.retrieval.retrieval_config import build_retrieval_runtime_config
from app.schemas.health_api import HealthResponse


router = APIRouter(tags=["health"])
logger = get_logger("health_api")


@router.get("/health", response_model=HealthResponse)
def health_check() -> HealthResponse:
    """Return backend liveness and active retrieval configuration.

    This endpoint intentionally avoids running retrieval, embedding, LLM calls,
    or vector-store checks. It is a lightweight contract for API clients and
    future UI layers to verify that the backend process is alive and which
    retrieval configuration is active.
    """

    settings = get_settings()
    try:
        retrieval_config = build_retrieval_runtime_config(settings)
    except ValueError as exc:
        logger.exception("Invalid retrieval runtime configuration during health check")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Invalid retrieval runtime configuration.",
        ) from exc

    return HealthResponse(
        status="ok",
        app_name=settings.app_name,
        environment=settings.environment,
        retrieval_mode=retrieval_config.retrieval_mode,
        hybrid_dense_weight=retrieval_config.hybrid_dense_weight,
        hybrid_lexical_weight=retrieval_config.hybrid_lexical_weight,
        hybrid_candidate_limit=retrieval_config.hybrid_candidate_limit,
        retrieval_min_top_score=settings.retrieval_min_top_score,
        qdrant_collection_name=settings.qdrant_collection_name,
        qdrant_required_for_current_mode=_requires_qdrant_collection(
            retrieval_config.retrieval_mode
        ),
    )


def _requires_qdrant_collection(retrieval_mode: str) -> bool:
    """Return whether the selected retrieval mode requires a Qdrant collection."""

    return retrieval_mode in {"dense", "hybrid"}