from __future__ import annotations

from pydantic import BaseModel


class HealthResponse(BaseModel):
    """Response contract for the backend health endpoint."""

    status: str
    app_name: str
    environment: str
    retrieval_mode: str
    hybrid_dense_weight: float
    hybrid_lexical_weight: float
    hybrid_candidate_limit: int
    retrieval_min_top_score: float
    qdrant_collection_name: str
    qdrant_required_for_current_mode: bool