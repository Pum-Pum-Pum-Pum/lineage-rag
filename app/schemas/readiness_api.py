from __future__ import annotations

from pydantic import BaseModel


class ReadinessCheck(BaseModel):
    """One dependency/artifact readiness check result."""

    name: str
    required: bool
    is_ready: bool
    detail: str


class ReadinessResponse(BaseModel):
    """Response contract for the backend readiness endpoint."""

    status: str
    is_ready: bool
    app_name: str
    environment: str
    retrieval_mode: str
    qdrant_required_for_current_mode: bool
    lexical_artifacts_required_for_current_mode: bool
    checks: list[ReadinessCheck]