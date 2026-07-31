from __future__ import annotations

from fastapi import FastAPI

from app.api.routes.conversations import router as conversations_router
from app.api.routes.health import router as health_router
from app.api.routes.query import router as query_router
from app.api.routes.readiness import router as readiness_router
from app.core.audit_sink import build_audit_sink
from app.core.config import get_settings
from app.core.request_observability import install_request_observability


def create_app() -> FastAPI:
    """Create the FastAPI application for the local RAG backend."""

    settings = get_settings()
    app = FastAPI(title=settings.app_name)
    audit_sink = build_audit_sink(settings)
    install_request_observability(app, audit_sink)
    app.include_router(health_router)
    app.include_router(readiness_router)
    app.include_router(query_router)
    app.include_router(conversations_router)
    return app


app = create_app()
