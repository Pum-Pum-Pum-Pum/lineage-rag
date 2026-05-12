from __future__ import annotations

from fastapi import FastAPI

from app.api.routes.health import router as health_router
from app.api.routes.query import router as query_router
from app.core.config import get_settings


def create_app() -> FastAPI:
    """Create the FastAPI application for the local RAG backend."""

    settings = get_settings()
    app = FastAPI(title=settings.app_name)
    app.include_router(health_router)
    app.include_router(query_router)
    return app


app = create_app()