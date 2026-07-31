from __future__ import annotations

import json
import re
from contextvars import ContextVar
from time import perf_counter
from typing import Callable
from uuid import uuid4

from fastapi import FastAPI, Request, Response

from app.core.audit_journal import ApiAuditEvent
from app.core.audit_sink import AuditSink
from app.core.logging import get_logger


REQUEST_ID_HEADER = "X-Request-ID"
REQUEST_ID_PATTERN = re.compile(r"^[A-Za-z0-9._:-]{1,128}$")
_request_id: ContextVar[str | None] = ContextVar(
    "request_id",
    default=None,
)
logger = get_logger("api_audit")


def install_request_observability(
    app: FastAPI,
    audit_sink: AuditSink | None = None,
) -> None:
    """Install request correlation, safe audit logging, and API headers."""

    @app.middleware("http")
    async def request_observability(
        request: Request,
        call_next: Callable,
    ) -> Response:
        request_id = _resolve_request_id(
            request.headers.get(REQUEST_ID_HEADER)
        )
        token = _request_id.set(request_id)
        started = perf_counter()
        status_code = 500
        try:
            response = await call_next(request)
            status_code = response.status_code
            _add_security_headers(response, request_id)
            return response
        finally:
            duration_ms = round((perf_counter() - started) * 1000, 3)
            route = request.scope.get("route")
            route_template = getattr(route, "path", "<unmatched>")
            event = ApiAuditEvent(
                event="api_request_completed",
                request_id=request_id,
                method=request.method,
                route=route_template,
                status_code=status_code,
                duration_ms=duration_ms,
            )
            logger.info(json.dumps(event.__dict__, separators=(",", ":")))
            if audit_sink is not None:
                try:
                    audit_sink.append(event)
                except Exception:
                    logger.critical(
                        json.dumps(
                            {
                                "event": "audit_journal_write_failed",
                                "request_id": request_id,
                            },
                            separators=(",", ":"),
                        )
                    )
            _request_id.reset(token)


def get_request_id() -> str | None:
    """Return the current request correlation ID, if called in a request."""

    return _request_id.get()


def _resolve_request_id(candidate: str | None) -> str:
    if candidate and REQUEST_ID_PATTERN.fullmatch(candidate):
        return candidate
    return str(uuid4())


def _add_security_headers(response: Response, request_id: str) -> None:
    response.headers[REQUEST_ID_HEADER] = request_id
    response.headers["Cache-Control"] = "no-store"
    response.headers["X-Content-Type-Options"] = "nosniff"
    response.headers["X-Frame-Options"] = "DENY"
    response.headers["Referrer-Policy"] = "no-referrer"
    response.headers["Permissions-Policy"] = (
        "camera=(), microphone=(), geolocation=()"
    )
