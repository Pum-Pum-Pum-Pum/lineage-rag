from __future__ import annotations

from typing import TypeVar

import httpx
from pydantic import BaseModel, ValidationError

from app.schemas.health_api import HealthResponse
from app.schemas.query_api import QueryRequest, QueryResponse
from app.schemas.readiness_api import ReadinessResponse


ResponseModel = TypeVar("ResponseModel", bound=BaseModel)


class UiApiError(RuntimeError):
    """Safe error contract for presentation layers calling the RAG API."""

    def __init__(
        self,
        *,
        code: str,
        message: str,
        status_code: int | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.status_code = status_code


class RagApiClient:
    """Typed HTTP boundary between a UI and the FastAPI RAG backend."""

    def __init__(
        self,
        base_url: str,
        *,
        timeout: float = 10.0,
        client: httpx.Client | None = None,
    ) -> None:
        self.base_url = _normalize_base_url(base_url)
        if timeout <= 0:
            raise ValueError("timeout must be greater than 0")
        self.timeout = timeout
        self._client = client or httpx.Client()
        self._owns_client = client is None

    def get_health(self) -> HealthResponse:
        return self._request("GET", "/health", response_model=HealthResponse)

    def get_readiness(self) -> ReadinessResponse:
        return self._request("GET", "/ready", response_model=ReadinessResponse)

    def query(self, request: QueryRequest) -> QueryResponse:
        return self._request(
            "POST",
            "/query",
            response_model=QueryResponse,
            json=request.model_dump(exclude_none=True),
        )

    def close(self) -> None:
        if self._owns_client:
            self._client.close()

    def __enter__(self) -> RagApiClient:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    def _request(
        self,
        method: str,
        path: str,
        *,
        response_model: type[ResponseModel],
        json: dict[str, object] | None = None,
    ) -> ResponseModel:
        try:
            response = self._client.request(
                method,
                f"{self.base_url}{path}",
                json=json,
                timeout=self.timeout,
            )
        except httpx.TimeoutException as exc:
            raise UiApiError(
                code="timeout",
                message="The RAG API timed out. Try again.",
            ) from exc
        except httpx.RequestError as exc:
            raise UiApiError(
                code="unavailable",
                message="The RAG API is unavailable. Verify that the backend is running.",
            ) from exc

        if response.status_code >= 400:
            if response.status_code == 503:
                raise UiApiError(
                    code="not_ready",
                    message="The RAG API is not ready. Check backend readiness and dependencies.",
                    status_code=response.status_code,
                )
            raise UiApiError(
                code="http_error",
                message=f"The RAG API request failed with HTTP {response.status_code}.",
                status_code=response.status_code,
            )

        try:
            payload = response.json()
            return response_model.model_validate(payload)
        except (ValueError, ValidationError) as exc:
            raise UiApiError(
                code="invalid_response",
                message="The RAG API returned an invalid response.",
                status_code=response.status_code,
            ) from exc


def _normalize_base_url(base_url: str) -> str:
    cleaned = base_url.strip().rstrip("/")
    if not cleaned:
        raise ValueError("base_url must not be blank")
    return cleaned
