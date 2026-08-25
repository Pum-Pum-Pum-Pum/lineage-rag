"""MCP-only safety adapter around :mod:`app.retrieval.knowledge_service`.

This module contains no HTTP client and never calls FastAPI.  It is deliberately
small: the shared retrieval service remains the authority for retrieval, ranking,
lineage, opaque IDs, limits, and source-catalog resolution.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any, Literal, TypeVar

from mcp_types import CallToolResult, TextContent
from pydantic import BaseModel

from app.core.config import Settings, get_settings
from app.retrieval.knowledge_service import (
    KnowledgeFetchResponse,
    KnowledgeRetrievalService,
    KnowledgeSearchResponse,
)
from app.retrieval.retrieval_config import build_retrieval_runtime_config
from app.vectorstore.qdrant_schema import create_persistent_qdrant_client


DISCLOSURE_DISABLED_MESSAGE = "Evidence disclosure is disabled."
RETRIEVAL_UNAVAILABLE_MESSAGE = "Retrieval is currently unavailable."
SOURCE_UNAVAILABLE_MESSAGE = "Requested source is unavailable."
KnowledgeMode = Literal["fdd", "code", "combined"]
McpResultModel = TypeVar("McpResultModel", bound=BaseModel)


class MCPRetrievalAdapter:
    """Enforce MCP disclosure policy before constructing any retrieval dependency."""

    def __init__(
        self,
        *,
        settings_provider: Callable[[], Settings] = get_settings,
        service_factory: Callable[[Settings], KnowledgeRetrievalService] | None = None,
    ) -> None:
        self._settings_provider = settings_provider
        self._service_factory = service_factory or _build_service

    def search(self, *, query: str, mode: KnowledgeMode) -> KnowledgeSearchResponse:
        settings = self._disclosure_enabled_settings()
        try:
            return self._service_factory(settings).search(query=query, mode=mode, limit=5)
        except PermissionError:
            # This feature state is intentionally safe for the MCP caller to know.
            raise MCPToolFailure("Code and combined knowledge modes are not activated.") from None
        except ValueError as exc:
            # Query validation errors are caller-correctable, but never include source details.
            raise MCPToolFailure(str(exc)) from None
        except Exception:
            raise MCPToolFailure(RETRIEVAL_UNAVAILABLE_MESSAGE) from None

    def fetch(self, *, public_id: str) -> KnowledgeFetchResponse:
        settings = self._disclosure_enabled_settings()
        try:
            return self._service_factory(settings).fetch(public_id)
        except LookupError:
            raise MCPToolFailure(SOURCE_UNAVAILABLE_MESSAGE) from None
        except Exception:
            raise MCPToolFailure(RETRIEVAL_UNAVAILABLE_MESSAGE) from None

    def _disclosure_enabled_settings(self) -> Settings:
        settings = self._settings_provider()
        if not settings.mcp_evidence_disclosure_enabled:
            raise MCPToolFailure(DISCLOSURE_DISABLED_MESSAGE)
        return settings


class MCPToolFailure(Exception):
    """An anticipated tool failure whose message is safe to expose."""


def encode_mcp_result(model: McpResultModel) -> CallToolResult:
    """Encode one validated model into coherent MCP structured and text content.

    A single canonical dictionary is created first.  The fallback text is a JSON
    rendering of exactly that dictionary; it is never independently rebuilt.
    """

    validated = type(model).model_validate(model)
    canonical = validated.model_dump(mode="json", by_alias=True)
    return CallToolResult(
        structured_content=canonical,
        content=[
            TextContent(
                type="text",
                text=json.dumps(canonical, ensure_ascii=False, separators=(",", ":")),
            )
        ],
    )


def encode_mcp_error(message: str) -> CallToolResult:
    """Return a tool error with no structured evidence payload."""

    return CallToolResult(
        content=[TextContent(type="text", text=message)],
        is_error=True,
    )


def _build_service(settings: Settings) -> KnowledgeRetrievalService:
    return KnowledgeRetrievalService(
        settings=settings,
        retrieval_config=build_retrieval_runtime_config(settings),
        qdrant_client_factory=create_persistent_qdrant_client,
    )
