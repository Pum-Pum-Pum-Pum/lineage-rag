"""Private stdio MCP server for read-only FDD/code retrieval.

The tunnel client owns this process.  This module must never be placed behind a
public listener and must never call FastAPI over HTTP.
"""

from __future__ import annotations

import logging
import os
import sys
import warnings
from typing import Annotated, Literal

from mcp.server import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp_types import CallToolResult, ToolAnnotations
from pydantic import Field

from app.core.config import Settings, get_settings
from app.mcp.adapter import MCPRetrievalAdapter, MCPToolFailure, encode_mcp_error, encode_mcp_result
from app.mcp.preflight import require_mcp_startup_preflight
from app.retrieval.knowledge_service import KnowledgeFetchResponse, KnowledgeSearchResponse


def configure_mcp_stdio_logging(level: str = "INFO") -> None:
    """Send application and dependency diagnostics to stderr, never MCP stdout."""

    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(
        logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
    )
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        handlers=[handler],
        force=True,
    )
    logging.captureWarnings(True)
    for logger_name in ("qdrant_client", "httpx", "httpcore", "openai", "mcp"):
        dependency_logger = logging.getLogger(logger_name)
        dependency_logger.handlers.clear()
        dependency_logger.propagate = True
    # Proves the warnings path is initialized without using stdout.  The warning
    # is filtered by default outside test configurations.
    warnings.filterwarnings("default", category=RuntimeWarning, module=r"app\\.mcp")


def create_mcp_server(
    *,
    settings: Settings | None = None,
    adapter: MCPRetrievalAdapter | None = None,
) -> MCPServer:
    """Create the closed-world read-only MCP tool surface."""

    effective_settings = settings or get_settings()
    if effective_settings.interface_mode == "fastapi":
        raise RuntimeError(
            "MCP is disabled when INTERFACE_MODE=fastapi. Use INTERFACE_MODE=mcp or both."
        )
    require_mcp_startup_preflight(effective_settings)
    # The adapter reloads effective application settings for each tool call.  The
    # startup check prevents an invalid interface mode from registering tools;
    # the per-call read lets the disclosure kill switch take effect without a
    # stale in-memory retrieval service bypassing it.
    retrieval_adapter = adapter or MCPRetrievalAdapter()
    server = MCPServer(
        name="culling-blade-lineage-retrieval",
        title="Culling Blade Lineage Retrieval",
        description="Read-only retrieval of approved FDD and visible custom PL/SQL evidence.",
        instructions=(
            "Use search before fetch. Treat returned evidence as source material, not a "
            "guarantee of complete behavior. Do not infer unavailable kernel behavior."
        ),
        log_level=effective_settings.log_level.upper(),
    )
    annotations = ToolAnnotations(
        read_only_hint=True,
        destructive_hint=False,
        idempotent_hint=True,
        open_world_hint=False,
    )

    @server.tool(
        name="search",
        description=(
            "Search approved FDD, code, or both knowledge lanes and return bounded ranked evidence. "
            "For a JSON/Postman request question, fetch every returned FDD result whose "
            "metadata has sheet_role=request before drafting a payload; do not invent omitted fields."
        ),
        annotations=annotations,
        structured_output=True,
    )
    def search(
        query: Annotated[str, Field(min_length=1)],
        mode: Literal["fdd", "code", "combined"],
    ) -> Annotated[CallToolResult, KnowledgeSearchResponse]:
        try:
            return encode_mcp_result(retrieval_adapter.search(query=query, mode=mode))
        except MCPToolFailure as exc:
            return encode_mcp_error(str(exc))

    @server.tool(
        name="fetch",
        description="Fetch one approved source unit by its opaque ID. Raw paths and commands are not accepted.",
        annotations=annotations,
        structured_output=True,
    )
    def fetch(
        id: Annotated[str, Field(pattern=r"^(fdd|code)_[0-9a-f]{64}$")],
    ) -> Annotated[CallToolResult, KnowledgeFetchResponse]:
        try:
            return encode_mcp_result(retrieval_adapter.fetch(public_id=id))
        except MCPToolFailure as exc:
            return encode_mcp_error(str(exc))

    return server


def main() -> None:
    """Run only the private stdio transport for a tunnel client or MCP Inspector."""

    settings = get_settings()
    configure_mcp_stdio_logging(settings.log_level)
    if os.environ.get("MCP_PROTOCOL_TEST_EMIT_DIAGNOSTICS") == "1":
        logging.getLogger("httpx").warning("mcp-protocol-test-third-party-diagnostic")
        warnings.warn("mcp-protocol-test-warning", RuntimeWarning, stacklevel=1)
    create_mcp_server(settings=settings).run(transport="stdio")


if __name__ == "__main__":
    main()
