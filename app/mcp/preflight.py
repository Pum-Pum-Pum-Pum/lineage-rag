"""Bounded, local-only startup checks for the private MCP child process."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.retrieval.retrieval_config import build_retrieval_runtime_config


@dataclass(frozen=True)
class MCPStartupCheck:
    name: str
    passed: bool
    detail: str


@dataclass(frozen=True)
class MCPStartupPreflightReport:
    passed: bool
    checks: tuple[MCPStartupCheck, ...]


def run_mcp_startup_preflight(settings: Any, *, environment: dict[str, str] | None = None) -> MCPStartupPreflightReport:
    """Validate effective MCP configuration without catalog, Qdrant, or API access.

    This is intentionally narrower than runtime readiness: startup proves that the
    private child has a coherent local configuration, while a tool call remains
    responsible for the disclosure gate and retrieval dependency checks.
    """

    effective_environment = os.environ if environment is None else environment
    checks: list[MCPStartupCheck] = []
    interface_mode = str(getattr(settings, "interface_mode", "")).strip().lower()
    checks.append(
        MCPStartupCheck(
            name="interface_mode",
            passed=interface_mode in {"mcp", "both"},
            detail=(
                "MCP interface mode is enabled."
                if interface_mode in {"mcp", "both"}
                else "Set INTERFACE_MODE to mcp or both before starting the MCP child."
            ),
        )
    )
    # Only presence is inspected: application code never reads, logs, traces, or
    # returns the tunnel control-plane key's value.
    checks.append(
        MCPStartupCheck(
            name="control_plane_key_isolation",
            passed="CONTROL_PLANE_API_KEY" not in effective_environment,
            detail=(
                "Tunnel control-plane key is not inherited by the MCP child."
                if "CONTROL_PLANE_API_KEY" not in effective_environment
                else "Tunnel control-plane key must not be inherited by the MCP child."
            ),
        )
    )
    try:
        build_retrieval_runtime_config(settings)
        checks.append(
            MCPStartupCheck(
                name="retrieval_configuration",
                passed=True,
                detail="Configured retrieval strategy is valid.",
            )
        )
    except (AttributeError, TypeError, ValueError):
        checks.append(
            MCPStartupCheck(
                name="retrieval_configuration",
                passed=False,
                detail="Configure a supported lexical, dense, or hybrid retrieval strategy.",
            )
        )

    fdd_directory = Path(getattr(settings, "fdd_retrieval_artifact_dir", ""))
    checks.append(
        MCPStartupCheck(
            name="fdd_lexical_artifact_directory",
            passed=fdd_directory.is_dir(),
            detail=(
                "FDD lexical artifact directory is available."
                if fdd_directory.is_dir()
                else "Configure an available FDD lexical artifact directory."
            ),
        )
    )
    if bool(getattr(settings, "code_modes_enabled", False)):
        checks.extend(
            (
                _existing_file_check("code_index_artifact", getattr(settings, "code_index_artifact_path", None)),
                _existing_directory_check("code_analysis_directory", getattr(settings, "code_analysis_directory", None)),
                _existing_file_check(
                    "reviewed_lineage_artifact",
                    getattr(settings, "fdd_code_lineage_artifact_path", None),
                ),
            )
        )
    return MCPStartupPreflightReport(
        passed=all(check.passed for check in checks),
        checks=tuple(checks),
    )


def require_mcp_startup_preflight(settings: Any, *, environment: dict[str, str] | None = None) -> MCPStartupPreflightReport:
    """Fail startup without exposing filesystem paths or dependency internals."""

    report = run_mcp_startup_preflight(settings, environment=environment)
    if not report.passed:
        failed_names = ", ".join(check.name for check in report.checks if not check.passed)
        raise RuntimeError(f"MCP startup preflight failed: {failed_names}.")
    return report


def _existing_file_check(name: str, value: object) -> MCPStartupCheck:
    exists = isinstance(value, Path) and value.is_file()
    return MCPStartupCheck(
        name=name,
        passed=exists,
        detail=(f"{name} is available." if exists else f"Configure an available {name}."),
    )


def _existing_directory_check(name: str, value: object) -> MCPStartupCheck:
    exists = isinstance(value, Path) and value.is_dir()
    return MCPStartupCheck(
        name=name,
        passed=exists,
        detail=(f"{name} is available." if exists else f"Configure an available {name}."),
    )
