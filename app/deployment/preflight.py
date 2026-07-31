from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from app.retrieval.retrieval_config import build_retrieval_runtime_config


@dataclass(frozen=True)
class DeploymentCheck:
    name: str
    passed: bool
    detail: str


@dataclass(frozen=True)
class DeploymentPreflightReport:
    passed: bool
    checks: tuple[DeploymentCheck, ...]


def run_deployment_preflight(
    settings: Any,
    *,
    project_root: str | Path,
    allow_development: bool = False,
    python_version: tuple[int, int] | None = None,
) -> DeploymentPreflightReport:
    """Validate native runtime prerequisites without calling model services."""

    root = Path(project_root)
    version = python_version or (sys.version_info.major, sys.version_info.minor)
    checks = [
        DeploymentCheck(
            name="python_runtime",
            passed=version == (3, 12),
            detail=(
                "Python 3.12 runtime is active."
                if version == (3, 12)
                else "Python 3.12 is required by the locked project."
            ),
        ),
        _required_files_check(root),
        _environment_check(settings, allow_development),
        _model_configuration_check(settings),
        _audit_integrity_check(settings, allow_development),
        _retrieval_state_check(settings),
        _writable_runtime_parent_check(settings),
    ]
    return DeploymentPreflightReport(
        passed=all(check.passed for check in checks),
        checks=tuple(checks),
    )


def _required_files_check(root: Path) -> DeploymentCheck:
    missing = [
        name
        for name in ("pyproject.toml", "uv.lock")
        if not (root / name).is_file()
    ]
    return DeploymentCheck(
        name="locked_project",
        passed=not missing,
        detail=(
            "Locked project files are present."
            if not missing
            else "Missing locked project files: " + ", ".join(missing)
        ),
    )


def _environment_check(
    settings: Any,
    allow_development: bool,
) -> DeploymentCheck:
    environment = str(getattr(settings, "environment", "")).strip().lower()
    is_development = environment in {"dev", "development", "test"}
    passed = bool(environment) and (allow_development or not is_development)
    if passed and is_development:
        detail = (
            "Development environment is allowed only for local package "
            "validation."
        )
    elif passed:
        detail = "Runtime environment label is deployment-safe."
    else:
        detail = "Set a non-development ENVIRONMENT value."
    return DeploymentCheck(
        name="environment",
        passed=passed,
        detail=detail,
    )


def _model_configuration_check(settings: Any) -> DeploymentCheck:
    missing = [
        name
        for name, value in (
            ("OPENAI_API_KEY", getattr(settings, "openai_api_key", "")),
            (
                "OPENAI_EMBEDDING_MODEL",
                getattr(settings, "openai_embedding_model", ""),
            ),
            ("OPENAI_CHAT_MODEL", getattr(settings, "openai_chat_model", "")),
        )
        if not str(value).strip()
    ]
    return DeploymentCheck(
        name="model_configuration",
        passed=not missing,
        detail=(
            "Required model configuration is present."
            if not missing
            else "Missing required model configuration: " + ", ".join(missing)
        ),
    )


def _audit_integrity_check(
    settings: Any,
    allow_development: bool,
) -> DeploymentCheck:
    enabled = bool(getattr(settings, "audit_journal_enabled", False))
    backend = str(
        getattr(settings, "audit_sink_backend", "hmac_jsonl")
    ).strip().lower()
    environment = str(getattr(settings, "environment", "")).strip().lower()
    is_development = environment in {"dev", "development", "test"}
    secret = getattr(settings, "audit_hmac_key", "")
    if hasattr(secret, "get_secret_value"):
        secret = secret.get_secret_value()
    key_is_strong = len(str(secret).encode("utf-8")) >= 32
    path = Path(
        getattr(
            settings,
            "audit_journal_path",
            Path(settings.exports_dir) / "audit" / "api_audit.jsonl",
        )
    )
    parent = path.parent
    parent_is_writable = parent.exists() and os.access(parent, os.W_OK)

    if allow_development and is_development and not enabled:
        return DeploymentCheck(
            name="audit_integrity",
            passed=True,
            detail=(
                "Audit journal is disabled only for local package validation."
            ),
        )
    backend_is_supported = backend == "hmac_jsonl"
    passed = (
        enabled
        and backend_is_supported
        and key_is_strong
        and parent_is_writable
    )
    if not enabled:
        detail = "Enable the audit journal for a deployed environment."
    elif not backend_is_supported:
        detail = "Configure a supported audit sink backend."
    elif not key_is_strong:
        detail = "Configure an audit HMAC key of at least 32 UTF-8 bytes."
    elif not parent_is_writable:
        detail = "Audit journal parent path is missing or not writable."
    else:
        detail = "Audit journal integrity configuration is present."
    return DeploymentCheck(
        name="audit_integrity",
        passed=passed,
        detail=detail,
    )


def _retrieval_state_check(settings: Any) -> DeploymentCheck:
    try:
        config = build_retrieval_runtime_config(settings)
    except ValueError:
        return DeploymentCheck(
            name="retrieval_state",
            passed=False,
            detail="Retrieval runtime configuration is invalid.",
        )

    missing: list[str] = []
    if config.retrieval_mode in {"lexical", "hybrid"}:
        processed = Path(settings.processed_dir)
        if not any(processed.glob("*.retrieval_ready.json")):
            missing.append("retrieval-ready artifacts")
    if config.retrieval_mode in {"dense", "hybrid"}:
        if not Path(settings.qdrant_local_path).exists():
            missing.append("local Qdrant state")
    return DeploymentCheck(
        name="retrieval_state",
        passed=not missing,
        detail=(
            f"Required {config.retrieval_mode} retrieval state is present."
            if not missing
            else "Missing runtime state: " + ", ".join(missing)
        ),
    )


def _writable_runtime_parent_check(settings: Any) -> DeploymentCheck:
    parents = {
        Path(settings.conversation_db_path).parent,
        Path(settings.exports_dir),
    }
    missing_or_read_only = [
        str(path)
        for path in sorted(parents, key=str)
        if not path.exists() or not os.access(path, os.W_OK)
    ]
    return DeploymentCheck(
        name="writable_runtime_paths",
        passed=not missing_or_read_only,
        detail=(
            "Conversation and trace locations are writable."
            if not missing_or_read_only
            else "Missing or non-writable runtime paths: "
            + ", ".join(missing_or_read_only)
        ),
    )
