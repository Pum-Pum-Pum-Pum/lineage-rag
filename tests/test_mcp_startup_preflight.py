from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from app.mcp.preflight import require_mcp_startup_preflight, run_mcp_startup_preflight


def _settings(tmp_path: Path, **overrides):
    fdd_directory = tmp_path / "processed"
    fdd_directory.mkdir(exist_ok=True)
    values = {
        "interface_mode": "mcp",
        "retrieval_mode": "lexical",
        "hybrid_dense_weight": 0.4,
        "hybrid_lexical_weight": 0.6,
        "hybrid_candidate_limit": 10,
        "fdd_retrieval_artifact_dir": fdd_directory,
        "code_modes_enabled": False,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_mcp_startup_preflight_is_local_and_passes_without_retrieval(tmp_path: Path) -> None:
    report = run_mcp_startup_preflight(
        _settings(tmp_path),
        environment={},
    )

    assert report.passed is True
    assert all(check.passed for check in report.checks)


def test_mcp_startup_preflight_rejects_inherited_control_plane_key_without_leaking_value(tmp_path: Path) -> None:
    report = run_mcp_startup_preflight(
        _settings(tmp_path),
        environment={"CONTROL_PLANE_API_KEY": "private-value"},
    )

    assert report.passed is False
    failed = next(check for check in report.checks if check.name == "control_plane_key_isolation")
    assert failed.detail == "Tunnel control-plane key must not be inherited by the MCP child."
    with pytest.raises(RuntimeError) as exc_info:
        require_mcp_startup_preflight(
            _settings(tmp_path),
            environment={"CONTROL_PLANE_API_KEY": "private-value"},
        )
    assert "private-value" not in str(exc_info.value)


def test_mcp_startup_preflight_rejects_missing_or_invalid_effective_configuration(tmp_path: Path) -> None:
    report = run_mcp_startup_preflight(
        _settings(
            tmp_path,
            interface_mode="fastapi",
            retrieval_mode="unsupported",
            fdd_retrieval_artifact_dir=tmp_path / "missing",
        ),
        environment={},
    )

    assert report.passed is False
    assert {check.name for check in report.checks if not check.passed} == {
        "interface_mode",
        "retrieval_configuration",
        "fdd_lexical_artifact_directory",
    }


def test_mcp_startup_preflight_checks_code_artifacts_when_code_modes_are_enabled(tmp_path: Path) -> None:
    code_index_artifact = tmp_path / "code-index.json"
    code_index_artifact.write_text("{}", encoding="utf-8")
    code_analysis_directory = tmp_path / "code-analysis"
    code_analysis_directory.mkdir()
    reviewed_lineage_artifact = tmp_path / "reviewed-lineage.json"
    reviewed_lineage_artifact.write_text("{}", encoding="utf-8")

    report = run_mcp_startup_preflight(
        _settings(
            tmp_path,
            code_modes_enabled=True,
            code_index_artifact_path=code_index_artifact,
            code_analysis_directory=code_analysis_directory,
            fdd_code_lineage_artifact_path=reviewed_lineage_artifact,
        ),
        environment={},
    )

    assert report.passed is True
    assert {check.name for check in report.checks} >= {
        "code_index_artifact",
        "code_analysis_directory",
        "reviewed_lineage_artifact",
    }
