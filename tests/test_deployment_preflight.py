from pathlib import Path
from types import SimpleNamespace

from app.deployment.preflight import run_deployment_preflight


def _settings(tmp_path: Path, **overrides):
    values = {
        "environment": "prod",
        "openai_api_key": "configured",
        "openai_embedding_model": "embedding-model",
        "openai_chat_model": "chat-model",
        "retrieval_mode": "lexical",
        "hybrid_dense_weight": 0.4,
        "hybrid_lexical_weight": 0.6,
        "hybrid_candidate_limit": 10,
        "processed_dir": tmp_path / "processed",
        "qdrant_local_path": tmp_path / "qdrant",
        "conversation_db_path": tmp_path / "state" / "chat.sqlite3",
        "exports_dir": tmp_path / "exports",
        "audit_journal_enabled": True,
        "audit_sink_backend": "hmac_jsonl",
        "audit_journal_path": tmp_path / "audit" / "api_audit.jsonl",
        "audit_hmac_key": "a" * 32,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _runtime_layout(tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text("[project]", encoding="utf-8")
    (tmp_path / "uv.lock").write_text("version = 1", encoding="utf-8")
    (tmp_path / "processed").mkdir()
    (tmp_path / "processed" / "doc.retrieval_ready.json").write_text(
        "{}",
        encoding="utf-8",
    )
    (tmp_path / "state").mkdir()
    (tmp_path / "exports").mkdir()
    (tmp_path / "audit").mkdir()


def test_deployment_preflight_passes_without_calling_external_services(
    tmp_path: Path,
) -> None:
    _runtime_layout(tmp_path)

    report = run_deployment_preflight(
        _settings(tmp_path),
        project_root=tmp_path,
        python_version=(3, 12),
    )

    assert report.passed is True
    assert all(check.passed for check in report.checks)


def test_deployment_preflight_reports_safe_actionable_failures(
    tmp_path: Path,
) -> None:
    settings = _settings(
        tmp_path,
        environment="dev",
        openai_api_key="",
        retrieval_mode="hybrid",
        audit_journal_enabled=False,
        audit_hmac_key="",
    )

    report = run_deployment_preflight(
        settings,
        project_root=tmp_path,
        python_version=(3, 13),
    )
    failures = {
        check.name: check.detail
        for check in report.checks
        if not check.passed
    }

    assert report.passed is False
    assert set(failures) == {
        "python_runtime",
        "locked_project",
        "environment",
        "model_configuration",
        "audit_integrity",
        "retrieval_state",
        "writable_runtime_paths",
    }
    assert "configured" not in " ".join(failures.values())
    assert "OPENAI_API_KEY" in failures["model_configuration"]
    assert "HMAC" not in failures["audit_integrity"]


def test_allow_development_does_not_describe_dev_as_production_safe(
    tmp_path: Path,
) -> None:
    _runtime_layout(tmp_path)
    report = run_deployment_preflight(
        _settings(tmp_path, environment="dev"),
        project_root=tmp_path,
        allow_development=True,
        python_version=(3, 12),
    )
    environment = next(
        check for check in report.checks if check.name == "environment"
    )

    assert environment.passed is True
    assert "only for local package validation" in environment.detail


def test_production_preflight_rejects_weak_audit_integrity_config(
    tmp_path: Path,
) -> None:
    _runtime_layout(tmp_path)
    report = run_deployment_preflight(
        _settings(tmp_path, audit_hmac_key="short"),
        project_root=tmp_path,
        python_version=(3, 12),
    )
    audit = next(
        check for check in report.checks if check.name == "audit_integrity"
    )

    assert audit.passed is False
    assert "32 UTF-8 bytes" in audit.detail
    assert "short" not in audit.detail


def test_production_preflight_rejects_unknown_audit_backend(
    tmp_path: Path,
) -> None:
    _runtime_layout(tmp_path)
    report = run_deployment_preflight(
        _settings(tmp_path, audit_sink_backend="unknown"),
        project_root=tmp_path,
        python_version=(3, 12),
    )
    audit = next(
        check for check in report.checks if check.name == "audit_integrity"
    )

    assert audit.passed is False
    assert audit.detail == "Configure a supported audit sink backend."
