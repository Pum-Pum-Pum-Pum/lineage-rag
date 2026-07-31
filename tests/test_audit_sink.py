from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import SecretStr

from app.core.audit_journal import ApiAuditEvent, verify_audit_journal
from app.core.audit_sink import HmacJsonlAuditSink, build_audit_sink


KEY = "production-test-key-material-32-bytes"


def _event() -> ApiAuditEvent:
    return ApiAuditEvent(
        event="api_request_completed",
        request_id="sink-test",
        method="POST",
        route="/query",
        status_code=200,
        duration_ms=2.5,
    )


def test_disabled_audit_builds_no_sink() -> None:
    settings = SimpleNamespace(audit_journal_enabled=False)

    assert build_audit_sink(settings) is None


def test_factory_builds_durable_hmac_jsonl_adapter(
    tmp_path: Path,
) -> None:
    path = tmp_path / "network-mount" / "audit.jsonl"
    settings = SimpleNamespace(
        audit_journal_enabled=True,
        audit_sink_backend="hmac_jsonl",
        audit_journal_path=path,
        audit_hmac_key=SecretStr(KEY),
    )

    sink = build_audit_sink(settings)
    assert isinstance(sink, HmacJsonlAuditSink)
    result = sink.append(_event())

    assert result.backend == "hmac_jsonl"
    assert result.durability == "durable_on_return"
    assert result.checkpoint is not None
    assert verify_audit_journal(path, KEY).valid is True


def test_factory_rejects_unknown_backend_without_echoing_value(
    tmp_path: Path,
) -> None:
    settings = SimpleNamespace(
        audit_journal_enabled=True,
        audit_sink_backend="secret-database-url",
        audit_journal_path=tmp_path / "audit.jsonl",
        audit_hmac_key=KEY,
    )

    with pytest.raises(ValueError, match="unsupported") as error:
        build_audit_sink(settings)

    assert "secret-database-url" not in str(error.value)
