import json
import logging
from datetime import UTC, datetime
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from app.core.audit_journal import (
    ApiAuditEvent,
    AuditJournal,
    verify_audit_journal,
)
from app.core.request_observability import install_request_observability


KEY = "production-test-key-material-32-bytes"


def _event(request_id: str = "request-1") -> ApiAuditEvent:
    return ApiAuditEvent(
        event="api_request_completed",
        request_id=request_id,
        method="GET",
        route="/health",
        status_code=200,
        duration_ms=1.25,
    )


def test_journal_appends_and_verifies_a_durable_chain(
    tmp_path: Path,
) -> None:
    path = tmp_path / "audit" / "requests.jsonl"
    fixed_time = datetime(2026, 7, 30, 12, 0, tzinfo=UTC)
    journal = AuditJournal(path, KEY, clock=lambda: fixed_time)

    first_hmac = journal.append(_event("request-1"))
    final_hmac = journal.append(_event("request-2"))
    result = verify_audit_journal(
        path,
        KEY,
        expected_record_count=2,
        expected_final_hmac=final_hmac,
    )

    assert result.valid is True
    assert result.record_count == 2
    assert result.final_hmac == final_hmac
    records = [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
    ]
    assert records[0]["previous_hmac"] == "0" * 64
    assert records[1]["previous_hmac"] == first_hmac
    assert records[0]["recorded_at_utc"] == "2026-07-30T12:00:00Z"


def test_content_tampering_is_detected_without_exposing_event_data(
    tmp_path: Path,
) -> None:
    path = tmp_path / "audit.jsonl"
    journal = AuditJournal(path, KEY)
    journal.append(_event())
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["event"]["status_code"] = 500
    path.write_text(json.dumps(payload) + "\n", encoding="utf-8")

    result = verify_audit_journal(path, KEY)

    assert result.valid is False
    assert result.errors == ("Line 1: Record HMAC is invalid.",)
    assert "request-1" not in str(result)
    with pytest.raises(ValueError, match="integrity verification"):
        AuditJournal(path, KEY)


def test_malformed_utf8_returns_safe_verification_failure(
    tmp_path: Path,
) -> None:
    path = tmp_path / "audit.jsonl"
    path.write_bytes(b"\xff\xfeprivate-bytes\n")

    result = verify_audit_journal(path, KEY)

    assert result.valid is False
    assert result.errors == ("Line 1 is not valid UTF-8.",)
    assert "private-bytes" not in str(result)


def test_trusted_checkpoint_detects_valid_suffix_deletion(
    tmp_path: Path,
) -> None:
    path = tmp_path / "audit.jsonl"
    journal = AuditJournal(path, KEY)
    journal.append(_event("request-1"))
    final_hmac = journal.append(_event("request-2"))
    first_line = path.read_text(encoding="utf-8").splitlines()[0]
    path.write_text(first_line + "\n", encoding="utf-8")

    without_checkpoint = verify_audit_journal(path, KEY)
    with_checkpoint = verify_audit_journal(
        path,
        KEY,
        expected_record_count=2,
        expected_final_hmac=final_hmac,
    )

    assert without_checkpoint.valid is True
    assert with_checkpoint.valid is False
    assert len(with_checkpoint.errors) == 2


def test_stale_second_writer_fails_instead_of_silently_forking_chain(
    tmp_path: Path,
) -> None:
    path = tmp_path / "audit.jsonl"
    first_writer = AuditJournal(path, KEY)
    second_writer = AuditJournal(path, KEY)
    first_writer.append(_event("request-1"))

    with pytest.raises(OSError, match="outside this writer"):
        second_writer.append(_event("request-2"))

    assert verify_audit_journal(path, KEY).valid is True


def test_request_succeeds_and_emits_critical_event_when_journal_fails(
    caplog,
) -> None:
    class FailingJournal:
        def append(self, event: ApiAuditEvent) -> str:
            raise OSError("private filesystem detail")

    app = FastAPI()
    install_request_observability(app, FailingJournal())  # type: ignore[arg-type]

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    caplog.set_level(logging.CRITICAL, logger="api_audit")
    response = TestClient(app).get(
        "/health",
        headers={"X-Request-ID": "failure-test"},
    )

    assert response.status_code == 200
    failure = json.loads(caplog.records[-1].message)
    assert failure == {
        "event": "audit_journal_write_failed",
        "request_id": "failure-test",
    }
    assert "private filesystem detail" not in caplog.text
