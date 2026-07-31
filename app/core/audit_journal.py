from __future__ import annotations

import hashlib
import hmac
import json
import os
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from threading import Lock
from typing import Callable


SCHEMA_VERSION = "1.0"
GENESIS_HMAC = "0" * 64


@dataclass(frozen=True)
class ApiAuditEvent:
    event: str
    request_id: str
    method: str
    route: str
    status_code: int
    duration_ms: float


@dataclass(frozen=True)
class AuditVerificationResult:
    valid: bool
    record_count: int
    final_hmac: str
    errors: tuple[str, ...]


class AuditJournal:
    """Append and verify a keyed, hash-chained JSONL audit journal.

    The journal detects modification and reordering when the HMAC key remains
    secret. A trusted external final HMAC/record count is still required to
    detect deletion of a valid suffix.
    """

    def __init__(
        self,
        path: str | Path,
        hmac_key: str,
        *,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        key = hmac_key.encode("utf-8")
        if len(key) < 32:
            raise ValueError("Audit HMAC key must be at least 32 UTF-8 bytes.")
        self.path = Path(path)
        self._key = key
        self._clock = clock or (lambda: datetime.now(UTC))
        self._lock = Lock()
        self.path.parent.mkdir(parents=True, exist_ok=True)

        verification = verify_audit_journal(self.path, hmac_key)
        if not verification.valid:
            raise ValueError(
                "Existing audit journal failed integrity verification: "
                + "; ".join(verification.errors)
            )
        self._sequence = verification.record_count
        self._previous_hmac = verification.final_hmac
        self._expected_size = (
            self.path.stat().st_size if self.path.exists() else 0
        )

    def append(self, event: ApiAuditEvent) -> str:
        """Durably append one event and return its HMAC checkpoint."""

        with self._lock:
            current_size = (
                self.path.stat().st_size if self.path.exists() else 0
            )
            if current_size != self._expected_size:
                raise OSError(
                    "Audit journal changed outside this writer process."
                )
            recorded_at = self._clock().astimezone(UTC).isoformat().replace(
                "+00:00",
                "Z",
            )
            unsigned = {
                "schema_version": SCHEMA_VERSION,
                "sequence": self._sequence + 1,
                "recorded_at_utc": recorded_at,
                "event": asdict(event),
                "previous_hmac": self._previous_hmac,
            }
            record_hmac = _record_hmac(unsigned, self._key)
            record = {**unsigned, "hmac_sha256": record_hmac}
            encoded = (
                json.dumps(
                    record,
                    ensure_ascii=False,
                    separators=(",", ":"),
                    sort_keys=True,
                )
                + "\n"
            ).encode("utf-8")

            with self.path.open("ab") as journal:
                journal.write(encoded)
                journal.flush()
                os.fsync(journal.fileno())

            self._sequence += 1
            self._previous_hmac = record_hmac
            self._expected_size += len(encoded)
            return record_hmac


def verify_audit_journal(
    path: str | Path,
    hmac_key: str,
    *,
    expected_record_count: int | None = None,
    expected_final_hmac: str | None = None,
) -> AuditVerificationResult:
    """Verify journal order, continuity, content HMACs, and an optional checkpoint."""

    journal_path = Path(path)
    key = hmac_key.encode("utf-8")
    if len(key) < 32:
        return AuditVerificationResult(
            valid=False,
            record_count=0,
            final_hmac=GENESIS_HMAC,
            errors=("Audit HMAC key must be at least 32 UTF-8 bytes.",),
        )
    if not journal_path.exists():
        count_error = (
            expected_record_count is not None and expected_record_count != 0
        )
        hmac_error = (
            expected_final_hmac is not None
            and not hmac.compare_digest(expected_final_hmac, GENESIS_HMAC)
        )
        errors = []
        if count_error:
            errors.append("Record count does not match the trusted checkpoint.")
        if hmac_error:
            errors.append("Final HMAC does not match the trusted checkpoint.")
        return AuditVerificationResult(
            valid=not errors,
            record_count=0,
            final_hmac=GENESIS_HMAC,
            errors=tuple(errors),
        )

    previous_hmac = GENESIS_HMAC
    count = 0
    errors: list[str] = []
    with journal_path.open("rb") as journal:
        for line_number, raw_line in enumerate(journal, start=1):
            try:
                line = raw_line.decode("utf-8")
            except UnicodeDecodeError:
                errors.append(f"Line {line_number} is not valid UTF-8.")
                break
            if not line.strip():
                errors.append(f"Line {line_number} is blank.")
                break
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                errors.append(f"Line {line_number} is not valid JSON.")
                break
            error = _validate_record(
                record,
                expected_sequence=line_number,
                expected_previous_hmac=previous_hmac,
                key=key,
            )
            if error:
                errors.append(f"Line {line_number}: {error}")
                break
            count += 1
            previous_hmac = record["hmac_sha256"]

    if expected_record_count is not None and count != expected_record_count:
        errors.append("Record count does not match the trusted checkpoint.")
    if expected_final_hmac is not None and not hmac.compare_digest(
        previous_hmac,
        expected_final_hmac,
    ):
        errors.append("Final HMAC does not match the trusted checkpoint.")
    return AuditVerificationResult(
        valid=not errors,
        record_count=count,
        final_hmac=previous_hmac,
        errors=tuple(errors),
    )


def _validate_record(
    record: object,
    *,
    expected_sequence: int,
    expected_previous_hmac: str,
    key: bytes,
) -> str | None:
    if not isinstance(record, dict):
        return "Record must be a JSON object."
    expected_fields = {
        "schema_version",
        "sequence",
        "recorded_at_utc",
        "event",
        "previous_hmac",
        "hmac_sha256",
    }
    if set(record) != expected_fields:
        return "Record fields do not match the audit schema."
    if record["schema_version"] != SCHEMA_VERSION:
        return "Unsupported schema version."
    if (
        not isinstance(record["sequence"], int)
        or isinstance(record["sequence"], bool)
        or record["sequence"] != expected_sequence
    ):
        return "Sequence is not contiguous."
    if (
        not _is_sha256(record["previous_hmac"])
        or record["previous_hmac"] != expected_previous_hmac
    ):
        return "Previous HMAC does not match the chain."
    if not isinstance(record["recorded_at_utc"], str):
        return "Recorded timestamp must be a string."
    event = record["event"]
    if not isinstance(event, dict) or set(event) != {
        "event",
        "request_id",
        "method",
        "route",
        "status_code",
        "duration_ms",
    }:
        return "Event fields do not match the safe request schema."
    if not all(
        isinstance(event[field], str)
        for field in ("event", "request_id", "method", "route")
    ):
        return "Event text fields must be strings."
    if (
        not isinstance(event["status_code"], int)
        or isinstance(event["status_code"], bool)
    ):
        return "Event status code must be an integer."
    if (
        not isinstance(event["duration_ms"], (int, float))
        or isinstance(event["duration_ms"], bool)
    ):
        return "Event duration must be numeric."
    supplied_hmac = record["hmac_sha256"]
    if not _is_sha256(supplied_hmac):
        return "Record HMAC must be a SHA-256 hexadecimal string."
    unsigned = {
        key_name: record[key_name]
        for key_name in record
        if key_name != "hmac_sha256"
    }
    expected_hmac = _record_hmac(unsigned, key)
    if not hmac.compare_digest(supplied_hmac, expected_hmac):
        return "Record HMAC is invalid."
    return None


def _record_hmac(record: dict[str, object], key: bytes) -> str:
    canonical = json.dumps(
        record,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hmac.new(key, canonical, hashlib.sha256).hexdigest()


def _is_sha256(value: object) -> bool:
    return (
        isinstance(value, str)
        and len(value) == 64
        and all(character in "0123456789abcdef" for character in value)
    )
