from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Protocol

from app.core.audit_journal import ApiAuditEvent, AuditJournal


AuditDurability = Literal["durable_on_return", "accepted_not_durable"]


@dataclass(frozen=True)
class AuditAppendResult:
    backend: str
    durability: AuditDurability
    checkpoint: str | None


class AuditSink(Protocol):
    """Storage-neutral boundary for privacy-safe API audit events."""

    backend: str
    durability: AuditDurability

    def append(self, event: ApiAuditEvent) -> AuditAppendResult:
        """Accept one event according to the declared durability contract."""


class HmacJsonlAuditSink:
    """Synchronous HMAC JSONL adapter; return means flush/fsync completed."""

    backend = "hmac_jsonl"
    durability: AuditDurability = "durable_on_return"

    def __init__(self, journal: AuditJournal) -> None:
        self._journal = journal

    def append(self, event: ApiAuditEvent) -> AuditAppendResult:
        checkpoint = self._journal.append(event)
        return AuditAppendResult(
            backend=self.backend,
            durability=self.durability,
            checkpoint=checkpoint,
        )


def build_audit_sink(settings: Any) -> AuditSink | None:
    """Build the configured sink without coupling FastAPI to its storage."""

    if not bool(getattr(settings, "audit_journal_enabled", False)):
        return None
    backend = str(
        getattr(settings, "audit_sink_backend", "hmac_jsonl")
    ).strip().lower()
    if backend != "hmac_jsonl":
        raise ValueError("Configured audit sink backend is unsupported.")

    secret = getattr(settings, "audit_hmac_key", "")
    if hasattr(secret, "get_secret_value"):
        secret = secret.get_secret_value()
    journal = AuditJournal(
        getattr(settings, "audit_journal_path"),
        str(secret),
    )
    return HmacJsonlAuditSink(journal)
