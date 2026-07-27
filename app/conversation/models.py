from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum


class MessageRole(StrEnum):
    USER = "user"
    ASSISTANT = "assistant"


def _require_non_blank(value: str, field_name: str) -> None:
    if not value.strip():
        raise ValueError(f"{field_name} must not be blank")


def _require_aware_timestamp(value: datetime, field_name: str) -> None:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError(f"{field_name} must be timezone-aware")


@dataclass(frozen=True)
class Conversation:
    conversation_id: str
    title: str
    created_at_utc: datetime
    updated_at_utc: datetime
    is_archived: bool = False

    def __post_init__(self) -> None:
        _require_non_blank(self.conversation_id, "conversation_id")
        _require_non_blank(self.title, "title")
        _require_aware_timestamp(self.created_at_utc, "created_at_utc")
        _require_aware_timestamp(self.updated_at_utc, "updated_at_utc")


@dataclass(frozen=True)
class ConversationMessage:
    message_id: str
    conversation_id: str
    sequence_number: int
    role: MessageRole
    content: str
    created_at_utc: datetime
    trace_id: str | None = None

    def __post_init__(self) -> None:
        _require_non_blank(self.message_id, "message_id")
        _require_non_blank(self.conversation_id, "conversation_id")
        if self.sequence_number <= 0:
            raise ValueError("sequence_number must be greater than 0")
        _require_non_blank(self.content, "content")
        _require_aware_timestamp(self.created_at_utc, "created_at_utc")
        if self.trace_id is not None:
            _require_non_blank(self.trace_id, "trace_id")


@dataclass(frozen=True)
class ConversationSummary:
    conversation_id: str
    summary_text: str
    summarized_through_sequence: int
    version: int
    updated_at_utc: datetime

    def __post_init__(self) -> None:
        _require_non_blank(self.conversation_id, "conversation_id")
        _require_non_blank(self.summary_text, "summary_text")
        if self.summarized_through_sequence <= 0:
            raise ValueError("summarized_through_sequence must be greater than 0")
        if self.version <= 0:
            raise ValueError("version must be greater than 0")
        _require_aware_timestamp(self.updated_at_utc, "updated_at_utc")
