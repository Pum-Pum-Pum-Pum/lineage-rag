from __future__ import annotations

import sqlite3
from contextlib import AbstractContextManager
from datetime import UTC, datetime
from pathlib import Path
from threading import RLock
from types import TracebackType
from typing import Protocol, runtime_checkable
from uuid import uuid4

from app.conversation.models import (
    Conversation,
    ConversationMessage,
    ConversationSummary,
    MessageRole,
)


class ConversationNotFoundError(LookupError):
    """Raised when a requested conversation does not exist."""


class ConversationArchivedError(RuntimeError):
    """Raised when attempting to mutate an archived conversation."""


class StoreClosedError(RuntimeError):
    """Raised when a closed conversation store is used."""


@runtime_checkable
class ConversationStore(Protocol):
    def create_conversation(self, title: str = "New conversation") -> Conversation: ...

    def get_conversation(self, conversation_id: str) -> Conversation | None: ...

    def list_conversations(
        self,
        *,
        include_archived: bool = False,
    ) -> list[Conversation]: ...

    def add_message(
        self,
        conversation_id: str,
        role: MessageRole,
        content: str,
        *,
        trace_id: str | None = None,
    ) -> ConversationMessage: ...

    def list_messages(self, conversation_id: str) -> list[ConversationMessage]: ...

    def save_summary(
        self,
        conversation_id: str,
        summary_text: str,
        *,
        summarized_through_sequence: int,
    ) -> ConversationSummary: ...

    def get_summary(self, conversation_id: str) -> ConversationSummary | None: ...

    def archive_conversation(self, conversation_id: str) -> Conversation: ...

    def close(self) -> None: ...


class SqliteConversationStore(
    AbstractContextManager["SqliteConversationStore"],
):
    """Local durable conversation store behind the replaceable store contract."""

    def __init__(self, database_path: str | Path) -> None:
        self._database_path = Path(database_path)
        if str(self._database_path) != ":memory:":
            self._database_path.parent.mkdir(parents=True, exist_ok=True)

        self._connection = sqlite3.connect(
            str(self._database_path),
            check_same_thread=False,
        )
        self._connection.row_factory = sqlite3.Row
        self._connection.execute("PRAGMA foreign_keys = ON")
        self._lock = RLock()
        self._is_closed = False
        self._create_schema()

    def _create_schema(self) -> None:
        with self._connection:
            self._connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS conversations (
                    conversation_id TEXT PRIMARY KEY,
                    title TEXT NOT NULL,
                    created_at_utc TEXT NOT NULL,
                    updated_at_utc TEXT NOT NULL,
                    is_archived INTEGER NOT NULL DEFAULT 0
                        CHECK (is_archived IN (0, 1))
                );

                CREATE TABLE IF NOT EXISTS conversation_messages (
                    message_id TEXT PRIMARY KEY,
                    conversation_id TEXT NOT NULL,
                    sequence_number INTEGER NOT NULL CHECK (sequence_number > 0),
                    role TEXT NOT NULL CHECK (role IN ('user', 'assistant')),
                    content TEXT NOT NULL,
                    created_at_utc TEXT NOT NULL,
                    trace_id TEXT,
                    UNIQUE (conversation_id, sequence_number),
                    FOREIGN KEY (conversation_id)
                        REFERENCES conversations(conversation_id)
                        ON DELETE CASCADE
                );

                CREATE TABLE IF NOT EXISTS conversation_summaries (
                    conversation_id TEXT PRIMARY KEY,
                    summary_text TEXT NOT NULL,
                    summarized_through_sequence INTEGER NOT NULL
                        CHECK (summarized_through_sequence > 0),
                    version INTEGER NOT NULL CHECK (version > 0),
                    updated_at_utc TEXT NOT NULL,
                    FOREIGN KEY (conversation_id)
                        REFERENCES conversations(conversation_id)
                        ON DELETE CASCADE
                );

                CREATE INDEX IF NOT EXISTS idx_conversations_updated
                    ON conversations(updated_at_utc DESC);
                CREATE INDEX IF NOT EXISTS idx_messages_conversation_sequence
                    ON conversation_messages(conversation_id, sequence_number);
                """
            )

    def create_conversation(self, title: str = "New conversation") -> Conversation:
        normalized_title = title.strip()
        if not normalized_title:
            raise ValueError("title must not be blank")

        with self._lock:
            self._ensure_open()
            now = datetime.now(UTC)
            conversation = Conversation(
                conversation_id=str(uuid4()),
                title=normalized_title,
                created_at_utc=now,
                updated_at_utc=now,
            )
            with self._connection:
                self._connection.execute(
                    """
                    INSERT INTO conversations (
                        conversation_id,
                        title,
                        created_at_utc,
                        updated_at_utc,
                        is_archived
                    ) VALUES (?, ?, ?, ?, 0)
                    """,
                    (
                        conversation.conversation_id,
                        conversation.title,
                        _serialize_datetime(conversation.created_at_utc),
                        _serialize_datetime(conversation.updated_at_utc),
                    ),
                )
            return conversation

    def get_conversation(self, conversation_id: str) -> Conversation | None:
        with self._lock:
            self._ensure_open()
            row = self._connection.execute(
                """
                SELECT conversation_id, title, created_at_utc, updated_at_utc,
                       is_archived
                FROM conversations
                WHERE conversation_id = ?
                """,
                (conversation_id,),
            ).fetchone()
            return _conversation_from_row(row) if row is not None else None

    def list_conversations(
        self,
        *,
        include_archived: bool = False,
    ) -> list[Conversation]:
        with self._lock:
            self._ensure_open()
            where_clause = "" if include_archived else "WHERE is_archived = 0"
            rows = self._connection.execute(
                f"""
                SELECT conversation_id, title, created_at_utc, updated_at_utc,
                       is_archived
                FROM conversations
                {where_clause}
                ORDER BY updated_at_utc DESC, conversation_id ASC
                """
            ).fetchall()
            return [_conversation_from_row(row) for row in rows]

    def add_message(
        self,
        conversation_id: str,
        role: MessageRole,
        content: str,
        *,
        trace_id: str | None = None,
    ) -> ConversationMessage:
        normalized_content = content.strip()
        if not normalized_content:
            raise ValueError("content must not be blank")
        if trace_id is not None and not trace_id.strip():
            raise ValueError("trace_id must not be blank")

        with self._lock:
            self._ensure_open()
            self._require_mutable_conversation(conversation_id)
            next_sequence = self._connection.execute(
                """
                SELECT COALESCE(MAX(sequence_number), 0) + 1
                FROM conversation_messages
                WHERE conversation_id = ?
                """,
                (conversation_id,),
            ).fetchone()[0]
            now = datetime.now(UTC)
            message = ConversationMessage(
                message_id=str(uuid4()),
                conversation_id=conversation_id,
                sequence_number=next_sequence,
                role=MessageRole(role),
                content=normalized_content,
                created_at_utc=now,
                trace_id=trace_id.strip() if trace_id is not None else None,
            )
            with self._connection:
                self._connection.execute(
                    """
                    INSERT INTO conversation_messages (
                        message_id,
                        conversation_id,
                        sequence_number,
                        role,
                        content,
                        created_at_utc,
                        trace_id
                    ) VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        message.message_id,
                        message.conversation_id,
                        message.sequence_number,
                        message.role.value,
                        message.content,
                        _serialize_datetime(message.created_at_utc),
                        message.trace_id,
                    ),
                )
                self._connection.execute(
                    """
                    UPDATE conversations
                    SET updated_at_utc = ?
                    WHERE conversation_id = ?
                    """,
                    (_serialize_datetime(now), conversation_id),
                )
            return message

    def list_messages(self, conversation_id: str) -> list[ConversationMessage]:
        with self._lock:
            self._ensure_open()
            self._require_conversation(conversation_id)
            rows = self._connection.execute(
                """
                SELECT message_id, conversation_id, sequence_number, role,
                       content, created_at_utc, trace_id
                FROM conversation_messages
                WHERE conversation_id = ?
                ORDER BY sequence_number ASC
                """,
                (conversation_id,),
            ).fetchall()
            return [_message_from_row(row) for row in rows]

    def save_summary(
        self,
        conversation_id: str,
        summary_text: str,
        *,
        summarized_through_sequence: int,
    ) -> ConversationSummary:
        normalized_summary = summary_text.strip()
        if not normalized_summary:
            raise ValueError("summary_text must not be blank")
        if summarized_through_sequence <= 0:
            raise ValueError("summarized_through_sequence must be greater than 0")

        with self._lock:
            self._ensure_open()
            self._require_mutable_conversation(conversation_id)
            latest_sequence = self._connection.execute(
                """
                SELECT COALESCE(MAX(sequence_number), 0)
                FROM conversation_messages
                WHERE conversation_id = ?
                """,
                (conversation_id,),
            ).fetchone()[0]
            if summarized_through_sequence > latest_sequence:
                raise ValueError(
                    "summarized_through_sequence cannot exceed the latest message"
                )

            existing = self.get_summary(conversation_id)
            if (
                existing is not None
                and summarized_through_sequence
                < existing.summarized_through_sequence
            ):
                raise ValueError("summary checkpoint cannot move backwards")

            now = datetime.now(UTC)
            summary = ConversationSummary(
                conversation_id=conversation_id,
                summary_text=normalized_summary,
                summarized_through_sequence=summarized_through_sequence,
                version=1 if existing is None else existing.version + 1,
                updated_at_utc=now,
            )
            with self._connection:
                self._connection.execute(
                    """
                    INSERT INTO conversation_summaries (
                        conversation_id,
                        summary_text,
                        summarized_through_sequence,
                        version,
                        updated_at_utc
                    ) VALUES (?, ?, ?, ?, ?)
                    ON CONFLICT(conversation_id) DO UPDATE SET
                        summary_text = excluded.summary_text,
                        summarized_through_sequence =
                            excluded.summarized_through_sequence,
                        version = excluded.version,
                        updated_at_utc = excluded.updated_at_utc
                    """,
                    (
                        summary.conversation_id,
                        summary.summary_text,
                        summary.summarized_through_sequence,
                        summary.version,
                        _serialize_datetime(summary.updated_at_utc),
                    ),
                )
            return summary

    def get_summary(self, conversation_id: str) -> ConversationSummary | None:
        with self._lock:
            self._ensure_open()
            self._require_conversation(conversation_id)
            row = self._connection.execute(
                """
                SELECT conversation_id, summary_text,
                       summarized_through_sequence, version, updated_at_utc
                FROM conversation_summaries
                WHERE conversation_id = ?
                """,
                (conversation_id,),
            ).fetchone()
            return _summary_from_row(row) if row is not None else None

    def archive_conversation(self, conversation_id: str) -> Conversation:
        with self._lock:
            self._ensure_open()
            self._require_conversation(conversation_id)
            now = datetime.now(UTC)
            with self._connection:
                self._connection.execute(
                    """
                    UPDATE conversations
                    SET is_archived = 1, updated_at_utc = ?
                    WHERE conversation_id = ?
                    """,
                    (_serialize_datetime(now), conversation_id),
                )
            archived = self.get_conversation(conversation_id)
            if archived is None:  # Defensive: the row existed in this transaction.
                raise ConversationNotFoundError(conversation_id)
            return archived

    def close(self) -> None:
        with self._lock:
            if not self._is_closed:
                self._connection.close()
                self._is_closed = True

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.close()

    def _ensure_open(self) -> None:
        if self._is_closed:
            raise StoreClosedError("conversation store is closed")

    def _require_conversation(self, conversation_id: str) -> Conversation:
        row = self._connection.execute(
            """
            SELECT conversation_id, title, created_at_utc, updated_at_utc,
                   is_archived
            FROM conversations
            WHERE conversation_id = ?
            """,
            (conversation_id,),
        ).fetchone()
        if row is None:
            raise ConversationNotFoundError(conversation_id)
        return _conversation_from_row(row)

    def _require_mutable_conversation(self, conversation_id: str) -> Conversation:
        conversation = self._require_conversation(conversation_id)
        if conversation.is_archived:
            raise ConversationArchivedError(conversation_id)
        return conversation


def _serialize_datetime(value: datetime) -> str:
    return value.astimezone(UTC).isoformat()


def _parse_datetime(value: str) -> datetime:
    return datetime.fromisoformat(value).astimezone(UTC)


def _conversation_from_row(row: sqlite3.Row) -> Conversation:
    return Conversation(
        conversation_id=row["conversation_id"],
        title=row["title"],
        created_at_utc=_parse_datetime(row["created_at_utc"]),
        updated_at_utc=_parse_datetime(row["updated_at_utc"]),
        is_archived=bool(row["is_archived"]),
    )


def _message_from_row(row: sqlite3.Row) -> ConversationMessage:
    return ConversationMessage(
        message_id=row["message_id"],
        conversation_id=row["conversation_id"],
        sequence_number=row["sequence_number"],
        role=MessageRole(row["role"]),
        content=row["content"],
        created_at_utc=_parse_datetime(row["created_at_utc"]),
        trace_id=row["trace_id"],
    )


def _summary_from_row(row: sqlite3.Row) -> ConversationSummary:
    return ConversationSummary(
        conversation_id=row["conversation_id"],
        summary_text=row["summary_text"],
        summarized_through_sequence=row["summarized_through_sequence"],
        version=row["version"],
        updated_at_utc=_parse_datetime(row["updated_at_utc"]),
    )
