from pathlib import Path

import pytest

from app.conversation.models import MessageRole
from app.conversation.store import (
    ConversationArchivedError,
    ConversationNotFoundError,
    ConversationStore,
    SqliteConversationStore,
    StoreClosedError,
)
from app.core.config import Settings


def test_sqlite_store_implements_conversation_store_protocol(tmp_path: Path) -> None:
    with SqliteConversationStore(tmp_path / "conversations.sqlite3") as store:
        assert isinstance(store, ConversationStore)


def test_conversation_database_path_is_configurable(
    monkeypatch,
    tmp_path: Path,
) -> None:
    database_path = tmp_path / "custom-conversations.sqlite3"
    monkeypatch.setenv("CONVERSATION_DB_PATH", str(database_path))

    settings = Settings(_env_file=None)

    assert settings.conversation_db_path == database_path


def test_local_sqlite3_conversation_data_is_git_ignored() -> None:
    gitignore = Path(".gitignore").read_text(encoding="utf-8")

    assert "*.sqlite3" in gitignore.splitlines()


def test_store_persists_ordered_isolated_conversations(tmp_path: Path) -> None:
    database_path = tmp_path / "conversations.sqlite3"

    with SqliteConversationStore(database_path) as store:
        first = store.create_conversation("First")
        second = store.create_conversation("Second")
        store.add_message(first.conversation_id, MessageRole.USER, "First question")
        assistant_message = store.add_message(
            first.conversation_id,
            MessageRole.ASSISTANT,
            "First answer",
            trace_id="trace-1",
        )
        store.add_message(second.conversation_id, MessageRole.USER, "Other question")

        first_messages = store.list_messages(first.conversation_id)
        second_messages = store.list_messages(second.conversation_id)

        assert [message.sequence_number for message in first_messages] == [1, 2]
        assert [message.content for message in first_messages] == [
            "First question",
            "First answer",
        ]
        assert assistant_message.trace_id == "trace-1"
        assert [message.content for message in second_messages] == ["Other question"]

    with SqliteConversationStore(database_path) as reopened:
        assert reopened.get_conversation(first.conversation_id) is not None
        assert len(reopened.list_messages(first.conversation_id)) == 2


def test_store_versions_forward_only_summary_checkpoints(tmp_path: Path) -> None:
    with SqliteConversationStore(tmp_path / "conversations.sqlite3") as store:
        conversation = store.create_conversation()
        store.add_message(conversation.conversation_id, MessageRole.USER, "Question")
        store.add_message(conversation.conversation_id, MessageRole.ASSISTANT, "Answer")

        first = store.save_summary(
            conversation.conversation_id,
            "The user asked a question.",
            summarized_through_sequence=1,
        )
        second = store.save_summary(
            conversation.conversation_id,
            "The user asked a question and received an answer.",
            summarized_through_sequence=2,
        )

        assert first.version == 1
        assert second.version == 2
        assert store.get_summary(conversation.conversation_id) == second

        with pytest.raises(ValueError, match="cannot move backwards"):
            store.save_summary(
                conversation.conversation_id,
                "Stale summary",
                summarized_through_sequence=1,
            )

        with pytest.raises(ValueError, match="cannot exceed"):
            store.save_summary(
                conversation.conversation_id,
                "Invented future summary",
                summarized_through_sequence=3,
            )


def test_archived_conversation_is_hidden_and_read_only(tmp_path: Path) -> None:
    with SqliteConversationStore(tmp_path / "conversations.sqlite3") as store:
        conversation = store.create_conversation("Archive me")
        store.add_message(conversation.conversation_id, MessageRole.USER, "Question")

        archived = store.archive_conversation(conversation.conversation_id)

        assert archived.is_archived is True
        assert store.list_conversations() == []
        assert store.list_conversations(include_archived=True) == [archived]
        assert len(store.list_messages(conversation.conversation_id)) == 1
        with pytest.raises(ConversationArchivedError):
            store.add_message(
                conversation.conversation_id,
                MessageRole.ASSISTANT,
                "Late answer",
            )


def test_store_rejects_missing_conversations_and_use_after_close(
    tmp_path: Path,
) -> None:
    store = SqliteConversationStore(tmp_path / "conversations.sqlite3")

    assert store.get_conversation("missing") is None
    with pytest.raises(ConversationNotFoundError):
        store.list_messages("missing")

    store.close()
    store.close()
    with pytest.raises(StoreClosedError):
        store.list_conversations()
