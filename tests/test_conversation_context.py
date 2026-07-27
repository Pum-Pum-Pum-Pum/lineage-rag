from pathlib import Path
from typing import Sequence

import pytest

from app.conversation.context import (
    ApproximateTokenCounter,
    ContextBudget,
    ContextBudgetExceededError,
    InvalidSummaryError,
    RollingConversationContextBuilder,
)
from app.conversation.models import ConversationMessage, MessageRole
from app.conversation.store import SqliteConversationStore
from app.core.config import Settings


class WordTokenCounter:
    def count_text(self, text: str) -> int:
        return len(text.split())


class RecordingSummarizer:
    def __init__(self, result: str = "rolled summary") -> None:
        self.result = result
        self.calls: list[
            tuple[str | None, tuple[int, ...], int]
        ] = []

    def summarize(
        self,
        *,
        previous_summary: str | None,
        messages: Sequence[ConversationMessage],
        max_tokens: int,
    ) -> str:
        self.calls.append(
            (
                previous_summary,
                tuple(message.sequence_number for message in messages),
                max_tokens,
            )
        )
        return self.result


def make_budget(conversation_tokens: int = 30) -> ContextBudget:
    return ContextBudget(
        max_context_tokens=conversation_tokens + 30,
        reserved_system_tokens=10,
        reserved_evidence_tokens=10,
        reserved_answer_tokens=10,
        summary_target_tokens=10,
    )


def add_messages(
    store: SqliteConversationStore,
    conversation_id: str,
    contents: Sequence[str],
) -> None:
    for index, content in enumerate(contents):
        role = MessageRole.USER if index % 2 == 0 else MessageRole.ASSISTANT
        store.add_message(conversation_id, role, content)


def test_budget_reserves_non_conversation_capacity() -> None:
    budget = make_budget(conversation_tokens=30)

    assert budget.conversation_tokens == 30

    with pytest.raises(ValueError, match="leave no conversation capacity"):
        ContextBudget(30, 10, 10, 10, 5)
    with pytest.raises(ValueError, match="smaller than conversation capacity"):
        ContextBudget(40, 10, 10, 10, 10)


def test_context_budget_settings_are_environment_configurable(
    monkeypatch,
) -> None:
    monkeypatch.setenv("CONVERSATION_MAX_CONTEXT_TOKENS", "100")
    monkeypatch.setenv("CONVERSATION_RESERVED_SYSTEM_TOKENS", "10")
    monkeypatch.setenv("CONVERSATION_RESERVED_EVIDENCE_TOKENS", "20")
    monkeypatch.setenv("CONVERSATION_RESERVED_ANSWER_TOKENS", "30")
    monkeypatch.setenv("CONVERSATION_SUMMARY_TARGET_TOKENS", "12")
    settings = Settings(_env_file=None)

    budget = ContextBudget(
        max_context_tokens=settings.conversation_max_context_tokens,
        reserved_system_tokens=settings.conversation_reserved_system_tokens,
        reserved_evidence_tokens=settings.conversation_reserved_evidence_tokens,
        reserved_answer_tokens=settings.conversation_reserved_answer_tokens,
        summary_target_tokens=settings.conversation_summary_target_tokens,
    )

    assert budget.conversation_tokens == 40
    assert budget.summary_target_tokens == 12


def test_approximate_counter_is_deterministic_and_utf8_aware() -> None:
    counter = ApproximateTokenCounter()

    assert counter.count_text("") == 0
    assert counter.count_text("abcd") == 1
    assert counter.count_text("界") == 1
    assert counter.count_text("界界") == 2


def test_context_below_budget_keeps_all_messages_without_summary(
    tmp_path: Path,
) -> None:
    summarizer = RecordingSummarizer()
    with SqliteConversationStore(tmp_path / "chat.sqlite3") as store:
        conversation = store.create_conversation()
        add_messages(store, conversation.conversation_id, ["one", "two"])
        builder = RollingConversationContextBuilder(
            store=store,
            summarizer=summarizer,
            token_counter=WordTokenCounter(),
            budget=make_budget(),
        )

        context = builder.build(conversation.conversation_id)

        assert context.summary_text is None
        assert [message.content for message in context.recent_messages] == [
            "one",
            "two",
        ]
        assert context.estimated_tokens == 10
        assert summarizer.calls == []
        assert store.get_summary(conversation.conversation_id) is None


def test_over_budget_rolls_old_prefix_and_keeps_recent_suffix(
    tmp_path: Path,
) -> None:
    summarizer = RecordingSummarizer("old intent")
    with SqliteConversationStore(tmp_path / "chat.sqlite3") as store:
        conversation = store.create_conversation()
        add_messages(
            store,
            conversation.conversation_id,
            [
                "one two three",
                "four five six",
                "seven eight nine",
                "ten eleven twelve",
            ],
        )
        builder = RollingConversationContextBuilder(
            store=store,
            summarizer=summarizer,
            token_counter=WordTokenCounter(),
            budget=make_budget(conversation_tokens=27),
        )

        context = builder.build(conversation.conversation_id)

        assert summarizer.calls == [(None, (1, 2), 6)]
        assert context.summary_text == "old intent"
        assert context.summarized_through_sequence == 2
        assert [
            message.sequence_number for message in context.recent_messages
        ] == [3, 4]
        assert context.estimated_tokens <= context.budget_tokens
        saved_summary = store.get_summary(conversation.conversation_id)
        assert saved_summary is not None
        assert saved_summary.summarized_through_sequence == 2
        assert saved_summary.version == 1


def test_next_roll_consolidates_previous_summary_and_advances_checkpoint(
    tmp_path: Path,
) -> None:
    summarizer = RecordingSummarizer("first summary")
    with SqliteConversationStore(tmp_path / "chat.sqlite3") as store:
        conversation = store.create_conversation()
        add_messages(
            store,
            conversation.conversation_id,
            [
                "one two three",
                "four five six",
                "seven eight nine",
                "ten eleven twelve",
            ],
        )
        builder = RollingConversationContextBuilder(
            store=store,
            summarizer=summarizer,
            token_counter=WordTokenCounter(),
            budget=make_budget(conversation_tokens=27),
        )
        first_context = builder.build(conversation.conversation_id)
        store.add_message(
            conversation.conversation_id,
            MessageRole.USER,
            "nine ten eleven twelve thirteen fourteen",
        )
        summarizer.result = "second summary"

        second_context = builder.build(conversation.conversation_id)

        assert first_context.summarized_through_sequence == 2
        assert summarizer.calls[-1] == ("first summary", (3,), 6)
        assert second_context.summarized_through_sequence == 3
        assert [
            message.sequence_number
            for message in second_context.recent_messages
        ] == [4, 5]
        assert store.get_summary(conversation.conversation_id).version == 2


def test_context_and_summary_are_isolated_between_conversations(
    tmp_path: Path,
) -> None:
    summarizer = RecordingSummarizer()
    with SqliteConversationStore(tmp_path / "chat.sqlite3") as store:
        first = store.create_conversation("First")
        second = store.create_conversation("Second")
        add_messages(store, first.conversation_id, ["one"] * 5)
        add_messages(store, second.conversation_id, ["private second"])
        builder = RollingConversationContextBuilder(
            store=store,
            summarizer=summarizer,
            token_counter=WordTokenCounter(),
            budget=make_budget(conversation_tokens=23),
        )

        first_context = builder.build(first.conversation_id)
        second_context = builder.build(second.conversation_id)

        assert first_context.summary_text == "rolled summary"
        assert second_context.summary_text is None
        assert [
            message.content for message in second_context.recent_messages
        ] == ["private second"]


def test_required_recent_messages_that_cannot_fit_fail_without_summary(
    tmp_path: Path,
) -> None:
    summarizer = RecordingSummarizer()
    with SqliteConversationStore(tmp_path / "chat.sqlite3") as store:
        conversation = store.create_conversation()
        add_messages(
            store,
            conversation.conversation_id,
            ["old", "too many words for recent", "also too many recent words"],
        )
        builder = RollingConversationContextBuilder(
            store=store,
            summarizer=summarizer,
            token_counter=WordTokenCounter(),
            budget=make_budget(conversation_tokens=20),
        )

        with pytest.raises(ContextBudgetExceededError, match="leave no room"):
            builder.build(conversation.conversation_id)

        assert summarizer.calls == []
        assert store.get_summary(conversation.conversation_id) is None


@pytest.mark.parametrize(
    ("result", "expected_message"),
    [
        (" ", "blank content"),
        ("one two three four five six seven", "exceeded its token target"),
    ],
)
def test_invalid_summary_does_not_advance_durable_checkpoint(
    tmp_path: Path,
    result: str,
    expected_message: str,
) -> None:
    summarizer = RecordingSummarizer(result)
    with SqliteConversationStore(tmp_path / "chat.sqlite3") as store:
        conversation = store.create_conversation()
        add_messages(store, conversation.conversation_id, ["one two"] * 5)
        builder = RollingConversationContextBuilder(
            store=store,
            summarizer=summarizer,
            token_counter=WordTokenCounter(),
            budget=make_budget(conversation_tokens=27),
        )

        with pytest.raises(InvalidSummaryError, match=expected_message):
            builder.build(conversation.conversation_id)

        assert store.get_summary(conversation.conversation_id) is None
