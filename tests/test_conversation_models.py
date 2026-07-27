from datetime import UTC, datetime

import pytest

from app.conversation.models import (
    Conversation,
    ConversationMessage,
    ConversationSummary,
    MessageRole,
)


NOW = datetime(2026, 7, 27, tzinfo=UTC)


def test_conversation_models_accept_valid_records() -> None:
    conversation = Conversation("conversation-1", "Release changes", NOW, NOW)
    message = ConversationMessage(
        "message-1",
        conversation.conversation_id,
        1,
        MessageRole.USER,
        "What changed in R24?",
        NOW,
    )
    summary = ConversationSummary(
        conversation.conversation_id,
        "The user is investigating R24.",
        1,
        1,
        NOW,
    )

    assert message.role is MessageRole.USER
    assert summary.summarized_through_sequence == 1


@pytest.mark.parametrize(
    ("factory", "expected_message"),
    [
        (
            lambda: Conversation("", "Title", NOW, NOW),
            "conversation_id",
        ),
        (
            lambda: Conversation("conversation-1", " ", NOW, NOW),
            "title",
        ),
        (
            lambda: ConversationMessage(
                "message-1",
                "conversation-1",
                0,
                MessageRole.USER,
                "Question",
                NOW,
            ),
            "sequence_number",
        ),
        (
            lambda: ConversationSummary(
                "conversation-1",
                "Summary",
                1,
                0,
                NOW,
            ),
            "version",
        ),
    ],
)
def test_conversation_models_reject_invalid_records(
    factory,
    expected_message: str,
) -> None:
    with pytest.raises(ValueError, match=expected_message):
        factory()


def test_conversation_models_reject_naive_timestamps() -> None:
    with pytest.raises(ValueError, match="timezone-aware"):
        Conversation(
            "conversation-1",
            "Title",
            datetime(2026, 7, 27),
            NOW,
        )
