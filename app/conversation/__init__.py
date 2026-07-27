"""Conversation-scoped memory contracts and persistence adapters."""

from app.conversation.models import (
    Conversation,
    ConversationMessage,
    ConversationSummary,
    MessageRole,
)
from app.conversation.store import (
    ConversationArchivedError,
    ConversationNotFoundError,
    ConversationStore,
    SqliteConversationStore,
    StoreClosedError,
)

__all__ = [
    "Conversation",
    "ConversationArchivedError",
    "ConversationMessage",
    "ConversationNotFoundError",
    "ConversationStore",
    "ConversationSummary",
    "MessageRole",
    "SqliteConversationStore",
    "StoreClosedError",
]
