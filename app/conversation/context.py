from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from typing import Protocol, Sequence, runtime_checkable

from app.conversation.models import ConversationMessage, ConversationSummary
from app.conversation.store import ConversationStore


MESSAGE_OVERHEAD_TOKENS = 4
SUMMARY_OVERHEAD_TOKENS = 4


class ContextBudgetExceededError(RuntimeError):
    """Raised when required conversation context cannot fit its allocation."""


class InvalidSummaryError(RuntimeError):
    """Raised when a summary generator violates its output contract."""


@runtime_checkable
class TokenCounter(Protocol):
    def count_text(self, text: str) -> int: ...


class ApproximateTokenCounter:
    """Conservative dependency-free estimate for preflight context budgeting."""

    def count_text(self, text: str) -> int:
        if not text:
            return 0
        return max(1, ceil(len(text.encode("utf-8")) / 4))


@runtime_checkable
class ConversationSummarizer(Protocol):
    def summarize(
        self,
        *,
        previous_summary: str | None,
        messages: Sequence[ConversationMessage],
        max_tokens: int,
    ) -> str: ...


@dataclass(frozen=True)
class ContextBudget:
    max_context_tokens: int
    reserved_system_tokens: int
    reserved_evidence_tokens: int
    reserved_answer_tokens: int
    summary_target_tokens: int

    def __post_init__(self) -> None:
        values = {
            "max_context_tokens": self.max_context_tokens,
            "reserved_system_tokens": self.reserved_system_tokens,
            "reserved_evidence_tokens": self.reserved_evidence_tokens,
            "reserved_answer_tokens": self.reserved_answer_tokens,
            "summary_target_tokens": self.summary_target_tokens,
        }
        for name, value in values.items():
            if value < 0:
                raise ValueError(f"{name} must not be negative")
        if self.max_context_tokens <= 0:
            raise ValueError("max_context_tokens must be greater than 0")
        if self.summary_target_tokens <= SUMMARY_OVERHEAD_TOKENS:
            raise ValueError(
                "summary_target_tokens must exceed summary framing overhead"
            )
        if self.conversation_tokens <= 0:
            raise ValueError("reserved tokens leave no conversation capacity")
        if self.summary_target_tokens >= self.conversation_tokens:
            raise ValueError(
                "summary_target_tokens must be smaller than conversation capacity"
            )

    @property
    def conversation_tokens(self) -> int:
        return self.max_context_tokens - (
            self.reserved_system_tokens
            + self.reserved_evidence_tokens
            + self.reserved_answer_tokens
        )


@dataclass(frozen=True)
class ConversationContext:
    summary_text: str | None
    summarized_through_sequence: int
    recent_messages: tuple[ConversationMessage, ...]
    estimated_tokens: int
    budget_tokens: int


class RollingConversationContextBuilder:
    """Build bounded memory and roll older messages into a durable summary."""

    def __init__(
        self,
        *,
        store: ConversationStore,
        summarizer: ConversationSummarizer,
        token_counter: TokenCounter,
        budget: ContextBudget,
        min_recent_messages: int = 2,
    ) -> None:
        if min_recent_messages <= 0:
            raise ValueError("min_recent_messages must be greater than 0")
        self._store = store
        self._summarizer = summarizer
        self._token_counter = token_counter
        self._budget = budget
        self._min_recent_messages = min_recent_messages

    def build(self, conversation_id: str) -> ConversationContext:
        messages = self._store.list_messages(conversation_id)
        existing_summary = self._store.get_summary(conversation_id)
        checkpoint = (
            existing_summary.summarized_through_sequence
            if existing_summary is not None
            else 0
        )
        unsummarized = [
            message
            for message in messages
            if message.sequence_number > checkpoint
        ]

        current_tokens = self._context_tokens(existing_summary, unsummarized)
        if current_tokens <= self._budget.conversation_tokens:
            return self._make_context(existing_summary, unsummarized, current_tokens)

        recent_start = self._choose_recent_start(unsummarized)
        older_messages = unsummarized[:recent_start]
        recent_messages = unsummarized[recent_start:]
        if not older_messages:
            raise ContextBudgetExceededError(
                "required recent messages exceed the conversation token budget"
            )

        summary_text = self._summarizer.summarize(
            previous_summary=(
                existing_summary.summary_text
                if existing_summary is not None
                else None
            ),
            messages=older_messages,
            max_tokens=(
                self._budget.summary_target_tokens - SUMMARY_OVERHEAD_TOKENS
            ),
        ).strip()
        if not summary_text:
            raise InvalidSummaryError("summary generator returned blank content")
        if self._summary_tokens(summary_text) > self._budget.summary_target_tokens:
            raise InvalidSummaryError("summary generator exceeded its token target")

        prospective_tokens = self._summary_tokens(
            summary_text
        ) + self._messages_tokens(recent_messages)
        if prospective_tokens > self._budget.conversation_tokens:
            raise ContextBudgetExceededError(
                "summary and required recent messages exceed the conversation "
                "token budget"
            )

        saved_summary = self._store.save_summary(
            conversation_id,
            summary_text,
            summarized_through_sequence=older_messages[-1].sequence_number,
        )
        return self._make_context(
            saved_summary,
            recent_messages,
            prospective_tokens,
        )

    def _choose_recent_start(
        self,
        messages: Sequence[ConversationMessage],
    ) -> int:
        if len(messages) <= self._min_recent_messages:
            return 0

        start = len(messages) - self._min_recent_messages
        required_tokens = (
            self._budget.summary_target_tokens
            + self._messages_tokens(messages[start:])
        )
        if required_tokens > self._budget.conversation_tokens:
            raise ContextBudgetExceededError(
                "required recent messages leave no room for a rolling summary"
            )

        while start > 0:
            candidate_tokens = (
                self._budget.summary_target_tokens
                + self._messages_tokens(messages[start - 1 :])
            )
            if candidate_tokens > self._budget.conversation_tokens:
                break
            start -= 1
        return start

    def _make_context(
        self,
        summary: ConversationSummary | None,
        messages: Sequence[ConversationMessage],
        estimated_tokens: int,
    ) -> ConversationContext:
        return ConversationContext(
            summary_text=summary.summary_text if summary is not None else None,
            summarized_through_sequence=(
                summary.summarized_through_sequence
                if summary is not None
                else 0
            ),
            recent_messages=tuple(messages),
            estimated_tokens=estimated_tokens,
            budget_tokens=self._budget.conversation_tokens,
        )

    def _context_tokens(
        self,
        summary: ConversationSummary | None,
        messages: Sequence[ConversationMessage],
    ) -> int:
        summary_tokens = (
            self._summary_tokens(summary.summary_text)
            if summary is not None
            else 0
        )
        return summary_tokens + self._messages_tokens(messages)

    def _messages_tokens(
        self,
        messages: Sequence[ConversationMessage],
    ) -> int:
        return sum(
            self._token_counter.count_text(message.content)
            + MESSAGE_OVERHEAD_TOKENS
            for message in messages
        )

    def _summary_tokens(self, summary_text: str) -> int:
        return (
            self._token_counter.count_text(summary_text)
            + SUMMARY_OVERHEAD_TOKENS
        )


def render_conversation_context(context: ConversationContext) -> str:
    """Render bounded memory as clearly marked, non-evidentiary prompt data."""

    summary = context.summary_text or "(none)"
    messages = "\n".join(
        (
            f'<message sequence="{message.sequence_number}" '
            f'role="{message.role.value}">\n'
            f"{message.content}\n"
            "</message>"
        )
        for message in context.recent_messages
    )
    return (
        "<conversation_memory>\n"
        f"<rolling_summary>\n{summary}\n</rolling_summary>\n"
        f"<recent_messages>\n{messages or '(none)'}\n</recent_messages>\n"
        "</conversation_memory>"
    )
