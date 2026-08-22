from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field, field_validator

from app.schemas.query_api import QueryResponse


class CreateConversationRequest(BaseModel):
    title: str = Field(default="New conversation", min_length=1, max_length=200)

    @field_validator("title")
    @classmethod
    def title_must_not_be_blank(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("title must not be blank")
        return cleaned


class ConversationMessageRequest(BaseModel):
    content: str = Field(..., min_length=1, max_length=20_000)
    knowledge_mode: Literal["fdd", "code", "combined"] = "fdd"
    analysis_kind: Literal["explanation", "impact_analysis"] = "explanation"
    limit: int = Field(default=5, gt=0, le=50)
    document_family: str | None = None
    release_label: str | None = None
    source_kind: str | None = None
    min_top_score: float | None = Field(default=None, ge=0)

    @field_validator("content")
    @classmethod
    def content_must_not_be_blank(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("content must not be blank")
        return cleaned

    @field_validator("source_kind")
    @classmethod
    def source_kind_must_be_supported(cls, value: str | None) -> str | None:
        if value is None:
            return None
        cleaned = value.strip().lower()
        if cleaned not in {"paragraph", "table"}:
            raise ValueError("source_kind must be either 'paragraph' or 'table'")
        return cleaned


class ConversationResponse(BaseModel):
    conversation_id: str
    title: str
    created_at_utc: datetime
    updated_at_utc: datetime
    is_archived: bool


class ConversationMessageResponse(BaseModel):
    message_id: str
    conversation_id: str
    sequence_number: int
    role: str
    content: str
    created_at_utc: datetime
    trace_id: str | None


class ConversationSummaryResponse(BaseModel):
    summary_text: str
    summarized_through_sequence: int
    version: int
    updated_at_utc: datetime


class ConversationDetailResponse(BaseModel):
    conversation: ConversationResponse
    messages: list[ConversationMessageResponse]
    summary: ConversationSummaryResponse | None


class ConversationTurnResponse(BaseModel):
    user_message: ConversationMessageResponse
    assistant_message: ConversationMessageResponse
    answer: QueryResponse
    context_estimated_tokens: int
    context_budget_tokens: int
    summarized_through_sequence: int
