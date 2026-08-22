from __future__ import annotations

from collections.abc import Iterator

from fastapi import APIRouter, Depends, HTTPException, Query, status

from app.api.routes.query import execute_query_request
from app.conversation.context import (
    ApproximateTokenCounter,
    ContextBudget,
    ContextBudgetExceededError,
    ConversationContext,
    InvalidSummaryError,
    RollingConversationContextBuilder,
    render_conversation_context,
)
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
)
from app.conversation.summarizer import OpenAIConversationSummarizer
from app.core.config import get_settings
from app.core.logging import get_logger
from app.schemas.conversation_api import (
    ConversationDetailResponse,
    ConversationMessageRequest,
    ConversationMessageResponse,
    ConversationResponse,
    ConversationSummaryResponse,
    ConversationTurnResponse,
    CreateConversationRequest,
)
from app.schemas.query_api import QueryRequest


router = APIRouter(prefix="/conversations", tags=["conversations"])
logger = get_logger("conversation_api")


def get_conversation_store() -> Iterator[ConversationStore]:
    settings = get_settings()
    with SqliteConversationStore(settings.conversation_db_path) as store:
        yield store


def build_conversation_context(
    store: ConversationStore,
    conversation_id: str,
) -> ConversationContext:
    settings = get_settings()
    budget = ContextBudget(
        max_context_tokens=settings.conversation_max_context_tokens,
        reserved_system_tokens=settings.conversation_reserved_system_tokens,
        reserved_evidence_tokens=settings.conversation_reserved_evidence_tokens,
        reserved_answer_tokens=settings.conversation_reserved_answer_tokens,
        summary_target_tokens=settings.conversation_summary_target_tokens,
    )
    builder = RollingConversationContextBuilder(
        store=store,
        summarizer=OpenAIConversationSummarizer(),
        token_counter=ApproximateTokenCounter(),
        budget=budget,
    )
    return builder.build(conversation_id)


@router.post(
    "",
    response_model=ConversationResponse,
    status_code=status.HTTP_201_CREATED,
)
def create_conversation(
    request: CreateConversationRequest,
    store: ConversationStore = Depends(get_conversation_store),
) -> ConversationResponse:
    return _conversation_response(store.create_conversation(request.title))


@router.get("", response_model=list[ConversationResponse])
def list_conversations(
    include_archived: bool = Query(default=False),
    store: ConversationStore = Depends(get_conversation_store),
) -> list[ConversationResponse]:
    return [
        _conversation_response(conversation)
        for conversation in store.list_conversations(
            include_archived=include_archived
        )
    ]


@router.get("/{conversation_id}", response_model=ConversationDetailResponse)
def get_conversation(
    conversation_id: str,
    store: ConversationStore = Depends(get_conversation_store),
) -> ConversationDetailResponse:
    try:
        conversation = store.get_conversation(conversation_id)
        if conversation is None:
            raise ConversationNotFoundError(conversation_id)
        return ConversationDetailResponse(
            conversation=_conversation_response(conversation),
            messages=[
                _message_response(message)
                for message in store.list_messages(conversation_id)
            ],
            summary=_summary_response(store.get_summary(conversation_id)),
        )
    except ConversationNotFoundError as exc:
        raise _not_found() from exc


@router.post(
    "/{conversation_id}/archive",
    response_model=ConversationResponse,
)
def archive_conversation(
    conversation_id: str,
    store: ConversationStore = Depends(get_conversation_store),
) -> ConversationResponse:
    try:
        return _conversation_response(
            store.archive_conversation(conversation_id)
        )
    except ConversationNotFoundError as exc:
        raise _not_found() from exc


@router.post(
    "/{conversation_id}/messages",
    response_model=ConversationTurnResponse,
)
def submit_message(
    conversation_id: str,
    request: ConversationMessageRequest,
    store: ConversationStore = Depends(get_conversation_store),
) -> ConversationTurnResponse:
    try:
        user_message = store.add_message(
            conversation_id,
            MessageRole.USER,
            request.content,
        )
        context = build_conversation_context(store, conversation_id)
        answer = execute_query_request(
            QueryRequest(
                query=request.content,
                knowledge_mode=request.knowledge_mode,
                analysis_kind=request.analysis_kind,
                limit=request.limit,
                document_family=request.document_family,
                release_label=request.release_label,
                source_kind=request.source_kind,
                min_top_score=request.min_top_score,
            ),
            conversation_context=render_conversation_context(context),
        )
        assistant_message = store.add_message(
            conversation_id,
            MessageRole.ASSISTANT,
            answer.answer,
            trace_id=answer.trace_id,
        )
        return ConversationTurnResponse(
            user_message=_message_response(user_message),
            assistant_message=_message_response(assistant_message),
            answer=answer,
            context_estimated_tokens=context.estimated_tokens,
            context_budget_tokens=context.budget_tokens,
            summarized_through_sequence=(
                context.summarized_through_sequence
            ),
        )
    except HTTPException:
        raise
    except ConversationNotFoundError as exc:
        raise _not_found() from exc
    except ConversationArchivedError as exc:
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Archived conversations are read-only.",
        ) from exc
    except ContextBudgetExceededError as exc:
        raise HTTPException(
            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            detail=(
                "Conversation context exceeds its configured token budget. "
                "Start a new conversation or reduce the message size."
            ),
        ) from exc
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    except InvalidSummaryError as exc:
        logger.exception("Conversation summary validation failed")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Conversation context processing failed.",
        ) from exc
    except Exception as exc:
        logger.exception("Unexpected conversation message failure")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal conversation processing error.",
        ) from exc


def _conversation_response(
    conversation: Conversation,
) -> ConversationResponse:
    return ConversationResponse(
        conversation_id=conversation.conversation_id,
        title=conversation.title,
        created_at_utc=conversation.created_at_utc,
        updated_at_utc=conversation.updated_at_utc,
        is_archived=conversation.is_archived,
    )


def _message_response(
    message: ConversationMessage,
) -> ConversationMessageResponse:
    return ConversationMessageResponse(
        message_id=message.message_id,
        conversation_id=message.conversation_id,
        sequence_number=message.sequence_number,
        role=message.role.value,
        content=message.content,
        created_at_utc=message.created_at_utc,
        trace_id=message.trace_id,
    )


def _summary_response(
    summary: ConversationSummary | None,
) -> ConversationSummaryResponse | None:
    if summary is None:
        return None
    return ConversationSummaryResponse(
        summary_text=summary.summary_text,
        summarized_through_sequence=summary.summarized_through_sequence,
        version=summary.version,
        updated_at_utc=summary.updated_at_utc,
    )


def _not_found() -> HTTPException:
    return HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail="Conversation not found.",
    )
