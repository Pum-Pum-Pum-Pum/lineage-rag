from __future__ import annotations

import os
from typing import Any, Literal, Sequence

from pydantic import ValidationError

from app.schemas.conversation_api import (
    ConversationMessageRequest,
    ConversationMessageResponse,
    ConversationResponse,
    ConversationTurnResponse,
    CreateConversationRequest,
)
from app.ui.api_client import RagApiClient, UiApiError


DEFAULT_API_BASE_URL = os.getenv(
    "RAG_API_BASE_URL",
    "http://127.0.0.1:8000",
)
ACTIVE_CONVERSATION_KEY = "active_conversation_id"
TURN_DEBUG_KEY = "conversation_turn_debug"
KnowledgeMode = Literal["fdd", "code", "combined"]


def main() -> None:
    import streamlit as st

    st.set_page_config(
        page_title="Culling Blade Lineage RAG",
        page_icon="🔎",
        layout="wide",
    )
    _initialize_session_state(st.session_state)
    st.title("Culling Blade Lineage RAG")
    st.caption(
        "Conversation memory helps interpret follow-ups. Functional-spec facts "
        "still require fresh retrieved evidence and citations."
    )

    api_base_url = st.sidebar.text_input(
        "Backend API URL",
        value=DEFAULT_API_BASE_URL,
    )
    timeout = st.sidebar.number_input(
        "Request timeout (seconds)",
        min_value=1.0,
        max_value=120.0,
        value=30.0,
        step=1.0,
    )
    show_debug = st.sidebar.toggle(
        "Show evidence and debug details",
        value=True,
    )
    knowledge_options = _render_knowledge_controls(st)

    try:
        with RagApiClient(api_base_url, timeout=float(timeout)) as api:
            _render_backend_check(
                st,
                api,
                knowledge_mode=knowledge_options["knowledge_mode"],
            )
            conversations = api.list_conversations()
            active_id = _render_conversation_controls(
                st,
                api,
                conversations,
            )
            if active_id is None:
                st.info(
                    "Create a conversation from the sidebar to begin a "
                    "grounded multi-turn chat."
                )
                return

            detail = api.get_conversation(active_id)
            st.subheader(detail.conversation.title)
            if detail.conversation.is_archived:
                st.warning(
                    "This conversation is archived and remains read-only."
                )

            unanswered = _find_unanswered_user_sequences(detail.messages)
            _render_history(
                st,
                detail.messages,
                unanswered_sequences=unanswered,
                show_debug=show_debug,
            )
            if show_debug and detail.summary is not None:
                st.caption(
                    "Rolling memory checkpoint: "
                    f"sequence {detail.summary.summarized_through_sequence}, "
                    f"version {detail.summary.version}."
                )

            request_options = _render_retrieval_controls(
                st,
                knowledge_mode=knowledge_options["knowledge_mode"],
            )
            prompt = st.chat_input(
                _chat_input_prompt(knowledge_options["knowledge_mode"]),
                disabled=detail.conversation.is_archived,
            )
            if prompt:
                try:
                    request = _build_message_request(
                        content=prompt,
                        **knowledge_options,
                        **request_options,
                    )
                    with st.spinner(
                        "Checking readiness, retrieving evidence, and "
                        "generating a grounded response..."
                    ):
                        turn = _run_ready_turn(api, active_id, request)
                    _cache_turn(st.session_state, turn)
                    st.rerun()
                except ValidationError:
                    st.error(
                        "Enter a non-blank message and valid retrieval filters."
                    )
                except (UiApiError, ValueError) as exc:
                    _render_safe_error(st, exc)
    except (UiApiError, ValueError) as exc:
        _render_safe_error(st, exc)


def _initialize_session_state(state: Any) -> None:
    if ACTIVE_CONVERSATION_KEY not in state:
        state[ACTIVE_CONVERSATION_KEY] = None
    if TURN_DEBUG_KEY not in state:
        state[TURN_DEBUG_KEY] = {}


def _render_backend_check(
    st: Any,
    api: RagApiClient,
    *,
    knowledge_mode: KnowledgeMode,
) -> None:
    if not st.sidebar.button("Check backend", use_container_width=True):
        return
    try:
        health = api.get_health()
        readiness = api.get_readiness(knowledge_mode=knowledge_mode)
    except (UiApiError, ValueError) as exc:
        _render_safe_error(st.sidebar, exc)
        return

    if readiness.is_ready:
        st.sidebar.success(
            f"Healthy · {health.retrieval_mode} retrieval · ready"
        )
    else:
        st.sidebar.warning(
            f"Healthy · {health.retrieval_mode} retrieval · not ready"
        )
    for check in readiness.checks:
        marker = "OK" if check.is_ready else "FAIL"
        st.sidebar.caption(f"{marker} · {check.name}: {check.detail}")


def _render_conversation_controls(
    st: Any,
    api: RagApiClient,
    conversations: Sequence[ConversationResponse],
) -> str | None:
    st.sidebar.divider()
    st.sidebar.subheader("Conversations")
    new_title = st.sidebar.text_input(
        "New chat title",
        value="New conversation",
        max_chars=200,
    )
    if st.sidebar.button(
        "New chat",
        type="primary",
        use_container_width=True,
    ):
        try:
            created = api.create_conversation(
                CreateConversationRequest(title=new_title)
            )
        except ValidationError:
            st.sidebar.error("Enter a non-blank conversation title.")
        else:
            st.session_state[ACTIVE_CONVERSATION_KEY] = (
                created.conversation_id
            )
            st.rerun()

    selected_id = _select_active_conversation_id(
        st.session_state.get(ACTIVE_CONVERSATION_KEY),
        conversations,
    )
    if selected_id is None:
        st.session_state[ACTIVE_CONVERSATION_KEY] = None
        return None

    conversation_by_id = {
        conversation.conversation_id: conversation
        for conversation in conversations
    }
    ordered_ids = [
        conversation.conversation_id for conversation in conversations
    ]
    selected_index = ordered_ids.index(selected_id)
    selected_id = st.sidebar.selectbox(
        "Active chat",
        options=ordered_ids,
        index=selected_index,
        format_func=lambda conversation_id: _conversation_label(
            conversation_by_id[conversation_id]
        ),
    )
    st.session_state[ACTIVE_CONVERSATION_KEY] = selected_id

    if st.sidebar.button(
        "Archive active chat",
        use_container_width=True,
    ):
        api.archive_conversation(selected_id)
        st.session_state[ACTIVE_CONVERSATION_KEY] = None
        st.rerun()
    return selected_id


def _render_knowledge_controls(st: Any) -> dict[str, str]:
    st.sidebar.divider()
    st.sidebar.subheader("Knowledge mode")
    knowledge_mode = st.sidebar.selectbox(
        "Authoritative evidence",
        options=["fdd", "code", "combined"],
        format_func=lambda value: {
            "fdd": "Functional documents",
            "code": "Visible custom code",
            "combined": "Documents + custom code",
        }[value],
        help=(
            "Code and combined modes fail closed unless deliberately activated "
            "by the backend operator."
        ),
    )
    analysis_kind = st.sidebar.selectbox(
        "Analysis type",
        options=["explanation", "impact_analysis"],
        format_func=lambda value: {
            "explanation": "Explanation",
            "impact_analysis": "Impact analysis",
        }[value],
        disabled=knowledge_mode == "fdd",
        help="Impact locations are candidates, not proven root causes.",
    )
    if knowledge_mode != "fdd":
        st.sidebar.caption(
            "Only approved visible custom code is available. Hidden kernel, "
            "runtime dynamic SQL, and external schema behavior may remain unknown."
        )
    return {
        "knowledge_mode": knowledge_mode,
        "analysis_kind": (
            "explanation" if knowledge_mode == "fdd" else analysis_kind
        ),
    }


def _render_retrieval_controls(
    st: Any,
    *,
    knowledge_mode: KnowledgeMode,
) -> dict[str, object]:
    with st.sidebar.expander("Retrieval controls"):
        limit = st.number_input(
            "Evidence limit",
            min_value=1,
            max_value=50,
            value=5,
            step=1,
        )
        if knowledge_mode != "fdd":
            document_family = ""
            release_label = ""
            source_kind = "Any"
            use_min_top_score = False
            min_top_score = 0.25
            st.caption(
                "FDD metadata and per-request score overrides currently apply "
                "only to functional-document mode. Code lanes use their "
                "approved runtime thresholds."
            )
        else:
            document_family = st.text_input("Document family (optional)")
            release_label = st.text_input("Release label (optional)")
            source_kind = st.selectbox(
                "FDD source kind",
                options=["Any", "paragraph", "table"],
            )
            use_min_top_score = st.checkbox("Override minimum top score")
            min_top_score = st.number_input(
                "Minimum top score",
                min_value=0.0,
                value=0.25,
                step=0.05,
                disabled=not use_min_top_score,
            )
    return {
        "limit": int(limit),
        "document_family": document_family,
        "release_label": release_label,
        "source_kind": source_kind,
        "use_min_top_score": use_min_top_score,
        "min_top_score": float(min_top_score),
    }


def _render_history(
    st: Any,
    messages: Sequence[ConversationMessageResponse],
    *,
    unanswered_sequences: set[int],
    show_debug: bool,
) -> None:
    if not messages:
        st.info(
            "No messages yet. Ask a question below; unsupported questions "
            "will receive a grounded refusal."
        )
        return

    debug_by_message_id = st.session_state.get(TURN_DEBUG_KEY, {})
    for message in messages:
        with st.chat_message(message.role):
            st.markdown(message.content)
            if message.sequence_number in unanswered_sequences:
                st.caption(
                    "No assistant response was persisted for this message. "
                    "The previous attempt may have failed and can be retried."
                )
            if message.role == "assistant" and message.trace_id:
                st.caption(f"Trace ID: {message.trace_id}")
            if show_debug and message.role == "assistant":
                turn = debug_by_message_id.get(message.message_id)
                if turn is not None:
                    _render_turn_debug(st, turn)


def _render_turn_debug(st: Any, turn: ConversationTurnResponse) -> None:
    response = turn.answer
    with st.expander("Evidence and debug details"):
        status_text = (
            "Grounded answer"
            if response.is_answered
            else "Safe refusal"
        )
        st.write(
            {
                "outcome": status_text,
                "knowledge_mode": response.knowledge_mode,
                "retrieval_mode": response.retrieval_mode,
                "requested_claim_supported": (
                    response.requested_claim_supported
                ),
                "related_grounded_context_provided": (
                    response.related_grounded_context_provided
                ),
                "evidence_results": response.sufficiency.result_count,
                "top_score": response.sufficiency.top_score,
                "trace_id": response.trace_id,
                "context_estimated_tokens": turn.context_estimated_tokens,
                "context_budget_tokens": turn.context_budget_tokens,
                "summarized_through_sequence": (
                    turn.summarized_through_sequence
                ),
            }
        )
        st.caption(
            f"Evidence sufficiency: {response.sufficiency.reason}"
        )
        if response.refusal_reason:
            st.warning(response.refusal_reason)

        if response.combined_sections:
            st.markdown("**Section support**")
            for name, section in response.combined_sections.items():
                label = name.replace("_", " ").title()
                marker = (
                    "SUPPORTED" if section.status == "answered" else "REFUSED"
                )
                st.caption(f"{marker} · {label}")
                if section.refusal_reason:
                    st.caption(f"Reason: {section.refusal_reason}")

        st.markdown("**FDD citations**")
        if not response.citations:
            st.info("No FDD citations were returned.")
        for index, citation in enumerate(response.citations, start=1):
            citation_prefix = (
                "F" if response.knowledge_mode == "combined" else "C"
            )
            label = (
                f"{citation_prefix}{index} · "
                f"{citation.document_family or 'unknown document'} · "
                f"{citation.release_label or 'unknown release'} · "
                f"score={citation.score:.4f}"
            )
            st.markdown(f"**{label}**")
            st.write(citation.text_preview)
            st.caption(
                f"unit={citation.unit_id} · "
                f"source={citation.source_kind or 'unknown'}"
            )

        st.markdown("**Code citations**")
        if not response.code_citations:
            st.info("No code citations were returned.")
        for index, citation in enumerate(response.code_citations, start=1):
            st.markdown(
                f"**C{index} · {citation.display_name} · {citation.source_path}:"
                f"{citation.start_line}-{citation.end_line} · "
                f"score={citation.score:.4f}**"
            )
            st.write(citation.text_preview)
            st.caption(
                f"unit={citation.unit_id} · snapshot={citation.snapshot_id} · "
                f"source={citation.source_kind}"
            )

        if response.usage is not None:
            st.write(
                {
                    "model": response.usage.model,
                    "prompt_tokens": response.usage.prompt_tokens,
                    "completion_tokens": response.usage.completion_tokens,
                    "total_tokens": response.usage.total_tokens,
                }
            )
        if response.cost is not None:
            st.write(
                {
                    "estimated_cost": response.cost.total_cost,
                    "currency": response.cost.currency,
                }
            )


def _build_message_request(
    *,
    content: str,
    knowledge_mode: KnowledgeMode,
    analysis_kind: Literal["explanation", "impact_analysis"],
    limit: int,
    document_family: str,
    release_label: str,
    source_kind: str,
    use_min_top_score: bool,
    min_top_score: float,
) -> ConversationMessageRequest:
    return ConversationMessageRequest(
        content=content,
        knowledge_mode=knowledge_mode,
        analysis_kind=analysis_kind,
        limit=limit,
        document_family=_optional_text(document_family),
        release_label=_optional_text(release_label),
        source_kind=None if source_kind == "Any" else source_kind,
        min_top_score=min_top_score if use_min_top_score else None,
    )


def _run_ready_turn(
    api: RagApiClient,
    conversation_id: str,
    request: ConversationMessageRequest,
) -> ConversationTurnResponse:
    try:
        readiness = api.get_readiness(
            knowledge_mode=request.knowledge_mode,
        )
    except UiApiError as exc:
        if exc.code == "not_ready" and request.knowledge_mode != "fdd":
            raise UiApiError(
                code="knowledge_mode_unavailable",
                message=(
                    f"{request.knowledge_mode.title()} mode is not ready or "
                    "has not been activated. Functional-document mode remains "
                    "available when its readiness checks pass."
                ),
                status_code=exc.status_code,
            ) from exc
        raise
    if not readiness.is_ready:
        raise UiApiError(
            code="not_ready",
            message=(
                "The RAG API is not ready. Check backend readiness and "
                "dependencies."
            ),
            status_code=503,
        )
    if readiness.knowledge_mode != request.knowledge_mode:
        raise UiApiError(
            code="readiness_mismatch",
            message="Backend readiness did not match the selected knowledge mode.",
            status_code=503,
        )
    return api.submit_conversation_message(conversation_id, request)


def _chat_input_prompt(knowledge_mode: KnowledgeMode) -> str:
    return {
        "fdd": "Ask a grounded question about documented functionality",
        "code": "Ask about visible custom-code structure or behavior",
        "combined": "Ask about documented functionality and visible implementation",
    }[knowledge_mode]


def _select_active_conversation_id(
    current_id: str | None,
    conversations: Sequence[ConversationResponse],
) -> str | None:
    ids = {
        conversation.conversation_id for conversation in conversations
    }
    if current_id in ids:
        return current_id
    if conversations:
        return conversations[0].conversation_id
    return None


def _find_unanswered_user_sequences(
    messages: Sequence[ConversationMessageResponse],
) -> set[int]:
    unanswered: set[int] = set()
    for index, message in enumerate(messages):
        if message.role != "user":
            continue
        next_message = (
            messages[index + 1] if index + 1 < len(messages) else None
        )
        if next_message is None or next_message.role != "assistant":
            unanswered.add(message.sequence_number)
    return unanswered


def _conversation_label(conversation: ConversationResponse) -> str:
    marker = "Archived · " if conversation.is_archived else ""
    return f"{marker}{conversation.title}"


def _cache_turn(state: Any, turn: ConversationTurnResponse) -> None:
    debug_by_message_id = dict(state.get(TURN_DEBUG_KEY, {}))
    debug_by_message_id[turn.assistant_message.message_id] = turn
    state[TURN_DEBUG_KEY] = debug_by_message_id


def _render_safe_error(st: Any, error: Exception) -> None:
    if isinstance(error, UiApiError):
        st.error(str(error))
        recovery = {
            "timeout": "Retry, or increase the request timeout.",
            "unavailable": (
                "Confirm the FastAPI backend URL and process, then retry."
            ),
            "not_ready": (
                "Run the readiness check and repair the failed dependency."
            ),
            "knowledge_mode_unavailable": (
                "Use Functional documents, or ask an operator to complete "
                "the deliberate code-mode activation procedure."
            ),
            "readiness_mismatch": (
                "Do not submit the request; verify backend routing and restart "
                "the services with the approved configuration."
            ),
            "not_found": (
                "Refresh the page and select an existing conversation."
            ),
            "archived": "Select an active chat or create a new one.",
            "context_too_large": (
                "Start a new chat or submit a shorter message."
            ),
        }.get(error.code)
        if recovery:
            st.caption(recovery)
        return
    st.error(
        "The UI configuration is invalid. Check the backend URL and timeout."
    )


def _optional_text(value: str) -> str | None:
    cleaned = value.strip()
    return cleaned or None


if __name__ == "__main__":
    main()
