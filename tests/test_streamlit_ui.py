from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from app.schemas.conversation_api import (
    ConversationMessageResponse,
    ConversationResponse,
)
from app.ui.api_client import UiApiError
from app.ui.streamlit_app import (
    _build_message_request,
    _chat_input_prompt,
    _find_unanswered_user_sequences,
    _run_ready_turn,
    _select_active_conversation_id,
)


NOW = datetime(2026, 7, 29, tzinfo=UTC)


def test_streamlit_ui_builds_validated_message_and_omits_blank_filters() -> None:
    request = _build_message_request(
        content="  What changed in branch reports?  ",
        knowledge_mode="fdd",
        analysis_kind="explanation",
        limit=3,
        document_family=" ",
        release_label=" R24 ",
        source_kind="Any",
        use_min_top_score=False,
        min_top_score=0.25,
    )

    assert request.model_dump(exclude_none=True) == {
        "content": "What changed in branch reports?",
        "knowledge_mode": "fdd",
        "analysis_kind": "explanation",
        "limit": 3,
        "release_label": "R24",
    }


def test_streamlit_ui_rejects_blank_message_before_api_call() -> None:
    with pytest.raises(ValidationError):
        _build_message_request(
            content=" ",
            knowledge_mode="fdd",
            analysis_kind="explanation",
            limit=5,
            document_family="",
            release_label="",
            source_kind="Any",
            use_min_top_score=False,
            min_top_score=0.25,
        )


def test_streamlit_ui_checks_readiness_before_submitting_turn() -> None:
    api = _FakeApi(is_ready=True)
    request = _build_message_request(
        content="branch reports",
        knowledge_mode="fdd",
        analysis_kind="explanation",
        limit=5,
        document_family="",
        release_label="",
        source_kind="Any",
        use_min_top_score=False,
        min_top_score=0.25,
    )

    response = _run_ready_turn(api, "conversation-1", request)

    assert response is api.response
    assert api.calls == ["ready", "submit"]
    assert api.conversation_id == "conversation-1"
    assert api.message_request == request


def test_streamlit_ui_blocks_turn_when_backend_is_not_ready() -> None:
    api = _FakeApi(is_ready=False)
    request = _build_message_request(
        content="branch reports",
        knowledge_mode="fdd",
        analysis_kind="explanation",
        limit=5,
        document_family="",
        release_label="",
        source_kind="Any",
        use_min_top_score=False,
        min_top_score=0.25,
    )

    with pytest.raises(UiApiError) as exc_info:
        _run_ready_turn(api, "conversation-1", request)

    assert exc_info.value.code == "not_ready"
    assert api.calls == ["ready"]


def test_streamlit_ui_preserves_explicit_code_mode_and_analysis_kind() -> None:
    request = _build_message_request(
        content="Where is the likely AML change location?",
        knowledge_mode="combined",
        analysis_kind="impact_analysis",
        limit=7,
        document_family="",
        release_label="",
        source_kind="Any",
        use_min_top_score=False,
        min_top_score=0.25,
    )

    assert request.knowledge_mode == "combined"
    assert request.analysis_kind == "impact_analysis"
    assert _chat_input_prompt("combined").startswith("Ask about documented")


def test_streamlit_ui_feature_gate_rollback_fails_closed() -> None:
    api = _FeatureGateApi(states=[False, True, False])
    request = _build_message_request(
        content="Explain visible custom code",
        knowledge_mode="code",
        analysis_kind="explanation",
        limit=5,
        document_family="",
        release_label="",
        source_kind="Any",
        use_min_top_score=False,
        min_top_score=0.25,
    )

    with pytest.raises(UiApiError, match="not ready or has not been activated"):
        _run_ready_turn(api, "conversation-1", request)
    assert _run_ready_turn(api, "conversation-1", request) is api.response
    with pytest.raises(UiApiError, match="not ready or has not been activated"):
        _run_ready_turn(api, "conversation-1", request)

    assert api.submit_count == 1
    assert api.requested_modes == ["code", "code", "code"]


def test_streamlit_ui_rejects_readiness_for_wrong_mode() -> None:
    api = _FakeApi(is_ready=True, readiness_mode="fdd")
    request = _build_message_request(
        content="Explain custom code",
        knowledge_mode="code",
        analysis_kind="explanation",
        limit=5,
        document_family="",
        release_label="",
        source_kind="Any",
        use_min_top_score=False,
        min_top_score=0.25,
    )

    with pytest.raises(UiApiError) as exc_info:
        _run_ready_turn(api, "conversation-1", request)

    assert exc_info.value.code == "readiness_mismatch"
    assert api.calls == ["ready"]


def test_streamlit_ui_selects_current_or_most_recent_conversation() -> None:
    conversations = [
        _conversation("conversation-2", "Second"),
        _conversation("conversation-1", "First"),
    ]

    assert (
        _select_active_conversation_id(
            "conversation-1",
            conversations,
        )
        == "conversation-1"
    )
    assert (
        _select_active_conversation_id("missing", conversations)
        == "conversation-2"
    )
    assert _select_active_conversation_id(None, []) is None


def test_streamlit_ui_identifies_partial_user_turns() -> None:
    messages = [
        _message(1, "user"),
        _message(2, "assistant"),
        _message(3, "user"),
        _message(4, "user"),
        _message(5, "assistant"),
        _message(6, "user"),
    ]

    assert _find_unanswered_user_sequences(messages) == {3, 6}


def test_streamlit_source_uses_chat_and_conversation_controls() -> None:
    source = Path("app/ui/streamlit_app.py").read_text(encoding="utf-8")

    assert "st.chat_message" in source
    assert "st.chat_input" in source
    assert '"New chat"' in source
    assert '"Archive active chat"' in source
    assert "api.list_conversations()" in source
    assert "api.get_conversation(active_id)" in source
    assert "api.submit_conversation_message" in source
    assert "No assistant response was persisted" in source
    assert "Evidence and debug details" in source
    assert '"Authoritative evidence"' in source
    assert "Section support" in source
    assert "FDD citations" in source
    assert "Code citations" in source
    assert "knowledge_mode_unavailable" in source


def test_readme_documents_streamlit_run_command_and_dependency() -> None:
    readme = Path("README.md").read_text(encoding="utf-8")
    pyproject = Path("pyproject.toml").read_text(encoding="utf-8")

    assert "uv run --locked streamlit run app/ui/streamlit_app.py" in readme
    assert '"streamlit>=1.60,<2"' in pyproject
    assert "**New chat**" in readme
    assert "**Archive active chat**" in readme
    assert "user-only" in readme
    assert "partial turn is rendered as retryable" in readme
    assert "only debug details returned during the current UI session" in readme


class _FakeApi:
    def __init__(self, *, is_ready: bool, readiness_mode: str | None = None) -> None:
        self.is_ready = is_ready
        self.readiness_mode = readiness_mode
        self.calls: list[str] = []
        self.conversation_id: str | None = None
        self.message_request = None
        self.response = SimpleNamespace(answer=SimpleNamespace(is_answered=True))

    def get_readiness(self, *, knowledge_mode="fdd"):
        self.calls.append("ready")
        return SimpleNamespace(
            is_ready=self.is_ready,
            knowledge_mode=self.readiness_mode or knowledge_mode,
        )

    def submit_conversation_message(self, conversation_id, request):
        self.calls.append("submit")
        self.conversation_id = conversation_id
        self.message_request = request
        return self.response


class _FeatureGateApi:
    def __init__(self, *, states: list[bool]) -> None:
        self.states = iter(states)
        self.requested_modes: list[str] = []
        self.submit_count = 0
        self.response = SimpleNamespace(answer=SimpleNamespace(is_answered=True))

    def get_readiness(self, *, knowledge_mode="fdd"):
        self.requested_modes.append(knowledge_mode)
        if not next(self.states):
            raise UiApiError(
                code="not_ready",
                message="The RAG API is not ready.",
                status_code=503,
            )
        return SimpleNamespace(is_ready=True, knowledge_mode=knowledge_mode)

    def submit_conversation_message(self, conversation_id, request):
        self.submit_count += 1
        return self.response


def _conversation(
    conversation_id: str,
    title: str,
) -> ConversationResponse:
    return ConversationResponse(
        conversation_id=conversation_id,
        title=title,
        created_at_utc=NOW,
        updated_at_utc=NOW,
        is_archived=False,
    )


def _message(
    sequence_number: int,
    role: str,
) -> ConversationMessageResponse:
    return ConversationMessageResponse(
        message_id=f"message-{sequence_number}",
        conversation_id="conversation-1",
        sequence_number=sequence_number,
        role=role,
        content=f"{role} content",
        created_at_utc=NOW,
        trace_id=None,
    )
