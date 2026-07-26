from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from app.schemas.query_api import QueryRequest
from app.ui.api_client import UiApiError
from app.ui.streamlit_app import _build_query_request, _run_ready_query


def test_streamlit_ui_builds_validated_query_and_omits_blank_filters() -> None:
    request = _build_query_request(
        query="  What changed in branch reports?  ",
        limit=3,
        document_family=" ",
        release_label=" R24 ",
        source_kind="Any",
        use_min_top_score=False,
        min_top_score=0.25,
    )

    assert request.model_dump(exclude_none=True) == {
        "query": "What changed in branch reports?",
        "limit": 3,
        "release_label": "R24",
    }


def test_streamlit_ui_rejects_blank_query_before_api_call() -> None:
    with pytest.raises(ValidationError):
        _build_query_request(
            query=" ",
            limit=5,
            document_family="",
            release_label="",
            source_kind="Any",
            use_min_top_score=False,
            min_top_score=0.25,
        )


def test_streamlit_ui_checks_readiness_before_query() -> None:
    api = _FakeApi(is_ready=True)
    request = QueryRequest(query="branch reports")

    response = _run_ready_query(api, request)

    assert response is api.response
    assert api.calls == ["ready", "query"]
    assert api.query_request == request


def test_streamlit_ui_blocks_query_when_backend_is_not_ready() -> None:
    api = _FakeApi(is_ready=False)

    with pytest.raises(UiApiError) as exc_info:
        _run_ready_query(api, QueryRequest(query="branch reports"))

    assert exc_info.value.code == "not_ready"
    assert api.calls == ["ready"]


def test_readme_documents_streamlit_run_command_and_dependency() -> None:
    readme = Path("README.md").read_text(encoding="utf-8")
    pyproject = Path("pyproject.toml").read_text(encoding="utf-8")

    assert "uv run --locked streamlit run app/ui/streamlit_app.py" in readme
    assert '"streamlit>=1.60,<2"' in pyproject


class _FakeApi:
    def __init__(self, *, is_ready: bool) -> None:
        self.is_ready = is_ready
        self.calls: list[str] = []
        self.query_request: QueryRequest | None = None
        self.response = SimpleNamespace(is_answered=True)

    def get_readiness(self):
        self.calls.append("ready")
        return SimpleNamespace(is_ready=self.is_ready)

    def query(self, request: QueryRequest):
        self.calls.append("query")
        self.query_request = request
        return self.response
