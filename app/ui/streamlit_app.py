from __future__ import annotations

import os
from typing import Any

from pydantic import ValidationError

from app.schemas.query_api import QueryRequest, QueryResponse
from app.ui.api_client import RagApiClient, UiApiError


DEFAULT_API_BASE_URL = os.getenv("RAG_API_BASE_URL", "http://127.0.0.1:8000")


def main() -> None:
    import streamlit as st

    st.set_page_config(
        page_title="Culling Blade Lineage RAG",
        page_icon="🔎",
        layout="wide",
    )
    st.title("Culling Blade Lineage RAG")
    st.caption(
        "Ask grounded questions about functional-specification lineage. "
        "Answers may be refused when retrieved evidence is insufficient."
    )

    api_base_url = st.sidebar.text_input("Backend API URL", value=DEFAULT_API_BASE_URL)
    timeout = st.sidebar.number_input(
        "Request timeout (seconds)",
        min_value=1.0,
        max_value=120.0,
        value=30.0,
        step=1.0,
    )
    if st.sidebar.button("Check backend", use_container_width=True):
        _render_backend_status(st, api_base_url, timeout)

    with st.form("grounded_query_form"):
        query = st.text_area(
            "Question",
            placeholder="What changed in branch reports?",
            height=100,
        )
        limit = st.number_input(
            "Retrieved evidence limit",
            min_value=1,
            max_value=20,
            value=5,
            step=1,
        )

        filter_columns = st.columns(3)
        document_family = filter_columns[0].text_input("Document family (optional)")
        release_label = filter_columns[1].text_input("Release label (optional)")
        source_kind = filter_columns[2].selectbox(
            "Source kind",
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
        submitted = st.form_submit_button(
            "Run grounded query",
            type="primary",
            use_container_width=True,
        )

    if not submitted:
        st.info("Check backend readiness, then submit a question.")
        return

    try:
        request = _build_query_request(
            query=query,
            limit=int(limit),
            document_family=document_family,
            release_label=release_label,
            source_kind=source_kind,
            use_min_top_score=use_min_top_score,
            min_top_score=float(min_top_score),
        )
    except ValidationError:
        st.error("Enter a non-blank question and valid retrieval filters.")
        return

    try:
        with st.spinner("Checking readiness and running grounded retrieval..."):
            with RagApiClient(api_base_url, timeout=float(timeout)) as api:
                response = _run_ready_query(api, request)
    except (UiApiError, ValueError) as exc:
        _render_safe_error(st, exc)
        return

    _render_query_response(st, response)


def _build_query_request(
    *,
    query: str,
    limit: int,
    document_family: str,
    release_label: str,
    source_kind: str,
    use_min_top_score: bool,
    min_top_score: float,
) -> QueryRequest:
    return QueryRequest(
        query=query,
        limit=limit,
        document_family=_optional_text(document_family),
        release_label=_optional_text(release_label),
        source_kind=None if source_kind == "Any" else source_kind,
        min_top_score=min_top_score if use_min_top_score else None,
    )


def _run_ready_query(api: RagApiClient, request: QueryRequest) -> QueryResponse:
    readiness = api.get_readiness()
    if not readiness.is_ready:
        raise UiApiError(
            code="not_ready",
            message="The RAG API is not ready. Check backend readiness and dependencies.",
            status_code=503,
        )
    return api.query(request)


def _render_backend_status(st: Any, api_base_url: str, timeout: float) -> None:
    try:
        with RagApiClient(api_base_url, timeout=float(timeout)) as api:
            health = api.get_health()
            readiness = api.get_readiness()
    except (UiApiError, ValueError) as exc:
        _render_safe_error(st, exc)
        return

    st.sidebar.success(
        f"Backend healthy · mode={health.retrieval_mode} · "
        f"ready={'yes' if readiness.is_ready else 'no'}"
    )
    for check in readiness.checks:
        icon = "✅" if check.is_ready else "❌"
        st.sidebar.caption(f"{icon} {check.name}: {check.detail}")


def _render_query_response(st: Any, response: QueryResponse) -> None:
    if response.is_answered:
        st.success("Grounded answer generated.")
    else:
        st.warning(response.refusal_reason or "The system declined to answer.")

    st.subheader("Answer")
    st.markdown(response.answer)

    metric_columns = st.columns(4)
    metric_columns[0].metric("Retrieval mode", response.retrieval_mode)
    metric_columns[1].metric("Evidence results", response.sufficiency.result_count)
    metric_columns[2].metric(
        "Top score",
        "N/A" if response.sufficiency.top_score is None else f"{response.sufficiency.top_score:.4f}",
    )
    metric_columns[3].metric("Trace ID", response.trace_id)
    st.caption(f"Evidence sufficiency: {response.sufficiency.reason}")

    st.subheader("Citations")
    if not response.citations:
        st.info("No citations were returned.")
    for index, citation in enumerate(response.citations, start=1):
        label = (
            f"C{index} · {citation.document_family or 'unknown document'} · "
            f"{citation.release_label or 'unknown release'} · score={citation.score:.4f}"
        )
        with st.expander(label):
            st.write(citation.text_preview)
            st.caption(
                f"unit={citation.unit_id} · source={citation.source_kind or 'unknown'}"
            )

    if response.usage is not None or response.cost is not None:
        with st.expander("Model usage and estimated cost"):
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
                        "model": response.cost.model,
                        "input_cost": response.cost.input_cost,
                        "output_cost": response.cost.output_cost,
                        "total_cost": response.cost.total_cost,
                        "currency": response.cost.currency,
                    }
                )


def _render_safe_error(st: Any, error: Exception) -> None:
    if isinstance(error, UiApiError):
        st.error(str(error))
        if error.code in {"timeout", "unavailable"}:
            st.caption("Confirm the FastAPI backend URL and process, then retry.")
        elif error.code == "not_ready":
            st.caption("Run the readiness check and repair the failed dependency before querying.")
        return
    st.error("The UI configuration is invalid. Check the backend URL and timeout.")


def _optional_text(value: str) -> str | None:
    cleaned = value.strip()
    return cleaned or None


if __name__ == "__main__":
    main()
