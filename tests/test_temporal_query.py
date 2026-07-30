from app.retrieval.temporal_query import (
    build_temporal_query_plan,
    latest_release_label,
    scope_results_to_temporal_plan,
)
from app.vectorstore.qdrant_search import QdrantSearchResult


def _result(unit_id: str, release_label: str) -> QdrantSearchResult:
    return QdrantSearchResult(
        point_id=unit_id,
        score=0.8,
        payload={
            "unit_id": unit_id,
            "release_label": release_label,
            "source_kind": "paragraph",
            "text": unit_id,
        },
    )


def test_current_query_uses_latest_retrieved_release_numerically() -> None:
    plan = build_temporal_query_plan(
        "How many teller and branch reports are there currently?"
    )

    scoped, updated = scope_results_to_temporal_plan(
        [
            _result("r2-background", "R2"),
            _result("r24-existing", "R24"),
            _result("r24-teller-table", "R24"),
            _result("r24-branch-table", "R24"),
        ],
        plan,
        limit=5,
    )

    assert updated.is_current_state is True
    assert updated.effective_release_label == "R24"
    assert updated.release_source == "retrieved_candidates"
    assert [item.point_id for item in scoped] == [
        "r24-existing",
        "r24-teller-table",
        "r24-branch-table",
    ]
    assert "resulting production state" in plan.retrieval_query


def test_referential_query_inherits_release_from_conversation_context() -> None:
    plan = build_temporal_query_plan(
        "Give me a summary of it. What is current?",
        conversation_context=(
            '<message role="user">What changed in R24?</message>'
            '<message role="assistant">R24 realigned reports.</message>'
        ),
    )

    assert plan.effective_release_label == "R24"
    assert plan.release_source == "conversation_context"
    assert "Resolved conversation release: R24" in plan.retrieval_query


def test_explicit_request_filter_overrides_query_and_context() -> None:
    plan = build_temporal_query_plan(
        "Compare it with R24",
        requested_release_label="R2",
        conversation_context="<conversation_memory>R24</conversation_memory>",
    )

    assert plan.effective_release_label == "R2"
    assert plan.release_source == "request_filter"


def test_non_current_query_does_not_auto_select_latest_release() -> None:
    plan = build_temporal_query_plan("What changed in branch reports?")
    scoped, updated = scope_results_to_temporal_plan(
        [_result("r2", "R2"), _result("r24", "R24")],
        plan,
        limit=1,
    )

    assert updated.effective_release_label is None
    assert [item.point_id for item in scoped] == ["r2"]


def test_current_application_date_is_not_latest_release_intent() -> None:
    plan = build_temporal_query_plan(
        "How is the current application date shown in the teller report?"
    )

    assert plan.is_current_state is False
    assert "resulting production state" not in plan.retrieval_query


def test_release_order_is_numeric_not_lexicographic() -> None:
    assert latest_release_label(["R2", "R10", "R9"]) == "R10"
