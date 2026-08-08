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


def test_current_query_does_not_filter_to_a_historical_release_mention() -> None:
    plan = build_temporal_query_plan(
        "Which current release contains the Teller and Branch Reports Re-alignment "
        "change, rather than the original R2 report specifications?"
    )

    scoped, updated = scope_results_to_temporal_plan(
        [_result("r2-baseline", "R2"), _result("r24-change", "R24")],
        plan,
        limit=5,
    )

    assert plan.is_current_state is True
    assert plan.effective_release_label is None
    assert plan.release_source is None
    assert plan.referenced_release_labels == ("R2",)
    assert updated.effective_release_label == "R24"
    assert updated.release_source == "retrieved_candidates"
    assert [item.point_id for item in scoped] == ["r2-baseline", "r24-change"]


def test_current_query_preserves_explicit_historical_and_latest_release_evidence() -> None:
    plan = build_temporal_query_plan(
        "Which release should answer whether an R2 branch report is produced in "
        "PDF, and which release should answer the current T-1 report name?"
    )

    scoped, updated = scope_results_to_temporal_plan(
        [
            _result("r1-unrelated", "R1"),
            _result("r2-pdf", "R2"),
            _result("r24-t1", "R24"),
        ],
        plan,
        limit=5,
    )

    assert updated.effective_release_label == "R24"
    assert updated.referenced_release_labels == ("R2",)
    assert [item.point_id for item in scoped] == ["r2-pdf", "r24-t1"]


def test_current_query_ignores_one_weak_newer_release_outlier() -> None:
    plan = build_temporal_query_plan(
        "What is the current rule for changing a unit holder from Deceased to Normal?"
    )

    scoped, updated = scope_results_to_temporal_plan(
        [
            _result("r21-1", "R21"),
            _result("r21-2", "R21"),
            _result("r21-3", "R21"),
            _result("r21-4", "R21"),
            _result("r21-5", "R21"),
            _result("r24-unrelated", "R24"),
        ],
        plan,
        limit=10,
    )

    assert updated.effective_release_label == "R21"
    assert [item.point_id for item in scoped] == [
        "r21-1",
        "r21-2",
        "r21-3",
        "r21-4",
        "r21-5",
    ]


def test_current_query_preserves_implicit_original_and_current_evidence() -> None:
    plan = build_temporal_query_plan(
        "Is the original branch report produced in PDF, and what is the current T-1 name?"
    )

    scoped, updated = scope_results_to_temporal_plan(
        [
            _result("r24-current", "R24"),
            _result("r2-original", "R2"),
            _result("r24-layout", "R24"),
            _result("r2-pdf", "R2"),
        ],
        plan,
        limit=10,
    )

    assert plan.historical_context_requested is True
    assert updated.effective_release_label == "R24"
    assert [item.point_id for item in scoped] == [
        "r24-current",
        "r2-original",
        "r24-layout",
        "r2-pdf",
    ]


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
