from pathlib import Path

from app.retrieval.evaluation import (
    RetrievalEvalCase,
    RetrievalEvalExpectation,
    RetrievalEvalFilters,
)
from app.retrieval.hybrid_evaluation import (
    HYBRID_OUTCOME_ALL_FAIL,
    HYBRID_OUTCOME_ALL_PASS,
    HYBRID_OUTCOME_DENSE_AND_HYBRID,
    HYBRID_OUTCOME_HYBRID_ONLY,
    build_hybrid_retrieval_eval_case_report,
    build_hybrid_retrieval_eval_report,
    classify_hybrid_outcome,
    write_hybrid_retrieval_eval_report_to_json,
)
from app.retrieval.hybrid_search import fuse_dense_and_lexical_results
from app.vectorstore.qdrant_search import QdrantSearchResult


def _result(unit_id: str, text: str, score: float = 0.9) -> QdrantSearchResult:
    return QdrantSearchResult(
        point_id=unit_id,
        score=score,
        payload={
            "unit_id": unit_id,
            "release_label": "R24",
            "source_kind": "paragraph",
            "text": text,
        },
    )


def _case(expected_text: str = "valid evidence") -> RetrievalEvalCase:
    return RetrievalEvalCase(
        case_id="case_1",
        query="valid query",
        filters=RetrievalEvalFilters(release_label="R24", source_kind="paragraph"),
        expectation=RetrievalEvalExpectation(
            expected_to_pass=True,
            min_results=1,
            expected_release_label="R24",
            expected_source_kind="paragraph",
            expected_top1_contains_any=[expected_text],
            expected_text_contains_any=[expected_text],
        ),
    )


def test_classify_hybrid_outcome() -> None:
    assert classify_hybrid_outcome(True, True, True) == HYBRID_OUTCOME_ALL_PASS
    assert classify_hybrid_outcome(True, False, True) == HYBRID_OUTCOME_DENSE_AND_HYBRID
    assert classify_hybrid_outcome(False, False, True) == HYBRID_OUTCOME_HYBRID_ONLY
    assert classify_hybrid_outcome(False, False, False) == HYBRID_OUTCOME_ALL_FAIL


def test_build_hybrid_retrieval_eval_case_report_uses_tightened_labels() -> None:
    case = RetrievalEvalCase(
        case_id="b01_layout",
        query="B-01 report layout",
        filters=RetrievalEvalFilters(release_label="R24", source_kind="table"),
        expectation=RetrievalEvalExpectation(
            expected_to_pass=False,
            min_results=1,
            expected_release_label="R24",
            expected_source_kind="table",
            expected_text_contains_any=["B-01", "layout"],
            unsupported_evidence_contains_any=["Sample Report", ".xlsx"],
        ),
    )
    marker_result = QdrantSearchResult(
        point_id="marker",
        score=1.0,
        payload={
            "unit_id": "marker",
            "release_label": "R24",
            "source_kind": "table",
            "text": "B-01 report layout Sample Report: B-01 Branch End of Day Report.xlsx",
        },
    )

    report = build_hybrid_retrieval_eval_case_report(
        case=case,
        dense_results=[],
        lexical_results=[marker_result],
        hybrid_results=[marker_result],
    )

    assert report.hybrid_evaluation.passed is False
    assert report.hybrid_evaluation.outcome_as_expected is True
    assert report.hybrid_outcome == HYBRID_OUTCOME_ALL_FAIL


def test_build_and_write_hybrid_retrieval_eval_report(tmp_path: Path) -> None:
    case = _case()
    dense_results = [_result("dense", "valid evidence")]
    lexical_results = [_result("lexical", "wrong text")]
    hybrid_results = fuse_dense_and_lexical_results(dense_results, lexical_results)
    case_report = build_hybrid_retrieval_eval_case_report(
        case=case,
        dense_results=dense_results,
        lexical_results=lexical_results,
        hybrid_results=hybrid_results,
    )
    report = build_hybrid_retrieval_eval_report([case_report])

    assert report.total_cases == 1
    assert report.dense_passed_count == 1
    assert report.hybrid_passed_count == 1

    output_file = write_hybrid_retrieval_eval_report_to_json(
        report,
        tmp_path / "generated" / "hybrid_report.json",
    )
    assert "hybrid_evaluation" in output_file.read_text(encoding="utf-8")