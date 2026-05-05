from pathlib import Path

from app.retrieval.evaluation import (
    RetrievalEvalCase,
    RetrievalEvalExpectation,
    RetrievalEvalFilters,
)
from app.retrieval.retrieval_comparison import (
    COMPARISON_BOTH_FAIL,
    COMPARISON_BOTH_PASS,
    COMPARISON_DENSE_ONLY,
    COMPARISON_LEXICAL_ONLY,
    build_retrieval_comparison_case_report,
    build_retrieval_comparison_report,
    classify_retrieval_comparison,
    write_retrieval_comparison_report_to_json,
)
from app.vectorstore.qdrant_search import QdrantSearchResult


def _result(
    text: str,
    unit_id: str = "unit-1",
    score: float = 0.9,
    release_label: str = "R24",
    source_kind: str = "paragraph",
) -> QdrantSearchResult:
    return QdrantSearchResult(
        point_id=unit_id,
        score=score,
        payload={
            "unit_id": unit_id,
            "release_label": release_label,
            "source_kind": source_kind,
            "text": text,
        },
    )


def _case(expected_text: str = "B-01 report layout") -> RetrievalEvalCase:
    return RetrievalEvalCase(
        case_id="b01_layout",
        query="B-01 report layout",
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


def test_classify_retrieval_comparison() -> None:
    assert classify_retrieval_comparison(True, True) == COMPARISON_BOTH_PASS
    assert classify_retrieval_comparison(True, False) == COMPARISON_DENSE_ONLY
    assert classify_retrieval_comparison(False, True) == COMPARISON_LEXICAL_ONLY
    assert classify_retrieval_comparison(False, False) == COMPARISON_BOTH_FAIL


def test_build_retrieval_comparison_case_report_detects_lexical_only_win() -> None:
    report = build_retrieval_comparison_case_report(
        case=_case(),
        dense_results=[_result("Generic Branch report text", unit_id="dense-generic")],
        lexical_results=[_result("B-01 report layout marker", unit_id="lexical-b01")],
    )

    assert report.dense_evaluation.passed is False
    assert report.lexical_evaluation.passed is True
    assert report.comparison_outcome == COMPARISON_LEXICAL_ONLY
    assert report.lexical_top_results[0]["payload"]["unit_id"] == "lexical-b01"


def test_build_retrieval_comparison_report_counts_outcomes() -> None:
    both_pass = build_retrieval_comparison_case_report(
        case=_case("Branch report evidence"),
        dense_results=[_result("Branch report evidence")],
        lexical_results=[_result("Branch report evidence")],
    )
    dense_only = build_retrieval_comparison_case_report(
        case=_case("dense marker"),
        dense_results=[_result("dense marker")],
        lexical_results=[_result("wrong text")],
    )
    lexical_only = build_retrieval_comparison_case_report(
        case=_case("lexical marker"),
        dense_results=[_result("wrong text")],
        lexical_results=[_result("lexical marker")],
    )
    both_fail = build_retrieval_comparison_case_report(
        case=_case("missing marker"),
        dense_results=[_result("wrong dense text")],
        lexical_results=[_result("wrong lexical text")],
    )

    report = build_retrieval_comparison_report(
        [both_pass, dense_only, lexical_only, both_fail]
    )

    assert report.total_cases == 4
    assert report.dense_passed_count == 2
    assert report.lexical_passed_count == 2
    assert report.both_pass_count == 1
    assert report.dense_only_count == 1
    assert report.lexical_only_count == 1
    assert report.both_fail_count == 1


def test_write_retrieval_comparison_report_to_json(tmp_path: Path) -> None:
    case_report = build_retrieval_comparison_case_report(
        case=_case(),
        dense_results=[_result("Generic Branch report text")],
        lexical_results=[_result("B-01 report layout marker")],
    )
    report = build_retrieval_comparison_report([case_report])

    output_file = write_retrieval_comparison_report_to_json(
        report,
        tmp_path / "generated" / "retrieval_comparison_report.json",
    )

    output_text = output_file.read_text(encoding="utf-8")
    assert "lexical_only" in output_text
    assert "B-01 report layout marker" in output_text