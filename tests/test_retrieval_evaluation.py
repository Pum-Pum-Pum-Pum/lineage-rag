from pathlib import Path

from app.retrieval.evaluation import (
    RetrievalEvalCaseReport,
    RetrievalEvalCase,
    RetrievalEvalExpectation,
    RetrievalEvalFilters,
    build_retrieval_eval_report,
    evaluate_retrieval_results,
    load_retrieval_eval_cases,
    serialize_search_result,
    write_retrieval_eval_report_to_json,
)
from app.vectorstore.qdrant_search import QdrantSearchResult


def test_load_retrieval_eval_cases(tmp_path: Path) -> None:
    eval_file = tmp_path / "eval.json"
    eval_file.write_text(
        """
        [
          {
            "case_id": "case_1",
            "query": "branch report",
            "filters": {"release_label": "R24"},
            "expectation": {"min_results": 1, "expected_release_label": "R24"}
          }
        ]
        """,
        encoding="utf-8",
    )

    cases = load_retrieval_eval_cases(eval_file)

    assert len(cases) == 1
    assert cases[0].case_id == "case_1"
    assert cases[0].filters.release_label == "R24"
    assert cases[0].expectation.min_results == 1


def test_load_retrieval_eval_cases_supports_marker_and_unsupported_expectations(tmp_path: Path) -> None:
    eval_file = tmp_path / "eval.json"
    eval_file.write_text(
        """
        [
          {
            "case_id": "case_1",
            "query": "B-01 report layout",
            "filters": {"release_label": "R24"},
            "expectation": {
              "expected_to_pass": false,
              "expected_marker_contains_any": ["B-01"],
              "unsupported_evidence_contains_any": ["Sample Report", ".xlsx"]
            }
          }
        ]
        """,
        encoding="utf-8",
    )

    cases = load_retrieval_eval_cases(eval_file)

    assert cases[0].expectation.expected_marker_contains_any == ["B-01"]
    assert cases[0].expectation.unsupported_evidence_contains_any == ["Sample Report", ".xlsx"]


def test_evaluate_retrieval_results_passes_when_expectations_match() -> None:
    case = RetrievalEvalCase(
        case_id="case_1",
        query="branch report",
        filters=RetrievalEvalFilters(release_label="R24", source_kind="paragraph"),
        expectation=RetrievalEvalExpectation(
            expected_to_pass=True,
            min_results=1,
            expected_release_label="R24",
            expected_source_kind="paragraph",
            expected_top1_contains_any=["Branch Reports"],
            expected_text_contains_any=["Branch Reports"],
        ),
    )
    results = [
        QdrantSearchResult(
            point_id="point_1",
            score=0.9,
            payload={
                "release_label": "R24",
                "source_kind": "paragraph",
                "text": "Teller and Branch Reports Re-alignment",
            },
        )
    ]

    evaluation = evaluate_retrieval_results(case, results)

    assert evaluation.passed is True
    assert evaluation.expected_to_pass is True
    assert evaluation.outcome_as_expected is True
    assert evaluation.failures == []


def test_evaluate_retrieval_results_reports_failures() -> None:
    case = RetrievalEvalCase(
        case_id="case_1",
        query="branch report",
        filters=RetrievalEvalFilters(release_label="R24"),
        expectation=RetrievalEvalExpectation(
            expected_to_pass=True,
            min_results=1,
            expected_release_label="R24",
            expected_top1_contains_any=["Branch Reports"],
            expected_text_contains_any=["Branch Reports"],
        ),
    )
    results = [
        QdrantSearchResult(
            point_id="point_1",
            score=0.5,
            payload={"release_label": "R2", "text": "Unrelated text"},
        )
    ]

    evaluation = evaluate_retrieval_results(case, results)

    assert evaluation.passed is False
    assert evaluation.outcome_as_expected is False
    assert len(evaluation.failures) == 3


def test_evaluate_retrieval_results_supports_expected_failure_case() -> None:
    case = RetrievalEvalCase(
        case_id="unsupported_case",
        query="missing layout",
        filters=RetrievalEvalFilters(release_label="R24", source_kind="table"),
        expectation=RetrievalEvalExpectation(
            expected_to_pass=False,
            min_results=1,
            expected_release_label="R24",
            expected_source_kind="table",
            expected_top1_contains_any=["B-01"],
            expected_text_contains_any=["B-01"],
        ),
    )
    results = [
        QdrantSearchResult(
            point_id="point_1",
            score=0.5,
            payload={"release_label": "R24", "source_kind": "table", "text": "Traceability Matrix"},
        )
    ]

    evaluation = evaluate_retrieval_results(case, results)

    assert evaluation.passed is False
    assert evaluation.expected_to_pass is False
    assert evaluation.outcome_as_expected is True


def test_evaluate_retrieval_results_requires_all_coverage_markers() -> None:
    case = RetrievalEvalCase(
        case_id="multi_fact_case",
        query="current teller and branch report counts",
        filters=RetrievalEvalFilters(release_label="R24"),
        expectation=RetrievalEvalExpectation(
            expected_text_contains_all=[
                "Currently a total of 23 reports",
                "T-2 and T-3 will be consolidated into T-1",
                "B-17 PNB Group Employee Transaction Report",
            ],
        ),
    )
    incomplete_results = [
        QdrantSearchResult(
            point_id="existing",
            score=0.9,
            payload={
                "release_label": "R24",
                "text": "Currently a total of 23 reports",
            },
        ),
        QdrantSearchResult(
            point_id="teller",
            score=0.8,
            payload={
                "release_label": "R24",
                "text": "T-2 and T-3 will be consolidated into T-1",
            },
        ),
    ]

    evaluation = evaluate_retrieval_results(case, incomplete_results)

    assert evaluation.passed is False
    assert "B-17 PNB Group Employee Transaction Report" in evaluation.failures[0]


def test_evaluate_retrieval_results_flags_unsupported_marker_only_evidence() -> None:
    case = RetrievalEvalCase(
        case_id="unsupported_attachment_case",
        query="B-01 report layout",
        filters=RetrievalEvalFilters(release_label="R24", source_kind="table"),
        expectation=RetrievalEvalExpectation(
            expected_to_pass=False,
            min_results=1,
            expected_release_label="R24",
            expected_source_kind="table",
            expected_marker_contains_any=["B-01", "layout"],
            expected_top1_contains_any=["B-01", "layout"],
            expected_text_contains_any=["B-01", "layout"],
            unsupported_evidence_contains_any=["Sample Report", ".xlsx"],
        ),
    )
    results = [
        QdrantSearchResult(
            point_id="point_1",
            score=0.9,
            payload={
                "release_label": "R24",
                "source_kind": "table",
                "text": "B-01 report layout Sample Report: B-01 Branch End of Day Report.xlsx",
            },
        )
    ]

    evaluation = evaluate_retrieval_results(case, results)

    assert evaluation.passed is False
    assert evaluation.expected_to_pass is False
    assert evaluation.outcome_as_expected is True
    assert any("unsupported marker/reference-only" in failure for failure in evaluation.failures)


def test_build_and_write_retrieval_eval_report(tmp_path: Path) -> None:
    case = RetrievalEvalCase(
        case_id="case_1",
        query="branch report",
        filters=RetrievalEvalFilters(release_label="R24"),
        expectation=RetrievalEvalExpectation(min_results=1),
    )
    result = QdrantSearchResult(
        point_id="point_1",
        score=0.9,
        payload={"release_label": "R24", "text": "Branch report evidence"},
    )
    evaluation = evaluate_retrieval_results(case, [result])
    report = build_retrieval_eval_report(
        [
            RetrievalEvalCaseReport(
                case=case,
                evaluation=evaluation,
                top_results=[serialize_search_result(result)],
            )
        ]
    )

    assert report.total_cases == 1
    assert report.passed_count == 1
    assert report.expected_outcome_count == 1

    output_file = write_retrieval_eval_report_to_json(
        report,
        tmp_path / "generated" / "retrieval_eval_report.json",
    )

    loaded_cases = output_file.read_text(encoding="utf-8")
    assert "Branch report evidence" in loaded_cases
