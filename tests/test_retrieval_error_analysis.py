from pathlib import Path

from app.retrieval.retrieval_error_analysis import (
    LABEL_DENSE_EXACT_IDENTIFIER_MISS,
    LABEL_EXPECTED_UNSUPPORTED_EVIDENCE,
    LABEL_EXPECTED_UNANSWERABLE,
    LABEL_LEXICAL_EXACT_TERM_FALSE_POSITIVE,
    LABEL_MARKER_MATCH_NOT_FULL_EVIDENCE,
    LABEL_UNSUPPORTED_ATTACHMENT_MARKER_MATCH,
    LABEL_WEAK_MARKER_EXPECTATION,
    SEVERITY_HIGH,
    SEVERITY_LOW,
    SEVERITY_MEDIUM,
    analyze_retrieval_comparison_report,
    write_retrieval_error_analysis_report_to_json,
)


def _case_payload(
    case_id: str,
    query: str,
    comparison_outcome: str,
    expected_to_pass: bool,
    dense_text: str,
    lexical_text: str,
    lexical_failures: list[str] | None = None,
    unsupported_evidence_contains_any: list[str] | None = None,
) -> dict:
    return {
        "case": {
            "case_id": case_id,
            "query": query,
            "filters": {"release_label": "R24"},
            "expectation": {
                "expected_to_pass": expected_to_pass,
                "unsupported_evidence_contains_any": unsupported_evidence_contains_any,
            },
            "notes": "Expected attachment limitation" if "b01" in case_id else "",
        },
        "dense_evaluation": {
            "case_id": case_id,
            "passed": comparison_outcome in {"dense_only", "both_pass"},
            "expected_to_pass": expected_to_pass,
            "outcome_as_expected": True,
            "result_count": 1,
            "failures": [],
        },
        "lexical_evaluation": {
            "case_id": case_id,
            "passed": comparison_outcome in {"lexical_only", "both_pass"},
            "expected_to_pass": expected_to_pass,
            "outcome_as_expected": True,
            "result_count": 1,
            "failures": lexical_failures or [],
        },
        "comparison_outcome": comparison_outcome,
        "dense_top_results": [
            {
                "point_id": "dense-point",
                "score": 0.5,
                "payload": {"unit_id": "dense-unit", "text": dense_text},
            }
        ],
        "lexical_top_results": [
            {
                "point_id": "lexical-point",
                "score": 10.0,
                "payload": {"unit_id": "lexical-unit", "text": lexical_text},
            }
        ],
    }


def test_analyze_retrieval_comparison_report_labels_dense_only_lexical_false_positive() -> None:
    report = analyze_retrieval_comparison_report(
        {
            "total_cases": 1,
            "cases": [
                _case_payload(
                    case_id="realignment_summary",
                    query="branch reports realignment",
                    comparison_outcome="dense_only",
                    expected_to_pass=True,
                    dense_text="Requirements Summary multiple Branch reports",
                    lexical_text="Annexure Branch reports layout document",
                    lexical_failures=["Top-1 result did not contain expected text markers"],
                )
            ],
        }
    )

    case = report.cases[0]
    assert case.severity == SEVERITY_MEDIUM
    assert LABEL_LEXICAL_EXACT_TERM_FALSE_POSITIVE in case.root_cause_labels
    assert report.label_counts[LABEL_LEXICAL_EXACT_TERM_FALSE_POSITIVE] == 1


def test_analyze_retrieval_comparison_report_labels_attachment_marker_match() -> None:
    report = analyze_retrieval_comparison_report(
        {
            "total_cases": 1,
            "cases": [
                _case_payload(
                    case_id="b01_layout",
                    query="B-01 report layout",
                    comparison_outcome="lexical_only",
                    expected_to_pass=False,
                    dense_text="Traceability Matrix",
                    lexical_text="B-01 report layout will be changed as per attached sample report",
                )
            ],
        }
    )

    labels = set(report.cases[0].root_cause_labels)
    assert report.cases[0].severity == SEVERITY_HIGH
    assert LABEL_DENSE_EXACT_IDENTIFIER_MISS in labels
    assert LABEL_UNSUPPORTED_ATTACHMENT_MARKER_MATCH in labels
    assert LABEL_MARKER_MATCH_NOT_FULL_EVIDENCE in labels
    assert LABEL_WEAK_MARKER_EXPECTATION in labels


def test_analyze_retrieval_comparison_report_labels_expected_unanswerable() -> None:
    report = analyze_retrieval_comparison_report(
        {
            "total_cases": 1,
            "cases": [
                _case_payload(
                    case_id="mobile_login",
                    query="mobile app login flow",
                    comparison_outcome="both_fail",
                    expected_to_pass=False,
                    dense_text="Unrelated approval table",
                    lexical_text="Unrelated background paragraph",
                )
            ],
        }
    )

    assert report.cases[0].severity == SEVERITY_LOW
    assert report.cases[0].root_cause_labels == [LABEL_EXPECTED_UNANSWERABLE]


def test_analyze_retrieval_comparison_report_labels_expected_unsupported_evidence() -> None:
    report = analyze_retrieval_comparison_report(
        {
            "total_cases": 1,
            "cases": [
                _case_payload(
                    case_id="b01_layout",
                    query="B-01 report layout",
                    comparison_outcome="both_fail",
                    expected_to_pass=False,
                    dense_text="Traceability Matrix",
                    lexical_text="B-01 report layout Sample Report: B-01 Branch End of Day Report.xlsx",
                    unsupported_evidence_contains_any=["Sample Report", ".xlsx"],
                )
            ],
        }
    )

    labels = set(report.cases[0].root_cause_labels)
    assert report.cases[0].severity == SEVERITY_HIGH
    assert LABEL_EXPECTED_UNSUPPORTED_EVIDENCE in labels
    assert LABEL_UNSUPPORTED_ATTACHMENT_MARKER_MATCH in labels


def test_analyze_retrieval_comparison_report_excludes_both_pass_by_default() -> None:
    report = analyze_retrieval_comparison_report(
        {
            "total_cases": 1,
            "cases": [
                _case_payload(
                    case_id="both_pass",
                    query="branch report",
                    comparison_outcome="both_pass",
                    expected_to_pass=True,
                    dense_text="Branch report evidence",
                    lexical_text="Branch report evidence",
                )
            ],
        }
    )

    assert report.total_cases == 1
    assert report.analyzed_case_count == 0
    assert report.cases == []


def test_write_retrieval_error_analysis_report_to_json(tmp_path: Path) -> None:
    report = analyze_retrieval_comparison_report(
        {
            "total_cases": 1,
            "cases": [
                _case_payload(
                    case_id="mobile_login",
                    query="mobile app login flow",
                    comparison_outcome="both_fail",
                    expected_to_pass=False,
                    dense_text="Unrelated approval table",
                    lexical_text="Unrelated background paragraph",
                )
            ],
        }
    )

    output_file = write_retrieval_error_analysis_report_to_json(
        report,
        tmp_path / "generated" / "retrieval_error_analysis_report.json",
    )

    assert "expected_unanswerable" in output_file.read_text(encoding="utf-8")