from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from app.vectorstore.qdrant_search import QdrantSearchResult


@dataclass(frozen=True)
class RetrievalEvalFilters:
    document_family: str | None = None
    release_label: str | None = None
    source_kind: str | None = None


@dataclass(frozen=True)
class RetrievalEvalExpectation:
    expected_to_pass: bool = True
    min_results: int = 1
    expected_release_label: str | None = None
    expected_source_kind: str | None = None
    expected_top1_contains_any: list[str] | None = None
    expected_text_contains_any: list[str] | None = None


@dataclass(frozen=True)
class RetrievalEvalCase:
    case_id: str
    query: str
    filters: RetrievalEvalFilters
    expectation: RetrievalEvalExpectation
    notes: str = ""


@dataclass(frozen=True)
class RetrievalEvalResult:
    case_id: str
    passed: bool
    expected_to_pass: bool
    outcome_as_expected: bool
    result_count: int
    failures: list[str]


@dataclass(frozen=True)
class RetrievalEvalCaseReport:
    case: RetrievalEvalCase
    evaluation: RetrievalEvalResult
    top_results: list[dict[str, Any]]


@dataclass(frozen=True)
class RetrievalEvalReport:
    total_cases: int
    passed_count: int
    expected_outcome_count: int
    cases: list[RetrievalEvalCaseReport]


def load_retrieval_eval_cases(path: str | Path) -> list[RetrievalEvalCase]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    cases: list[RetrievalEvalCase] = []

    for item in payload:
        filters_payload = item.get("filters", {})
        expectation_payload = item.get("expectation", {})
        cases.append(
            RetrievalEvalCase(
                case_id=item["case_id"],
                query=item["query"],
                filters=RetrievalEvalFilters(**filters_payload),
                expectation=RetrievalEvalExpectation(**expectation_payload),
                notes=item.get("notes", ""),
            )
        )

    return cases


def evaluate_retrieval_results(
    case: RetrievalEvalCase,
    results: list[QdrantSearchResult],
) -> RetrievalEvalResult:
    failures: list[str] = []
    expectation = case.expectation

    if len(results) < expectation.min_results:
        failures.append(
            f"Expected at least {expectation.min_results} results, got {len(results)}"
        )

    if expectation.expected_release_label is not None:
        bad_releases = [
            result.payload.get("release_label")
            for result in results
            if result.payload.get("release_label") != expectation.expected_release_label
        ]
        if bad_releases:
            failures.append(
                "Found results with unexpected release_label values: "
                f"{bad_releases}"
            )

    if expectation.expected_source_kind is not None:
        bad_sources = [
            result.payload.get("source_kind")
            for result in results
            if result.payload.get("source_kind") != expectation.expected_source_kind
        ]
        if bad_sources:
            failures.append(
                "Found results with unexpected source_kind values: "
                f"{bad_sources}"
            )

    if expectation.expected_text_contains_any:
        combined_text = "\n".join(str(result.payload.get("text", "")) for result in results)
        if not any(expected_text in combined_text for expected_text in expectation.expected_text_contains_any):
            failures.append(
                "None of the expected text markers were found: "
                f"{expectation.expected_text_contains_any}"
            )

    if expectation.expected_top1_contains_any:
        if not results:
            failures.append(
                "Top-1 expectation could not be checked because no results were returned."
            )
        else:
            top1_text = str(results[0].payload.get("text", ""))
            if not any(expected_text in top1_text for expected_text in expectation.expected_top1_contains_any):
                failures.append(
                    "Top-1 result did not contain expected text markers: "
                    f"{expectation.expected_top1_contains_any}"
                )

    passed = not failures

    return RetrievalEvalResult(
        case_id=case.case_id,
        passed=passed,
        expected_to_pass=expectation.expected_to_pass,
        outcome_as_expected=passed == expectation.expected_to_pass,
        result_count=len(results),
        failures=failures,
    )


def serialize_search_result(result: QdrantSearchResult) -> dict[str, Any]:
    """Convert one Qdrant search result into JSON-friendly output."""

    return {
        "point_id": result.point_id,
        "score": result.score,
        "payload": result.payload,
    }


def build_retrieval_eval_report(
    case_reports: list[RetrievalEvalCaseReport],
) -> RetrievalEvalReport:
    """Build a summary report from evaluated retrieval cases."""

    return RetrievalEvalReport(
        total_cases=len(case_reports),
        passed_count=sum(int(report.evaluation.passed) for report in case_reports),
        expected_outcome_count=sum(
            int(report.evaluation.outcome_as_expected) for report in case_reports
        ),
        cases=case_reports,
    )


def write_retrieval_eval_report_to_json(
    report: RetrievalEvalReport,
    output_path: str | Path,
) -> Path:
    """Persist a retrieval evaluation report as JSON."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(asdict(report), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return path
