from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

from app.retrieval.evaluation import (
    RetrievalEvalCase,
    RetrievalEvalResult,
    evaluate_retrieval_results,
    serialize_search_result,
)
from app.vectorstore.qdrant_search import QdrantSearchResult


COMPARISON_BOTH_PASS = "both_pass"
COMPARISON_DENSE_ONLY = "dense_only"
COMPARISON_LEXICAL_ONLY = "lexical_only"
COMPARISON_BOTH_FAIL = "both_fail"


@dataclass(frozen=True)
class RetrievalComparisonCaseReport:
    case: RetrievalEvalCase
    dense_evaluation: RetrievalEvalResult
    lexical_evaluation: RetrievalEvalResult
    comparison_outcome: str
    dense_top_results: list[dict[str, Any]]
    lexical_top_results: list[dict[str, Any]]


@dataclass(frozen=True)
class RetrievalComparisonReport:
    total_cases: int
    dense_passed_count: int
    lexical_passed_count: int
    both_pass_count: int
    dense_only_count: int
    lexical_only_count: int
    both_fail_count: int
    cases: list[RetrievalComparisonCaseReport]


def classify_retrieval_comparison(
    dense_passed: bool,
    lexical_passed: bool,
) -> str:
    """Classify how dense and lexical retrieval performed for one eval case."""

    if dense_passed and lexical_passed:
        return COMPARISON_BOTH_PASS
    if dense_passed:
        return COMPARISON_DENSE_ONLY
    if lexical_passed:
        return COMPARISON_LEXICAL_ONLY
    return COMPARISON_BOTH_FAIL


def build_retrieval_comparison_case_report(
    case: RetrievalEvalCase,
    dense_results: Sequence[Any],
    lexical_results: Sequence[Any],
) -> RetrievalComparisonCaseReport:
    """Evaluate one case for dense and lexical retrieval and compare outcomes."""

    normalized_dense_results = normalize_search_results(dense_results)
    normalized_lexical_results = normalize_search_results(lexical_results)

    dense_evaluation = evaluate_retrieval_results(case, normalized_dense_results)
    lexical_evaluation = evaluate_retrieval_results(case, normalized_lexical_results)
    comparison_outcome = classify_retrieval_comparison(
        dense_passed=dense_evaluation.passed,
        lexical_passed=lexical_evaluation.passed,
    )

    return RetrievalComparisonCaseReport(
        case=case,
        dense_evaluation=dense_evaluation,
        lexical_evaluation=lexical_evaluation,
        comparison_outcome=comparison_outcome,
        dense_top_results=[serialize_search_result(result) for result in normalized_dense_results],
        lexical_top_results=[serialize_search_result(result) for result in normalized_lexical_results],
    )


def build_retrieval_comparison_report(
    case_reports: list[RetrievalComparisonCaseReport],
) -> RetrievalComparisonReport:
    """Build aggregate dense-vs-lexical comparison counts."""

    return RetrievalComparisonReport(
        total_cases=len(case_reports),
        dense_passed_count=sum(int(report.dense_evaluation.passed) for report in case_reports),
        lexical_passed_count=sum(int(report.lexical_evaluation.passed) for report in case_reports),
        both_pass_count=sum(
            int(report.comparison_outcome == COMPARISON_BOTH_PASS)
            for report in case_reports
        ),
        dense_only_count=sum(
            int(report.comparison_outcome == COMPARISON_DENSE_ONLY)
            for report in case_reports
        ),
        lexical_only_count=sum(
            int(report.comparison_outcome == COMPARISON_LEXICAL_ONLY)
            for report in case_reports
        ),
        both_fail_count=sum(
            int(report.comparison_outcome == COMPARISON_BOTH_FAIL)
            for report in case_reports
        ),
        cases=case_reports,
    )


def write_retrieval_comparison_report_to_json(
    report: RetrievalComparisonReport,
    output_path: str | Path,
) -> Path:
    """Persist a dense-vs-lexical comparison report as JSON."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(asdict(report), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return path


def normalize_search_results(results: Sequence[Any]) -> list[QdrantSearchResult]:
    """Normalize dense or lexical search result objects to the shared eval shape."""

    normalized_results: list[QdrantSearchResult] = []
    for result in results:
        normalized_results.append(
            QdrantSearchResult(
                point_id=str(result.point_id),
                score=float(result.score),
                payload=dict(result.payload),
            )
        )
    return normalized_results