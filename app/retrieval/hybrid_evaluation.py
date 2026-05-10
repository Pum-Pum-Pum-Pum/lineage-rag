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
from app.retrieval.retrieval_comparison import normalize_search_results


HYBRID_OUTCOME_ALL_PASS = "all_pass"
HYBRID_OUTCOME_DENSE_AND_HYBRID = "dense_and_hybrid"
HYBRID_OUTCOME_LEXICAL_AND_HYBRID = "lexical_and_hybrid"
HYBRID_OUTCOME_HYBRID_ONLY = "hybrid_only"
HYBRID_OUTCOME_HYBRID_MISSED_BOTH = "hybrid_missed_both"
HYBRID_OUTCOME_HYBRID_MISSED_DENSE = "hybrid_missed_dense"
HYBRID_OUTCOME_HYBRID_MISSED_LEXICAL = "hybrid_missed_lexical"
HYBRID_OUTCOME_ALL_FAIL = "all_fail"


@dataclass(frozen=True)
class HybridRetrievalEvalCaseReport:
    case: RetrievalEvalCase
    dense_evaluation: RetrievalEvalResult
    lexical_evaluation: RetrievalEvalResult
    hybrid_evaluation: RetrievalEvalResult
    hybrid_outcome: str
    dense_top_results: list[dict[str, Any]]
    lexical_top_results: list[dict[str, Any]]
    hybrid_top_results: list[dict[str, Any]]


@dataclass(frozen=True)
class HybridRetrievalEvalReport:
    total_cases: int
    dense_passed_count: int
    lexical_passed_count: int
    hybrid_passed_count: int
    all_pass_count: int
    hybrid_only_count: int
    dense_and_hybrid_count: int
    lexical_and_hybrid_count: int
    hybrid_missed_dense_count: int
    hybrid_missed_lexical_count: int
    hybrid_missed_both_count: int
    all_fail_count: int
    cases: list[HybridRetrievalEvalCaseReport]


def classify_hybrid_outcome(
    dense_passed: bool,
    lexical_passed: bool,
    hybrid_passed: bool,
) -> str:
    """Classify dense-vs-lexical-vs-hybrid behavior for one case."""

    if dense_passed and lexical_passed and hybrid_passed:
        return HYBRID_OUTCOME_ALL_PASS
    if dense_passed and not lexical_passed and hybrid_passed:
        return HYBRID_OUTCOME_DENSE_AND_HYBRID
    if not dense_passed and lexical_passed and hybrid_passed:
        return HYBRID_OUTCOME_LEXICAL_AND_HYBRID
    if not dense_passed and not lexical_passed and hybrid_passed:
        return HYBRID_OUTCOME_HYBRID_ONLY
    if dense_passed and lexical_passed and not hybrid_passed:
        return HYBRID_OUTCOME_HYBRID_MISSED_BOTH
    if dense_passed and not lexical_passed and not hybrid_passed:
        return HYBRID_OUTCOME_HYBRID_MISSED_DENSE
    if not dense_passed and lexical_passed and not hybrid_passed:
        return HYBRID_OUTCOME_HYBRID_MISSED_LEXICAL
    return HYBRID_OUTCOME_ALL_FAIL


def build_hybrid_retrieval_eval_case_report(
    case: RetrievalEvalCase,
    dense_results: Sequence[Any],
    lexical_results: Sequence[Any],
    hybrid_results: Sequence[Any],
) -> HybridRetrievalEvalCaseReport:
    """Evaluate dense, lexical, and hybrid outputs for one eval case."""

    normalized_dense_results = normalize_search_results(dense_results)
    normalized_lexical_results = normalize_search_results(lexical_results)
    normalized_hybrid_results = normalize_search_results(hybrid_results)

    dense_evaluation = evaluate_retrieval_results(case, normalized_dense_results)
    lexical_evaluation = evaluate_retrieval_results(case, normalized_lexical_results)
    hybrid_evaluation = evaluate_retrieval_results(case, normalized_hybrid_results)
    hybrid_outcome = classify_hybrid_outcome(
        dense_passed=dense_evaluation.passed,
        lexical_passed=lexical_evaluation.passed,
        hybrid_passed=hybrid_evaluation.passed,
    )

    return HybridRetrievalEvalCaseReport(
        case=case,
        dense_evaluation=dense_evaluation,
        lexical_evaluation=lexical_evaluation,
        hybrid_evaluation=hybrid_evaluation,
        hybrid_outcome=hybrid_outcome,
        dense_top_results=[serialize_search_result(result) for result in normalized_dense_results],
        lexical_top_results=[serialize_search_result(result) for result in normalized_lexical_results],
        hybrid_top_results=[serialize_search_result(result) for result in normalized_hybrid_results],
    )


def build_hybrid_retrieval_eval_report(
    case_reports: list[HybridRetrievalEvalCaseReport],
) -> HybridRetrievalEvalReport:
    """Build aggregate hybrid retrieval evaluation counts."""

    return HybridRetrievalEvalReport(
        total_cases=len(case_reports),
        dense_passed_count=sum(int(report.dense_evaluation.passed) for report in case_reports),
        lexical_passed_count=sum(int(report.lexical_evaluation.passed) for report in case_reports),
        hybrid_passed_count=sum(int(report.hybrid_evaluation.passed) for report in case_reports),
        all_pass_count=_count_outcome(case_reports, HYBRID_OUTCOME_ALL_PASS),
        hybrid_only_count=_count_outcome(case_reports, HYBRID_OUTCOME_HYBRID_ONLY),
        dense_and_hybrid_count=_count_outcome(case_reports, HYBRID_OUTCOME_DENSE_AND_HYBRID),
        lexical_and_hybrid_count=_count_outcome(case_reports, HYBRID_OUTCOME_LEXICAL_AND_HYBRID),
        hybrid_missed_dense_count=_count_outcome(case_reports, HYBRID_OUTCOME_HYBRID_MISSED_DENSE),
        hybrid_missed_lexical_count=_count_outcome(case_reports, HYBRID_OUTCOME_HYBRID_MISSED_LEXICAL),
        hybrid_missed_both_count=_count_outcome(case_reports, HYBRID_OUTCOME_HYBRID_MISSED_BOTH),
        all_fail_count=_count_outcome(case_reports, HYBRID_OUTCOME_ALL_FAIL),
        cases=case_reports,
    )


def write_hybrid_retrieval_eval_report_to_json(
    report: HybridRetrievalEvalReport,
    output_path: str | Path,
) -> Path:
    """Persist a dense-vs-lexical-vs-hybrid evaluation report as JSON."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(asdict(report), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return path


def _count_outcome(
    case_reports: list[HybridRetrievalEvalCaseReport],
    outcome: str,
) -> int:
    return sum(int(report.hybrid_outcome == outcome) for report in case_reports)