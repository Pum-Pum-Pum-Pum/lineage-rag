from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

from app.llm.fdd_grounded_evaluation import FddGroundedEvalCase
from app.services.query_retrieval import PlannedRetrievalResult


@dataclass(frozen=True)
class RetrievalIdentity:
    rank: int
    point_id: str
    unit_id: str
    document_id: str | None
    release_label: str | None
    source_kind: str | None
    score: float


@dataclass(frozen=True)
class FddRetrievalGateCaseReport:
    case_id: str
    question: str
    should_abstain: bool
    positive_gate_applicable: bool
    positive_gate_passed: bool | None
    expected_document_ids: list[str]
    retrieved_document_ids: list[str]
    missing_document_ids: list[str]
    document_recall_at_k: float | None
    expected_release_labels: list[str]
    retrieved_release_labels: list[str]
    missing_release_labels: list[str]
    effective_release_label: str | None
    release_source: str | None
    current_state_requested: bool
    historical_context_requested: bool
    final_results: list[RetrievalIdentity]
    dense_candidates: list[RetrievalIdentity]
    lexical_candidates: list[RetrievalIdentity]


def build_fdd_retrieval_gate_case_report(
    *,
    case: FddGroundedEvalCase,
    planned: PlannedRetrievalResult,
) -> FddRetrievalGateCaseReport:
    retrieved_document_ids = _unique_nonblank(
        result.payload.get("document_id") for result in planned.results
    )
    retrieved_release_labels = _unique_nonblank(
        result.payload.get("release_label") for result in planned.results
    )
    missing_document_ids = sorted(
        set(case.expected_document_ids).difference(retrieved_document_ids)
    )
    missing_release_labels = sorted(
        set(case.expected_release_labels).difference(retrieved_release_labels)
    )
    expected_document_count = len(case.expected_document_ids)
    document_recall = (
        (expected_document_count - len(missing_document_ids)) / expected_document_count
        if expected_document_count
        else None
    )
    positive_gate_applicable = not case.should_abstain
    positive_gate_passed = (
        not missing_document_ids and not missing_release_labels
        if positive_gate_applicable
        else None
    )
    return FddRetrievalGateCaseReport(
        case_id=case.case_id,
        question=case.question,
        should_abstain=case.should_abstain,
        positive_gate_applicable=positive_gate_applicable,
        positive_gate_passed=positive_gate_passed,
        expected_document_ids=list(case.expected_document_ids),
        retrieved_document_ids=retrieved_document_ids,
        missing_document_ids=missing_document_ids,
        document_recall_at_k=document_recall,
        expected_release_labels=list(case.expected_release_labels),
        retrieved_release_labels=retrieved_release_labels,
        missing_release_labels=missing_release_labels,
        effective_release_label=planned.temporal_plan.effective_release_label,
        release_source=planned.temporal_plan.release_source,
        current_state_requested=planned.temporal_plan.is_current_state,
        historical_context_requested=planned.temporal_plan.historical_context_requested,
        final_results=_summarize(planned.results),
        dense_candidates=_summarize(planned.routed.dense_candidates),
        lexical_candidates=_summarize(planned.routed.lexical_candidates),
    )


def build_fdd_retrieval_gate_report(
    *,
    metadata: dict[str, Any],
    cases: Sequence[FddRetrievalGateCaseReport],
    minimum_document_recall: float,
) -> dict[str, Any]:
    if not 0.0 <= minimum_document_recall <= 1.0:
        raise ValueError("minimum_document_recall must be between 0 and 1")
    positive_cases = [case for case in cases if case.positive_gate_applicable]
    expected_documents = sum(len(case.expected_document_ids) for case in positive_cases)
    missing_documents = sum(len(case.missing_document_ids) for case in positive_cases)
    document_recall = (
        (expected_documents - missing_documents) / expected_documents
        if expected_documents
        else 0.0
    )
    positive_passed = sum(case.positive_gate_passed is True for case in positive_cases)
    reviewed_manifest = bool(metadata.get("reviewed_manifest"))
    retrieval_threshold_passed = document_recall >= minimum_document_recall
    return {
        "schema_version": "fdd_retrieval_gate_v1",
        "metadata": metadata,
        "summary": {
            "total_cases": len(cases),
            "positive_cases": len(positive_cases),
            "abstention_diagnostic_cases": len(cases) - len(positive_cases),
            "positive_cases_passed": positive_passed,
            "positive_case_pass_rate": (
                positive_passed / len(positive_cases) if positive_cases else 0.0
            ),
            "expected_document_occurrences": expected_documents,
            "missing_document_occurrences": missing_documents,
            "document_recall_at_k": document_recall,
            "minimum_document_recall": minimum_document_recall,
            "retrieval_threshold_passed": retrieval_threshold_passed,
            "reviewed_manifest": reviewed_manifest,
            "release_gate_eligible": reviewed_manifest and retrieval_threshold_passed,
        },
        "cases": [asdict(case) for case in cases],
    }


def write_fdd_retrieval_gate_report(report: dict[str, Any], output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def _summarize(results: Sequence[Any]) -> list[RetrievalIdentity]:
    return [
        RetrievalIdentity(
            rank=rank,
            point_id=str(result.point_id),
            unit_id=str(result.payload.get("unit_id", "")),
            document_id=_optional_string(result.payload.get("document_id")),
            release_label=_optional_string(result.payload.get("release_label")),
            source_kind=_optional_string(result.payload.get("source_kind")),
            score=float(result.score),
        )
        for rank, result in enumerate(results, start=1)
    ]


def _unique_nonblank(values: Sequence[Any] | Any) -> list[str]:
    unique: list[str] = []
    for value in values:
        normalized = str(value).strip() if value is not None else ""
        if normalized and normalized not in unique:
            unique.append(normalized)
    return unique


def _optional_string(value: Any) -> str | None:
    normalized = str(value).strip() if value is not None else ""
    return normalized or None
