from __future__ import annotations

import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


LABEL_BOTH_RETRIEVERS_PASSED = "both_retrievers_passed"
LABEL_LEXICAL_RANKING_FAILURE = "lexical_ranking_failure"
LABEL_LEXICAL_EXACT_TERM_FALSE_POSITIVE = "lexical_exact_term_false_positive"
LABEL_LEXICAL_TOP1_FAILURE = "lexical_top1_failure"
LABEL_DENSE_RANKING_FAILURE = "dense_ranking_failure"
LABEL_DENSE_EXACT_IDENTIFIER_MISS = "dense_exact_identifier_miss"
LABEL_UNSUPPORTED_ATTACHMENT_MARKER_MATCH = "unsupported_attachment_marker_match"
LABEL_MARKER_MATCH_NOT_FULL_EVIDENCE = "marker_match_not_full_evidence"
LABEL_WEAK_MARKER_EXPECTATION = "weak_marker_expectation"
LABEL_EXPECTED_UNANSWERABLE = "expected_unanswerable"
LABEL_MISSING_OR_UNSUPPORTED_EVIDENCE = "missing_or_unsupported_evidence"
LABEL_EXPECTED_UNSUPPORTED_EVIDENCE = "expected_unsupported_evidence"
LABEL_UNCLASSIFIED_COMPARISON_FAILURE = "unclassified_comparison_failure"

SEVERITY_INFO = "info"
SEVERITY_LOW = "low"
SEVERITY_MEDIUM = "medium"
SEVERITY_HIGH = "high"


@dataclass(frozen=True)
class RetrievalErrorAnalysisCase:
    case_id: str
    query: str
    comparison_outcome: str
    expected_to_pass: bool
    severity: str
    root_cause_labels: list[str]
    rationale: str
    recommended_next_action: str
    dense_top_unit_id: str | None
    lexical_top_unit_id: str | None
    dense_top_preview: str | None
    lexical_top_preview: str | None


@dataclass(frozen=True)
class RetrievalErrorAnalysisReport:
    total_cases: int
    analyzed_case_count: int
    high_severity_count: int
    medium_severity_count: int
    low_severity_count: int
    info_severity_count: int
    label_counts: dict[str, int]
    cases: list[RetrievalErrorAnalysisCase]


def load_retrieval_comparison_report(path: str | Path) -> dict[str, Any]:
    """Load a persisted dense-vs-lexical comparison report."""

    return json.loads(Path(path).read_text(encoding="utf-8"))


def analyze_retrieval_comparison_report(
    comparison_report_payload: dict[str, Any],
    include_both_pass: bool = False,
) -> RetrievalErrorAnalysisReport:
    """Classify root-cause labels for dense-vs-lexical comparison cases."""

    analyzed_cases: list[RetrievalErrorAnalysisCase] = []

    for case_payload in comparison_report_payload.get("cases", []):
        if case_payload.get("comparison_outcome") == "both_pass" and not include_both_pass:
            continue
        analyzed_cases.append(analyze_retrieval_comparison_case(case_payload))

    severity_counts = Counter(case.severity for case in analyzed_cases)
    label_counts: Counter[str] = Counter()
    for case in analyzed_cases:
        label_counts.update(case.root_cause_labels)

    return RetrievalErrorAnalysisReport(
        total_cases=int(comparison_report_payload.get("total_cases", 0)),
        analyzed_case_count=len(analyzed_cases),
        high_severity_count=severity_counts[SEVERITY_HIGH],
        medium_severity_count=severity_counts[SEVERITY_MEDIUM],
        low_severity_count=severity_counts[SEVERITY_LOW],
        info_severity_count=severity_counts[SEVERITY_INFO],
        label_counts=dict(sorted(label_counts.items())),
        cases=analyzed_cases,
    )


def analyze_retrieval_comparison_case(
    case_payload: dict[str, Any],
) -> RetrievalErrorAnalysisCase:
    """Assign deterministic error-analysis labels to one comparison case."""

    case = case_payload.get("case", {})
    expectation = case.get("expectation", {})
    case_id = str(case.get("case_id", ""))
    query = str(case.get("query", ""))
    notes = str(case.get("notes", ""))
    expected_to_pass = bool(expectation.get("expected_to_pass", True))
    has_unsupported_expectation = bool(expectation.get("unsupported_evidence_contains_any"))
    comparison_outcome = str(case_payload.get("comparison_outcome", ""))
    dense_evaluation = case_payload.get("dense_evaluation", {})
    lexical_evaluation = case_payload.get("lexical_evaluation", {})
    dense_top = _first_result(case_payload.get("dense_top_results", []))
    lexical_top = _first_result(case_payload.get("lexical_top_results", []))
    dense_top_text = _result_text(dense_top)
    lexical_top_text = _result_text(lexical_top)

    labels: list[str] = []
    severity = SEVERITY_INFO
    rationale = "Both retrievers passed this comparison case."
    recommended_next_action = "Keep this case as a regression check."

    if comparison_outcome == "dense_only":
        labels.extend([LABEL_LEXICAL_RANKING_FAILURE, LABEL_LEXICAL_TOP1_FAILURE])
        if _has_query_overlap(query, lexical_top_text):
            labels.append(LABEL_LEXICAL_EXACT_TERM_FALSE_POSITIVE)
        severity = SEVERITY_MEDIUM
        rationale = (
            "Dense retrieval passed while lexical retrieval failed. The lexical top result appears to share "
            "surface query terms but did not satisfy the expected top-1 evidence markers."
        )
        recommended_next_action = (
            "Inspect lexical ranking, noisy exact-term matches, and whether domain-specific stopwords or "
            "section-aware metadata could reduce false positives before hybrid fusion."
        )
    elif comparison_outcome == "lexical_only":
        labels.append(LABEL_DENSE_RANKING_FAILURE)
        if _query_has_identifier(query):
            labels.append(LABEL_DENSE_EXACT_IDENTIFIER_MISS)
        if _looks_like_attachment_marker(case_id, query, notes, lexical_top_text):
            labels.extend(
                [
                    LABEL_UNSUPPORTED_ATTACHMENT_MARKER_MATCH,
                    LABEL_MARKER_MATCH_NOT_FULL_EVIDENCE,
                ]
            )
        if not expected_to_pass:
            labels.append(LABEL_WEAK_MARKER_EXPECTATION)
        severity = SEVERITY_HIGH if not expected_to_pass else SEVERITY_MEDIUM
        rationale = (
            "Lexical retrieval passed while dense retrieval failed. This may indicate an exact-identifier advantage, "
            "but if the case was expected to fail or points to an attachment, the pass may only be a marker match."
        )
        recommended_next_action = (
            "Inspect the source/processed artifacts to confirm whether full evidence exists, then tighten evaluation "
            "labels so marker matches are separated from complete evidence matches."
        )
    elif comparison_outcome == "both_fail":
        if not expected_to_pass and has_unsupported_expectation:
            labels.extend(
                [
                    LABEL_EXPECTED_UNSUPPORTED_EVIDENCE,
                    LABEL_MARKER_MATCH_NOT_FULL_EVIDENCE,
                    LABEL_UNSUPPORTED_ATTACHMENT_MARKER_MATCH,
                ]
            )
            severity = SEVERITY_HIGH
            rationale = (
                "Both retrievers failed on a case expected not to pass because retrieved marker/reference content "
                "does not prove the underlying unsupported evidence was extracted."
            )
            recommended_next_action = (
                "Inspect whether full evidence exists in extracted artifacts and tighten labels so marker/reference "
                "matches are not treated as complete answer evidence."
            )
        elif not expected_to_pass:
            labels.append(LABEL_EXPECTED_UNANSWERABLE)
            severity = SEVERITY_LOW
            rationale = (
                "Both retrievers failed on a case that is expected not to pass, which is consistent with an "
                "unanswerable or unsupported-evidence scenario."
            )
            recommended_next_action = (
                "Keep this as an abstention/no-evidence regression case and verify answer generation refuses safely."
            )
        else:
            labels.append(LABEL_MISSING_OR_UNSUPPORTED_EVIDENCE)
            severity = SEVERITY_HIGH
            rationale = (
                "Both retrievers failed on a case expected to pass. The issue may be missing indexed evidence, "
                "unsupported source content, metadata filters, chunking, or weak ingestion."
            )
            recommended_next_action = (
                "Trace the evidence from source document to processed artifact, retrieval-ready artifact, embedding cache, "
                "and Qdrant payload before changing retrieval logic."
            )
    elif comparison_outcome == "both_pass":
        labels.append(LABEL_BOTH_RETRIEVERS_PASSED)
    else:
        labels.append(LABEL_UNCLASSIFIED_COMPARISON_FAILURE)
        severity = SEVERITY_HIGH
        rationale = f"Unknown comparison outcome: {comparison_outcome}"
        recommended_next_action = "Inspect the comparison report schema and update the analyzer."

    labels = sorted(set(labels))

    return RetrievalErrorAnalysisCase(
        case_id=case_id,
        query=query,
        comparison_outcome=comparison_outcome,
        expected_to_pass=expected_to_pass,
        severity=severity,
        root_cause_labels=labels,
        rationale=rationale,
        recommended_next_action=recommended_next_action,
        dense_top_unit_id=_result_unit_id(dense_top),
        lexical_top_unit_id=_result_unit_id(lexical_top),
        dense_top_preview=_preview(dense_top_text),
        lexical_top_preview=_preview(lexical_top_text),
    )


def write_retrieval_error_analysis_report_to_json(
    report: RetrievalErrorAnalysisReport,
    output_path: str | Path,
) -> Path:
    """Persist retrieval comparison error analysis as JSON."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(asdict(report), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return path


def _first_result(results: list[dict[str, Any]]) -> dict[str, Any] | None:
    return results[0] if results else None


def _result_text(result: dict[str, Any] | None) -> str:
    if result is None:
        return ""
    return str(result.get("payload", {}).get("text", ""))


def _result_unit_id(result: dict[str, Any] | None) -> str | None:
    if result is None:
        return None
    return str(result.get("payload", {}).get("unit_id", result.get("point_id", "")))


def _preview(text: str, max_length: int = 250) -> str | None:
    if not text:
        return None
    return text[:max_length].replace("\n", " ")


def _has_query_overlap(query: str, text: str) -> bool:
    query_terms = {term for term in _simple_terms(query) if len(term) > 2}
    text_terms = set(_simple_terms(text))
    return bool(query_terms & text_terms)


def _query_has_identifier(query: str) -> bool:
    terms = _simple_terms(query)
    return any("-" in term or any(character.isdigit() for character in term) for term in terms)


def _looks_like_attachment_marker(
    case_id: str,
    query: str,
    notes: str,
    lexical_top_text: str,
) -> bool:
    combined = " ".join([case_id, query, notes, lexical_top_text]).lower()
    marker_terms = ["attach", "attached", "attachment", "sample report", "layout"]
    return any(term in combined for term in marker_terms)


def _simple_terms(text: str) -> list[str]:
    return [token.strip(".,:;()[]{}\"'“”‘’").lower() for token in text.split() if token.strip()]