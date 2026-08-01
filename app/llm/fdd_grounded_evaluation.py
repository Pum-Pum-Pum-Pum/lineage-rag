from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence

from app.llm.answer_contract import GroundedAnswerResponse


REQUIRED_CASE_FIELDS = frozenset(
    {
        "case_id",
        "question",
        "expected_claims",
        "expected_evidence",
        "expected_document_ids",
        "expected_release_labels",
        "required_citation_document_ids",
        "should_abstain",
        "sme_reviewed",
        "review_status",
    }
)


@dataclass(frozen=True)
class FddGroundedEvalCase:
    case_id: str
    question: str
    expected_claims: list[str]
    expected_evidence: list[dict[str, str]]
    expected_document_ids: list[str]
    expected_release_labels: list[str]
    required_citation_document_ids: list[str]
    should_abstain: bool
    sme_reviewed: bool
    review_status: str


@dataclass(frozen=True)
class FddGroundedEvalResult:
    case_id: str
    structural_passed: bool
    failures: list[str]
    claim_review_required: bool
    expected_claims: list[str]
    answer: str
    is_answered: bool
    refusal_reason: str | None
    citation_document_ids: list[str | None]
    citation_release_labels: list[str | None]


def load_fdd_grounded_eval_cases(path: str | Path) -> list[FddGroundedEvalCase]:
    """Load and strictly validate the repository's JSONL grounded-evaluation format."""

    cases: list[FddGroundedEvalCase] = []
    case_ids: set[str] = set()
    for line_number, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"Invalid evaluation JSON at line {line_number}: {error.msg}") from error
        _validate_case_payload(payload, line_number)
        case = FddGroundedEvalCase(**payload)
        if case.case_id in case_ids:
            raise ValueError(f"Duplicate evaluation case_id at line {line_number}: {case.case_id}")
        case_ids.add(case.case_id)
        cases.append(case)

    if not cases:
        raise ValueError("Evaluation file did not contain any cases")
    return cases


def require_reviewed_cases(
    cases: Sequence[FddGroundedEvalCase],
    *,
    allow_unreviewed: bool,
) -> None:
    """Prevent an unreviewed benchmark from being misreported as a quality gate."""

    unreviewed = [case.case_id for case in cases if not case.sme_reviewed]
    if unreviewed and not allow_unreviewed:
        raise ValueError(
            "Evaluation contains unreviewed cases and cannot be used as a release-quality gate: "
            + ", ".join(unreviewed)
            + ". Obtain SME approval or pass --allow-unreviewed for a draft baseline only."
        )


def evaluate_fdd_grounded_response(
    case: FddGroundedEvalCase,
    response: GroundedAnswerResponse,
) -> FddGroundedEvalResult:
    """Score deterministic grounded-response contracts; leave claim entailment to SMEs."""

    failures: list[str] = []
    citation_document_ids = [citation.document_id for citation in response.citations]
    citation_release_labels = [citation.release_label for citation in response.citations]

    if case.should_abstain:
        if response.is_answered:
            failures.append("Expected a safe abstention, but the system answered the question.")
        if not response.refusal_reason:
            failures.append("Expected a machine-readable refusal reason for abstention.")
    else:
        if not response.is_answered:
            failures.append(
                "Expected a grounded answer, but the system abstained: "
                f"{response.refusal_reason or 'no refusal reason returned'}"
            )
        if not response.citations:
            failures.append("Grounded answer did not include any citations.")

        missing_document_ids = [
            document_id
            for document_id in case.required_citation_document_ids
            if document_id not in citation_document_ids
        ]
        if missing_document_ids:
            failures.append(
                "Required citation document IDs were missing: "
                f"{missing_document_ids}"
            )

        missing_release_labels = [
            release_label
            for release_label in case.expected_release_labels
            if release_label not in citation_release_labels
        ]
        if missing_release_labels:
            failures.append(
                "Expected citation release labels were missing: "
                f"{missing_release_labels}"
            )

    return FddGroundedEvalResult(
        case_id=case.case_id,
        structural_passed=not failures,
        failures=failures,
        claim_review_required=not case.should_abstain,
        expected_claims=case.expected_claims,
        answer=response.answer,
        is_answered=response.is_answered,
        refusal_reason=response.refusal_reason,
        citation_document_ids=citation_document_ids,
        citation_release_labels=citation_release_labels,
    )


def write_fdd_grounded_eval_report(
    *,
    output_path: str | Path,
    report_metadata: dict[str, Any],
    results: Sequence[FddGroundedEvalResult],
) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "metadata": report_metadata,
        "summary": {
            "total_cases": len(results),
            "structural_passed_count": sum(result.structural_passed for result in results),
            "claim_review_required_count": sum(result.claim_review_required for result in results),
        },
        "cases": [asdict(result) for result in results],
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def _validate_case_payload(payload: object, line_number: int) -> None:
    if not isinstance(payload, dict):
        raise ValueError(f"Evaluation case at line {line_number} must be a JSON object")
    missing = sorted(REQUIRED_CASE_FIELDS.difference(payload))
    if missing:
        raise ValueError(f"Evaluation case at line {line_number} is missing required fields: {missing}")

    for field_name in (
        "case_id",
        "question",
        "review_status",
    ):
        if not isinstance(payload[field_name], str) or not payload[field_name].strip():
            raise ValueError(f"Evaluation case at line {line_number} has invalid {field_name}")
    for field_name in (
        "expected_claims",
        "expected_evidence",
        "expected_document_ids",
        "expected_release_labels",
        "required_citation_document_ids",
    ):
        if not isinstance(payload[field_name], list):
            raise ValueError(f"Evaluation case at line {line_number} has invalid {field_name}; expected a list")
    if not isinstance(payload["should_abstain"], bool) or not isinstance(payload["sme_reviewed"], bool):
        raise ValueError(f"Evaluation case at line {line_number} must use boolean should_abstain and sme_reviewed")

    if payload["should_abstain"]:
        answer_fields = (
            "expected_claims",
            "expected_evidence",
            "expected_document_ids",
            "expected_release_labels",
            "required_citation_document_ids",
        )
        if any(payload[field_name] for field_name in answer_fields):
            raise ValueError(
                f"Abstention case at line {line_number} must not declare answer evidence or required citations"
            )
    elif not payload["required_citation_document_ids"]:
        raise ValueError(
            f"Answered case at line {line_number} must include required_citation_document_ids"
        )
