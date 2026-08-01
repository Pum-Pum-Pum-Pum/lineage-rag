from pathlib import Path

import pytest

from app.llm.answer_contract import Citation, GroundedAnswerResponse
from app.llm.fdd_grounded_evaluation import (
    FddGroundedEvalCase,
    evaluate_fdd_grounded_response,
    load_fdd_grounded_eval_cases,
    require_reviewed_cases,
)


def test_load_fdd_grounded_cases_rejects_unanswered_case_with_evidence(tmp_path: Path) -> None:
    path = tmp_path / "invalid.jsonl"
    path.write_text(
        '{"case_id":"abstain","question":"Unknown?","expected_claims":["bad"],'
        '"expected_evidence":[],"expected_document_ids":[],"expected_release_labels":[],'
        '"required_citation_document_ids":[],"should_abstain":true,'
        '"sme_reviewed":true,"review_status":"approved"}\n',
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="Abstention case"):
        load_fdd_grounded_eval_cases(path)


def test_require_reviewed_cases_blocks_release_gate_but_allows_explicit_draft() -> None:
    case = _answered_case(sme_reviewed=False)

    with pytest.raises(ValueError, match="unreviewed cases"):
        require_reviewed_cases([case], allow_unreviewed=False)

    require_reviewed_cases([case], allow_unreviewed=True)


def test_answered_case_requires_expected_document_and_release_citations() -> None:
    response = _response(answered=True, document_id="FS_ASNB_R1_Fund", release_label="R1")

    result = evaluate_fdd_grounded_response(_answered_case(), response)

    assert result.structural_passed is True
    assert result.claim_review_required is True


def test_answered_case_reports_missing_required_document_identity() -> None:
    response = _response(answered=True, document_id="FS_ASNB_R1_Other", release_label="R1")

    result = evaluate_fdd_grounded_response(_answered_case(), response)

    assert result.structural_passed is False
    assert "Required citation document IDs" in result.failures[0]


def test_abstention_case_requires_refusal_state() -> None:
    case = FddGroundedEvalCase(
        case_id="abstain",
        question="Unknown feature?",
        expected_claims=[],
        expected_evidence=[],
        expected_document_ids=[],
        expected_release_labels=[],
        required_citation_document_ids=[],
        should_abstain=True,
        sme_reviewed=True,
        review_status="approved",
    )

    result = evaluate_fdd_grounded_response(case, _response(answered=True, document_id="FS_ASNB_R1_Fund", release_label="R1"))

    assert result.structural_passed is False
    assert "Expected a safe abstention" in result.failures[0]


def _answered_case(*, sme_reviewed: bool = True) -> FddGroundedEvalCase:
    return FddGroundedEvalCase(
        case_id="fund-rule",
        question="What is the fund rule?",
        expected_claims=["Expected claim."],
        expected_evidence=[{"document_id": "FS_ASNB_R1_Fund", "evidence": "Expected evidence."}],
        expected_document_ids=["FS_ASNB_R1_Fund"],
        expected_release_labels=["R1"],
        required_citation_document_ids=["FS_ASNB_R1_Fund"],
        should_abstain=False,
        sme_reviewed=sme_reviewed,
        review_status="approved" if sme_reviewed else "pending_sme_approval",
    )


def _response(*, answered: bool, document_id: str, release_label: str) -> GroundedAnswerResponse:
    return GroundedAnswerResponse(
        query="Question",
        answer="Answer" if answered else "I could not find sufficient evidence.",
        is_answered=answered,
        refusal_reason=None if answered else "Top score is below threshold.",
        citations=[
            Citation(
                unit_id="FS_ASNB_R1_Fund.docx::chunk_1",
                document_id=document_id,
                document_family="FS_ASNB",
                release_label=release_label,
                source_kind="paragraph",
                score=0.9,
                text_preview="Evidence",
            )
        ],
    )
