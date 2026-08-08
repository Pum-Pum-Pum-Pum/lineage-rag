import json
from pathlib import Path

from app.llm.fdd_eval_manifest_validation import validate_fdd_eval_manifest
from app.llm.fdd_grounded_evaluation import FddGroundedEvalCase


def test_reviewed_answer_case_matching_artifact_is_release_ready(tmp_path: Path) -> None:
    manifest_path = tmp_path / "eval.jsonl"
    manifest_path.write_text("{}\n", encoding="utf-8")
    artifact_directory = _write_artifact(tmp_path, document_id="doc-r21", release_label="R21")

    report = validate_fdd_eval_manifest(
        cases=[_answer_case()],
        manifest_path=manifest_path,
        artifact_directory=artifact_directory,
    )

    assert report.release_gate_ready is True
    assert report.errors == []
    assert report.gate_blockers == []
    assert report.indexed_document_count == 1


def test_unreviewed_case_is_gate_blocker_not_structural_error(tmp_path: Path) -> None:
    manifest_path = tmp_path / "eval.jsonl"
    manifest_path.write_text("{}\n", encoding="utf-8")
    artifact_directory = _write_artifact(tmp_path, document_id="doc-r21", release_label="R21")
    case = _answer_case(sme_reviewed=False, review_status="pending_sme_approval")

    report = validate_fdd_eval_manifest(
        cases=[case],
        manifest_path=manifest_path,
        artifact_directory=artifact_directory,
    )

    assert report.release_gate_ready is False
    assert report.errors == []
    assert [issue.code for issue in report.gate_blockers] == ["pending_sme_review"]


def test_missing_expected_document_fails_closed(tmp_path: Path) -> None:
    manifest_path = tmp_path / "eval.jsonl"
    manifest_path.write_text("{}\n", encoding="utf-8")
    artifact_directory = _write_artifact(tmp_path, document_id="different-doc", release_label="R21")

    report = validate_fdd_eval_manifest(
        cases=[_answer_case()],
        manifest_path=manifest_path,
        artifact_directory=artifact_directory,
    )

    assert report.release_gate_ready is False
    assert "document_not_indexed" in {issue.code for issue in report.errors}


def test_required_citation_must_be_an_expected_document(tmp_path: Path) -> None:
    manifest_path = tmp_path / "eval.jsonl"
    manifest_path.write_text("{}\n", encoding="utf-8")
    artifact_directory = _write_artifact(tmp_path, document_id="doc-r21", release_label="R21")
    case = FddGroundedEvalCase(
        **{
            **_answer_case().__dict__,
            "required_citation_document_ids": ["other-doc"],
        }
    )

    report = validate_fdd_eval_manifest(
        cases=[case],
        manifest_path=manifest_path,
        artifact_directory=artifact_directory,
    )

    assert "required_citation_not_expected" in {issue.code for issue in report.errors}


def test_expected_evidence_may_use_a_structured_source_locator(tmp_path: Path) -> None:
    manifest_path = tmp_path / "eval.jsonl"
    manifest_path.write_text("{}\n", encoding="utf-8")
    artifact_directory = _write_artifact(tmp_path, document_id="doc-r21", release_label="R21")
    case = FddGroundedEvalCase(
        **{
            **_answer_case().__dict__,
            "expected_evidence": [
                {"document_id": "doc-r21", "release_label": "R21", "source_kind": "table"}
            ],
        }
    )

    report = validate_fdd_eval_manifest(
        cases=[case],
        manifest_path=manifest_path,
        artifact_directory=artifact_directory,
    )

    assert report.release_gate_ready is True
    assert report.errors == []


def test_release_label_in_user_question_fails_closed(tmp_path: Path) -> None:
    manifest_path = tmp_path / "eval.jsonl"
    manifest_path.write_text("{}\n", encoding="utf-8")
    artifact_directory = _write_artifact(tmp_path, document_id="doc-r21", release_label="R21")
    case = FddGroundedEvalCase(
        **{
            **_answer_case().__dict__,
            "question": "What does R21 support?",
        }
    )

    report = validate_fdd_eval_manifest(
        cases=[case],
        manifest_path=manifest_path,
        artifact_directory=artifact_directory,
    )

    assert report.release_gate_ready is False
    assert "release_label_in_user_question" in {issue.code for issue in report.errors}


def _answer_case(
    *, sme_reviewed: bool = True, review_status: str = "approved_by_sme"
) -> FddGroundedEvalCase:
    return FddGroundedEvalCase(
        case_id="r21-case",
        question="What behavior is supported?",
        expected_claims=["R21 supports the documented behavior."],
        expected_evidence=[{"document_id": "doc-r21", "evidence": "Documented behavior."}],
        expected_document_ids=["doc-r21"],
        expected_release_labels=["R21"],
        required_citation_document_ids=["doc-r21"],
        should_abstain=False,
        sme_reviewed=sme_reviewed,
        review_status=review_status,
    )


def _write_artifact(tmp_path: Path, *, document_id: str, release_label: str) -> Path:
    directory = tmp_path / "processed"
    directory.mkdir()
    payload = {
        "document_name": f"{document_id}.docx",
        "document_family": "family",
        "release_label": release_label,
        "total_units": 1,
        "units": [
            {
                "unit_id": f"{document_id}::chunk_0",
                "document_id": document_id,
                "release_label": release_label,
                "text": "Evidence",
            }
        ],
    }
    (directory / f"{document_id}.retrieval_ready.json").write_text(
        json.dumps(payload), encoding="utf-8"
    )
    return directory
