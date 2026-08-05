import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from scripts import run_fdd_grounded_eval
from app.llm.fdd_grounded_evaluation import FddGroundedEvalCase


def test_fdd_grounded_eval_script_help_runs() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/run_fdd_grounded_eval.py", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert "--allow-unreviewed" in result.stdout
    assert "--dry-run" in result.stdout
    assert "--case-id" in result.stdout
    assert "--collection-name" in result.stdout
    assert "--lexical-artifact-directory" in result.stdout
    assert "--resume-trace-directory" in result.stdout


def test_resolve_evaluation_target_uses_paired_staged_overrides(tmp_path: Path) -> None:
    args = run_fdd_grounded_eval.parse_args(
        [
            "--collection-name",
            "functional_specs_v4",
            "--lexical-artifact-directory",
            str(tmp_path / "stage" / "processed"),
        ]
    )
    settings = SimpleNamespace(
        qdrant_collection_name="functional_specs_v2",
        processed_dir=tmp_path / "active-processed",
    )

    target = run_fdd_grounded_eval.resolve_evaluation_target(args=args, settings=settings)

    assert target.collection_name == "functional_specs_v4"
    assert target.lexical_artifact_directory == tmp_path / "stage" / "processed"


def test_resolve_evaluation_target_rejects_mixed_generation_override(tmp_path: Path) -> None:
    args = run_fdd_grounded_eval.parse_args(["--collection-name", "functional_specs_v4"])
    settings = SimpleNamespace(
        qdrant_collection_name="functional_specs_v2",
        processed_dir=tmp_path / "active-processed",
    )

    with pytest.raises(ValueError, match="must be supplied together"):
        run_fdd_grounded_eval.resolve_evaluation_target(args=args, settings=settings)


def test_resume_reuses_one_valid_trace_without_model_call(tmp_path: Path) -> None:
    case = FddGroundedEvalCase(
        case_id="case-1",
        question="What does R1 support?",
        expected_claims=["R1 support"],
        expected_evidence=[],
        expected_document_ids=[],
        expected_release_labels=["R1"],
        required_citation_document_ids=["R1-doc"],
        should_abstain=False,
        sme_reviewed=False,
        review_status="draft",
    )
    trace_directory = tmp_path / "traces"
    trace_directory.mkdir()
    (trace_directory / "run-case-1.json").write_text(
        '''{
          "query": "What does R1 support?",
          "answer_response": {
            "query": "What does R1 support?",
            "answer": "R1 supports the documented function [C1].",
            "is_answered": true,
            "refusal_reason": null,
            "citations": [{
              "unit_id": "unit-1",
              "document_family": "FS_ASNB",
              "release_label": "R1",
              "source_kind": "paragraph",
              "score": 0.8,
              "text_preview": "documented function",
              "document_id": "R1-doc"
            }]
          }
        }''',
        encoding="utf-8",
    )

    results = run_fdd_grounded_eval.load_resumed_evaluation_results([case], trace_directory)

    assert results["case-1"].structural_passed is True


def test_resume_rejects_trace_for_different_question(tmp_path: Path) -> None:
    case = FddGroundedEvalCase(
        case_id="case-1",
        question="Expected question",
        expected_claims=[],
        expected_evidence=[],
        expected_document_ids=[],
        expected_release_labels=[],
        required_citation_document_ids=[],
        should_abstain=True,
        sme_reviewed=False,
        review_status="draft",
    )
    trace_directory = tmp_path / "traces"
    trace_directory.mkdir()
    (trace_directory / "run-case-1.json").write_text(
        '{"query":"Wrong question","answer_response":{"query":"Wrong question"}}',
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="does not match"):
        run_fdd_grounded_eval.load_resumed_evaluation_results([case], trace_directory)
