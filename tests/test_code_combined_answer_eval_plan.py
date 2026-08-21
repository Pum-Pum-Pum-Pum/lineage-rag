from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.fdd_code_lineage.evaluation import CodeCombinedEvalCase
from scripts.prepare_code_combined_answer_eval import main


def test_paid_answer_plan_fails_closed_when_retrieval_gate_fails(
    tmp_path: Path,
) -> None:
    manifest = _write_manifest(tmp_path)
    report = _write_report(tmp_path, threshold_passed=False)
    with pytest.raises(ValueError, match="blocked by the retrieval gate"):
        main(
            [
                "--eval-file",
                str(manifest),
                "--retrieval-report",
                str(report),
            ]
        )


def test_paid_answer_plan_records_scope_but_performs_no_external_calls(
    tmp_path: Path,
) -> None:
    manifest = _write_manifest(tmp_path)
    report = _write_report(tmp_path, threshold_passed=True)
    output = tmp_path / "plan.json"
    assert (
        main(
            [
                "--eval-file",
                str(manifest),
                "--retrieval-report",
                str(report),
                "--output-file",
                str(output),
            ]
        )
        == 0
    )
    plan = json.loads(output.read_text(encoding="utf-8"))
    assert plan["status"] == "awaiting_explicit_paid_authorization"
    assert plan["external_api_calls_performed"] == 0
    assert plan["answer_generation_request_count"] == 1
    assert plan["release_gate_eligible"] is False
    with pytest.raises(FileExistsError):
        main(
            [
                "--eval-file",
                str(manifest),
                "--retrieval-report",
                str(report),
                "--output-file",
                str(output),
            ]
        )


def _write_manifest(tmp_path: Path) -> Path:
    case = CodeCombinedEvalCase(
        case_id="code-reviewed-001",
        mode="code",
        question="Where is the approved AML routine implemented?",
        expected_claims=("Identify the visible routine.",),
        expected_code_paths=("pkgaml_custom.sql",),
        expected_code_symbols=("process_aml",),
        sme_reviewed=True,
        review_status="reviewed",
        rationale="The SME reviewed this exact visible custom-code expectation.",
    )
    path = tmp_path / "eval.jsonl"
    path.write_text(case.model_dump_json() + "\n", encoding="utf-8")
    return path


def _write_report(tmp_path: Path, *, threshold_passed: bool) -> Path:
    path = tmp_path / f"report-{threshold_passed}.json"
    path.write_text(
        json.dumps(
            {
                "summary": {"retrieval_threshold_passed": threshold_passed},
                "cases": [{"case_id": "code-reviewed-001"}],
            }
        ),
        encoding="utf-8",
    )
    return path
