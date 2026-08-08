import json
from pathlib import Path

import pytest

from app.llm.fdd_sme_remediation import build_remediation_report


def _write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_report_requires_missing_lineage_source_and_replay(tmp_path: Path) -> None:
    ledger = tmp_path / "ledger.json"
    plan = tmp_path / "plan.json"
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    (artifacts / "R21_FDS.retrieval_ready.json").write_text("{}", encoding="utf-8")
    _write_json(
        ledger,
        {
            "decisions": [
                {"case_id": "cheque", "verdict": "expected_case_incorrect"},
                {"case_id": "lineage", "verdict": "conditionally_accepted"},
            ]
        },
    )
    _write_json(
        plan,
        {
            "actions": [
                {
                    "case_id": "cheque",
                    "status": "benchmark_revised_pending_replay",
                    "required_artifact_patterns": ["*R21*"],
                },
                {
                    "case_id": "lineage",
                    "status": "blocked_missing_source",
                    "required_artifact_patterns": ["*R2*Death*Claim*"],
                },
            ]
        },
    )

    report = build_remediation_report(
        review_ledger_path=ledger,
        remediation_plan_path=plan,
        artifact_directory=artifacts,
    )

    assert report["phase_1_gate_status"] == "pending_material_remediation"
    assert report["blocking_case_ids"] == ["cheque", "lineage"]
    assert report["actions"][1]["missing_artifact_patterns"] == ["*R2*Death*Claim*"]


def test_report_fails_if_an_unresolved_review_is_omitted(tmp_path: Path) -> None:
    ledger = tmp_path / "ledger.json"
    plan = tmp_path / "plan.json"
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    (artifacts / "R21.json").write_text("{}", encoding="utf-8")
    _write_json(ledger, {"decisions": [{"case_id": "gap", "verdict": "other"}]})
    _write_json(plan, {"actions": []})

    with pytest.raises(ValueError, match="cover every unresolved"):
        build_remediation_report(
            review_ledger_path=ledger,
            remediation_plan_path=plan,
            artifact_directory=artifacts,
        )


def test_paid_replay_pending_semantic_review_remains_blocking(tmp_path: Path) -> None:
    ledger = tmp_path / "ledger.json"
    plan = tmp_path / "plan.json"
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    (artifacts / "R21.retrieval_ready.json").write_text("{}", encoding="utf-8")
    _write_json(
        ledger,
        {"decisions": [{"case_id": "cheque", "verdict": "expected_case_incorrect"}]},
    )
    _write_json(
        plan,
        {
            "actions": [
                {
                    "case_id": "cheque",
                    "status": "paid_replay_pending_semantic_review",
                    "required_artifact_patterns": ["*R21*"],
                }
            ]
        },
    )

    report = build_remediation_report(
        review_ledger_path=ledger,
        remediation_plan_path=plan,
        artifact_directory=artifacts,
    )

    assert report["blocking_case_ids"] == ["cheque"]
    assert report["phase_1_gate_status"] == "pending_material_remediation"


def test_accepted_paid_replay_closes_gate(tmp_path: Path) -> None:
    ledger = tmp_path / "ledger.json"
    plan = tmp_path / "plan.json"
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    (artifacts / "R21.retrieval_ready.json").write_text("{}", encoding="utf-8")
    _write_json(
        ledger,
        {"decisions": [{"case_id": "cheque", "verdict": "expected_case_incorrect"}]},
    )
    _write_json(
        plan,
        {
            "actions": [
                {
                    "case_id": "cheque",
                    "status": "accepted_after_paid_replay",
                    "required_artifact_patterns": ["*R21*"],
                }
            ]
        },
    )

    report = build_remediation_report(
        review_ledger_path=ledger,
        remediation_plan_path=plan,
        artifact_directory=artifacts,
    )

    assert report["blocking_case_ids"] == []
    assert report["phase_1_gate_status"] == "passed"
