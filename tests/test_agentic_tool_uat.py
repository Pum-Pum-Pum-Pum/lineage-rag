from __future__ import annotations

import sys
from pathlib import Path

import pytest

from app.agentic_tools.evaluation import execute_local_lexical_tools
from app.agentic_tools.policy import load_agentic_tools_policy
from app.agentic_tools.uat import build_local_uat_report, write_local_uat_report_no_overwrite
from scripts import run_local_bounded_tool_uat
from tests.test_agentic_tool_evaluation import (
    _code_artifact,
    _fdd_documents,
    _lineage,
)


def test_local_uat_report_is_source_marked_and_no_overwrite(tmp_path: Path) -> None:
    artifact = _code_artifact()
    lineage = _lineage(artifact)
    policy = load_agentic_tools_policy()
    execution = execute_local_lexical_tools(
        knowledge_mode="combined",
        question="Explain the visible AML transaction integration implementation.",
        limit=5,
        policy=policy,
        fdd_documents=_fdd_documents(),
        fdd_generation="functional_specs_v5",
        code_artifact=artifact,
        lineage_artifact=lineage,
    )
    report = build_local_uat_report(
        knowledge_mode="combined",
        question="Explain the visible AML transaction integration implementation.",
        fdd_generation="functional_specs_v5",
        code_snapshot_id=artifact.snapshot_id,
        lineage_artifact_identity_sha256=lineage.artifact_identity_sha256,
        policy_sha256=policy.sha256,
        execution=execution,
    )
    assert report.contains_internal_source_text is True
    assert report.external_api_calls == 0
    assert report.execution.trace.automatic_routing_used is False
    output = tmp_path / "uat.json"
    write_local_uat_report_no_overwrite(report, output)
    with pytest.raises(FileExistsError, match="overwrite"):
        write_local_uat_report_no_overwrite(report, output)


def test_local_uat_cli_requires_internal_evidence_acknowledgement(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_local_bounded_tool_uat.py",
            "--mode",
            "fdd",
            "--question",
            "Explain the documented AML integration behavior.",
            "--output-file",
            str(tmp_path / "uat.json"),
        ],
    )
    with pytest.raises(PermissionError, match="internal source text"):
        run_local_bounded_tool_uat.main()
