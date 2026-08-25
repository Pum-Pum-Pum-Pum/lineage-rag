from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from app.agentic_tools.evaluation import execute_local_lexical_tools
from app.agentic_tools.policy import load_agentic_tools_policy
from app.agentic_tools.uat import (
    ManualToolUatCase,
    build_local_uat_report,
    build_manual_uat_batch_report,
    build_manual_uat_case_summary,
    load_manual_uat_cases,
    write_manual_uat_packet_no_overwrite,
)
from app.agentic_tools.uat_review import promote_manual_uat_global_acceptance
from tests.test_agentic_tool_evaluation import _code_artifact, _fdd_documents, _lineage


def _case() -> ManualToolUatCase:
    return ManualToolUatCase(
        case_id="uat-combined-test-001",
        source_reviewed_case_id="reviewed-source-001",
        knowledge_mode="combined",
        question="Explain the AML transaction integration and visible custom code.",
        limit=5,
        expected_outcome="evidence",
        expected_fdd_document_ids=("FDD-AML-R24",),
        expected_code_paths=("pkg_aml_custom.sql",),
        expected_code_symbols=("SP_PROCESS_AML",),
        require_reviewed_lineage=True,
        rationale="Tests the batch UAT identity and check boundary.",
    )


def test_manual_uat_manifest_rejects_duplicate_ids(tmp_path: Path) -> None:
    path = tmp_path / "manifest.jsonl"
    line = _case().model_dump_json()
    path.write_text(f"{line}\n{line}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unique"):
        load_manual_uat_cases(path)


def test_manual_uat_batch_is_draft_source_minimized_and_no_overwrite(
    tmp_path: Path,
) -> None:
    case = _case()
    artifact = _code_artifact()
    lineage = _lineage(artifact)
    policy = load_agentic_tools_policy()
    execution = execute_local_lexical_tools(
        knowledge_mode="combined",
        question=case.question,
        limit=case.limit,
        policy=policy,
        fdd_documents=_fdd_documents(),
        fdd_generation="functional_specs_v5",
        code_artifact=artifact,
        lineage_artifact=lineage,
    )
    report = build_local_uat_report(
        knowledge_mode="combined",
        question=case.question,
        fdd_generation="functional_specs_v5",
        code_snapshot_id=artifact.snapshot_id,
        lineage_artifact_identity_sha256=lineage.artifact_identity_sha256,
        policy_sha256=policy.sha256,
        execution=execution,
    )
    summary = build_manual_uat_case_summary(
        case=case, report=report, report_file=tmp_path / "case.json"
    )
    assert summary.diagnostic_passed is True
    batch = build_manual_uat_batch_report(manifest_sha256="a" * 64, summaries=(summary,))
    assert batch.all_cases_reviewed is False
    assert batch.external_api_calls == 0
    packet = tmp_path / "review.md"
    write_manual_uat_packet_no_overwrite(cases=(case,), batch=batch, path=packet)
    text = packet.read_text(encoding="utf-8")
    assert "PROCEDURE SP_PROCESS_AML" not in text
    assert "SME verdict:" in text
    with pytest.raises(FileExistsError, match="overwrite"):
        write_manual_uat_packet_no_overwrite(cases=(case,), batch=batch, path=packet)


def test_review_ledger_hash_matches_persisted_manifest_bytes(tmp_path: Path) -> None:
    case = _case()
    draft = tmp_path / "draft.jsonl"
    draft.write_text(case.model_dump_json() + "\n", encoding="utf-8")
    manifest_hash = hashlib.sha256(draft.read_bytes()).hexdigest()
    batch_identity = "b" * 64
    batch = {
        "manifest_sha256": manifest_hash,
        "batch_identity_sha256": batch_identity,
        "diagnostic_passes": 1,
        "diagnostic_total": 1,
        "cases": [{"case_id": case.case_id, "diagnostic_passed": True}],
    }
    batch_path = tmp_path / "batch.json"
    batch_path.write_text(json.dumps(batch), encoding="utf-8")
    packet = tmp_path / "packet.md"
    packet.write_text(f"- Batch identity: `{batch_identity}`\n", encoding="utf-8")
    reviewed = tmp_path / "reviewed.jsonl"
    ledger_path = tmp_path / "ledger.json"
    ledger = promote_manual_uat_global_acceptance(
        draft_manifest=draft,
        batch_report=batch_path,
        review_packet=packet,
        reviewed_manifest=reviewed,
        ledger_file=ledger_path,
        reviewer="Reviewer",
        approval_note="The exact one-case UAT packet and disclosure are approved.",
        paid_use_authorized=True,
        internal_evidence_disclosure_authorized=True,
    )
    assert hashlib.sha256(reviewed.read_bytes()).hexdigest() == ledger[
        "reviewed_manifest_sha256"
    ]
