from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

from app.agentic_tools.evaluation import (
    BoundedToolEvalCase,
    evaluate_bounded_tools,
    load_eval_cases,
    write_eval_report_no_overwrite,
    write_sme_review_packet_no_overwrite,
)
from app.agentic_tools.policy import load_agentic_tools_policy
from app.code_indexing.models import CodeIndexArtifact, CodeIndexRecord
from app.code_ingestion.plsql_models import SourceMap
from app.fdd_code_lineage.models import (
    FddCodeTarget,
    build_lineage_artifact,
    create_mapping,
)
from app.retrieval.lexical_search import LexicalSearchDocument
from scripts import run_bounded_tool_eval


FDD_ID = "FDD-AML-R24"


def _code_artifact() -> CodeIndexArtifact:
    source_map = SourceMap(
        source_path="pkg_aml_custom.sql",
        start_line=10,
        end_line=20,
        start_offset=100,
        end_offset=300,
    )
    record = CodeIndexRecord(
        unit_id="code-unit-aml",
        point_id="11111111-1111-5111-8111-111111111111",
        unit_index=0,
        snapshot_id="code-snapshot-r1",
        module_id="fci-custom",
        source_path="pkg_aml_custom.sql",
        source_kind="procedure",
        display_name="SP_PROCESS_AML",
        package_name="PKG_AML_CUSTOM",
        source_map=source_map,
        parser_state="full_parse",
        conditional_state="unconditional",
        citation_text="PROCEDURE sp_process_aml IS BEGIN NULL; END;",
        embedding_text="Package PKG_AML_CUSTOM process AML integration FlagRight",
        content_sha256="a" * 64,
        cache_key="b" * 64,
        embedding_model="text-embedding-3-large",
        embedding_status="embedded",
        vector=(1.0, 0.0),
    )
    return CodeIndexArtifact(
        status="embedded",
        snapshot_id="code-snapshot-r1",
        snapshot_content_sha256="c" * 64,
        parse_generation="plsql_antlr_4_13_2_analysis_test",
        analysis_policy_sha256="d" * 64,
        dependency_review_status="reviewed",
        dependency_review_packet_sha256="e" * 64,
        dependency_review_ledger_sha256="f" * 64,
        module_id="fci-custom",
        embedding_model="text-embedding-3-large",
        vector_dimension=2,
        total_records=1,
        artifact_identity_sha256="1" * 64,
        records=(record,),
    )


def _lineage(artifact: CodeIndexArtifact, *, reviewed: bool = True):
    target = FddCodeTarget(
        module_id="fci-custom",
        path="pkg_aml_custom.sql",
        selector_scope="file",
        rationale="The reviewed package implements the visible AML flow.",
    )
    mapping = create_mapping(
        fdd_document_id=FDD_ID,
        fdd_release_label="R24",
        code_snapshot_id=artifact.snapshot_id,
        targets=(target,),
        rationale="The package is linked to the approved AML document.",
        mapping_status="reviewed" if reviewed else "candidate",
        reviewer="AIAgentSmith" if reviewed else None,
    )
    return build_lineage_artifact(
        fdd_generation="functional_specs_v5",
        code_artifact=artifact,
        mappings=(mapping,),
        source_candidate_artifact_identity_sha256="2" * 64 if reviewed else None,
        review_packet_sha256="3" * 64 if reviewed else None,
        reviewer="AIAgentSmith" if reviewed else None,
    )


def _fdd_documents() -> list[LexicalSearchDocument]:
    return [
        LexicalSearchDocument(
            document_name="AML.docx",
            document_id=FDD_ID,
            unit_id="fdd-unit-aml",
            unit_index=0,
            source_kind="paragraph",
            document_family="AML",
            release_label="R24",
            text="FCIS transaction integration with FlagRight.",
            retrieval_text="FCIS transaction integration with FlagRight AML process.",
        )
    ]


def _case(*, reviewed: bool = False) -> BoundedToolEvalCase:
    return BoundedToolEvalCase(
        case_id="tool-combined-test-001",
        knowledge_mode="combined",
        question="Explain the AML process integration with FlagRight.",
        tools=("fdd_search", "code_search", "impact_graph"),
        limit=1,
        expected_fdd_document_ids=(FDD_ID,),
        expected_code_paths=("pkg_aml_custom.sql",),
        expected_code_symbols=("SP_PROCESS_AML",),
        require_reviewed_lineage=True,
        review_status="reviewed" if reviewed else "draft",
        sme_reviewed=reviewed,
        rationale="Tests all three bounded tools against reviewed local lineage.",
    )


def test_manifest_rejects_duplicate_ids_and_wrong_tool_order(tmp_path: Path) -> None:
    case = _case()
    duplicate = tmp_path / "duplicate.jsonl"
    line = case.model_dump_json()
    duplicate.write_text(f"{line}\n{line}\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unique"):
        load_eval_cases(duplicate)

    with pytest.raises(ValidationError, match="mode contract"):
        BoundedToolEvalCase.model_validate(
            {**case.model_dump(mode="json"), "tools": ["impact_graph", "code_search", "fdd_search"]}
        )


def test_draft_manifest_requires_explicit_cli_override(tmp_path: Path, monkeypatch) -> None:
    manifest = tmp_path / "draft.jsonl"
    manifest.write_text(_case().model_dump_json() + "\n", encoding="utf-8")
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_bounded_tool_eval.py",
            "--manifest",
            str(manifest),
            "--output-file",
            str(tmp_path / "report.json"),
        ],
    )
    with pytest.raises(PermissionError, match="--allow-draft"):
        run_bounded_tool_eval.main()


def test_local_combined_tool_evaluation_passes_and_trace_excludes_source(tmp_path: Path) -> None:
    artifact = _code_artifact()
    report = evaluate_bounded_tools(
        cases=(_case(),),
        manifest_sha256="4" * 64,
        policy=load_agentic_tools_policy(),
        fdd_documents=_fdd_documents(),
        fdd_generation="functional_specs_v5",
        code_artifact=artifact,
        lineage_artifact=_lineage(artifact),
    )
    assert report.positive_passes == 1
    assert report.safety_passes == report.safety_total == 5
    assert report.all_cases_reviewed is False
    assert report.release_gate_eligible is False
    assert report.external_api_calls == 0
    trace_text = json.dumps(report.cases[0].execution_trace)
    assert "FCIS transaction integration" not in trace_text
    assert "PROCEDURE sp_process_aml" not in trace_text

    path = tmp_path / "report.json"
    write_eval_report_no_overwrite(report, path)
    with pytest.raises(FileExistsError, match="overwrite"):
        write_eval_report_no_overwrite(report, path)

    packet = tmp_path / "review.md"
    write_sme_review_packet_no_overwrite(cases=(_case(),), report=report, path=packet)
    packet_text = packet.read_text(encoding="utf-8")
    assert "SME verdict: accepted | corrected | needs_more_context" in packet_text
    assert "PROCEDURE sp_process_aml" not in packet_text
    with pytest.raises(FileExistsError, match="overwrite"):
        write_sme_review_packet_no_overwrite(cases=(_case(),), report=report, path=packet)


def test_reviewed_passing_cases_become_gate_eligible() -> None:
    artifact = _code_artifact()
    report = evaluate_bounded_tools(
        cases=(_case(reviewed=True),),
        manifest_sha256="5" * 64,
        policy=load_agentic_tools_policy(),
        fdd_documents=_fdd_documents(),
        fdd_generation="functional_specs_v5",
        code_artifact=artifact,
        lineage_artifact=_lineage(artifact),
    )
    assert report.release_gate_eligible is True


def test_candidate_lineage_fails_before_tool_evaluation() -> None:
    artifact = _code_artifact()
    with pytest.raises(ValueError, match="reviewed lineage"):
        evaluate_bounded_tools(
            cases=(_case(),),
            manifest_sha256="6" * 64,
            policy=load_agentic_tools_policy(),
            fdd_documents=_fdd_documents(),
            fdd_generation="functional_specs_v5",
            code_artifact=artifact,
            lineage_artifact=_lineage(artifact, reviewed=False),
        )
