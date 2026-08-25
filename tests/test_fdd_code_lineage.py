from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from app.code_indexing.models import CodeIndexArtifact, CodeIndexRecord
from app.code_ingestion.code_analysis_models import (
    CodeStaticAnalysisArtifact,
    CodeSymbol,
    OracleIdentifier,
)
from app.code_ingestion.plsql_models import SourceMap
from app.fdd_code_lineage.combined_answer import (
    CombinedAnswerDraft,
    CombinedSectionDraft,
    finalize_combined_answer,
)
from app.fdd_code_lineage.combined_retrieval import retrieve_combined_evidence
from app.fdd_code_lineage.models import (
    FddCodeTarget,
    build_lineage_artifact,
    create_mapping,
    validate_lineage_artifact,
    write_lineage_artifact_no_overwrite,
)
from app.fdd_code_lineage.evaluation import CodeCombinedEvalCase
from app.fdd_code_lineage.paid_evaluation import (
    CODE_SYSTEM_PROMPT,
    COMBINED_SYSTEM_PROMPT,
    combined_response_contract_aligned,
    generate_grounded_answer,
)


FDD_ID = "FS_FCIS_14.7.0.0.0$ASNB_R22_Neo_AML_v1.2"
REVIEW_BINDINGS = {
    "source_candidate_artifact_identity_sha256": "7" * 64,
    "review_packet_sha256": "8" * 64,
    "reviewer": "AIAgentSmith",
}


def _code_artifact() -> CodeIndexArtifact:
    source_map = SourceMap(
        source_path="pkgaml_custom.sql",
        start_line=10,
        end_line=20,
        start_offset=100,
        end_offset=300,
    )
    record = CodeIndexRecord(
        unit_id="unit-aml",
        point_id="11111111-1111-5111-8111-111111111111",
        unit_index=0,
        snapshot_id="fci-custom-r1-abc",
        module_id="fci-custom",
        source_path="pkgaml_custom.sql",
        source_kind="procedure",
        display_name="process_aml",
        package_name="PKG_AML_CUSTOM",
        source_map=source_map,
        parser_state="full_parse",
        conditional_state="unconditional",
        citation_text="PROCEDURE process_aml IS BEGIN NULL; END;",
        embedding_text="Package PKG_AML_CUSTOM process AML integration",
        content_sha256="a" * 64,
        cache_key="b" * 64,
        embedding_model="text-embedding-3-large",
        embedding_status="embedded",
        vector=(1.0, 0.0),
    )
    return CodeIndexArtifact(
        status="embedded",
        snapshot_id="fci-custom-r1-abc",
        snapshot_content_sha256="c" * 64,
        parse_generation="plsql_antlr_4_13_2_analysis_v13",
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


def _analysis_directory(tmp_path: Path) -> Path:
    directory = tmp_path / "analysis"
    directory.mkdir()
    source_map = SourceMap(
        source_path="pkgaml_custom.sql",
        start_line=10,
        end_line=20,
        start_offset=100,
        end_offset=300,
    )
    symbol = CodeSymbol(
        occurrence_id="2" * 64,
        symbol_key="3" * 64,
        source_node_id="node-1",
        module_id="fci-custom",
        snapshot_id="fci-custom-r1-abc",
        source_path="pkgaml_custom.sql",
        source_map=source_map,
        occurrence_role="implementation",
        symbol_kind="procedure",
        name=OracleIdentifier(
            display_name="process_aml",
            canonical_name="PROCESS_AML",
            is_quoted=False,
        ),
        qualified_display_name="PKG_AML_CUSTOM.process_aml",
        canonical_qualified_name="PKG_AML_CUSTOM.PROCESS_AML",
        overload_discriminator_hash="4" * 64,
        declaration_signature_hash="5" * 64,
        conditional_state="unconditional",
    )
    analysis = CodeStaticAnalysisArtifact(
        module_id="fci-custom",
        snapshot_id="fci-custom-r1-abc",
        source_path="pkgaml_custom.sql",
        source_sha256="6" * 64,
        analysis_policy_sha256="d" * 64,
        parser_state="full_parse",
        symbols=(symbol,),
    )
    (directory / "aml.json").write_text(
        json.dumps(analysis.model_dump(mode="json")), encoding="utf-8"
    )
    return directory


def _mapping(status: str = "candidate"):
    target = FddCodeTarget(
        module_id="fci-custom",
        path="pkgaml_custom.sql",
        qualified_name="PKG_AML_CUSTOM.PROCESS_AML",
        symbol_kind="procedure",
        overload_discriminator_hash="4" * 64,
        selector_scope="overload",
        rationale="This exact overload implements the reviewed AML flow.",
    )
    return create_mapping(
        fdd_document_id=FDD_ID,
        fdd_release_label="R22",
        code_snapshot_id="fci-custom-r1-abc",
        targets=[target],
        rationale="The SME associates the documented AML flow with this implementation.",
        mapping_status=status,
        reviewer="AIAgentSmith" if status == "reviewed" else None,
    )


def _fdd_result():
    return SimpleNamespace(
        score=0.9,
        payload={
            "unit_id": "fdd-unit",
            "document_id": FDD_ID,
            "document_family": "FS_FCIS_ASNB_NEO_AML",
            "release_label": "R22",
            "source_kind": "paragraph",
            "text": "The integration sends the AML flag for screening.",
        },
    )


def test_mapping_contract_validates_exact_overload_and_is_immutable(tmp_path: Path) -> None:
    code = _code_artifact()
    artifact = build_lineage_artifact(
        fdd_generation="functional_specs_v5",
        code_artifact=code,
        mappings=[_mapping("reviewed")],
        **REVIEW_BINDINGS,
    )
    summary = validate_lineage_artifact(
        artifact,
        fdd_document_ids={FDD_ID},
        code_artifact=code,
        analysis_directory=_analysis_directory(tmp_path),
    )
    assert summary == {"status": "reviewed", "mappings": 1, "targets": 1}
    path = write_lineage_artifact_no_overwrite(artifact, tmp_path / "lineage.json")
    assert path.is_file()
    with pytest.raises(FileExistsError):
        write_lineage_artifact_no_overwrite(artifact, path)


def test_mapping_contract_rejects_unknown_document_path_or_overload(tmp_path: Path) -> None:
    code = _code_artifact()
    analysis = _analysis_directory(tmp_path)
    artifact = build_lineage_artifact(
        fdd_generation="functional_specs_v5",
        code_artifact=code,
        mappings=[_mapping("reviewed")],
        **REVIEW_BINDINGS,
    )
    with pytest.raises(ValueError, match="Unknown FDD"):
        validate_lineage_artifact(
            artifact,
            fdd_document_ids={"different-document"},
            code_artifact=code,
            analysis_directory=analysis,
        )
    bad_target = artifact.mappings[0].targets[0].model_copy(
        update={"overload_discriminator_hash": "9" * 64}
    )
    bad_mapping = artifact.mappings[0].model_copy(update={"targets": (bad_target,)})
    bad_artifact = build_lineage_artifact(
        fdd_generation="functional_specs_v5",
        code_artifact=code,
        mappings=[bad_mapping],
        **REVIEW_BINDINGS,
    )
    with pytest.raises(ValueError, match="no code symbol"):
        validate_lineage_artifact(
            bad_artifact,
            fdd_document_ids={FDD_ID},
            code_artifact=code,
            analysis_directory=analysis,
        )


def test_combined_retrieval_keeps_lanes_separate_and_follows_only_reviewed_links(
    tmp_path: Path,
) -> None:
    code = _code_artifact()
    analysis = _analysis_directory(tmp_path)
    candidate = build_lineage_artifact(
        fdd_generation="functional_specs_v5",
        code_artifact=code,
        mappings=[_mapping("candidate")],
    )
    candidate_result = retrieve_combined_evidence(
        query="Explain the AML integration",
        fdd_results=[_fdd_result()],
        fdd_generation="functional_specs_v5",
        known_fdd_document_ids={FDD_ID},
        code_artifact=code,
        lineage_artifact=candidate,
        analysis_directory=analysis,
        code_mode="lexical",
    )
    assert candidate_result.fdd_evidence
    assert candidate_result.direct_code_evidence
    assert not candidate_result.mapped_code_evidence
    assert not candidate_result.reviewed_lineage
    assert "No reviewed FDD-to-code mapping" in candidate_result.unknowns[0]

    reviewed = build_lineage_artifact(
        fdd_generation="functional_specs_v5",
        code_artifact=code,
        mappings=[_mapping("reviewed")],
        **REVIEW_BINDINGS,
    )
    reviewed_result = retrieve_combined_evidence(
        query="Explain the AML integration",
        fdd_results=[_fdd_result()],
        fdd_generation="functional_specs_v5",
        known_fdd_document_ids={FDD_ID},
        code_artifact=code,
        lineage_artifact=reviewed,
        analysis_directory=analysis,
        code_mode="lexical",
    )
    assert reviewed_result.mapped_code_evidence[0].unit_id == "unit-aml"
    assert reviewed_result.reviewed_lineage[0].mapping_id == reviewed.mappings[0].mapping_id
    assert reviewed_result.fdd_evidence[0].document_id == FDD_ID
    assert reviewed_result.direct_lexical_candidates
    assert reviewed_result.mapped_lexical_candidates


def test_combined_retrieval_caps_merged_direct_and_mapped_evidence(
    tmp_path: Path,
) -> None:
    code = _code_artifact()
    reviewed = build_lineage_artifact(
        fdd_generation="functional_specs_v5",
        code_artifact=code,
        mappings=[_mapping("reviewed")],
        **REVIEW_BINDINGS,
    )
    result = retrieve_combined_evidence(
        query="Explain the AML integration",
        fdd_results=[_fdd_result()],
        fdd_generation="functional_specs_v5",
        known_fdd_document_ids={FDD_ID},
        code_artifact=code,
        lineage_artifact=reviewed,
        analysis_directory=_analysis_directory(tmp_path),
        code_mode="lexical",
        code_limit=1,
    )
    assert len(result.code_evidence) == 1


def test_combined_answer_enforces_lane_specific_citations(tmp_path: Path) -> None:
    code = _code_artifact()
    reviewed = build_lineage_artifact(
        fdd_generation="functional_specs_v5",
        code_artifact=code,
        mappings=[_mapping("reviewed")],
        **REVIEW_BINDINGS,
    )
    retrieval = retrieve_combined_evidence(
        query="Explain the AML integration",
        fdd_results=[_fdd_result()],
        fdd_generation="functional_specs_v5",
        known_fdd_document_ids={FDD_ID},
        code_artifact=code,
        lineage_artifact=reviewed,
        analysis_directory=_analysis_directory(tmp_path),
        code_mode="lexical",
    )
    draft = CombinedAnswerDraft(
        requested_claim_supported=True,
        documented_functionality=CombinedSectionDraft(
            status="answered", text="The documented integration sends an AML flag [F1]."
        ),
        visible_custom_implementation=CombinedSectionDraft(
            status="answered", text="The visible custom routine contains the flow [C1]."
        ),
        impact_and_likely_change_locations=CombinedSectionDraft(
            status="answered", text="This routine is a candidate change location [C1]."
        ),
        unknown_or_unavailable_behavior=CombinedSectionDraft(
            status="answered", text="Kernel behavior is not represented."
        ),
    )
    answer = finalize_combined_answer(retrieval=retrieval, draft=draft)
    assert answer.requested_claim_supported is True
    assert answer.related_grounded_context_provided is False
    assert answer.documented_functionality.status == "answered"
    assert answer.visible_custom_implementation.status == "answered"
    assert answer.fdd_citations[0].citation_id == "F1"
    assert answer.code_citations[0].citation_id == "C1"
    assert answer.reviewed_mapping_ids == (reviewed.mappings[0].mapping_id,)
    assert answer.patch_generation_allowed is False

    crossed = draft.model_copy(
        update={
            "documented_functionality": CombinedSectionDraft(
                status="answered", text="Unsupported cross-lane statement [C1]."
            )
        }
    )
    refused = finalize_combined_answer(retrieval=retrieval, draft=crossed)
    assert refused.documented_functionality.status == "refused"
    assert refused.documented_functionality.refusal_reason == "invalid_or_cross_lane_citation"

    helpful_refusal = finalize_combined_answer(
        retrieval=retrieval,
        draft=draft.model_copy(update={"requested_claim_supported": False}),
    )
    assert helpful_refusal.requested_claim_supported is False
    assert helpful_refusal.related_grounded_context_provided is True


def test_combined_impact_rejects_patch_output(tmp_path: Path) -> None:
    code = _code_artifact()
    reviewed = build_lineage_artifact(
        fdd_generation="functional_specs_v5",
        code_artifact=code,
        mappings=[_mapping("reviewed")],
        **REVIEW_BINDINGS,
    )
    retrieval = retrieve_combined_evidence(
        query="Change AML integration",
        fdd_results=[_fdd_result()],
        fdd_generation="functional_specs_v5",
        known_fdd_document_ids={FDD_ID},
        code_artifact=code,
        lineage_artifact=reviewed,
        analysis_directory=_analysis_directory(tmp_path),
        code_mode="lexical",
    )
    draft = CombinedAnswerDraft(
        requested_claim_supported=True,
        documented_functionality=CombinedSectionDraft(status="refused", text="Not requested."),
        visible_custom_implementation=CombinedSectionDraft(
            status="answered", text="The custom routine is visible [C1]."
        ),
        impact_and_likely_change_locations=CombinedSectionDraft(
            status="answered", text="diff --git a/pkg.sql b/pkg.sql\nCandidate [C1]."
        ),
        unknown_or_unavailable_behavior=CombinedSectionDraft(
            status="answered", text="No kernel source is available."
        ),
    )
    answer = finalize_combined_answer(retrieval=retrieval, draft=draft)
    assert answer.impact_and_likely_change_locations.status == "refused"
    assert answer.impact_and_likely_change_locations.refusal_reason == "patch_generation_not_allowed"


@pytest.mark.parametrize(
    ("raw_response", "expected_error"),
    [
        ("not valid json", "JSONDecodeError"),
        (
            json.dumps(
                {
                    "requested_claim_supported": {
                        "status": "answered",
                        "text": "Incorrect historical response shape.",
                    },
                    "documented_functionality": {"status": "refused", "text": "None."},
                    "visible_custom_implementation": {"status": "refused", "text": "None."},
                    "impact_and_likely_change_locations": {
                        "status": "refused",
                        "text": "None.",
                    },
                    "unknown_or_unavailable_behavior": {
                        "status": "answered",
                        "text": "The response contract is invalid.",
                    },
                }
            ),
            "ValidationError",
        ),
    ],
    ids=("invalid-json", "claim-support-object"),
)
def test_malformed_combined_model_output_becomes_traceable_safe_refusal(
    tmp_path: Path, raw_response: str, expected_error: str
) -> None:
    code = _code_artifact()
    reviewed = build_lineage_artifact(
        fdd_generation="functional_specs_v5",
        code_artifact=code,
        mappings=[_mapping("reviewed")],
        **REVIEW_BINDINGS,
    )
    retrieval = retrieve_combined_evidence(
        query="Explain the AML integration",
        fdd_results=[_fdd_result()],
        fdd_generation="functional_specs_v5",
        known_fdd_document_ids={FDD_ID},
        code_artifact=code,
        lineage_artifact=reviewed,
        analysis_directory=_analysis_directory(tmp_path),
        code_mode="lexical",
    )
    response = SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=raw_response))],
        usage=SimpleNamespace(prompt_tokens=10, completion_tokens=3, total_tokens=13),
        _request_id="answer-malformed-1",
    )
    captured: dict = {}

    def create(**kwargs):
        captured.update(kwargs)
        return response

    client = SimpleNamespace(
        chat=SimpleNamespace(
            completions=SimpleNamespace(create=create)
        )
    )
    case = CodeCombinedEvalCase(
        case_id="combined-contract-failure-001",
        mode="combined",
        question="Explain the visible AML integration flow.",
        expected_code_paths=("pkgaml_custom.sql",),
        expected_fdd_document_ids=(FDD_ID,),
        rationale="Tests fail-closed handling of malformed combined model output.",
    )

    answer, call = generate_grounded_answer(
        client=client,
        model="test-chat",
        case=case,
        retrieval=retrieval,
    )

    assert answer.requested_claim_supported is False
    assert answer.documented_functionality.status == "refused"
    assert answer.visible_custom_implementation.status == "refused"
    assert not answer.fdd_citations
    assert not answer.code_citations
    assert call["contract_valid"] is False
    assert call["contract_error"] == expected_error
    assert call["request_id"] == "answer-malformed-1"
    assert call["raw_response"] == raw_response
    assert call["response_format"] == "json_schema"
    assert len(call["response_schema_sha256"]) == 64
    response_format = captured["response_format"]
    assert response_format["type"] == "json_schema"
    assert response_format["json_schema"]["strict"] is True
    schema = response_format["json_schema"]["schema"]
    assert schema["properties"]["requested_claim_supported"]["type"] == "boolean"
    assert schema["additionalProperties"] is False


def test_combined_structured_output_contract_is_unambiguous() -> None:
    assert combined_response_contract_aligned() is True
    assert "requested_claim_supported is one JSON boolean" in COMBINED_SYSTEM_PROMPT
    assert "Each value must be an object" not in COMBINED_SYSTEM_PROMPT


def test_code_prompt_requires_exact_bracketed_citation_syntax() -> None:
    assert "exact square-bracket syntax" in CODE_SYSTEM_PROMPT
    assert "Never\nwrite a bare citation" in CODE_SYSTEM_PROMPT
    assert '"Evidence: C2"' in CODE_SYSTEM_PROMPT
