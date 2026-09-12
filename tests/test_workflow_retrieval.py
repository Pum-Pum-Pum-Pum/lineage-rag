from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from app.code_indexing.models import CodeIndexArtifact, CodeIndexRecord
from app.code_ingestion.code_analysis_models import (
    CodeStaticAnalysisArtifact,
    CodeSymbol,
    DependencyEdge,
    OracleIdentifier,
)
from app.code_ingestion.plsql_models import SourceMap
from app.code_retrieval.models import CodeEvidence
from app.fdd_code_lineage.workflow_retrieval import (
    discover_explicit_routine_workflow,
    enumerate_package_inventory,
    inventory_for_explicit_package_query,
    merge_workflow_fdd_candidates,
    promote_workflow_code_context,
)
from app.fdd_code_lineage.combined_retrieval import retrieve_combined_evidence
from app.fdd_code_lineage.models import build_lineage_artifact
from app.retrieval.lexical_search import LexicalSearchDocument
from app.vectorstore.qdrant_search import QdrantSearchResult


def _map(path: str, start: int, end: int) -> SourceMap:
    return SourceMap(
        source_path=path,
        start_line=start // 10 + 1,
        end_line=end // 10 + 1,
        start_offset=start,
        end_offset=end,
    )


def _symbol(*, name: str, occurrence: str, source_map: SourceMap, kind: str = "procedure") -> CodeSymbol:
    return CodeSymbol(
        occurrence_id=occurrence,
        symbol_key="a" * 64,
        source_node_id=f"node-{name}",
        module_id="fci-custom",
        snapshot_id="snapshot-r2",
        source_path=source_map.source_path,
        source_map=source_map,
        occurrence_role="implementation",
        symbol_kind=kind,
        name=OracleIdentifier(display_name=name, canonical_name=name.upper(), is_quoted=False),
        qualified_display_name=f"PKG_UH.{name}",
        canonical_qualified_name=f"PKG_UH.{name.upper()}",
        overload_discriminator_hash="b" * 64,
        declaration_signature_hash="c" * 64,
        conditional_state="unconditional",
    )


def _record(*, unit_id: str, name: str, source_map: SourceMap, text: str) -> CodeIndexRecord:
    return CodeIndexRecord(
        unit_id=unit_id,
        point_id=f"point-{unit_id}",
        unit_index=0,
        snapshot_id="snapshot-r2",
        module_id="fci-custom",
        source_path=source_map.source_path,
        source_kind="procedure",
        display_name=name,
        package_name="PKG_UH",
        source_map=source_map,
        parent_source_map=source_map,
        parser_state="full_parse",
        conditional_state="unconditional",
        citation_text=text,
        embedding_text=text,
        content_sha256="d" * 64,
        cache_key="e" * 64,
        embedding_model="test-model",
        embedding_status="embedded",
        vector=(1.0, 0.0),
    )


def _evidence(record: CodeIndexRecord) -> CodeEvidence:
    return CodeEvidence(
        unit_id=record.unit_id,
        point_id=record.point_id,
        score=0.9,
        retrieval_method="lexical",
        snapshot_id=record.snapshot_id,
        module_id=record.module_id,
        source_path=record.source_path,
        source_kind=record.source_kind,
        display_name=record.display_name,
        parent_unit_id=record.parent_unit_id,
        package_name=record.package_name,
        start_line=record.source_map.start_line,
        end_line=record.source_map.end_line,
        parser_state=record.parser_state,
        conditional_state=record.conditional_state,
        text=record.citation_text,
    )


def _documents() -> list[LexicalSearchDocument]:
    return [
        LexicalSearchDocument(
            document_name="Death Claim.docx",
            document_id="DEATH_CLAIM_R2",
            unit_id="death-22",
            unit_index=22,
            source_kind="paragraph",
            document_family="ASNB",
            release_label="R2",
            text="Khairat reversal is allowed only on the same day when transmission details are absent.",
        ),
        LexicalSearchDocument(
            document_name="Death Claim.docx",
            document_id="DEATH_CLAIM_R2",
            unit_id="death-23",
            unit_index=23,
            source_kind="paragraph",
            document_family="ASNB",
            release_label="R2",
            text="When RPO changes from Deceased to Normal, clear Khairat details and delete claimant details on authorization.",
        ),
        LexicalSearchDocument(
            document_name="Generic.docx",
            document_id="GENERIC_R1",
            unit_id="generic-1",
            unit_index=1,
            source_kind="paragraph",
            document_family="ASNB",
            release_label="R1",
            text="A maker may delete an unauthorized record before authorization.",
        ),
    ]


def _inputs(tmp_path: Path):
    path = "utpks_utduh_custom.sql"
    target_map = _map(path, 100, 200)
    caller_map = _map(path, 220, 360)
    target = _symbol(name="SpDeleteUnauthKhairatInfo", occurrence="1" * 64, source_map=target_map)
    caller = _symbol(name="Fn_Post_Upload_Db", occurrence="2" * 64, source_map=caller_map, kind="function")
    edge = DependencyEdge(
        edge_id="3" * 64,
        dependency_kind="routine_call",
        source_symbol_occurrence_id=caller.occurrence_id,
        source_path=path,
        source_map=_map(path, 270, 280),
        target_display_name=target.name.display_name,
        target_canonical_name=target.name.canonical_name,
        resolution_state="resolved_in_snapshot",
        candidate_symbol_occurrence_ids=(target.occurrence_id,),
        extraction_method="antlr_tokens",
        confidence="high",
    )
    analysis = CodeStaticAnalysisArtifact(
        module_id="fci-custom",
        snapshot_id="snapshot-r2",
        source_path=path,
        source_sha256="4" * 64,
        analysis_policy_sha256="5" * 64,
        parser_state="full_parse",
        symbols=(target, caller),
        dependencies=(edge,),
    )
    directory = tmp_path / "analysis"
    directory.mkdir()
    (directory / "unit.json").write_text(json.dumps(analysis.model_dump(mode="json")), encoding="utf-8")
    target_record = _record(
        unit_id="target", name=target.name.display_name, source_map=target_map,
        text=(
            "PROCEDURE SpDeleteUnauthKhairatInfo IS\n"
            "DBG('RPO changed from Deceased back to Normal');\n"
            "DELETE claimant details;"
        ),
    )
    caller_record = _record(
        unit_id="caller", name=caller.name.display_name, source_map=_map(path, 270, 280),
        text="IF Khairat was captured same day and Transmission is absent THEN SpDeleteUnauthKhairatInfo; END IF;",
    )
    validation_record = _record(
        unit_id="validation", name=caller.name.display_name, source_map=_map(path, 260, 340),
        text=(
            "IF RPO changes to Normal after Khairat THEN SpDeleteUnauthKhairatInfo; "
            "RAISE 'Khairat was claimed not on same day or Transmission/Hibah details exist'; END IF;"
        ),
    )
    artifact = CodeIndexArtifact(
        status="embedded",
        snapshot_id="snapshot-r2",
        snapshot_content_sha256="6" * 64,
        parse_generation="parse-v1",
        analysis_policy_sha256="5" * 64,
        dependency_review_status="reviewed",
        dependency_review_packet_sha256="7" * 64,
        dependency_review_ledger_sha256="8" * 64,
        module_id="fci-custom",
        embedding_model="test-model",
        vector_dimension=2,
        total_records=3,
        artifact_identity_sha256="9" * 64,
        records=(target_record, caller_record, validation_record),
    )
    return directory, artifact, target_record, caller_record


def test_inventory_is_complete_parser_output_not_ranked_search(tmp_path: Path) -> None:
    directory, _, _, _ = _inputs(tmp_path)

    inventory = enumerate_package_inventory(
        analysis_directory=directory, source_path="utpks_utduh_custom.sql"
    )

    assert inventory.procedures == ("SpDeleteUnauthKhairatInfo",)
    assert inventory.functions == ("Fn_Post_Upload_Db",)
    assert inventory.parser_states == ("full_parse",)


def test_inventory_requires_an_explicit_retrieved_logical_source_name(tmp_path: Path) -> None:
    directory, _, target, _ = _inputs(tmp_path)

    assert inventory_for_explicit_package_query(
        query="List all procedures in utpks_utduh_custom.sql",
        direct_code_evidence=(_evidence(target),),
        analysis_directory=directory,
    ) is not None
    assert inventory_for_explicit_package_query(
        query="List all procedures in this package",
        direct_code_evidence=(_evidence(target),),
        analysis_directory=directory,
    ) is None


def test_exact_routine_discovers_fdd_and_adjacent_context_without_fdd_title(tmp_path: Path) -> None:
    directory, artifact, target, caller = _inputs(tmp_path)

    discovery = discover_explicit_routine_workflow(
        query="Does SpDeleteUnauthKhairatInfo have any documented functional link?",
        direct_code_evidence=(_evidence(target),),
        code_artifact=artifact,
        analysis_directory=directory,
        fdd_documents=_documents(),
        fdd_limit=5,
    )

    assert discovery is not None
    assert discovery.target.name.display_name == "SpDeleteUnauthKhairatInfo"
    assert [item.name.display_name for item in discovery.caller_symbols] == ["Fn_Post_Upload_Db"]
    assert discovery.caller_evidence[0].unit_id == caller.unit_id
    assert discovery.validation_evidence[0].unit_id == "validation"
    assert [item.payload["unit_id"] for item in discovery.fdd_candidates] == ["death-23", "death-22"]
    assert discovery.fdd_candidates[0].payload["retrieval_relation"] == "workflow_fdd_candidate"
    assert discovery.fdd_candidates[1].payload["retrieval_relation"] == "same_document_adjacent_context"


def test_missing_documentation_stays_explicitly_unreviewed(tmp_path: Path) -> None:
    directory, artifact, target, _ = _inputs(tmp_path)

    discovery = discover_explicit_routine_workflow(
        query="Does SpDeleteUnauthKhairatInfo have any documented functional link?",
        direct_code_evidence=(_evidence(target),),
        code_artifact=artifact,
        analysis_directory=directory,
        fdd_documents=(),
        fdd_limit=5,
    )

    assert discovery is not None
    assert discovery.fdd_candidates == ()
    assert discovery.status == "unreviewed_documentation_candidate"


def test_generic_query_cannot_trigger_package_workflow_scan(tmp_path: Path) -> None:
    directory, artifact, target, _ = _inputs(tmp_path)

    discovery = discover_explicit_routine_workflow(
        query="How does the deletion workflow work?",
        direct_code_evidence=(_evidence(target),),
        code_artifact=artifact,
        analysis_directory=directory,
        fdd_documents=_documents(),
        fdd_limit=5,
    )

    assert discovery is None


def test_workflow_context_is_bounded_and_clearly_marked(tmp_path: Path) -> None:
    directory, artifact, target, caller = _inputs(tmp_path)
    discovery = discover_explicit_routine_workflow(
        query="SpDeleteUnauthKhairatInfo",
        direct_code_evidence=(_evidence(target),),
        code_artifact=artifact,
        analysis_directory=directory,
        fdd_documents=_documents(),
        fdd_limit=2,
    )
    assert discovery is not None

    promoted = promote_workflow_code_context(
        current=(_evidence(target),), discovery=discovery, limit=2
    )
    assert [item.unit_id for item in promoted] == [target.unit_id, caller.unit_id]
    assert promoted[0].retrieval_metadata["retrieval_relation"] == "workflow_target_routine"
    assert promoted[1].retrieval_metadata["retrieval_relation"] == "workflow_caller_context"

    existing = [QdrantSearchResult(point_id="other", score=0.1, payload={"unit_id": "other"})]
    merged = merge_workflow_fdd_candidates(
        existing=existing, candidates=discovery.fdd_candidates, limit=2
    )
    assert [item.payload["unit_id"] for item in merged] == ["death-23", "death-22"]


def test_combined_retrieval_exposes_candidate_boundary_not_reviewed_lineage(
    tmp_path: Path,
) -> None:
    directory, artifact, _, _ = _inputs(tmp_path)
    lineage = build_lineage_artifact(
        fdd_generation="functional_specs_v9", code_artifact=artifact, mappings=()
    )
    generic_result = SimpleNamespace(
        point_id="generic-1",
        score=10.0,
        payload={
            "unit_id": "generic-1",
            "document_id": "GENERIC_R1",
            "document_family": "ASNB",
            "release_label": "R1",
            "source_kind": "paragraph",
            "text": "A maker may delete an unauthorized record before authorization.",
        },
    )

    result = retrieve_combined_evidence(
        query="Does SpDeleteUnauthKhairatInfo have any documented functional link?",
        fdd_results=(generic_result,),
        fdd_generation="functional_specs_v9",
        known_fdd_document_ids={"DEATH_CLAIM_R2", "GENERIC_R1"},
        code_artifact=artifact,
        lineage_artifact=lineage,
        analysis_directory=directory,
        code_mode="lexical",
        fdd_limit=2,
        fdd_documents=_documents(),
    )

    assert {item.document_id for item in result.fdd_evidence} == {"DEATH_CLAIM_R2"}
    assert result.fdd_evidence[0].retrieval_metadata["workflow_status"] == (
        "unreviewed_documentation_candidate"
    )
    assert len(result.fdd_evidence) <= 2
    assert len(result.code_evidence) <= 5
    assert not result.reviewed_lineage
    assert "unreviewed documentation candidate" in result.unknowns[0]
