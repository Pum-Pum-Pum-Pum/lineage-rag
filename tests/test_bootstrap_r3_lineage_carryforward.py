from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.code_indexing.models import CodeIndexArtifact, CodeIndexRecord
from app.code_ingestion.code_analysis_models import CodeStaticAnalysisArtifact, CodeSymbol, OracleIdentifier
from app.code_ingestion.plsql_models import SourceMap
from app.fdd_code_lineage.models import FddCodeTarget, build_lineage_artifact, create_mapping
from app.fdd_code_lineage.reviewed_bundle import build_bundle, write_bundle
from scripts.bootstrap_r3_lineage_carryforward import main


FDD = "FS_FCIS_14.7.0.0.0$ASNB_R22_Test_v1.0"


def _artifact(snapshot: str, *, content: str = "same") -> CodeIndexArtifact:
    source_map = SourceMap(source_path="pkgaml.sql", start_line=1, end_line=3, start_offset=0, end_offset=20)
    record = CodeIndexRecord(
        unit_id=f"unit-{snapshot}", point_id="11111111-1111-5111-8111-111111111111" if snapshot.endswith("r2") else "22222222-2222-5222-8222-222222222222",
        unit_index=0, snapshot_id=snapshot, module_id="fci-custom", source_path="pkgaml.sql",
        source_kind="procedure", display_name="spAML", package_name="PKGAML", source_map=source_map,
        parser_state="full_parse", conditional_state="unconditional", citation_text=content,
        embedding_text=content, content_sha256="a" * 64 if content == "same" else "b" * 64,
        cache_key="c" * 64 if content == "same" else "d" * 64,
        embedding_model="text-embedding-3-large", embedding_status="embedded", vector=(1.0, 0.0),
    )
    return CodeIndexArtifact(
        status="embedded", snapshot_id=snapshot, snapshot_content_sha256="e" * 64,
        parse_generation="plsql_antlr_4_13_2_analysis_v15", analysis_policy_sha256="f" * 64,
        dependency_review_status="reviewed", dependency_review_packet_sha256="1" * 64,
        dependency_review_ledger_sha256="2" * 64, module_id="fci-custom",
        embedding_model="text-embedding-3-large", vector_dimension=2, total_records=1,
        artifact_identity_sha256="3" * 64 if snapshot.endswith("r2") else "4" * 64, records=(record,),
    )


def _write_artifact(path: Path, artifact: CodeIndexArtifact) -> None:
    path.write_text(json.dumps(artifact.model_dump(mode="json")), encoding="utf-8")


def _analysis(path: Path, snapshot: str) -> None:
    source_map = SourceMap(source_path="pkgaml.sql", start_line=1, end_line=3, start_offset=0, end_offset=20)
    symbol = CodeSymbol(
        occurrence_id="5" * 64, symbol_key="6" * 64, source_node_id="node", module_id="fci-custom",
        snapshot_id=snapshot, source_path="pkgaml.sql", source_map=source_map, occurrence_role="implementation",
        symbol_kind="procedure", name=OracleIdentifier(display_name="spAML", canonical_name="SPAML", is_quoted=False),
        qualified_display_name="PKGAML.spAML", canonical_qualified_name="PKGAML.SPAML",
        overload_discriminator_hash="7" * 64, declaration_signature_hash="8" * 64,
        conditional_state="unconditional",
    )
    item = CodeStaticAnalysisArtifact(
        module_id="fci-custom", snapshot_id=snapshot, source_path="pkgaml.sql", source_sha256="9" * 64,
        analysis_policy_sha256="f" * 64, parser_state="full_parse", symbols=(symbol,),
    )
    path.mkdir()
    (path / "analysis.json").write_text(json.dumps(item.model_dump(mode="json")), encoding="utf-8")


def _reviewed_source_lineage(path: Path, source: CodeIndexArtifact) -> None:
    target = FddCodeTarget(module_id="fci-custom", path="pkgaml.sql", qualified_name="PKGAML.SPAML",
        symbol_kind="procedure", selector_scope="overload", overload_discriminator_hash="7" * 64,
        rationale="The reviewed AML requirement maps to this exact source routine.")
    mapping = create_mapping(fdd_document_id=FDD, fdd_release_label="R22", code_snapshot_id=source.snapshot_id,
        targets=(target,), rationale="Original reviewed R2 AML implementation mapping.", mapping_status="reviewed", reviewer="Pum")
    reviewed = build_lineage_artifact(fdd_generation="functional_specs_v9", code_artifact=source,
        mappings=(mapping,), source_candidate_artifact_identity_sha256="a" * 64,
        review_packet_sha256="b" * 64, reviewer="Pum")
    path.write_text(json.dumps(reviewed.model_dump(mode="json")), encoding="utf-8")


def test_bootstrap_emits_r3_candidate_only_when_mapped_records_are_identical(tmp_path: Path) -> None:
    r2, r3 = _artifact("fci-custom-r2"), _artifact("fci-custom-r3")
    r2_path, r3_path = tmp_path / "r2.json", tmp_path / "r3.json"
    _write_artifact(r2_path, r2)
    _write_artifact(r3_path, r3)
    source_lineage = tmp_path / "r2_lineage.json"
    _reviewed_source_lineage(source_lineage, r2)
    analysis = tmp_path / "analysis"
    _analysis(analysis, r3.snapshot_id)
    fdd = tmp_path / "fdd"
    fdd.mkdir()
    (fdd / "one.retrieval_ready.json").write_text(json.dumps({"document_id": FDD}), encoding="utf-8")
    output = tmp_path / "r3_candidate.json"

    assert main([
        "--source-lineage", str(source_lineage), "--source-code-artifact", str(r2_path),
        "--target-code-artifact", str(r3_path), "--analysis-directory", str(analysis),
        "--fdd-processed-directory", str(fdd), "--output", str(output),
    ]) == 0

    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["status"] == "candidate"
    assert payload["code_snapshot_id"] == "fci-custom-r3"
    assert payload["mappings"][0]["mapping_status"] == "candidate"
    assert "R3 carry-forward candidate" in payload["mappings"][0]["rationale"]


def test_bootstrap_rejects_changed_mapped_retrieval_records(tmp_path: Path) -> None:
    r2, r3 = _artifact("fci-custom-r2"), _artifact("fci-custom-r3", content="changed")
    r2_path, r3_path = tmp_path / "r2.json", tmp_path / "r3.json"
    _write_artifact(r2_path, r2)
    _write_artifact(r3_path, r3)
    source_lineage = tmp_path / "r2_lineage.json"
    _reviewed_source_lineage(source_lineage, r2)
    analysis = tmp_path / "analysis"
    _analysis(analysis, r3.snapshot_id)
    fdd = tmp_path / "fdd"
    fdd.mkdir()
    (fdd / "one.retrieval_ready.json").write_text(json.dumps({"document_id": FDD}), encoding="utf-8")

    with pytest.raises(ValueError, match="unchanged retrieval records"):
        main([
            "--source-lineage", str(source_lineage), "--source-code-artifact", str(r2_path),
            "--target-code-artifact", str(r3_path), "--analysis-directory", str(analysis),
            "--fdd-processed-directory", str(fdd), "--output", str(tmp_path / "out.json"),
        ])
