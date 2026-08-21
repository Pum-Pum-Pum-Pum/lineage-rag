from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from qdrant_client.models import PointStruct

from app.code_indexing.contract import (
    build_code_index_artifact,
    build_code_point_id,
    verify_prepared_code_index_artifact,
    write_code_index_artifact_no_overwrite,
)
from app.code_indexing.embedding import embed_code_index_artifact
from app.code_indexing.lexical import search_code_lexical_artifact
from app.code_indexing.qdrant import (
    index_code_artifact_new_collection,
    verify_code_collection,
)
from app.code_retrieval.answer_contract import (
    CodeUnknownBoundary,
    finalize_code_answer,
)
from app.code_retrieval.models import CodeEvidence
from app.code_retrieval.service import retrieve_code_evidence
from app.code_retrieval.service import _select_parent_diverse
from app.code_ingestion.plsql_models import (
    CodeParseStageManifest,
    CodeRetrievalArtifact,
    CodeRetrievalUnit,
    SourceMap,
)
from app.code_ingestion.dependency_review_ledger import DependencyReviewLedger
from app.vectorstore.qdrant_schema import create_local_qdrant_client


def _prepared(tmp_path: Path):
    stage = tmp_path / "parse"
    (stage / "retrieval").mkdir(parents=True)
    source_map = SourceMap(
        source_path="pkg_claim.sql",
        start_line=10,
        end_line=12,
        start_offset=100,
        end_offset=150,
    )
    retrieval = CodeRetrievalArtifact(
        snapshot_id="fci-custom-r1-abc",
        source_path="pkg_claim.sql",
        total_units=2,
        max_unit_characters=6000,
        overlap_characters=400,
        units=(
            CodeRetrievalUnit(
                unit_id="unit-a",
                source_kind="procedure",
                snapshot_id="fci-custom-r1-abc",
                source_path="pkg_claim.sql",
                source_map=source_map,
                display_name="process_claim",
                package_name="pkg_claim",
                text="PROCEDURE process_claim IS BEGIN NULL; END;",
                retrieval_text="Package: pkg_claim\nPROCEDURE process_claim IS BEGIN NULL; END;",
                parser_state="full_parse",
                conditional_state="unconditional",
            ),
            CodeRetrievalUnit(
                unit_id="unit-b",
                source_kind="procedure",
                snapshot_id="fci-custom-r1-abc",
                source_path="pkg_claim.sql",
                source_map=source_map.model_copy(update={"start_line": 20, "end_line": 22, "start_offset": 200, "end_offset": 250}),
                display_name="audit_claim",
                package_name="pkg_claim",
                text="PROCEDURE audit_claim IS BEGIN NULL; END;",
                retrieval_text="Package: pkg_claim\nPROCEDURE audit_claim IS BEGIN NULL; END;",
                parser_state="full_parse",
                conditional_state="unconditional",
            ),
        ),
    )
    retrieval_path = stage / "retrieval/code.json"
    retrieval_path.write_text(json.dumps(retrieval.model_dump(mode="json")), encoding="utf-8")
    manifest = CodeParseStageManifest(
        status="complete",
        snapshot_id="fci-custom-r1-abc",
        snapshot_content_sha256="a" * 64,
        analysis_policy_sha256="b" * 64,
        file_count=1,
        state_counts={"full_parse": 1, "segmented_parse": 0, "fallback_parse": 0, "failed": 0},
        parse_artifacts=("parse/code.json",),
        retrieval_artifacts=("retrieval/code.json",),
        analysis_artifacts=("analysis/code.json",),
        timeout_seconds=120,
        memory_limit_bytes=1024,
        max_segment_characters=500,
        max_retrieval_unit_characters=6000,
        retrieval_overlap_characters=400,
    )
    (stage / "parse_stage_manifest.json").write_text(
        json.dumps(manifest.model_dump(mode="json")), encoding="utf-8"
    )
    return build_code_index_artifact(stage, embedding_model="text-embedding-3-large")


class _FakeEmbeddings:
    def __init__(self):
        self.calls = []

    def create(self, *, model, input):
        self.calls.append((model, list(input)))
        return SimpleNamespace(
            data=[
                SimpleNamespace(index=index, embedding=[float(index + 1), 0.5])
                for index in reversed(range(len(input)))
            ]
        )


def test_code_contract_is_deterministic_and_lexically_searchable(tmp_path: Path) -> None:
    first = _prepared(tmp_path)
    second = build_code_index_artifact(tmp_path / "parse", embedding_model="text-embedding-3-large")

    assert first == second
    assert first.total_records == 2
    assert len({record.point_id for record in first.records}) == 2
    assert first.records[0].point_id == build_code_point_id(first.snapshot_id, first.records[0].unit_id)
    result = search_code_lexical_artifact(first, "process_claim", limit=1)
    assert result[0].payload["unit_id"] == "unit-a"
    summary = verify_prepared_code_index_artifact(
        first,
        tmp_path / "parse",
        expected_policy_sha256="b" * 64,
    )
    assert summary["status"] == "pass"


def test_prepared_contract_verifier_rejects_wrong_policy_or_tampering(tmp_path: Path) -> None:
    prepared = _prepared(tmp_path)
    with pytest.raises(RuntimeError, match="policy hash"):
        verify_prepared_code_index_artifact(
            prepared,
            tmp_path / "parse",
            expected_policy_sha256="c" * 64,
        )

    tampered_record = prepared.records[0].model_copy(update={"citation_text": "tampered"})
    tampered = prepared.model_copy(
        update={"records": (tampered_record, *prepared.records[1:])}
    )
    with pytest.raises(RuntimeError, match="deterministic rebuild"):
        verify_prepared_code_index_artifact(
            tampered,
            tmp_path / "parse",
            expected_policy_sha256="b" * 64,
        )


def test_reviewed_contract_is_bound_to_matching_dependency_ledger(tmp_path: Path) -> None:
    _prepared(tmp_path)
    ledger = DependencyReviewLedger.model_construct(
        status="reviewed",
        reviewer="project-sme",
        snapshot_id="fci-custom-r1-abc",
        parser_generation="plsql_antlr_4_13_2_analysis_v13",
        analysis_policy_sha256="b" * 64,
        packet_identity_sha256="d" * 64,
        packet_json_sha256="e" * 64,
        reviewed_markdown_sha256="f" * 64,
        ledger_identity_sha256="1" * 64,
        decisions=(),
        external_calls_performed=False,
    )
    artifact = build_code_index_artifact(
        tmp_path / "parse",
        embedding_model="text-embedding-3-large",
        dependency_review_ledger=ledger,
    )

    assert artifact.dependency_review_status == "reviewed"
    assert artifact.dependency_review_packet_sha256 == "d" * 64
    assert artifact.dependency_review_ledger_sha256 == "1" * 64
    assert verify_prepared_code_index_artifact(
        artifact,
        tmp_path / "parse",
        expected_policy_sha256="b" * 64,
        dependency_review_ledger=ledger,
    )["status"] == "pass"

    wrong = ledger.model_copy(update={"parser_generation": "analysis-wrong"})
    with pytest.raises(ValueError, match="does not match"):
        build_code_index_artifact(
            tmp_path / "parse",
            embedding_model="text-embedding-3-large",
            dependency_review_ledger=wrong,
        )

def test_code_embedding_maps_reordered_response_and_reuses_cache(tmp_path: Path) -> None:
    prepared = _prepared(tmp_path)
    embeddings = _FakeEmbeddings()
    client = SimpleNamespace(embeddings=embeddings)

    embedded, summary = embed_code_index_artifact(prepared, client=client, request_batch_size=1)

    assert embedded.status == "embedded"
    assert embedded.vector_dimension == 2
    assert summary.request_count == 2
    cache_path = write_code_index_artifact_no_overwrite(embedded, tmp_path / "embedded")
    cached, cached_summary = embed_code_index_artifact(
        prepared,
        client=SimpleNamespace(embeddings=_FakeEmbeddings()),
        cache_artifact_paths=[cache_path],
    )
    assert cached_summary.cached_records == 2
    assert cached_summary.request_count == 0
    assert [record.vector for record in cached.records] == [record.vector for record in embedded.records]


def test_qdrant_generation_is_exact_and_does_not_touch_rollback_collection(tmp_path: Path) -> None:
    prepared = _prepared(tmp_path)
    embedded, _ = embed_code_index_artifact(
        prepared,
        client=SimpleNamespace(embeddings=_FakeEmbeddings()),
    )
    client = create_local_qdrant_client()
    client.create_collection(
        collection_name="code_custom_rollback_v0",
        vectors_config={"size": 2, "distance": "Cosine"},
    )

    result = index_code_artifact_new_collection(
        client,
        collection_name="code_custom_r1_v1",
        artifact=embedded,
    )

    assert result.verified_points == 2
    assert client.collection_exists("code_custom_rollback_v0")
    assert client.count("code_custom_rollback_v0", exact=True).count == 0
    verify_code_collection(client, collection_name="code_custom_r1_v1", artifact=embedded)
    with pytest.raises(FileExistsError, match="will not be modified"):
        index_code_artifact_new_collection(
            client,
            collection_name="code_custom_r1_v1",
            artifact=embedded,
        )


def test_qdrant_verifier_rejects_extra_point(tmp_path: Path) -> None:
    prepared = _prepared(tmp_path)
    embedded, _ = embed_code_index_artifact(
        prepared,
        client=SimpleNamespace(embeddings=_FakeEmbeddings()),
    )
    client = create_local_qdrant_client()
    index_code_artifact_new_collection(
        client,
        collection_name="code_custom_r1_v1",
        artifact=embedded,
    )
    client.upsert(
        collection_name="code_custom_r1_v1",
        points=[PointStruct(id="46b00d6c-63de-4ff0-bcea-f68646e5d14d", vector=[0.0, 1.0], payload={})],
        wait=True,
    )

    with pytest.raises(RuntimeError, match="point count mismatch"):
        verify_code_collection(client, collection_name="code_custom_r1_v1", artifact=embedded)


def _reviewed_embedded(tmp_path: Path):
    prepared = _prepared(tmp_path)
    embedded, _ = embed_code_index_artifact(
        prepared,
        client=SimpleNamespace(embeddings=_FakeEmbeddings()),
    )
    return embedded.model_copy(
        update={
            "dependency_review_status": "reviewed",
            "dependency_review_packet_sha256": "d" * 64,
            "dependency_review_ledger_sha256": "e" * 64,
        }
    )


def test_code_retrieval_modes_preserve_exact_artifact_provenance(tmp_path: Path) -> None:
    artifact = _reviewed_embedded(tmp_path)
    client = create_local_qdrant_client()
    index_code_artifact_new_collection(
        client,
        collection_name="code_custom_test_v1",
        artifact=artifact,
    )

    lexical = retrieve_code_evidence(
        artifact=artifact,
        query="process_claim",
        mode="lexical",
        limit=1,
    )
    dense = retrieve_code_evidence(
        artifact=artifact,
        query="process claim",
        mode="dense",
        limit=1,
        client=client,
        collection_name="code_custom_test_v1",
        query_vector=artifact.records[0].vector,
    )
    hybrid = retrieve_code_evidence(
        artifact=artifact,
        query="process_claim",
        mode="hybrid",
        limit=1,
        client=client,
        collection_name="code_custom_test_v1",
        query_vector=artifact.records[0].vector,
    )

    for result in (lexical, dense, hybrid):
        assert result.evidence[0].unit_id == "unit-a"
        assert result.evidence[0].source_path == "pkg_claim.sql"
        assert result.evidence[0].start_line == 10
        assert result.evidence[0].text.startswith("PROCEDURE process_claim")
    assert hybrid.evidence[0].retrieval_metadata["contributing_retrievers"] == [
        "dense",
        "lexical",
    ]
    assert hybrid.dense_candidates and hybrid.lexical_candidates


def test_code_retrieval_limits_repeated_children_per_parent(tmp_path: Path) -> None:
    artifact = _reviewed_embedded(tmp_path)
    base = artifact.records[0]
    records = []
    for index in range(3):
        records.append(
            base.model_copy(
                update={
                    "unit_id": f"child-{index}",
                    "point_id": f"point-child-{index}",
                    "unit_index": index,
                    "display_name": "large_parent_routine",
                    "parent_unit_id": "shared-parent",
                    "source_map": base.source_map.model_copy(
                        update={
                            "start_line": 10 + index * 10,
                            "end_line": 19 + index * 10,
                            "start_offset": 100 + index * 100,
                            "end_offset": 199 + index * 100,
                        }
                    ),
                    "embedding_text": "offline AML transaction " * 8,
                    "citation_text": "offline AML transaction " * 8,
                }
            )
        )
    records.append(
        base.model_copy(
            update={
                "unit_id": "target-unit",
                "point_id": "point-target",
                "unit_index": 3,
                "display_name": "target_offline_routine",
                "parent_unit_id": None,
                "source_map": base.source_map.model_copy(
                    update={
                        "start_line": 50,
                        "end_line": 60,
                        "start_offset": 500,
                        "end_offset": 600,
                    }
                ),
                "embedding_text": "offline AML transaction target routine",
                "citation_text": "offline AML transaction target routine",
            }
        )
    )
    diverse_artifact = artifact.model_copy(
        update={"records": tuple(records), "total_records": len(records)}
    )

    result = retrieve_code_evidence(
        artifact=diverse_artifact,
        query="offline AML transaction",
        mode="lexical",
        limit=3,
        candidate_limit=4,
        max_units_per_parent=2,
    )

    assert sum(item.parent_unit_id == "shared-parent" for item in result.evidence) == 2
    assert "target_offline_routine" in {item.display_name for item in result.evidence}
    assert len(result.lexical_candidates) == 4


def test_code_retrieval_rejects_invalid_parent_limit(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="max_units_per_parent"):
        retrieve_code_evidence(
            artifact=_reviewed_embedded(tmp_path),
            query="claim",
            mode="lexical",
            max_units_per_parent=0,
        )


def test_parent_diversity_prefers_unseen_parent_before_second_child(
    tmp_path: Path,
) -> None:
    artifact = _reviewed_embedded(tmp_path)
    base = artifact.records[0]
    records = {
        "a1": base.model_copy(update={"unit_id": "a1", "parent_unit_id": "parent-a"}),
        "b1": base.model_copy(update={"unit_id": "b1", "parent_unit_id": "parent-b"}),
        "a2": base.model_copy(update={"unit_id": "a2", "parent_unit_id": "parent-a"}),
        "c1": base.model_copy(update={"unit_id": "c1", "parent_unit_id": "parent-c"}),
    }
    ranked = [
        SimpleNamespace(payload={"unit_id": unit_id})
        for unit_id in ("a1", "b1", "a2", "c1")
    ]

    selected = _select_parent_diverse(
        ranked, records, limit=3, max_units_per_parent=2
    )

    assert [item.payload["unit_id"] for item in selected] == ["a1", "b1", "c1"]


def test_code_dense_retrieval_fails_closed_on_generation_or_dimension_error(
    tmp_path: Path,
) -> None:
    artifact = _reviewed_embedded(tmp_path)
    client = create_local_qdrant_client()
    index_code_artifact_new_collection(
        client,
        collection_name="code_custom_test_v1",
        artifact=artifact,
    )

    with pytest.raises(ValueError, match="dimension"):
        retrieve_code_evidence(
            artifact=artifact,
            query="claim",
            mode="dense",
            client=client,
            collection_name="code_custom_test_v1",
            query_vector=[1.0],
        )
    with pytest.raises(RuntimeError, match="does not exist"):
        retrieve_code_evidence(
            artifact=artifact,
            query="claim",
            mode="dense",
            client=client,
            collection_name="code_custom_missing_v1",
            query_vector=artifact.records[0].vector,
        )

    client.set_payload(
        collection_name="code_custom_test_v1",
        payload={"source_path": "tampered.sql"},
        points=[artifact.records[0].point_id],
        wait=True,
    )
    with pytest.raises(RuntimeError, match="identity mismatch"):
        retrieve_code_evidence(
            artifact=artifact,
            query="claim",
            mode="dense",
            limit=1,
            client=client,
            collection_name="code_custom_test_v1",
            query_vector=artifact.records[0].vector,
        )


def _evidence(**updates) -> CodeEvidence:
    values = {
        "unit_id": "unit-a",
        "point_id": "point-a",
        "score": 0.9,
        "retrieval_method": "hybrid",
        "snapshot_id": "fci-custom-r1-abc",
        "module_id": "fci-custom",
        "source_path": "pkg_claim.sql",
        "source_kind": "procedure",
        "display_name": "process_claim",
        "package_name": "pkg_claim",
        "start_line": 10,
        "end_line": 12,
        "parser_state": "full_parse",
        "conditional_state": "unconditional",
        "text": "PROCEDURE process_claim IS BEGIN NULL; END;",
    }
    values.update(updates)
    return CodeEvidence(**values)


def test_code_answer_contract_uses_original_lines_and_valid_citations() -> None:
    response = finalize_code_answer(
        query="How is a claim processed?",
        generated_content=(
            "DECISION: ANSWER\nThe visible custom procedure currently contains "
            "a no-op implementation [C1]."
        ),
        evidence=[_evidence()],
    )

    assert response.is_answered is True
    assert response.citations[0].source_path == "pkg_claim.sql"
    assert (response.citations[0].start_line, response.citations[0].end_line) == (10, 12)
    assert response.citations[0].text_preview.startswith("PROCEDURE process_claim")


@pytest.mark.parametrize(
    ("content", "reason"),
    [
        ("The procedure does something [C1].", "invalid_answer_contract"),
        ("DECISION: ANSWER\nUnsupported claim.", "invalid_or_missing_citation"),
        ("DECISION: ANSWER\nUnsupported [C9].", "invalid_or_missing_citation"),
        ("DECISION: ANSWER\ndiff --git a/a b/a", "patch_generation_not_allowed"),
    ],
)
def test_code_answer_contract_fails_closed(content: str, reason: str) -> None:
    response = finalize_code_answer(
        query="Explain the code",
        generated_content=content,
        evidence=[_evidence()],
    )
    assert response.is_answered is False
    assert response.refusal_reason == reason


def test_code_answer_contract_preserves_unknowns_and_impact_limitations() -> None:
    response = finalize_code_answer(
        query="Where should this defect be fixed?",
        generated_content="DECISION: ANSWER\nA candidate location is shown in [C1].",
        evidence=[
            _evidence(
                parser_state="fallback_parse",
                conditional_state="conditional_unknown",
            )
        ],
        analysis_kind="impact_analysis",
        additional_unknowns=[
            CodeUnknownBoundary(
                kind="kernel_unavailable",
                detail="The called kernel package is not in the approved snapshot.",
                unit_id="unit-a",
            )
        ],
    )

    assert response.is_answered is True
    assert {item.kind for item in response.unknowns} == {
        "parser_degradation",
        "conditional_unknown",
        "kernel_unavailable",
    }
    assert "not proven root causes" in response.impact_limitation
    assert response.patch_generation_allowed is False


def test_code_answer_contract_refuses_when_no_evidence() -> None:
    response = finalize_code_answer(
        query="Explain unavailable package",
        generated_content="DECISION: ANSWER\nInvented answer [C1].",
        evidence=[],
    )
    assert response.is_answered is False
    assert response.refusal_reason == "no_code_evidence"
    assert response.unknowns[0].kind == "missing_snapshot"
