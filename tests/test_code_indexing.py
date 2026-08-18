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
        parser_generation="plsql_antlr_4_13_2_analysis_v12",
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
