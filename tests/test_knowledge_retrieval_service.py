from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from app.core.config import Settings
from app.retrieval import knowledge_service
from app.retrieval.knowledge_service import (
    KnowledgeRetrievalExecution,
    KnowledgeRetrievalService,
    SourceCatalog,
    _add_request_workbook_companions,
)
from app.services.query_retrieval import PlannedRetrievalResult
from app.retrieval.lexical_search import LexicalSearchDocument
from app.retrieval.retrieval_config import RetrievalRuntimeConfig
from app.vectorstore.qdrant_search import QdrantSearchResult


def _fdd_document(*, unit_id: str = "FDD::chunk_1", text: str = "Evidence text") -> LexicalSearchDocument:
    return LexicalSearchDocument(
        document_name="fdd.docx",
        document_id="FDD_R24.docx",
        unit_id=unit_id,
        unit_index=1,
        source_kind="paragraph",
        document_family="ASNB",
        release_label="R24",
        text=text,
    )


def test_settings_accepts_retrieval_index_alias_and_rejects_conflict(tmp_path: Path) -> None:
    shared = tmp_path / "processed"
    settings = Settings(
        INTERFACE_MODE="mcp",
        PROCESSED_DIR=shared,
        RETRIEVAL_INDEX_PATH=shared,
    )

    assert settings.interface_mode == "mcp"
    assert settings.fdd_retrieval_artifact_dir == shared.resolve()
    assert settings.mcp_evidence_disclosure_enabled is False

    with pytest.raises(ValidationError, match="RETRIEVAL_INDEX_PATH"):
        Settings(
            PROCESSED_DIR=tmp_path / "processed-a",
            RETRIEVAL_INDEX_PATH=tmp_path / "processed-b",
        )


def test_source_catalog_uses_stable_opaque_ids_and_rejects_unknown_fetch() -> None:
    catalog = SourceCatalog.build(
        fdd_documents=[_fdd_document()],
        code_artifact=None,
        fdd_generation="functional_specs_v5",
    )
    source = next(iter(catalog.by_public_id.values()))

    assert source.public_id.startswith("fdd_")
    assert source.internal_unit_id not in source.public_id
    assert source.internal_unit_id not in source.source_reference
    fetched = catalog.fetch(source.public_id)
    assert fetched.text == "Evidence text"
    assert fetched.metadata["fdd_generation"] == "functional_specs_v5"

    with pytest.raises(LookupError, match="unavailable"):
        catalog.fetch("fdd_" + "0" * 64)


def test_source_catalog_rejects_duplicate_active_identity() -> None:
    with pytest.raises(RuntimeError, match="duplicate source identity"):
        SourceCatalog.build(
            fdd_documents=[_fdd_document(), _fdd_document()],
            code_artifact=None,
            fdd_generation="functional_specs_v5",
        )


def test_search_formats_results_from_active_catalog_without_exposing_internal_id(
    monkeypatch,
    tmp_path: Path,
) -> None:
    document = _fdd_document()
    raw = QdrantSearchResult(
        point_id="point-1",
        score=0.91,
        payload={"unit_id": document.unit_id},
    )
    planned = SimpleNamespace(results=[raw])
    settings = SimpleNamespace(
        fdd_generation="functional_specs_v5",
        processed_dir=tmp_path / "processed",
    )
    service = KnowledgeRetrievalService(
        settings=settings,
        retrieval_config=RetrievalRuntimeConfig("lexical", 0.4, 0.6, 10),
        fdd_document_loader=lambda _: [document],
    )
    monkeypatch.setattr(
        service,
        "retrieve",
        lambda **_: KnowledgeRetrievalExecution(
            mode="fdd",
            query="branch report",
            retrieval_mode="lexical",
            fdd=planned,
        ),
    )

    response = service.search(query="branch report", mode="fdd")

    assert response.results[0].id.startswith("fdd_")
    assert document.unit_id not in response.model_dump_json()
    assert response.results[0].short_excerpt == "Evidence text"


def test_request_json_query_retains_same_workbook_request_sheet_and_exposes_sheet_metadata() -> None:
    version = LexicalSearchDocument(
        document_name="rest-api.docx",
        document_id="REST_API_V2_31",
        unit_id="rest::workbook_1::version",
        unit_index=1,
        source_kind="embedded_workbook",
        document_family="ASNB",
        release_label="R4",
        text="Re-Query Transaction Inquiry Service changed in version 1.11.",
        attachment_path="word/embeddings/workbook_23.xlsx",
        attachment_sha256="a" * 64,
        sheet_name="Version",
        sheet_role="version",
        source_range="Version!1:4",
    )
    request = LexicalSearchDocument(
        document_name="rest-api.docx",
        document_id="REST_API_V2_31",
        unit_id="rest::workbook_1::request",
        unit_index=2,
        source_kind="embedded_workbook",
        document_family="ASNB",
        release_label="R4",
        text="row 1: B1=Field Name\nrow 2: B2=upload_requery_txn\nrow 3: B3=channeltype",
        attachment_path="word/embeddings/workbook_23.xlsx",
        attachment_sha256="a" * 64,
        sheet_name="Request",
        sheet_role="request",
        source_range="Request!1:25",
    )
    unrelated_request = LexicalSearchDocument(
        document_name="rest-api.docx",
        document_id="REST_API_V2_31",
        unit_id="rest::workbook_2::request",
        unit_index=3,
        source_kind="embedded_workbook",
        document_family="ASNB",
        release_label="R4",
        text="row 1: B1=Field Name\nrow 2: B2=unrelated_service",
        attachment_path="word/embeddings/workbook_24.xlsx",
        attachment_sha256="b" * 64,
        sheet_name="Request",
        sheet_role="request",
        source_range="Request!1:10",
    )
    original = QdrantSearchResult(point_id="p1", score=0.9, payload={"unit_id": version.unit_id})
    planned = PlannedRetrievalResult(
        routed=SimpleNamespace(),
        results=[original],
        temporal_plan=SimpleNamespace(),
        retrieval_candidate_limit=5,
    )

    enriched = _add_request_workbook_companions(
        planned=planned,
        documents=[version, request, unrelated_request],
        query="What will the JSON request look like for Re-Query Transaction Inquiry Service?",
        limit=5,
    )

    assert [item.payload["unit_id"] for item in enriched.results] == [
        request.unit_id,
        version.unit_id,
    ]
    assert enriched.results[0].payload["retrieval_relation"] == "same_workbook_request_companion"
    catalog = SourceCatalog.build(
        fdd_documents=[version, request], code_artifact=None, fdd_generation="functional_specs_v7"
    )
    request_source = catalog.fdd_by_internal_id[request.unit_id]
    assert request_source.metadata["sheet_role"] == "request"
    assert request_source.metadata["source_range"] == "Request!1:25"


def test_request_companion_requires_json_request_intent_and_service_affinity() -> None:
    version = LexicalSearchDocument(
        document_name="rest-api.docx",
        document_id="REST_API_V2_31",
        unit_id="rest::workbook_1::version",
        unit_index=1,
        source_kind="embedded_workbook",
        document_family="ASNB",
        release_label="R4",
        text="Re-Query Transaction Inquiry Service changed in version 1.11.",
        attachment_path="word/embeddings/workbook_23.xlsx",
        attachment_sha256="a" * 64,
        sheet_name="Version",
        sheet_role="version",
    )
    planned = PlannedRetrievalResult(
        routed=SimpleNamespace(),
        results=[QdrantSearchResult(point_id="p1", score=0.9, payload={"unit_id": version.unit_id})],
        temporal_plan=SimpleNamespace(),
        retrieval_candidate_limit=5,
    )

    assert _add_request_workbook_companions(
        planned=planned,
        documents=[version],
        query="What validations are added for Re-Query Transaction Inquiry Service?",
        limit=5,
    ) is planned


class _FakeQdrant:
    def __init__(self) -> None:
        self.closed = False

    def collection_exists(self, name: str) -> bool:
        return name == "functional_specs_test"

    def close(self) -> None:
        self.closed = True


def test_fdd_dense_retrieval_embeds_once_and_closes_qdrant(monkeypatch, tmp_path: Path) -> None:
    qdrant = _FakeQdrant()
    calls = {"embedding": 0}
    captured: dict[str, object] = {}
    planned = SimpleNamespace(results=[], routed=SimpleNamespace(retrieval_mode="dense"))

    def fake_embed(*, client, model, question, expected_dimension):
        calls["embedding"] += 1
        captured["question"] = question
        assert expected_dimension == 3
        return [0.1, 0.2, 0.3], {"request_id": "embed-1"}

    def fake_planned(**kwargs):
        captured["query_vector"] = kwargs["query_vector"]
        return planned

    monkeypatch.setattr(knowledge_service, "embed_one_query", fake_embed)
    monkeypatch.setattr(knowledge_service, "retrieve_planned_query_evidence", fake_planned)
    settings = SimpleNamespace(
        qdrant_local_path=tmp_path / "qdrant",
        qdrant_collection_name="functional_specs_test",
        qdrant_vector_size=3,
        openai_embedding_model="test-embedding",
        processed_dir=tmp_path / "processed",
        fdd_retrieval_artifact_dir=tmp_path / "processed",
    )
    service = KnowledgeRetrievalService(
        settings=settings,
        retrieval_config=RetrievalRuntimeConfig("dense", 0.4, 0.6, 10),
        qdrant_client_factory=lambda _: qdrant,
        embedding_client_factory=lambda: object(),
    )

    result = service.retrieve(query="current report", mode="fdd")

    assert calls == {"embedding": 1}
    assert captured["question"] == "current report"
    assert captured["query_vector"] == [0.1, 0.2, 0.3]
    assert result.fdd is planned
    assert result.embedding_call == {"request_id": "embed-1"}
    assert qdrant.closed is True


def test_combined_hybrid_reuses_one_query_embedding_for_both_lanes(
    monkeypatch,
    tmp_path: Path,
) -> None:
    calls = {"embedding": 0}
    captured: dict[str, object] = {}
    code_client = _FakeQdrant()
    fdd_client = _FakeQdrant()
    artifact = SimpleNamespace(vector_dimension=3)
    document = _fdd_document()
    planned = SimpleNamespace(results=[])
    combined = SimpleNamespace()
    lineage = SimpleNamespace(status="reviewed", fdd_generation="functional_specs_v5")

    def fake_embed(**kwargs):
        calls["embedding"] += 1
        return [0.1, 0.2, 0.3], {"request_id": "embed-1"}

    def fake_combined(**kwargs):
        captured["combined_vector"] = kwargs["query_vector"]
        return combined

    monkeypatch.setattr(knowledge_service, "embed_one_query", fake_embed)
    monkeypatch.setattr(knowledge_service, "retrieve_planned_query_evidence", lambda **_: planned)
    monkeypatch.setattr(knowledge_service, "retrieve_combined_evidence", fake_combined)
    monkeypatch.setattr(
        knowledge_service.FddCodeLineageArtifact,
        "model_validate_json",
        lambda _: lineage,
    )
    monkeypatch.setattr(knowledge_service, "validate_lineage_artifact", lambda *args, **kwargs: None)
    lineage_path = tmp_path / "lineage.json"
    lineage_path.write_text("{}", encoding="utf-8")
    settings = SimpleNamespace(
        code_modes_enabled=True,
        code_index_artifact_path=tmp_path / "code.json",
        code_qdrant_local_path=tmp_path / "code-qdrant",
        code_qdrant_collection_name="functional_specs_test",
        qdrant_local_path=tmp_path / "fdd-qdrant",
        qdrant_collection_name="functional_specs_test",
        qdrant_vector_size=3,
        openai_embedding_model="test-embedding",
        processed_dir=tmp_path / "processed",
        fdd_retrieval_artifact_dir=tmp_path / "processed",
        fdd_code_lineage_artifact_path=lineage_path,
        code_analysis_directory=tmp_path / "analysis",
        fdd_generation="functional_specs_v5",
    )
    created = []

    def client_factory(path):
        created.append(path)
        return code_client if path == settings.code_qdrant_local_path else fdd_client

    service = KnowledgeRetrievalService(
        settings=settings,
        retrieval_config=RetrievalRuntimeConfig("hybrid", 0.4, 0.6, 10),
        qdrant_client_factory=client_factory,
        embedding_client_factory=lambda: object(),
        code_artifact_loader=lambda _: artifact,
        fdd_document_loader=lambda _: [document],
    )

    result = service.retrieve(query="AML batch", mode="combined")

    assert calls == {"embedding": 1}
    assert captured["combined_vector"] == [0.1, 0.2, 0.3]
    assert result.combined is combined
    assert code_client.closed is True
    assert fdd_client.closed is True
