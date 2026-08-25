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
)
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
