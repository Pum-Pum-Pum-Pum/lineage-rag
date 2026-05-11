import json
from pathlib import Path

from app.embeddings.embedding_contract import EmbeddingBatch, EmbeddingRecord
from app.retrieval.retrieval_config import RetrievalRuntimeConfig
from app.services.query_retrieval import retrieve_query_evidence
from app.vectorstore.qdrant_schema import (
    QdrantCollectionConfig,
    create_local_qdrant_client,
    ensure_collection,
)
from app.vectorstore.qdrant_upsert import upsert_embedding_batch


class FakeEmbeddingItem:
    def __init__(self, embedding: list[float]) -> None:
        self.embedding = embedding


class FakeEmbeddingResponse:
    def __init__(self, data: list[FakeEmbeddingItem]) -> None:
        self.data = data


class FakeEmbeddingsAPI:
    def create(self, model: str, input: list[str]) -> FakeEmbeddingResponse:
        return FakeEmbeddingResponse([FakeEmbeddingItem([1.0, 0.0])])


class FakeOpenAIClient:
    def __init__(self) -> None:
        self.embeddings = FakeEmbeddingsAPI()


def _retrieval_config(mode: str = "hybrid") -> RetrievalRuntimeConfig:
    return RetrievalRuntimeConfig(
        retrieval_mode=mode,
        hybrid_dense_weight=0.6,
        hybrid_lexical_weight=0.4,
        hybrid_candidate_limit=10,
    )


def _record(unit_id: str, vector: list[float], text: str) -> EmbeddingRecord:
    return EmbeddingRecord(
        unit_id=unit_id,
        unit_index=0,
        source_kind="paragraph",
        document_family="FS_FCIS_14.7.0.0.0$ASNB",
        release_label="R24",
        content_hash=f"hash-{unit_id}",
        artifact_version="v1",
        cache_key=f"cache-{unit_id}",
        text=text,
        embedding_model="text-embedding-3-large",
        embedding_status="embedded",
        vector=vector,
    )


def _build_qdrant_client_with_dense_records() -> tuple[object, str]:
    client = create_local_qdrant_client()
    config = QdrantCollectionConfig(
        collection_name="test_functional_specs",
        vector_size=2,
    )
    ensure_collection(client, config)
    batch = EmbeddingBatch(
        document_name="example.docx",
        total_records=2,
        records=[
            _record("dense-shared", [1.0, 0.0], "Branch report dense evidence"),
            _record("dense-far", [0.0, 1.0], "Unrelated dense evidence"),
        ],
    )
    upsert_embedding_batch(client, config.collection_name, batch)
    return client, config.collection_name


def _write_retrieval_ready_artifact(tmp_path: Path) -> Path:
    artifact_file = tmp_path / "example.retrieval_ready.json"
    artifact_file.write_text(
        json.dumps(
            {
                "document_name": "example.docx",
                "document_family": "FS_FCIS_14.7.0.0.0$ASNB",
                "release_label": "R24",
                "total_units": 2,
                "units": [
                    {
                        "unit_id": "dense-shared",
                        "unit_index": 0,
                        "source_kind": "paragraph",
                        "text": "Branch report lexical evidence",
                        "document_family": "FS_FCIS_14.7.0.0.0$ASNB",
                        "release_label": "R24",
                    },
                    {
                        "unit_id": "lexical-only",
                        "unit_index": 1,
                        "source_kind": "paragraph",
                        "text": "Branch report exact lexical only evidence",
                        "document_family": "FS_FCIS_14.7.0.0.0$ASNB",
                        "release_label": "R24",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    return tmp_path


def test_retrieve_query_evidence_dense_mode_uses_qdrant(tmp_path: Path) -> None:
    client, collection_name = _build_qdrant_client_with_dense_records()
    artifact_dir = _write_retrieval_ready_artifact(tmp_path)

    routed = retrieve_query_evidence(
        qdrant_client=client,
        collection_name=collection_name,
        query_text="branch report",
        embedding_model="text-embedding-3-large",
        embedding_client=FakeOpenAIClient(),
        retrieval_config=_retrieval_config("dense"),
        lexical_artifact_directory=artifact_dir,
        limit=1,
    )

    assert routed.retrieval_mode == "dense"
    assert routed.results[0].payload["unit_id"] == "dense-shared"


def test_retrieve_query_evidence_lexical_mode_uses_artifacts(tmp_path: Path) -> None:
    client, collection_name = _build_qdrant_client_with_dense_records()
    artifact_dir = _write_retrieval_ready_artifact(tmp_path)

    routed = retrieve_query_evidence(
        qdrant_client=client,
        collection_name=collection_name,
        query_text="exact lexical only",
        embedding_model="text-embedding-3-large",
        embedding_client=FakeOpenAIClient(),
        retrieval_config=_retrieval_config("lexical"),
        lexical_artifact_directory=artifact_dir,
        limit=1,
    )

    assert routed.retrieval_mode == "lexical"
    assert routed.results[0].payload["unit_id"] == "lexical-only"
    assert routed.results[0].payload["retrieval_method"] == "lexical"


def test_retrieve_query_evidence_hybrid_mode_fuses_dense_and_lexical(tmp_path: Path) -> None:
    client, collection_name = _build_qdrant_client_with_dense_records()
    artifact_dir = _write_retrieval_ready_artifact(tmp_path)

    routed = retrieve_query_evidence(
        qdrant_client=client,
        collection_name=collection_name,
        query_text="branch report",
        embedding_model="text-embedding-3-large",
        embedding_client=FakeOpenAIClient(),
        retrieval_config=_retrieval_config("hybrid"),
        lexical_artifact_directory=artifact_dir,
        limit=2,
    )

    assert routed.retrieval_mode == "hybrid"
    assert routed.results[0].payload["retrieval_method"] == "hybrid"
    assert "hybrid_score" in routed.results[0].payload
    assert routed.results[0].payload["unit_id"] == "dense-shared"


def test_retrieve_query_evidence_respects_metadata_filters(tmp_path: Path) -> None:
    client, collection_name = _build_qdrant_client_with_dense_records()
    artifact_dir = _write_retrieval_ready_artifact(tmp_path)

    routed = retrieve_query_evidence(
        qdrant_client=client,
        collection_name=collection_name,
        query_text="branch report",
        embedding_model="text-embedding-3-large",
        embedding_client=FakeOpenAIClient(),
        retrieval_config=_retrieval_config("lexical"),
        lexical_artifact_directory=artifact_dir,
        release_label="R999",
        limit=5,
    )

    assert routed.retrieval_mode == "lexical"
    assert routed.results == []


def test_retrieve_query_evidence_rejects_invalid_limit(tmp_path: Path) -> None:
    client, collection_name = _build_qdrant_client_with_dense_records()
    artifact_dir = _write_retrieval_ready_artifact(tmp_path)

    try:
        retrieve_query_evidence(
            qdrant_client=client,
            collection_name=collection_name,
            query_text="branch report",
            embedding_model="text-embedding-3-large",
            embedding_client=FakeOpenAIClient(),
            retrieval_config=_retrieval_config("dense"),
            lexical_artifact_directory=artifact_dir,
            limit=0,
        )
    except ValueError as exc:
        assert "greater than 0" in str(exc)
    else:
        raise AssertionError("Expected ValueError for invalid retrieval limit")