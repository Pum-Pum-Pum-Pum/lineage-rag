from app.embeddings.embedding_contract import EmbeddingBatch, EmbeddingRecord
from app.retrieval.query_search import embed_query_text, search_query_text
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


class BadFakeEmbeddingsAPI:
    def create(self, model: str, input: list[str]) -> FakeEmbeddingResponse:
        return FakeEmbeddingResponse([])


class FakeOpenAIClient:
    def __init__(self) -> None:
        self.embeddings = FakeEmbeddingsAPI()


class BadFakeOpenAIClient:
    def __init__(self) -> None:
        self.embeddings = BadFakeEmbeddingsAPI()


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


def test_embed_query_text_returns_one_vector() -> None:
    vector = embed_query_text(
        query_text="branch report",
        embedding_model="text-embedding-3-large",
        embedding_client=FakeOpenAIClient(),
    )

    assert vector == [1.0, 0.0]


def test_embed_query_text_rejects_empty_query() -> None:
    try:
        embed_query_text(
            query_text="   ",
            embedding_model="text-embedding-3-large",
            embedding_client=FakeOpenAIClient(),
        )
    except ValueError as exc:
        assert "must not be empty" in str(exc)
    else:
        raise AssertionError("Expected ValueError for empty query")


def test_embed_query_text_rejects_bad_embedding_response_count() -> None:
    try:
        embed_query_text(
            query_text="branch report",
            embedding_model="text-embedding-3-large",
            embedding_client=BadFakeOpenAIClient(),
        )
    except RuntimeError as exc:
        assert "Expected exactly one query embedding" in str(exc)
    else:
        raise AssertionError("Expected RuntimeError for bad query embedding response count")


def test_search_query_text_returns_expected_payload() -> None:
    qdrant_client = create_local_qdrant_client()
    config = QdrantCollectionConfig(
        collection_name="test_functional_specs",
        vector_size=2,
    )
    ensure_collection(qdrant_client, config)

    batch = EmbeddingBatch(
        document_name="example.docx",
        total_records=2,
        records=[
            _record("unit-close", [1.0, 0.0], "Branch report evidence"),
            _record("unit-far", [0.0, 1.0], "Unrelated evidence"),
        ],
    )
    upsert_embedding_batch(qdrant_client, config.collection_name, batch)

    results = search_query_text(
        qdrant_client=qdrant_client,
        collection_name=config.collection_name,
        query_text="branch report",
        embedding_model="text-embedding-3-large",
        embedding_client=FakeOpenAIClient(),
        limit=1,
    )

    assert len(results) == 1
    assert results[0].payload["unit_id"] == "unit-close"
    assert results[0].payload["text"] == "Branch report evidence"
