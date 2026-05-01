from qdrant_client.models import Distance

from app.vectorstore.qdrant_schema import (
    QdrantCollectionConfig,
    create_local_qdrant_client,
    create_persistent_qdrant_client,
    ensure_collection,
)


def test_ensure_collection_creates_local_in_memory_collection() -> None:
    client = create_local_qdrant_client()
    config = QdrantCollectionConfig(
        collection_name="test_functional_specs",
        vector_size=3072,
        distance=Distance.COSINE,
    )

    ensure_collection(client, config)

    collection_info = client.get_collection(config.collection_name)

    assert collection_info.config.params.vectors.size == 3072
    assert collection_info.config.params.vectors.distance == Distance.COSINE


def test_ensure_collection_is_idempotent() -> None:
    client = create_local_qdrant_client()
    config = QdrantCollectionConfig(
        collection_name="test_functional_specs",
        vector_size=3072,
    )

    ensure_collection(client, config)
    ensure_collection(client, config)

    assert client.collection_exists(config.collection_name) is True


def test_persistent_qdrant_collection_survives_client_reopen(tmp_path) -> None:
    storage_path = tmp_path / "qdrant_local"
    config = QdrantCollectionConfig(
        collection_name="persistent_functional_specs",
        vector_size=3072,
    )

    first_client = create_persistent_qdrant_client(storage_path)
    ensure_collection(first_client, config)
    assert first_client.collection_exists(config.collection_name) is True
    first_client.close()

    second_client = create_persistent_qdrant_client(storage_path)
    assert second_client.collection_exists(config.collection_name) is True
    second_client.close()
