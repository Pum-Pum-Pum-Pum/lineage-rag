import json
from pathlib import Path

from app.embeddings.embedding_artifact_writer import write_embedding_batch_to_json
from app.embeddings.embedding_cache import find_cached_embedding, load_embedding_cache
from app.embeddings.embedding_contract import EmbeddingBatch, EmbeddingRecord


def _record(cache_key: str, status: str, vector: list[float] | None) -> EmbeddingRecord:
    return EmbeddingRecord(
        unit_id=f"unit::{cache_key}",
        unit_index=0,
        source_kind="paragraph",
        document_family="FS_FCIS_14.7.0.0.0$ASNB",
        release_label="R24",
        content_hash=f"content-{cache_key}",
        artifact_version="v1",
        cache_key=cache_key,
        text="Example text",
        embedding_model="text-embedding-3-large",
        embedding_status=status,
        vector=vector,
    )


def test_load_embedding_cache_returns_only_embedded_records(tmp_path: Path) -> None:
    cache_dir = tmp_path / "embeddings"
    batch = EmbeddingBatch(
        document_name="example.docx",
        total_records=2,
        records=[
            _record("embedded-key", "embedded", [0.1, 0.2]),
            _record("pending-key", "pending", None),
        ],
    )
    write_embedding_batch_to_json(batch, cache_dir)

    cache = load_embedding_cache(cache_dir)

    assert list(cache.keys()) == ["embedded-key"]
    assert cache["embedded-key"].vector == [0.1, 0.2]


def test_find_cached_embedding_returns_record_by_cache_key(tmp_path: Path) -> None:
    cache_dir = tmp_path / "embeddings"
    batch = EmbeddingBatch(
        document_name="example.docx",
        total_records=1,
        records=[_record("cache-key-1", "embedded", [0.3, 0.4])],
    )
    write_embedding_batch_to_json(batch, cache_dir)

    cached = find_cached_embedding("cache-key-1", cache_dir)


    assert cached is not None
    assert cached.cache_key == "cache-key-1"
    assert cached.vector == [0.3, 0.4]


def test_load_embedding_cache_missing_directory_returns_empty_cache(tmp_path: Path) -> None:
    missing_dir = tmp_path / "missing"

    assert load_embedding_cache(missing_dir) == {}
    assert find_cached_embedding("any-key", missing_dir) is None


def test_load_embedding_cache_raises_on_conflicting_duplicate_cache_key(tmp_path: Path) -> None:
    cache_dir = tmp_path / "embeddings"
    cache_dir.mkdir()

    first = EmbeddingBatch(
        document_name="first.docx",
        total_records=1,
        records=[_record("same-key", "embedded", [0.1, 0.2])],
    )
    second_payload = {
        "document_name": "second.docx",
        "total_records": 1,
        "records": [
            {
                **json.loads(json.dumps(first.records[0].__dict__)),
                "vector": [9.9, 9.8],
            }
        ],
    }

    write_embedding_batch_to_json(first, cache_dir)
    (cache_dir / "conflicting.embeddings.json").write_text(
        json.dumps(second_payload),
        encoding="utf-8",
    )

    try:
        load_embedding_cache(cache_dir)
    except RuntimeError as exc:
        assert "Conflicting cached embeddings" in str(exc)
    else:
        raise AssertionError("Expected RuntimeError for conflicting duplicate cache key")
