from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from scripts import check_qdrant_index, run_qdrant_indexing


def test_qdrant_indexing_closes_client_on_success(monkeypatch, tmp_path: Path) -> None:
    fake_client = _IndexingClient()
    captured: dict[str, object] = {}

    def fake_index(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(
            collection_name="lineage_chunks",
            attempted_records=2,
            upserted_points=2,
            skipped_records=0,
        )

    monkeypatch.setattr(run_qdrant_indexing, "get_settings", lambda: _settings(tmp_path))
    monkeypatch.setattr(
        run_qdrant_indexing,
        "create_persistent_qdrant_client",
        lambda path: fake_client,
    )
    monkeypatch.setattr(run_qdrant_indexing, "index_embedding_cache_directory", fake_index)

    run_qdrant_indexing.main()

    assert captured["client"] is fake_client
    assert captured["cache_directory"] == tmp_path / "cache" / "embeddings"
    assert fake_client.closed is True


def test_qdrant_indexing_closes_client_when_indexing_fails(
    monkeypatch,
    tmp_path: Path,
) -> None:
    fake_client = _IndexingClient()

    def fail_index(**kwargs):
        raise RuntimeError("Qdrant upsert failed")

    monkeypatch.setattr(run_qdrant_indexing, "get_settings", lambda: _settings(tmp_path))
    monkeypatch.setattr(
        run_qdrant_indexing,
        "create_persistent_qdrant_client",
        lambda path: fake_client,
    )
    monkeypatch.setattr(run_qdrant_indexing, "index_embedding_cache_directory", fail_index)

    with pytest.raises(RuntimeError, match="Qdrant upsert failed"):
        run_qdrant_indexing.main()

    assert fake_client.closed is True


def test_qdrant_indexing_rejects_legacy_rebuild_cli_before_collection_access() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/run_qdrant_indexing.py", "--rebuild"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode != 0
    assert "unsupported for embedded local Qdrant" in result.stderr


def test_qdrant_index_check_closes_client_when_collection_is_missing(
    monkeypatch,
    tmp_path: Path,
) -> None:
    fake_client = _CheckClient(collection_exists=False)
    _patch_check_script(monkeypatch, tmp_path, fake_client)

    check_qdrant_index.main()

    assert fake_client.collection_exists_calls == ["lineage_chunks"]
    assert fake_client.closed is True


def test_qdrant_index_check_closes_client_on_success(monkeypatch, tmp_path: Path) -> None:
    fake_client = _CheckClient(collection_exists=True)
    _patch_check_script(monkeypatch, tmp_path, fake_client)

    check_qdrant_index.main()

    assert fake_client.get_collection_calls == ["lineage_chunks"]
    assert fake_client.count_calls == ["lineage_chunks"]
    assert fake_client.scroll_calls == ["lineage_chunks"]
    assert fake_client.closed is True


def test_qdrant_index_check_closes_client_when_inspection_fails(
    monkeypatch,
    tmp_path: Path,
) -> None:
    fake_client = _CheckClient(collection_exists=True, fail_get_collection=True)
    _patch_check_script(monkeypatch, tmp_path, fake_client)

    with pytest.raises(RuntimeError, match="collection inspection failed"):
        check_qdrant_index.main()

    assert fake_client.closed is True


class _IndexingClient:
    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


class _CheckClient:
    def __init__(
        self,
        *,
        collection_exists: bool,
        fail_get_collection: bool = False,
    ) -> None:
        self._collection_exists = collection_exists
        self.fail_get_collection = fail_get_collection
        self.collection_exists_calls: list[str] = []
        self.get_collection_calls: list[str] = []
        self.count_calls: list[str] = []
        self.scroll_calls: list[str] = []
        self.closed = False

    def collection_exists(self, collection_name: str) -> bool:
        self.collection_exists_calls.append(collection_name)
        return self._collection_exists

    def get_collection(self, collection_name: str):
        self.get_collection_calls.append(collection_name)
        if self.fail_get_collection:
            raise RuntimeError("collection inspection failed")
        return SimpleNamespace(
            config=SimpleNamespace(
                params=SimpleNamespace(
                    vectors=SimpleNamespace(size=3072, distance="Cosine"),
                )
            )
        )

    def count(self, collection_name: str):
        self.count_calls.append(collection_name)
        return SimpleNamespace(count=2)

    def scroll(self, *, collection_name: str, **kwargs):
        self.scroll_calls.append(collection_name)
        return [], None

    def close(self) -> None:
        self.closed = True


def _patch_check_script(monkeypatch, tmp_path: Path, fake_client: _CheckClient) -> None:
    monkeypatch.setattr(check_qdrant_index, "get_settings", lambda: _settings(tmp_path))
    monkeypatch.setattr(
        check_qdrant_index,
        "create_persistent_qdrant_client",
        lambda path: fake_client,
    )


def _settings(tmp_path: Path) -> SimpleNamespace:
    return SimpleNamespace(
        log_level="INFO",
        cache_dir=tmp_path / "cache",
        qdrant_local_path=tmp_path / "qdrant",
        qdrant_collection_name="lineage_chunks",
        qdrant_vector_size=3072,
    )
