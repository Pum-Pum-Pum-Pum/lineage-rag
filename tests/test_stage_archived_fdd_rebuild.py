from pathlib import Path
from types import SimpleNamespace
import json

import pytest

from app.embeddings.embedding_contract import EmbeddingRecord
from scripts import stage_archived_fdd_rebuild


class FakeQdrantClient:
    def __init__(self, exists: bool) -> None:
        self.exists = exists
        self.closed = False

    def collection_exists(self, collection_name: str) -> bool:
        return self.exists

    def close(self) -> None:
        self.closed = True


def test_dry_run_hashes_archived_sources_without_creating_stage(monkeypatch, tmp_path: Path) -> None:
    source_directory = tmp_path / "docs_embedded"
    source_directory.mkdir()
    source = source_directory / "FS_ASNB_R25_Staged.docx"
    source.write_bytes(b"archived-source")
    stage_directory = tmp_path / "staging" / "table_context_v1"
    client = FakeQdrantClient(exists=False)
    settings = SimpleNamespace(
        data_dir=tmp_path / "data",
        embedded_docs_dir=source_directory,
        artifact_version="v1",
        log_level="INFO",
        qdrant_collection_name="functional_specs_v2",
        qdrant_local_path=tmp_path / "qdrant",
        cache_dir=tmp_path / "cache",
    )

    monkeypatch.setattr(stage_archived_fdd_rebuild, "get_settings", lambda: settings)
    monkeypatch.setattr(
        stage_archived_fdd_rebuild,
        "create_persistent_qdrant_client",
        lambda path: client,
    )

    stage_archived_fdd_rebuild.main(
        ["--dry-run", "--stage-directory", str(stage_directory), "--collection-name", "functional_specs_v3"]
    )

    assert not stage_directory.exists()
    assert client.closed is True


def test_stage_rejects_active_collection_before_any_stage_write(monkeypatch, tmp_path: Path) -> None:
    stage_directory = tmp_path / "stage"
    client = FakeQdrantClient(exists=False)
    monkeypatch.setattr(
        stage_archived_fdd_rebuild,
        "create_persistent_qdrant_client",
        lambda path: client,
    )

    with pytest.raises(ValueError, match="differ from the active collection"):
        stage_archived_fdd_rebuild.validate_stage_targets(
            stage_directory=stage_directory,
            target_collection_name="functional_specs_v2",
            active_collection_name="functional_specs_v2",
            qdrant_local_path=tmp_path / "qdrant",
        )

    assert not stage_directory.exists()


def test_stage_rejects_existing_target_collection_without_deleting_it(monkeypatch, tmp_path: Path) -> None:
    stage_directory = tmp_path / "stage"
    client = FakeQdrantClient(exists=True)
    monkeypatch.setattr(
        stage_archived_fdd_rebuild,
        "create_persistent_qdrant_client",
        lambda path: client,
    )

    with pytest.raises(FileExistsError, match="already exists"):
        stage_archived_fdd_rebuild.validate_stage_targets(
            stage_directory=stage_directory,
            target_collection_name="functional_specs_v3",
            active_collection_name="functional_specs_v2",
            qdrant_local_path=tmp_path / "qdrant",
        )

    assert not stage_directory.exists()
    assert client.closed is True


def test_vector_validation_fails_before_qdrant_for_wrong_dimension() -> None:
    batch = stage_archived_fdd_rebuild.EmbeddingBatch(
        document_name="example.docx",
        total_records=1,
        records=[
            EmbeddingRecord(
                unit_id="unit-1",
                unit_index=1,
                source_kind="table",
                document_family="FS_ASNB",
                release_label="R21",
                content_hash="hash",
                artifact_version="v1",
                cache_key="key",
                text="Parent context: fields",
                embedding_model="text-embedding-3-large",
                embedding_status="embedded",
                vector=[0.1, 0.2],
            )
        ],
    )

    with pytest.raises(RuntimeError, match="dimension validation failed"):
        stage_archived_fdd_rebuild._validate_vector_dimensions(batch, expected_vector_size=3)


def test_manifest_separates_embedding_input_from_index_generation(tmp_path: Path) -> None:
    manifest_path = tmp_path / "stage_manifest.json"
    stage_archived_fdd_rebuild._write_manifest(
        manifest_path,
        status="running",
        sources=[],
        collection_name="functional_specs_v3",
        embedding_input_version="v1",
        index_generation="table_context_v1",
        embedding_model="text-embedding-3-large",
    )

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    assert manifest["embedding_input_version"] == "v1"
    assert manifest["embedding_record_artifact_version"] == "v1"
    assert manifest["index_generation"] == "table_context_v1"
