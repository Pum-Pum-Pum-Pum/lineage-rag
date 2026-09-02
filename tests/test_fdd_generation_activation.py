from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.activation.fdd_generation import (
    apply_fdd_generation_activation,
    build_fdd_generation_activation_plan,
)


def _stage(tmp_path: Path, generation: str = "functional_specs_v7") -> Path:
    stage = tmp_path / "staging" / generation
    processed = stage / "processed"
    processed.mkdir(parents=True)
    (processed / "artifact.retrieval_ready.json").write_text('{"id":"artifact"}\n', encoding="utf-8")
    (stage / "stage_manifest.json").write_text(
        json.dumps(
            {
                "status": "verified",
                "collection_name": generation,
                "sources": [{"document_id": "one"}],
                "qdrant": {"verified_records": 1},
            }
        ),
        encoding="utf-8",
    )
    return stage


def _env(tmp_path: Path) -> Path:
    path = tmp_path / ".env"
    path.write_text(
        "QDRANT_COLLECTION_NAME=functional_specs_v5\n"
        "PROCESSED_DIR=data/indexes/functional_specs_v5/processed\n"
        "FDD_GENERATION=functional_specs_v5\n"
        "RETRIEVAL_INDEX_PATH=data/indexes/functional_specs_v5/processed\n"
        "OPENAI_API_KEY=not-a-test-secret\n",
        encoding="utf-8",
    )
    return path


def test_fdd_generation_activation_promotes_verified_pair_and_updates_env(tmp_path: Path) -> None:
    generation = "functional_specs_v7"
    stage = _stage(tmp_path, generation)
    env_path = _env(tmp_path)
    plan = build_fdd_generation_activation_plan(
        generation=generation,
        stage_directory=stage,
        indexes_directory=tmp_path / "indexes",
        env_path=env_path,
    )

    assert plan.target_configuration == {
        "QDRANT_COLLECTION_NAME": generation,
        "PROCESSED_DIR": "data/indexes/functional_specs_v7/processed",
        "FDD_GENERATION": generation,
        "RETRIEVAL_INDEX_PATH": "data/indexes/functional_specs_v7/processed",
    }
    evidence = apply_fdd_generation_activation(
        plan=plan,
        env_path=env_path,
        evidence_directory=tmp_path / "exports" / "fdd",
    )

    assert (tmp_path / "indexes" / generation / "processed" / "artifact.retrieval_ready.json").is_file()
    text = env_path.read_text(encoding="utf-8")
    assert "QDRANT_COLLECTION_NAME=functional_specs_v7" in text
    assert "PROCESSED_DIR=data/indexes/functional_specs_v7/processed" in text
    assert "FDD_GENERATION=functional_specs_v7" in text
    assert "RETRIEVAL_INDEX_PATH=data/indexes/functional_specs_v7/processed" in text
    payload = json.loads(evidence.read_text(encoding="utf-8"))
    assert payload["activation_complete"] is True
    assert payload["restart_required"] is True
    assert payload["rollback_configuration"]["QDRANT_COLLECTION_NAME"] == "functional_specs_v5"


@pytest.mark.parametrize(
    ("manifest_update", "message"),
    [
        ({"status": "staged"}, "Stage is not verified"),
        ({"collection_name": "functional_specs_v6"}, "does not match"),
        ({"qdrant": {"verified_records": 0}}, "exact-Qdrant"),
    ],
)
def test_fdd_generation_activation_refuses_unverified_stage(
    tmp_path: Path, manifest_update: dict, message: str
) -> None:
    stage = _stage(tmp_path)
    manifest_path = stage / "stage_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest.update(manifest_update)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    env_path = _env(tmp_path)

    with pytest.raises(ValueError, match=message):
        build_fdd_generation_activation_plan(
            generation="functional_specs_v7",
            stage_directory=stage,
            indexes_directory=tmp_path / "indexes",
            env_path=env_path,
        )
    assert "functional_specs_v5" in env_path.read_text(encoding="utf-8")


def test_fdd_generation_activation_refuses_existing_target_or_duplicate_env(tmp_path: Path) -> None:
    stage = _stage(tmp_path)
    env_path = _env(tmp_path)
    indexes = tmp_path / "indexes"
    (indexes / "functional_specs_v7").mkdir(parents=True)
    with pytest.raises(FileExistsError, match="overwrite"):
        build_fdd_generation_activation_plan(
            generation="functional_specs_v7", stage_directory=stage, indexes_directory=indexes, env_path=env_path
        )

    (indexes / "functional_specs_v7").rmdir()
    env_path.write_text(env_path.read_text(encoding="utf-8") + "PROCESSED_DIR=duplicate\n", encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate PROCESSED_DIR"):
        build_fdd_generation_activation_plan(
            generation="functional_specs_v7", stage_directory=stage, indexes_directory=indexes, env_path=env_path
        )
