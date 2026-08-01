from scripts.run_embedding_smoke_test import quarantine_embedding_artifact


def test_quarantine_embedding_artifact_preserves_and_deactivates_conflicting_file(tmp_path) -> None:
    artifact = tmp_path / "R21.embeddings.json"
    artifact.write_text('{"records": []}', encoding="utf-8")

    quarantined = quarantine_embedding_artifact(artifact)

    assert quarantined is not None
    assert not artifact.exists()
    assert quarantined.is_file()
    assert quarantined.read_text(encoding="utf-8") == '{"records": []}'
    assert list(tmp_path.glob("*.embeddings.json")) == []


def test_quarantine_embedding_artifact_ignores_missing_file(tmp_path) -> None:
    assert quarantine_embedding_artifact(tmp_path / "missing.embeddings.json") is None
