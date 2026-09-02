from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]


def _read(name: str) -> str:
    return (ROOT_DIR / "scripts" / name).read_text(encoding="utf-8")


def test_fdd_launcher_exposes_guarded_stages_and_existing_scripts() -> None:
    content = _read("run_fdd_generation.ps1")

    assert "'prepare', 'embed-index', 'evaluate', 'activate'" in content
    assert "scripts/master_ingestion_embedding_docs.py" in content
    assert "scripts/stage_archived_fdd_rebuild.py" in content
    assert "scripts/run_fdd_retrieval_gate.py" in content
    assert "Type APPROVE to continue" in content
    assert "scripts/activate_fdd_generation.py" in content
    assert "ACTIVATE $TargetGeneration" in content
    assert "--apply" in content
    assert "Toggle the Desktop-owned MCP server off and on" in content


def test_code_launcher_exposes_guarded_stages_and_existing_scripts() -> None:
    content = _read("run_code_generation.ps1")

    assert "'intake-parse', 'prepare-index', 'embed-index', 'evaluate'" in content
    for script in (
        "scripts/build_code_snapshot.py",
        "scripts/parse_code_snapshot.py",
        "scripts/check_code_preindex_gate.py",
        "scripts/prepare_code_index_artifacts.py",
        "scripts/verify_prepared_code_index.py",
        "scripts/embed_code_index_artifacts.py",
        "scripts/index_code_qdrant.py",
        "scripts/verify_code_qdrant.py",
    ):
        assert script in content
    assert "I_AUTHORIZE_OPENAI_CODE_DISCLOSURE_AND_COST" in content
    assert "Type APPROVE to continue" in content
    assert "scripts/run_code_combined_retrieval_eval.py" in content
    assert "This launcher never creates query embeddings" in content
    assert "do not overwrite review evidence" in content
    assert "'intake-parse', 'prepare-index', 'embed-index', 'evaluate', 'activate'" in content
    assert "NO ACTIVATION PERFORMED" in content


def test_launcher_runbooks_document_exact_commands_and_boundaries() -> None:
    fdd = (ROOT_DIR / "docs" / "FDD_Generation_Launcher_Runbook.md").read_text(encoding="utf-8")
    code = (ROOT_DIR / "docs" / "Code_Generation_Launcher_Runbook.md").read_text(encoding="utf-8")

    assert "run_fdd_generation.ps1 -Generation functional_specs_v6 -Stage prepare" in fdd
    assert "run_fdd_generation.ps1 -Generation functional_specs_v6 -Stage embed-index" in fdd
    assert "run_code_generation.ps1 -SnapshotRequest fci-custom-r2 -Stage intake-parse" in code
    assert "run_code_generation.ps1 -SnapshotRequest fci-custom-r2 -Stage embed-index" in code
    assert "ACTIVATE functional_specs_v6" in fdd
    assert "atomically updates" in fdd
    assert "complete custom-code snapshot" in code
