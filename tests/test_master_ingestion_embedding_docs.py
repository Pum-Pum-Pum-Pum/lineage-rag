from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace

import pytest

from scripts import master_ingestion_embedding_docs


def test_master_script_help_runs() -> None:
    result = subprocess.run(
        [sys.executable, "scripts/master_ingestion_embedding_docs.py", "--help"],
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0
    assert "--dry-run" in result.stdout
    assert "--request-batch-size" in result.stdout


def test_master_runs_existing_stages_then_archives_verified_documents(monkeypatch, tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    first = _write_docx_placeholder(settings.raw_specs_dir, "FS_ASNB_R25_First.docx")
    second = _write_docx_placeholder(settings.raw_specs_dir, "FS_ASNB_R26_Second.docx")
    commands: list[list[str]] = []

    def fake_run(command, **kwargs) -> None:
        commands.append(command)
        assert kwargs["cwd"] == master_ingestion_embedding_docs.ROOT_DIR
        assert kwargs["check"] is True

    monkeypatch.setattr(master_ingestion_embedding_docs, "get_settings", lambda: settings)
    monkeypatch.setattr(master_ingestion_embedding_docs.subprocess, "run", fake_run)

    master_ingestion_embedding_docs.main(["--request-batch-size", "32"])

    assert not first.exists()
    assert not second.exists()
    assert (settings.embedded_docs_dir / first.name).is_file()
    assert (settings.embedded_docs_dir / second.name).is_file()
    assert len(commands) == 5
    assert commands[0][1].endswith("run_ingestion_pipeline.py")
    assert commands[1][1].endswith("run_embedding_smoke_test.py")
    assert commands[1][-1] == "32"
    assert commands[2][1].endswith("run_embedding_smoke_test.py")
    assert commands[3][1].endswith("run_qdrant_indexing.py")
    assert commands[4][1].endswith("check_qdrant_index.py")
    assert commands[4].count("--embedding-artifact") == 2


def test_master_dry_run_changes_nothing(monkeypatch, tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    source = _write_docx_placeholder(settings.raw_specs_dir, "FS_ASNB_R25_Dry_Run.docx")

    def fail_if_called(*args, **kwargs) -> None:
        raise AssertionError("Dry run must not start a child process")

    monkeypatch.setattr(master_ingestion_embedding_docs, "get_settings", lambda: settings)
    monkeypatch.setattr(master_ingestion_embedding_docs.subprocess, "run", fail_if_called)

    master_ingestion_embedding_docs.main(["--dry-run"])

    assert source.is_file()
    assert not settings.embedded_docs_dir.exists()


def test_master_failure_keeps_documents_in_raw_specs(monkeypatch, tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    source = _write_docx_placeholder(settings.raw_specs_dir, "FS_ASNB_R25_Failure.docx")

    def fail_at_qdrant(command, **kwargs) -> None:
        if command[1].endswith("run_qdrant_indexing.py"):
            raise subprocess.CalledProcessError(returncode=1, cmd=command)

    monkeypatch.setattr(master_ingestion_embedding_docs, "get_settings", lambda: settings)
    monkeypatch.setattr(master_ingestion_embedding_docs.subprocess, "run", fail_at_qdrant)

    with pytest.raises(subprocess.CalledProcessError):
        master_ingestion_embedding_docs.main([])

    assert source.is_file()
    assert not (settings.embedded_docs_dir / source.name).exists()


def test_master_rejects_existing_archive_destination_before_running_stages(
    monkeypatch,
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path)
    source = _write_docx_placeholder(settings.raw_specs_dir, "FS_ASNB_R25_Conflict.docx")
    settings.embedded_docs_dir.mkdir(parents=True)
    (settings.embedded_docs_dir / source.name).write_text("already archived", encoding="utf-8")

    def fail_if_called(*args, **kwargs) -> None:
        raise AssertionError("Archive conflict must block all child stages")

    monkeypatch.setattr(master_ingestion_embedding_docs, "get_settings", lambda: settings)
    monkeypatch.setattr(master_ingestion_embedding_docs.subprocess, "run", fail_if_called)

    with pytest.raises(FileExistsError, match="duplicate FDD filename"):
        master_ingestion_embedding_docs.main([])

    assert source.is_file()


def test_master_dry_run_rejects_case_insensitive_archived_filename_before_stages(
    monkeypatch,
    tmp_path: Path,
) -> None:
    settings = _settings(tmp_path)
    source = _write_docx_placeholder(settings.raw_specs_dir, "FS_ASNB_R25_Duplicate.docx")
    settings.embedded_docs_dir.mkdir(parents=True)
    _write_docx_placeholder(settings.embedded_docs_dir, source.name.upper())

    def fail_if_called(*args, **kwargs) -> None:
        raise AssertionError("Archive duplicate must block all child stages")

    monkeypatch.setattr(master_ingestion_embedding_docs, "get_settings", lambda: settings)
    monkeypatch.setattr(master_ingestion_embedding_docs.subprocess, "run", fail_if_called)

    with pytest.raises(FileExistsError, match="duplicate FDD filename"):
        master_ingestion_embedding_docs.main(["--dry-run"])

    assert source.is_file()


def test_master_rejects_legacy_rebuild_flag_before_child_stages(monkeypatch, tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    _write_docx_placeholder(settings.raw_specs_dir, "FS_ASNB_R21_Repair.docx")
    monkeypatch.setattr(master_ingestion_embedding_docs, "get_settings", lambda: settings)
    monkeypatch.setattr(
        master_ingestion_embedding_docs.subprocess,
        "run",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("No child stage expected")),
    )

    with pytest.raises(SystemExit, match="unsupported for embedded local Qdrant"):
        master_ingestion_embedding_docs.main(["--rebuild-qdrant"])



def _settings(tmp_path: Path) -> SimpleNamespace:
    return SimpleNamespace(
        log_level="INFO",
        raw_specs_dir=tmp_path / "raw_specs",
        embedded_docs_dir=tmp_path / "docs_embedded",
        cache_dir=tmp_path / "cache",
    )


def _write_docx_placeholder(directory: Path, name: str) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / name
    path.write_bytes(b"placeholder")
    return path
