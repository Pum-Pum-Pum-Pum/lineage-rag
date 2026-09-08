from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.code_ingestion import source_import
from app.code_ingestion.source_import import SourceImportError, stage_external_code_source


def test_external_source_import_copies_only_configured_plsql_extensions(tmp_path: Path) -> None:
    external = tmp_path / "svn-backend"
    intake = tmp_path / "intake"
    external.mkdir()
    intake.mkdir()
    (external / "nested").mkdir()
    included = {
        "nested/pkg_a.sql": "package pkg_a is end;\n",
        "nested/pkg_b.spc": "package pkg_b is end;\n",
        "fn.fnc": "function fn return number is begin return 1; end;\n",
        "pr.prc": "procedure pr is begin null; end;\n",
    }
    for relative_path, content in included.items():
        path = external / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    (external / "schema.ddl").write_text("create table stale_t (id number);", encoding="utf-8")
    (external / "comment.cmt").write_text("comment", encoding="utf-8")
    (external / "notes.txt").write_text("notes", encoding="utf-8")
    source_before = {
        path.relative_to(external).as_posix(): path.read_bytes()
        for path in external.rglob("*")
        if path.is_file()
    }

    receipt = stage_external_code_source(external, intake)

    copied = {
        path.relative_to(intake / "source").as_posix(): path.read_text(encoding="utf-8")
        for path in (intake / "source").rglob("*")
        if path.is_file()
    }
    assert copied == included
    assert receipt["selected_extensions"] == [".fnc", ".prc", ".spc", ".sql"]
    assert receipt["skipped_file_counts_by_extension"] == {".cmt": 1, ".ddl": 1, ".txt": 1}
    assert receipt["writes_to_external_source"] is False
    assert {
        path.relative_to(external).as_posix(): path.read_bytes()
        for path in external.rglob("*")
        if path.is_file()
    } == source_before
    stored_receipt = json.loads((intake / "source_import_receipt.json").read_text(encoding="utf-8"))
    assert stored_receipt["selected_tree_sha256"] == receipt["selected_tree_sha256"]


def test_external_source_import_refuses_to_overwrite_existing_intake(tmp_path: Path) -> None:
    external = tmp_path / "svn-backend"
    intake = tmp_path / "intake"
    external.mkdir()
    (intake / "source").mkdir(parents=True)
    (external / "pkg.sql").write_text("select 1 from dual;", encoding="utf-8")

    with pytest.raises(SourceImportError, match="already exists"):
        stage_external_code_source(external, intake)


def test_external_source_import_rejects_source_mutation_during_copy(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    external = tmp_path / "svn-backend"
    intake = tmp_path / "intake"
    external.mkdir()
    intake.mkdir()
    source_file = external / "pkg.sql"
    source_file.write_text("select 1 from dual;", encoding="utf-8")
    original_sha256 = source_import._sha256
    calls = 0

    def mutate_after_copy(path: Path) -> str:
        nonlocal calls
        digest = original_sha256(path)
        calls += 1
        if calls == 2:
            source_file.write_text("select 2 from dual;", encoding="utf-8")
        return digest

    monkeypatch.setattr(source_import, "_sha256", mutate_after_copy)

    with pytest.raises(SourceImportError, match="changed during read-only import"):
        stage_external_code_source(external, intake)

    assert not (intake / "source").exists()
    assert not (intake / "source_import_receipt.json").exists()
