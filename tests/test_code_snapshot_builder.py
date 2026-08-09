from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.code_ingestion.intake_validation import validate_code_intake
from app.code_ingestion.snapshot_builder import (
    build_code_snapshot,
    build_snapshot_diff,
    load_snapshot_manifest,
)


def _write_intake(
    root: Path,
    *,
    revision: str,
    files: dict[str, str],
    base_snapshot_id: str | None = None,
    expected: list[str] | None = None,
) -> Path:
    intake = root / f"intake-{revision}"
    source = intake / "source"
    source.mkdir(parents=True)
    for relative_path, content in files.items():
        path = source / relative_path
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8", newline="")
    (intake / "snapshot_request.json").write_text(
        json.dumps(
            {
                "schema_version": "code_snapshot_request_v1",
                "module_set": "fci-custom",
                "svn_revision": revision,
                "application_build": f"14.7.{revision}",
                "reviewer": "Phase 2 SME",
                "base_snapshot_id": base_snapshot_id,
                "expected_changed_packages": expected or [],
                "compiler_context": {"oracle_version": None, "plsql_ccflags": None},
            }
        ),
        encoding="utf-8",
    )
    return intake


def test_snapshot_publication_is_content_addressed_verified_and_no_overwrite(tmp_path: Path) -> None:
    intake = _write_intake(
        tmp_path,
        revision="100",
        files={"packages/pkg_a.prc": "procedure a is begin null; end;\n"},
    )
    snapshot_root = tmp_path / "snapshots"

    manifest = build_code_snapshot(intake, snapshot_root)
    snapshot_directory = snapshot_root / manifest.snapshot_id

    assert manifest.snapshot_id.startswith("fci-custom-r100-")
    assert (snapshot_directory / "source/packages/pkg_a.prc").is_file()
    assert load_snapshot_manifest(snapshot_directory) == manifest
    with pytest.raises(FileExistsError, match="never overwritten"):
        build_code_snapshot(intake, snapshot_root)


def test_snapshot_identity_does_not_change_with_large_file_warning_policy(tmp_path: Path) -> None:
    intake = _write_intake(
        tmp_path,
        revision="99",
        files={"packages/pkg_a.prc": "procedure a is begin null; end;\n"},
    )

    default_policy = build_code_snapshot(intake, tmp_path / "default-snapshots")
    warning_policy = build_code_snapshot(
        intake,
        tmp_path / "warning-snapshots",
        large_file_warning_bytes=1,
    )

    assert default_policy.snapshot_id == warning_policy.snapshot_id
    assert default_policy.snapshot_content_sha256 == warning_policy.snapshot_content_sha256
    assert default_policy.files[0].is_large_file is False
    assert warning_policy.files[0].is_large_file is True


def test_snapshot_detects_added_modified_deleted_unchanged_and_exact_rename(tmp_path: Path) -> None:
    snapshot_root = tmp_path / "snapshots"
    base = build_code_snapshot(
        _write_intake(
            tmp_path,
            revision="100",
            files={
                "pkg_keep.prc": "keep\n",
                "pkg_modify.prc": "old\n",
                "pkg_delete.prc": "delete\n",
                "old_name.fnc": "same renamed content\n",
            },
        ),
        snapshot_root,
    )
    current = build_code_snapshot(
        _write_intake(
            tmp_path,
            revision="101",
            base_snapshot_id=base.snapshot_id,
            expected=["pkg_modify.prc", "pkg_add.ddl", "new_name.fnc", "missing.sql"],
            files={
                "pkg_keep.prc": "keep\n",
                "pkg_modify.prc": "new\n",
                "pkg_add.ddl": "create table t (id number);\n",
                "new_name.fnc": "same renamed content\n",
            },
        ),
        snapshot_root,
    )

    diff = current.diff
    assert diff.added == ("pkg_add.ddl",)
    assert diff.modified == ("pkg_modify.prc",)
    assert diff.deleted == ("pkg_delete.prc",)
    assert diff.unchanged == ("pkg_keep.prc",)
    assert [(item.old_path, item.new_path) for item in diff.exact_renames] == [
        ("old_name.fnc", "new_name.fnc")
    ]
    assert diff.missing_expected_changes == ("missing.sql",)
    assert diff.unexpected_changed_files == ("pkg_delete.prc",)


def test_line_ending_only_change_is_reported_as_formatting_only(tmp_path: Path) -> None:
    base_dir = tmp_path / "base"
    current_dir = tmp_path / "current"
    base_dir.mkdir()
    current_dir.mkdir()
    (base_dir / "pkg.prc").write_bytes(b"begin\r\nnull;\r\nend;\r\n")
    (current_dir / "pkg.prc").write_bytes(b"begin\nnull;\nend;\n")
    base_files = validate_code_intake(base_dir).files
    current_files = validate_code_intake(current_dir).files

    base_intake = _write_intake(tmp_path, revision="200", files={"placeholder.sql": "x\n"})
    base_manifest = build_code_snapshot(base_intake, tmp_path / "snapshots")
    base_manifest = base_manifest.model_copy(update={"files": base_files})
    diff = build_snapshot_diff(current_files=current_files, base_manifest=base_manifest)

    assert diff.modified == ("pkg.prc",)
    assert diff.formatting_only_modified == ("pkg.prc",)


def test_snapshot_verification_detects_manual_source_tampering(tmp_path: Path) -> None:
    snapshot_root = tmp_path / "snapshots"
    manifest = build_code_snapshot(
        _write_intake(tmp_path, revision="300", files={"pkg.sql": "select 1 from dual;\n"}),
        snapshot_root,
    )
    source = snapshot_root / manifest.snapshot_id / "source/pkg.sql"
    source.write_text("select 2 from dual;\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="Immutable snapshot source verification failed"):
        load_snapshot_manifest(snapshot_root / manifest.snapshot_id)


def test_modified_intake_between_validation_and_copy_cannot_publish(monkeypatch, tmp_path: Path) -> None:
    from app.code_ingestion import snapshot_builder

    intake = _write_intake(tmp_path, revision="400", files={"pkg.sql": "original\n"})
    snapshot_root = tmp_path / "snapshots"
    real_copytree = snapshot_builder.shutil.copytree

    def mutate_then_copy(source: Path, target: Path) -> Path:
        (source / "pkg.sql").write_text("mutated\n", encoding="utf-8")
        return real_copytree(source, target)

    monkeypatch.setattr(snapshot_builder.shutil, "copytree", mutate_then_copy)

    with pytest.raises(RuntimeError, match="do not match"):
        build_code_snapshot(intake, snapshot_root)

    assert not list(snapshot_root.glob("fci-custom-*"))
