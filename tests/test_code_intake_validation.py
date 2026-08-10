from __future__ import annotations

from pathlib import Path

import pytest

from app.code_ingestion.intake_validation import (
    CodeIntakeValidationError,
    validate_code_intake,
)


def test_intake_accepts_allowlisted_extensions_case_insensitively(tmp_path: Path) -> None:
    (tmp_path / "PACKAGE.PRC").write_text("procedure p is begin null; end;\n", encoding="utf-8")
    (tmp_path / "function.FnC").write_text("function f return number is begin return 1; end;\n", encoding="utf-8")
    (tmp_path / "schema.DDL").write_text("create table t (id number);\n", encoding="utf-8")
    (tmp_path / "mixed.Sql").write_text("select 1 from dual;\n", encoding="utf-8")

    report = validate_code_intake(tmp_path)

    assert [entry.extension for entry in report.files] == [".fnc", ".sql", ".prc", ".ddl"]
    assert all(entry.encoding == "utf-8" for entry in report.files)
    assert {entry.source_handler for entry in report.files} == {"plsql", "ddl"}


def test_line_ending_normalization_does_not_change_normalized_hash(tmp_path: Path) -> None:
    left = tmp_path / "left"
    right = tmp_path / "right"
    left.mkdir()
    right.mkdir()
    (left / "pkg.prc").write_bytes(b"begin\r\n  null;\r\nend;\r\n")
    (right / "pkg.prc").write_bytes(b"begin\n  null;\nend;\n")

    left_entry = validate_code_intake(left, stream_chunk_bytes=7).files[0]
    right_entry = validate_code_intake(right, stream_chunk_bytes=5).files[0]

    assert left_entry.sha256 != right_entry.sha256
    assert left_entry.normalized_text_sha256 == right_entry.normalized_text_sha256
    assert left_entry.line_count == right_entry.line_count == 3


def test_file_above_warning_threshold_is_accepted_not_rejected(tmp_path: Path) -> None:
    source = tmp_path / "large.sql"
    source.write_text("begin\n" + ("  null;\n" * 20) + "end;\n", encoding="utf-8")

    report = validate_code_intake(tmp_path, large_file_warning_bytes=32, stream_chunk_bytes=11)

    assert report.files[0].is_large_file is True
    assert report.files[0].warnings == ("large_file",)
    assert report.warnings[0].code == "large_file"


def test_secret_scan_fails_closed_without_echoing_secret_value(tmp_path: Path) -> None:
    secret = "super-secret-password"
    (tmp_path / "unsafe.sql").write_text(
        f"password := '{secret}';\n",
        encoding="utf-8",
    )

    with pytest.raises(CodeIntakeValidationError) as exc_info:
        validate_code_intake(tmp_path, stream_chunk_bytes=4)

    assert "potential_secret_detected" in str(exc_info.value)
    assert "assigned_password" in str(exc_info.value)
    assert secret not in str(exc_info.value)


def test_binary_content_is_rejected_even_with_allowlisted_extension(tmp_path: Path) -> None:
    (tmp_path / "binary.sql").write_bytes(b"\x00\x01\x02\x03" * 50)

    with pytest.raises(CodeIntakeValidationError, match="binary_content_detected"):
        validate_code_intake(tmp_path)


def test_non_allowlisted_file_fails_the_whole_intake(tmp_path: Path) -> None:
    (tmp_path / "valid.sql").write_text("select 1 from dual;", encoding="utf-8")
    (tmp_path / "settings.ini").write_text("not allowed", encoding="utf-8")

    with pytest.raises(CodeIntakeValidationError, match="extension_not_allowed"):
        validate_code_intake(tmp_path)


def test_windows_1252_source_is_accepted_with_visible_warning(tmp_path: Path) -> None:
    (tmp_path / "legacy.prc").write_bytes("-- café\nbegin null; end;\n".encode("cp1252"))

    report = validate_code_intake(tmp_path)

    assert report.files[0].encoding == "cp1252"
    assert "legacy_encoding" in report.files[0].warnings
