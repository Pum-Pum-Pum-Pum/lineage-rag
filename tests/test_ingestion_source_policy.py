from __future__ import annotations

from pathlib import Path

import pytest
from docx import Document

from app.code_ingestion.intake_validation import validate_code_intake
from app.core.ingestion_policy import load_ingestion_source_policy
from app.ingestion.docx_loader import discover_docx_files
from app.ingestion.docx_text_extractor import extract_docx_text


def _write_policy(
    path: Path,
    *,
    fdd_extensions: str = '".docx" = "docx"',
    code_extensions: str = '\n'.join(
        [
            '".sql" = "plsql"',
            '".prc" = "plsql"',
            '".fnc" = "plsql"',
            '".ddl" = "ddl"',
        ]
    ),
) -> Path:
    path.write_text(
        "schema_version = \"ingestion_source_policy_v1\"\n\n"
        "[fdd.extensions]\n"
        f"{fdd_extensions}\n\n"
        "[code.extensions]\n"
        f"{code_extensions}\n",
        encoding="utf-8",
    )
    return path


def test_new_extension_for_existing_code_handler_requires_only_policy_change(tmp_path: Path) -> None:
    policy = load_ingestion_source_policy(
        _write_policy(
            tmp_path / "policy.toml",
            code_extensions='".sql" = "plsql"\n".pkb" = "plsql"\n".ddl" = "ddl"',
        )
    )
    source = tmp_path / "source"
    source.mkdir()
    (source / "pkg_customer.pkb").write_text("package body pkg_customer is end;\n", encoding="utf-8")

    report = validate_code_intake(source, source_policy=policy)

    assert report.files[0].extension == ".pkb"
    assert report.files[0].source_handler == "plsql"
    assert report.ingestion_policy_sha256 == policy.policy_sha256


def test_new_extension_for_existing_fdd_handler_is_used_by_discovery_and_extraction(tmp_path: Path) -> None:
    policy = load_ingestion_source_policy(
        _write_policy(
            tmp_path / "policy.toml",
            fdd_extensions='".fddx" = "docx"',
        )
    )
    source = tmp_path / "FS_ASNB_R25_Test.fddx"
    document = Document()
    document.add_paragraph("Configured extension works")
    document.save(source)

    discovered = discover_docx_files(tmp_path, source_policy=policy)
    extracted = extract_docx_text(source, source_policy=policy)

    assert [item.file_name for item in discovered] == [source.name]
    assert extracted.full_text == "Configured extension works"


def test_policy_rejects_extension_mapped_to_unimplemented_handler(tmp_path: Path) -> None:
    path = _write_policy(
        tmp_path / "policy.toml",
        fdd_extensions='".pdf" = "pdf"',
    )

    with pytest.raises(ValueError, match="not implemented"):
        load_ingestion_source_policy(path)


def test_policy_rejects_wildcard_extension(tmp_path: Path) -> None:
    path = _write_policy(
        tmp_path / "policy.toml",
        code_extensions='"*.sql" = "plsql"',
    )

    with pytest.raises(ValueError, match="Invalid configured extension"):
        load_ingestion_source_policy(path)


def test_policy_hash_depends_on_meaning_not_toml_order_or_comments(tmp_path: Path) -> None:
    first = _write_policy(tmp_path / "first.toml")
    second = tmp_path / "second.toml"
    second.write_text(
        "# reordered but equivalent\n"
        "schema_version = \"ingestion_source_policy_v1\"\n"
        "[code.extensions]\n"
        '".ddl" = "ddl"\n".fnc" = "plsql"\n".prc" = "plsql"\n".sql" = "plsql"\n'
        "[fdd.extensions]\n"
        '".docx" = "docx"\n',
        encoding="utf-8",
    )

    assert load_ingestion_source_policy(first).policy_sha256 == load_ingestion_source_policy(second).policy_sha256

