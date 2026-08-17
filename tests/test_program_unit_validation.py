from __future__ import annotations

import hashlib

import pytest

from app.code_ingestion.plsql_parser_core import parse_plsql_source
from app.code_ingestion.program_unit_validation import (
    ProgramUnitPolicyError,
    validate_custom_program_unit,
)


SUFFIXES = ("_CUSTOM", "_MAIN")


def _parse(source: str, path: str):
    return parse_plsql_source(
        source,
        snapshot_id="snapshot-validation",
        source_path=path,
        source_sha256=hashlib.sha256(source.encode("utf-8")).hexdigest(),
    )


@pytest.mark.parametrize(
    ("path", "source", "owner"),
    [
        (
            "pkg_txn_p_custom.sql",
            "CREATE OR REPLACE PACKAGE BODY pkg_txn_p_custom AS PROCEDURE plain_member IS BEGIN NULL; END; END; /",
            "PKG_TXN_P_CUSTOM",
        ),
        (
            "utpks_txn_main.spc",
            "CREATE OR REPLACE PACKAGE utpks_txn_main AS FUNCTION plain_member RETURN NUMBER; END; /",
            "UTPKS_TXN_MAIN",
        ),
        (
            "sfcheck_p_custom.fnc",
            "CREATE OR REPLACE FUNCTION sfcheck_p_custom RETURN NUMBER IS BEGIN RETURN 1; END; /",
            "SFCHECK_P_CUSTOM",
        ),
        (
            "spclear_main.prc",
            "CREATE OR REPLACE PROCEDURE spclear_main IS BEGIN NULL; END; /",
            "SPCLEAR_MAIN",
        ),
    ],
)
def test_declared_custom_program_units_are_accepted(path: str, source: str, owner: str) -> None:
    assert validate_custom_program_unit(
        _parse(source, path),
        source_handler="plsql",
        allowed_suffixes=SUFFIXES,
    ) == owner


def test_package_member_names_do_not_require_custom_suffix() -> None:
    source = """CREATE OR REPLACE PACKAGE BODY pkg_owner_custom AS
  PROCEDURE validate_transaction IS BEGIN NULL; END;
  FUNCTION calculate_amount RETURN NUMBER IS BEGIN RETURN 1; END;
END;
/
"""
    assert validate_custom_program_unit(
        _parse(source, "pkg_owner_custom.sql"),
        source_handler="plsql",
        allowed_suffixes=SUFFIXES,
    ) == "PKG_OWNER_CUSTOM"


@pytest.mark.parametrize(
    ("path", "source", "message"),
    [
        (
            "pkg_kernel.sql",
            "CREATE OR REPLACE PACKAGE pkg_kernel AS PROCEDURE run; END; /",
            "must end",
        ),
        (
            "wrong_custom.prc",
            "CREATE OR REPLACE PROCEDURE actual_custom IS BEGIN NULL; END; /",
            "does not match",
        ),
    ],
)
def test_kernel_or_filename_mismatch_fails_closed(path: str, source: str, message: str) -> None:
    with pytest.raises(ProgramUnitPolicyError, match=message):
        validate_custom_program_unit(
            _parse(source, path),
            source_handler="plsql",
            allowed_suffixes=SUFFIXES,
        )


def test_ddl_is_not_filtered_by_program_unit_suffix() -> None:
    source = "CREATE TABLE transaction_master (id NUMBER);"
    assert validate_custom_program_unit(
        _parse(source, "transaction_master.ddl"),
        source_handler="ddl",
        allowed_suffixes=SUFFIXES,
    ) is None
