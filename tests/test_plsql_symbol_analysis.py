from __future__ import annotations

import hashlib

from app.code_ingestion.plsql_parser_core import parse_plsql_source
from app.code_ingestion.plsql_symbol_analysis import diagnose_symbol_groups, extract_symbols


def _symbols(source: str):
    parsed = parse_plsql_source(
        source,
        snapshot_id="snapshot-156",
        source_path="packages/pkg_overload.sql",
        source_sha256=hashlib.sha256(source.encode("utf-8")).hexdigest(),
    )
    assert parsed.parser_state == "full_parse"
    return extract_symbols(parsed, module_id="fci-custom")


def test_overloads_receive_distinct_keys_from_parameter_type() -> None:
    source = """CREATE OR REPLACE PACKAGE pkg_overload AS
  PROCEDURE process_claim(p_id NUMBER);
  PROCEDURE process_claim(p_id VARCHAR2);
END pkg_overload;
/
"""

    symbols = _symbols(source)

    assert len(symbols) == 2
    assert len({symbol.symbol_key for symbol in symbols}) == 2
    assert [symbol.parameters[0].type_family for symbol in symbols] == [
        "numeric",
        "character",
    ]
    assert not diagnose_symbol_groups(symbols)


def test_mode_only_overloads_collide_without_overwriting_occurrences() -> None:
    source = """CREATE OR REPLACE PACKAGE pkg_overload AS
  PROCEDURE update_value(p_value IN NUMBER);
  PROCEDURE update_value(p_value OUT NUMBER);
END pkg_overload;
/
"""

    symbols = _symbols(source)
    diagnostics = diagnose_symbol_groups(symbols)

    assert len(symbols) == 2
    assert symbols[0].symbol_key == symbols[1].symbol_key
    assert symbols[0].declaration_signature_hash != symbols[1].declaration_signature_hash
    assert len({symbol.occurrence_id for symbol in symbols}) == 2
    assert [item.code for item in diagnostics] == ["overload_symbol_collision"]


def test_return_type_only_function_overloads_are_reported_as_collisions() -> None:
    source = """CREATE OR REPLACE PACKAGE pkg_overload AS
  FUNCTION current_value(p_id NUMBER) RETURN NUMBER;
  FUNCTION current_value(p_id NUMBER) RETURN VARCHAR2;
END pkg_overload;
/
"""

    symbols = _symbols(source)
    diagnostics = diagnose_symbol_groups(symbols)

    assert symbols[0].overload_discriminator_hash == symbols[1].overload_discriminator_hash
    assert symbols[0].canonical_return_type != symbols[1].canonical_return_type
    assert diagnostics[0].code == "overload_symbol_collision"


def test_quoted_and_unquoted_identifiers_never_merge() -> None:
    source = """CREATE OR REPLACE PACKAGE pkg_names AS
  PROCEDURE foo(p_id NUMBER);
  PROCEDURE "FOO"(p_id NUMBER);
END pkg_names;
/
"""

    symbols = _symbols(source)

    assert symbols[0].name.canonical_name == "FOO"
    assert symbols[0].name.is_quoted is False
    assert symbols[1].name.canonical_name == '"FOO"'
    assert symbols[1].name.is_quoted is True
    assert symbols[0].symbol_key != symbols[1].symbol_key


def test_matching_declaration_and_implementation_share_key_without_collision() -> None:
    source = """CREATE OR REPLACE PACKAGE pkg_pair AS
  PROCEDURE run_job(p_id NUMBER DEFAULT 1);
END pkg_pair;
/
CREATE OR REPLACE PACKAGE BODY pkg_pair AS
  PROCEDURE run_job(p_id NUMBER) IS BEGIN NULL; END;
END pkg_pair;
/
"""

    symbols = _symbols(source)

    assert {symbol.occurrence_role for symbol in symbols} == {"declaration", "implementation"}
    assert len({symbol.symbol_key for symbol in symbols}) == 1
    assert len({symbol.declaration_signature_hash for symbol in symbols}) == 2
    assert not diagnose_symbol_groups(symbols)


def test_nested_routine_qualified_name_includes_parent_scope() -> None:
    source = """CREATE OR REPLACE PACKAGE BODY pkg_nested AS
  PROCEDURE outer_proc IS
    PROCEDURE local_proc(p_id NUMBER) IS BEGIN NULL; END;
  BEGIN
    local_proc(1);
  END outer_proc;
END pkg_nested;
/
"""

    symbols = _symbols(source)
    local = next(symbol for symbol in symbols if symbol.name.display_name == "local_proc")

    assert local.canonical_qualified_name == "PKG_NESTED.OUTER_PROC.LOCAL_PROC"
