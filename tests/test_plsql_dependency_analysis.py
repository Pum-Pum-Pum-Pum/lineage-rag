from __future__ import annotations

import hashlib

from app.code_ingestion.analysis_policy import AnalysisBoundaries, CodeAnalysisPolicy
from app.code_ingestion.code_analysis_models import SchemaObject
from app.code_ingestion.oracle_identifiers import oracle_identifier
from app.code_ingestion.plsql_dependency_analysis import extract_dependencies
from app.code_ingestion.plsql_models import SourceMap
from app.code_ingestion.plsql_parser_core import parse_plsql_source
from app.code_ingestion.plsql_symbol_analysis import extract_symbols


POLICY = CodeAnalysisPolicy(
    boundaries=AnalysisBoundaries(
        kernel_package_prefixes=("KERNEL_",),
        external_package_prefixes=("DBMS_", "UTL_"),
        ignored_builtin_calls=("COUNT", "NVL"),
    )
)


def _parse(source: str):
    return parse_plsql_source(
        source,
        snapshot_id="snapshot-157",
        source_path="packages/pkg_dependency.sql",
        source_sha256=hashlib.sha256(source.encode("utf-8")).hexdigest(),
    )


def _schema_object(name: str) -> SchemaObject:
    return SchemaObject(
        object_id="a" * 64,
        object_kind="table",
        name=oracle_identifier(name.rsplit(".", 1)[-1]),
        schema_name=oracle_identifier(name.split(".", 1)[0]) if "." in name else None,
        canonical_qualified_name=name.upper(),
        source_path="ddl/tables.ddl",
        source_map=SourceMap(
            source_path="ddl/tables.ddl",
            start_line=1,
            end_line=1,
            start_offset=0,
            end_offset=10,
        ),
    )


def test_calls_keep_all_plausible_overload_candidates() -> None:
    source = """CREATE OR REPLACE PACKAGE BODY pkg_dependency AS
  PROCEDURE process_claim(p_value NUMBER) IS BEGIN NULL; END;
  PROCEDURE process_claim(p_value VARCHAR2) IS BEGIN NULL; END;
  PROCEDURE caller(p_value NUMBER) IS
  BEGIN
    process_claim(p_value);
  END caller;
END pkg_dependency;
/
"""
    parsed = _parse(source)
    symbols = extract_symbols(parsed, module_id="fci-custom")

    edges = extract_dependencies(
        source,
        parsed,
        file_symbols=symbols,
        all_symbols=symbols,
        schema_objects=(),
        policy=POLICY,
    )
    call = next(
        edge
        for edge in edges
        if edge.dependency_kind == "routine_call"
        and edge.target_canonical_name == "PROCESS_CLAIM"
    )

    assert call.resolution_state == "ambiguous"
    assert len(call.candidate_symbol_occurrence_ids) == 2


def test_kernel_external_and_unresolved_calls_have_distinct_boundaries() -> None:
    source = """CREATE OR REPLACE PROCEDURE run_claim IS
BEGIN
  kernel_claim.validate_claim(1);
  DBMS_OUTPUT.PUT_LINE('x');
  mystery_package.run_it();
END;
/
"""
    parsed = _parse(source)
    symbols = extract_symbols(parsed, module_id="fci-custom")

    edges = extract_dependencies(
        source,
        parsed,
        file_symbols=symbols,
        all_symbols=symbols,
        schema_objects=(),
        policy=POLICY,
    )
    by_target = {edge.target_canonical_name: edge for edge in edges}

    assert by_target["KERNEL_CLAIM.VALIDATE_CLAIM"].dependency_kind == "kernel_boundary"
    assert by_target["KERNEL_CLAIM.VALIDATE_CLAIM"].resolution_state == "kernel_unavailable"
    assert by_target["DBMS_OUTPUT.PUT_LINE"].dependency_kind == "external_package"
    assert by_target["DBMS_OUTPUT.PUT_LINE"].resolution_state == "external_schema"
    assert by_target["MYSTERY_PACKAGE.RUN_IT"].resolution_state == "unresolved"


def test_dynamic_sql_is_explicit_unknown_and_static_tables_still_resolve() -> None:
    source = """CREATE OR REPLACE PROCEDURE update_customer(p_sql VARCHAR2) IS
  v_id NUMBER;
BEGIN
  SELECT customer_id INTO v_id FROM app.customer;
  UPDATE app.customer SET status = 'A';
  EXECUTE IMMEDIATE p_sql;
END;
/
"""
    parsed = _parse(source)
    symbols = extract_symbols(parsed, module_id="fci-custom")

    edges = extract_dependencies(
        source,
        parsed,
        file_symbols=symbols,
        all_symbols=symbols,
        schema_objects=(_schema_object("APP.CUSTOMER"),),
        policy=POLICY,
    )

    dynamic = next(edge for edge in edges if edge.dependency_kind == "dynamic_sql")
    table_edges = [edge for edge in edges if edge.target_canonical_name == "APP.CUSTOMER"]
    assert dynamic.resolution_state == "dynamic_unknown"
    assert dynamic.target_canonical_name == "P_SQL"
    assert {edge.dependency_kind for edge in table_edges} == {"table_read", "table_write"}
    assert all(edge.resolution_state == "resolved_in_snapshot" for edge in table_edges)


def test_package_declaration_references_are_resolved_without_copying_source() -> None:
    source = """CREATE OR REPLACE PACKAGE BODY pkg_context AS
  TYPE claim_rec IS RECORD (claim_id NUMBER);
  c_status CONSTANT VARCHAR2(1) := 'A';
  g_count NUMBER := 0;
  CURSOR c_claim IS SELECT claim_id FROM claims;
  PROCEDURE use_context(p_claim claim_rec) IS
  BEGIN
    g_count := g_count + 1;
    IF c_status = 'A' THEN OPEN c_claim; END IF;
  END use_context;
END pkg_context;
/
"""
    parsed = _parse(source)
    symbols = extract_symbols(parsed, module_id="fci-custom")

    edges = extract_dependencies(
        source,
        parsed,
        file_symbols=symbols,
        all_symbols=symbols,
        schema_objects=(),
        policy=POLICY,
    )
    reference_edges = {
        edge.target_canonical_name: edge.dependency_kind
        for edge in edges
        if edge.dependency_kind.endswith("reference")
    }

    assert reference_edges == {
        "CLAIM_REC": "type_reference",
        "C_STATUS": "constant_reference",
        "G_COUNT": "global_reference",
        "C_CLAIM": "cursor_reference",
    }


def test_nested_routine_body_is_not_attributed_to_outer_routine() -> None:
    source = """CREATE OR REPLACE PACKAGE BODY pkg_nested AS
  PROCEDURE outer_proc IS
    PROCEDURE local_proc IS BEGIN kernel_claim.hidden_call(); END;
  BEGIN
    local_proc();
  END outer_proc;
END pkg_nested;
/
"""
    parsed = _parse(source)
    symbols = extract_symbols(parsed, module_id="fci-custom")

    edges = extract_dependencies(
        source,
        parsed,
        file_symbols=symbols,
        all_symbols=symbols,
        schema_objects=(),
        policy=POLICY,
    )
    kernel = [edge for edge in edges if edge.dependency_kind == "kernel_boundary"]
    local_symbol = next(symbol for symbol in symbols if symbol.name.display_name == "local_proc")

    assert len(kernel) == 1
    assert kernel[0].source_symbol_occurrence_id == local_symbol.occurrence_id


def test_cursor_queries_and_comma_join_tables_are_not_silently_omitted() -> None:
    source = """CREATE OR REPLACE PACKAGE BODY pkg_tables AS
  CURSOR c_join IS SELECT a.id FROM app.table_a a, app.table_b b WHERE a.id = b.id;
  PROCEDURE read_join IS
    v_id NUMBER;
  BEGIN
    SELECT a.id INTO v_id FROM app.table_a a, app.table_b b WHERE a.id = b.id;
  END read_join;
END pkg_tables;
/
"""
    parsed = _parse(source)
    symbols = extract_symbols(parsed, module_id="fci-custom")

    edges = extract_dependencies(
        source,
        parsed,
        file_symbols=symbols,
        all_symbols=symbols,
        schema_objects=(
            _schema_object("APP.TABLE_A"),
            _schema_object("APP.TABLE_B").model_copy(update={"object_id": "b" * 64}),
        ),
        policy=POLICY,
    )

    procedure_edges = [edge for edge in edges if edge.source_symbol_occurrence_id]
    cursor_edges = [edge for edge in edges if edge.source_symbol_occurrence_id is None]
    assert {edge.target_canonical_name for edge in procedure_edges if edge.dependency_kind == "table_read"} == {
        "APP.TABLE_A",
        "APP.TABLE_B",
    }
    assert {edge.target_canonical_name for edge in cursor_edges} == {
        "APP.TABLE_A",
        "APP.TABLE_B",
    }
