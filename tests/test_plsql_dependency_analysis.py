from __future__ import annotations

import hashlib

from app.code_ingestion.analysis_policy import AnalysisBoundaries, CodeAnalysisPolicy
from app.code_ingestion.code_analysis_models import SchemaObject
from app.code_ingestion.oracle_identifiers import oracle_identifier
from app.code_ingestion.plsql_dependency_analysis import (
    build_symbol_lookup,
    extract_dependencies,
)
from app.code_ingestion.plsql_models import SourceMap
from app.code_ingestion.plsql_parser_core import parse_plsql_source
from app.code_ingestion.plsql_symbol_analysis import extract_symbols


POLICY = CodeAnalysisPolicy(
    boundaries=AnalysisBoundaries(
        custom_program_unit_suffixes=("_CUSTOM", "_MAIN"),
        infer_noncustom_qualified_packages_as_kernel=False,
        kernel_package_names=("MYSTERY_PACKAGE", "PKG_KERNEL"),
        kernel_package_prefixes=("KERNEL_",),
        external_package_prefixes=("DBMS_", "UTL_"),
        infrastructure_utility_calls=(
            "DEBUG.PR_DEBUG",
            "GLOBAL.PR_INIT",
            "ISDEBUG.WRITELINE",
            "PKGGLOBAL.PR_INIT",
            "PR_DEBUG",
        ),
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


def test_custom_kernel_external_and_unresolved_calls_have_distinct_boundaries() -> None:
    source = """CREATE OR REPLACE PROCEDURE run_claim IS
BEGIN
  kernel_claim.validate_claim(1);
  DBMS_OUTPUT.PUT_LINE('x');
  mystery_package_custom.run_it();
  mystery_package.run_it();
  unknown_local();
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
    assert by_target["MYSTERY_PACKAGE_CUSTOM.RUN_IT"].resolution_state == "custom_source_missing"
    assert by_target["MYSTERY_PACKAGE.RUN_IT"].resolution_state == "kernel_unavailable"
    assert by_target["UNKNOWN_LOCAL"].resolution_state == "unresolved"


def test_schema_qualified_custom_package_and_all_tables_are_retained() -> None:
    source = """CREATE OR REPLACE PROCEDURE run_report IS
BEGIN
  app.pkg_report_custom.build_report();
  app.pkg_kernel.build_report();
  app.calculate_custom();
  SELECT id FROM app.business_table;
  SELECT id FROM app.audit_custom;
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

    assert by_target["APP.PKG_REPORT_CUSTOM.BUILD_REPORT"].resolution_state == "custom_source_missing"
    assert by_target["APP.PKG_REPORT_CUSTOM.BUILD_REPORT"].dependency_kind == "routine_call"
    assert by_target["APP.PKG_KERNEL.BUILD_REPORT"].resolution_state == "kernel_unavailable"
    assert by_target["APP.PKG_KERNEL.BUILD_REPORT"].dependency_kind == "kernel_boundary"
    assert by_target["APP.CALCULATE_CUSTOM"].resolution_state == "custom_source_missing"
    assert by_target["APP.CALCULATE_CUSTOM"].dependency_kind == "routine_call"
    assert by_target["APP.BUSINESS_TABLE"].dependency_kind == "table_read"
    assert by_target["APP.AUDIT_CUSTOM"].dependency_kind == "table_read"


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


def test_sql_syntax_and_collection_assignments_are_not_routine_calls() -> None:
    source = """CREATE OR REPLACE PACKAGE BODY pkg_noise AS
  PROCEDURE real_helper(p_id NUMBER) IS BEGIN NULL; END;
  PROCEDURE process_rows(p_id NUMBER) IS
    TYPE row_list IS TABLE OF NUMBER INDEX BY PLS_INTEGER;
    rows row_list;
    l_value NUMBER;
  BEGIN
    IF p_id IN (1, 2) AND EXISTS (SELECT 1 FROM claim_table) THEN
      l_value := TRUNC(SYSDATE);
    END IF;
    rows(1) := p_id;
    INSERT INTO claim_table (claim_id) VALUES (p_id);
    INSERT INTO claim_table (claim_id) VALUES rows(1);
    real_helper(p_id);
    mystery_package.run_it(p_id);
  END process_rows;
END pkg_noise;
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

    calls = {
        edge.target_canonical_name: edge
        for edge in edges
        if edge.dependency_kind == "routine_call"
    }
    assert set(calls) == {"REAL_HELPER"}
    assert calls["REAL_HELPER"].resolution_state == "resolved_in_snapshot"
    kernel = next(
        edge for edge in edges if edge.target_canonical_name == "MYSTERY_PACKAGE.RUN_IT"
    )
    assert kernel.dependency_kind == "kernel_boundary"
    assert kernel.resolution_state == "kernel_unavailable"
    assert all(
        target not in calls
        for target in {"IN", "AND", "EXISTS", "TRUNC", "ROWS", "CLAIM_TABLE"}
    )
    table = next(
        edge
        for edge in edges
        if edge.dependency_kind == "table_write"
        and edge.target_canonical_name == "CLAIM_TABLE"
    )
    assert table.resolution_state == "unresolved"


def test_outer_join_cursor_and_infrastructure_utility_calls_are_separated() -> None:
    source = """CREATE OR REPLACE PACKAGE BODY pkg_report_custom AS
  CURSOR cur_fin_txn(p_fund_id VARCHAR2) IS
    SELECT alc.transactionnumber
      FROM allocation_table alc, transaction_table txn
     WHERE txn.transactionnumber = alc.transactionnumber(+);
  PROCEDURE run_report(p_fund_id VARCHAR2) IS
  BEGIN
    FOR row_item IN cur_fin_txn(p_fund_id) LOOP
      isdebug.writeline(row_item.transactionnumber);
      GLOBAL.pr_init('000', 'USER');
      debug.pr_debug('done');
    END LOOP;
  END run_report;
END pkg_report_custom;
/
"""
    parsed = _parse(source)
    parsed = parsed.model_copy(
        update={
            "extracted_nodes": tuple(
                node for node in parsed.extracted_nodes if node.node_kind != "cursor"
            )
        }
    )
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

    assert "ALC.TRANSACTIONNUMBER" not in by_target
    assert by_target["CUR_FIN_TXN"].dependency_kind == "cursor_reference"
    assert by_target["CUR_FIN_TXN"].resolution_state == "resolved_in_snapshot"
    assert by_target["ISDEBUG.WRITELINE"].dependency_kind == "infrastructure_utility"
    assert by_target["GLOBAL.PR_INIT"].dependency_kind == "infrastructure_utility"
    assert by_target["DEBUG.PR_DEBUG"].dependency_kind == "infrastructure_utility"
    assert all(
        edge.resolution_state == "external_schema"
        for target, edge in by_target.items()
        if target in {"ISDEBUG.WRITELINE", "GLOBAL.PR_INIT", "DEBUG.PR_DEBUG"}
    )


def test_symbol_lookup_indexes_thousands_of_program_units_without_changing_resolution() -> None:
    source = """CREATE OR REPLACE PROCEDURE caller_custom IS
BEGIN
  pkg_3999_custom.do_work();
END;
/
"""
    parsed = _parse(source)
    caller = extract_symbols(parsed, module_id="fci-custom")[0]
    generated = []
    for index in range(4_000):
        digest = hashlib.sha256(f"symbol-{index}".encode()).hexdigest()
        generated.append(
            caller.model_copy(
                update={
                    "occurrence_id": digest,
                    "symbol_key": hashlib.sha256(f"key-{index}".encode()).hexdigest(),
                    "canonical_qualified_name": f"PKG_{index}_CUSTOM.DO_WORK",
                    "qualified_display_name": f"PKG_{index}_CUSTOM.DO_WORK",
                }
            )
        )
    all_symbols = (caller, *generated)
    lookup = build_symbol_lookup(all_symbols)

    edges = extract_dependencies(
        source,
        parsed,
        file_symbols=(caller,),
        all_symbols=all_symbols,
        schema_objects=(),
        policy=POLICY,
        symbol_lookup=lookup,
    )

    call = next(edge for edge in edges if edge.target_canonical_name == "PKG_3999_CUSTOM.DO_WORK")
    assert call.resolution_state == "resolved_in_snapshot"
    assert call.candidate_symbol_occurrence_ids == (generated[-1].occurrence_id,)
    assert len(lookup.by_canonical_name) == 4_001
