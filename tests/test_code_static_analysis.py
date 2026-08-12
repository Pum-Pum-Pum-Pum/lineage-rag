from __future__ import annotations

import hashlib

from app.code_ingestion.analysis_policy import AnalysisBoundaries, CodeAnalysisPolicy
from app.code_ingestion.code_static_analysis import StaticAnalysisInput, analyze_snapshot_sources
from app.code_ingestion.plsql_parser_core import parse_plsql_source


POLICY = CodeAnalysisPolicy(
    boundaries=AnalysisBoundaries(
        kernel_package_prefixes=("KERNEL_",),
        external_package_prefixes=("DBMS_",),
        ignored_builtin_calls=("NVL",),
    )
)


def _input(source: str, path: str) -> StaticAnalysisInput:
    parsed = parse_plsql_source(
        source,
        snapshot_id="snapshot-static",
        source_path=path,
        source_sha256=hashlib.sha256(source.encode("utf-8")).hexdigest(),
    )
    return StaticAnalysisInput(parse_artifact=parsed, source_text=source)


def test_cross_file_synonym_and_table_dependencies_resolve_at_snapshot_scope() -> None:
    ddl = "CREATE TABLE app.customer (customer_id NUMBER);\n"
    synonym = "CREATE SYNONYM app.customer_alias FOR app.customer;\n"
    package = """CREATE OR REPLACE PROCEDURE read_customer IS
  v_id NUMBER;
BEGIN
  SELECT customer_id INTO v_id FROM app.customer;
END;
/
"""

    artifacts = analyze_snapshot_sources(
        (
            _input(ddl, "ddl/customer.ddl"),
            _input(synonym, "ddl/synonym.sql"),
            _input(package, "packages/read_customer.prc"),
        ),
        module_id="fci-custom",
        policy=POLICY,
    )
    by_path = {artifact.source_path: artifact for artifact in artifacts}

    resolved_synonym = by_path["ddl/synonym.sql"].synonyms[0]
    table_edge = next(
        edge
        for edge in by_path["packages/read_customer.prc"].dependencies
        if edge.dependency_kind == "table_read"
    )
    assert resolved_synonym.resolution_state == "resolved_in_snapshot"
    assert resolved_synonym.resolved_object_id == by_path["ddl/customer.ddl"].schema_objects[0].object_id
    assert table_edge.resolution_state == "resolved_in_snapshot"


def test_cross_file_symbol_collision_is_recorded_in_each_affected_artifact() -> None:
    first = "CREATE OR REPLACE PROCEDURE duplicate_proc(p_id NUMBER) IS BEGIN NULL; END; /\n"
    second = "CREATE OR REPLACE PROCEDURE duplicate_proc(p_id NUMBER) IS BEGIN NULL; END; /\n"

    artifacts = analyze_snapshot_sources(
        (
            _input(first, "packages/first.prc"),
            _input(second, "packages/second.prc"),
        ),
        module_id="fci-custom",
        policy=POLICY,
    )

    assert all(
        any(diagnostic.code == "overload_symbol_collision" for diagnostic in artifact.diagnostics)
        for artifact in artifacts
    )
    assert len({artifact.symbols[0].occurrence_id for artifact in artifacts}) == 2


def test_unknown_boundaries_become_durable_diagnostics() -> None:
    source = """CREATE OR REPLACE PROCEDURE boundary_proc(p_sql VARCHAR2) IS
BEGIN
  kernel_hidden.run_it();
  EXECUTE IMMEDIATE p_sql;
END;
/
"""

    artifact = analyze_snapshot_sources(
        (_input(source, "packages/boundary.prc"),),
        module_id="fci-custom",
        policy=POLICY,
    )[0]

    assert {diagnostic.code for diagnostic in artifact.diagnostics} >= {
        "dependency_kernel_unavailable",
        "dependency_dynamic_unknown",
    }
    assert artifact.analysis_policy_sha256 == POLICY.sha256


def test_duplicate_schema_identity_is_an_error_not_last_file_wins() -> None:
    first = "CREATE TABLE app.customer (id NUMBER);\n"
    second = "CREATE TABLE app.customer (id NUMBER, code VARCHAR2(10));\n"

    artifacts = analyze_snapshot_sources(
        (
            _input(first, "ddl/first.ddl"),
            _input(second, "ddl/second.ddl"),
        ),
        module_id="fci-custom",
        policy=POLICY,
    )

    assert all(
        any(
            diagnostic.code == "duplicate_schema_object_identity"
            and diagnostic.severity == "error"
            for diagnostic in artifact.diagnostics
        )
        for artifact in artifacts
    )
