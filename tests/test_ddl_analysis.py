from __future__ import annotations

import hashlib

from app.code_ingestion.ddl_analysis import extract_ddl_structures, resolve_synonyms
from app.code_ingestion.plsql_parser_core import parse_plsql_source


def _parse(source: str, path: str = "ddl/schema.ddl"):
    return parse_plsql_source(
        source,
        snapshot_id="snapshot-158",
        source_path=path,
        source_sha256=hashlib.sha256(source.encode("utf-8")).hexdigest(),
    )


def _extract(source: str, path: str = "ddl/schema.ddl"):
    parsed = _parse(source, path)
    assert parsed.parser_state == "full_parse"
    return extract_ddl_structures(source, parsed)


def test_tables_columns_constraints_and_other_schema_objects_are_structured() -> None:
    source = """CREATE TABLE app.customer_stage (
  customer_id NUMBER NOT NULL,
  customer_code VARCHAR2(20) DEFAULT 'NEW',
  parent_id NUMBER,
  CONSTRAINT pk_customer_stage PRIMARY KEY (customer_id),
  CONSTRAINT fk_customer_stage FOREIGN KEY (parent_id) REFERENCES app.customer(customer_id)
);
CREATE VIEW app.customer_view AS SELECT customer_id FROM app.customer_stage;
CREATE SEQUENCE app.customer_seq START WITH 1;
CREATE INDEX app.ix_customer_code ON app.customer_stage(customer_code);
CREATE TYPE app.customer_ids AS TABLE OF NUMBER;
"""

    objects, synonyms, diagnostics = _extract(source)
    table = next(item for item in objects if item.object_kind == "table")

    assert not synonyms
    assert not diagnostics
    assert {item.object_kind for item in objects} == {
        "table",
        "view",
        "sequence",
        "index",
        "collection_type",
    }
    assert table.canonical_qualified_name == "APP.CUSTOMER_STAGE"
    assert [column.name.canonical_name for column in table.columns] == [
        "CUSTOMER_ID",
        "CUSTOMER_CODE",
        "PARENT_ID",
    ]
    assert table.columns[0].nullable is False
    assert table.columns[1].default_expression == "'NEW'"
    assert {constraint.constraint_kind for constraint in table.constraints} == {
        "not_null",
        "primary_key",
        "foreign_key",
    }
    foreign_key = next(
        constraint for constraint in table.constraints if constraint.constraint_kind == "foreign_key"
    )
    assert foreign_key.referenced_object == "app.customer"


def test_synonyms_resolve_only_against_approved_snapshot_objects() -> None:
    source = """CREATE TABLE app.customer (customer_id NUMBER);
CREATE SYNONYM app.customer_alias FOR app.customer;
CREATE SYNONYM app.customer_alias_2 FOR app.customer_alias;
CREATE SYNONYM external_alias FOR core.customer;
CREATE SYNONYM remote_alias FOR core.customer@prod_link;
"""

    objects, synonyms, _ = _extract(source)
    resolved = resolve_synonyms(objects, synonyms)
    by_name = {item.name.canonical_name: item for item in resolved}

    assert by_name["CUSTOMER_ALIAS"].resolution_state == "resolved_in_snapshot"
    assert by_name["CUSTOMER_ALIAS"].resolved_object_id == objects[0].object_id
    assert by_name["CUSTOMER_ALIAS_2"].resolution_state == "resolved_in_snapshot"
    assert by_name["EXTERNAL_ALIAS"].resolution_state == "external_schema"
    assert by_name["REMOTE_ALIAS"].resolution_state == "database_link"
    assert by_name["REMOTE_ALIAS"].database_link == "prod_link"


def test_synonym_cycles_and_ambiguous_unqualified_targets_fail_closed() -> None:
    source = """CREATE TABLE app_a.customer (id NUMBER);
CREATE TABLE app_b.customer (id NUMBER);
CREATE SYNONYM customer_alias FOR customer;
CREATE SYNONYM cycle_a FOR cycle_b;
CREATE SYNONYM cycle_b FOR cycle_a;
"""

    objects, synonyms, _ = _extract(source)
    resolved = resolve_synonyms(objects, synonyms)
    by_name = {item.name.canonical_name: item for item in resolved}

    assert by_name["CUSTOMER_ALIAS"].resolution_state == "ambiguous"
    assert by_name["CYCLE_A"].resolution_state == "cyclic"
    assert by_name["CYCLE_B"].resolution_state == "cyclic"


def test_quoted_schema_objects_do_not_merge_with_unquoted_names() -> None:
    source = """CREATE TABLE app.foo (id NUMBER);
CREATE TABLE app."FOO" (id NUMBER);
"""

    objects, _, _ = _extract(source)

    assert {item.canonical_qualified_name for item in objects} == {
        "APP.FOO",
        'APP."FOO"',
    }


def test_degraded_parse_emits_no_schema_claims() -> None:
    source = "not valid DDL\n" * 10
    parsed = _parse(source)
    assert parsed.parser_state == "fallback_parse"

    objects, synonyms, diagnostics = extract_ddl_structures(source, parsed)

    assert not objects
    assert not synonyms
    assert diagnostics[0].code == "ddl_extraction_skipped_for_degraded_parse"
