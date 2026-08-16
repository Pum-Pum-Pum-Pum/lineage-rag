from __future__ import annotations

import hashlib

from app.code_ingestion.plsql_parser_core import parse_plsql_source
from app.code_ingestion.plsql_segmentation import build_fallback_segments, find_routine_segments
from app.code_ingestion.snapshot_models import CompilerContext


PACKAGE_SOURCE = """CREATE OR REPLACE PACKAGE pkg_customer AS
  TYPE id_list IS TABLE OF NUMBER;
  c_enabled CONSTANT VARCHAR2(1) := 'Y';
  g_count NUMBER := 0;
  CURSOR c_customers IS SELECT customer_id FROM customers;
  PROCEDURE update_customer(p_id NUMBER);
END pkg_customer;
/
CREATE OR REPLACE PACKAGE BODY pkg_customer AS
  PROCEDURE update_customer(p_id NUMBER) IS
    l_value NUMBER;
    PROCEDURE local_helper IS BEGIN NULL; END;
  BEGIN
    l_value := g_count;
    local_helper;
  END update_customer;
END pkg_customer;
/
"""


def _parse(source: str):
    return parse_plsql_source(
        source,
        snapshot_id="snapshot-1",
        source_path="packages/pkg_customer.prc",
        source_sha256=hashlib.sha256(source.encode("utf-8")).hexdigest(),
    )


def test_full_parser_extracts_package_routines_and_top_level_declarations() -> None:
    artifact = _parse(PACKAGE_SOURCE)

    assert artifact.parser_state == "full_parse"
    assert artifact.syntax_error_count == 0
    kinds = [node.node_kind for node in artifact.extracted_nodes]
    assert kinds.count("package") == 1
    assert kinds.count("package_body") == 1
    assert "type" in kinds
    assert "constant" in kinds
    assert "global_variable" in kinds
    assert "cursor" in kinds
    assert "procedure_spec" in kinds
    assert kinds.count("procedure") == 2
    assert not any(node.display_name == "l_value" for node in artifact.extracted_nodes)
    update = next(
        node
        for node in artifact.extracted_nodes
        if node.node_kind == "procedure" and node.display_name == "update_customer"
    )
    assert update.package_name == "pkg_customer"
    assert update.signature_text == "PROCEDURE update_customer(p_id NUMBER)"
    assert PACKAGE_SOURCE[update.source_map.start_offset : update.source_map.end_offset].startswith(
        "PROCEDURE update_customer"
    )


def test_conditional_type_uses_parse_view_and_retains_branch_state() -> None:
    source = """CREATE OR REPLACE PACKAGE pkg_types AS
$IF $$ENABLE_TYPES $THEN
  TYPE number_list IS TABLE OF NUMBER;
$END
END pkg_types;
/
"""
    artifact = parse_plsql_source(
        source,
        snapshot_id="snapshot-1",
        source_path="pkg_types.sql",
        source_sha256=hashlib.sha256(source.encode()).hexdigest(),
        compiler_context=CompilerContext(plsql_ccflags="enable_types:true"),
    )

    assert artifact.parser_state == "full_parse"
    assert artifact.syntax_error_count > 0
    assert any(diagnostic.code == "conditional_parse_view_used" for diagnostic in artifact.diagnostics)
    type_node = next(node for node in artifact.extracted_nodes if node.node_kind == "type")
    assert type_node.conditional_state == "active"
    assert source[type_node.source_map.start_offset : type_node.source_map.end_offset].strip().startswith(
        "TYPE number_list"
    )


def test_invalid_file_degrades_to_successfully_parsed_routine_segments() -> None:
    source = """THIS IS NOT VALID PL/SQL;
PROCEDURE first_ok IS
BEGIN
  IF 1 = 1 THEN
    NULL;
  END IF;
END first_ok;
BROKEN TOKENS HERE;
PROCEDURE second_ok IS BEGIN NULL; END second_ok;
"""
    artifact = _parse(source)

    assert artifact.parser_state == "segmented_parse"
    assert [segment.display_name for segment in artifact.segments] == ["first_ok", "second_ok"]
    assert all(segment.parse_succeeded for segment in artifact.segments)
    assert {node.display_name for node in artifact.extracted_nodes} == {"first_ok", "second_ok"}


def test_oversized_routine_is_retained_as_explicit_token_structure() -> None:
    source = "BROKEN TOKENS;\nPROCEDURE large_one IS BEGIN\n" + ("NULL;\n" * 100) + "END large_one;\n"
    artifact = parse_plsql_source(
        source,
        snapshot_id="snapshot-1",
        source_path="large.prc",
        source_sha256=hashlib.sha256(source.encode("utf-8")).hexdigest(),
        max_segment_characters=100,
    )

    assert artifact.parser_state == "segmented_parse"
    assert artifact.segments[0].parse_succeeded is False
    assert artifact.segments[0].degradation_reason == "segment_character_limit_exceeded"
    node = artifact.extracted_nodes[0]
    assert node.display_name == "large_one"
    assert node.extraction_method == "token_structural"
    assert node.signature_text == "PROCEDURE large_one"
    assert source[node.source_map.start_offset : node.source_map.end_offset].startswith(
        "PROCEDURE large_one"
    )


def test_unrecoverable_file_uses_bounded_original_line_fallback() -> None:
    source = "\n".join(f"invalid line {index}" for index in range(450))
    artifact = _parse(source)

    assert artifact.parser_state == "fallback_parse"
    assert len(artifact.segments) == 3
    assert artifact.segments[0].source_map.start_line == 1
    assert artifact.segments[0].source_map.end_line == 200
    assert artifact.segments[1].source_map.start_line == 181
    assert all(segment.segment_kind == "fallback_chunk" for segment in artifact.segments)


def test_token_segmentation_ignores_procedure_words_in_comments_and_strings() -> None:
    source = """-- PROCEDURE fake IS BEGIN NULL; END;
PROCEDURE real_one IS
BEGIN
  dbms_output.put_line('FUNCTION fake return number');
  NULL;
END real_one;
"""
    segments = find_routine_segments(source, source_path="safe.prc")

    assert [segment.display_name for segment in segments] == ["real_one"]


def test_fallback_bounds_are_validated() -> None:
    try:
        build_fallback_segments("x", source_path="x.sql", max_lines=20, overlap_lines=20)
    except ValueError as exc:
        assert "bounds are invalid" in str(exc)
    else:
        raise AssertionError("Expected invalid fallback bounds to fail")
