from __future__ import annotations

import hashlib

import pytest

from app.code_ingestion.code_retrieval_artifact import (
    DERIVED_CONTEXT_MARKER,
    build_code_retrieval_artifact,
)
from app.code_ingestion.plsql_parser_core import parse_plsql_source
from app.code_ingestion.plsql_models import CodeRetrievalArtifact


PACKAGE_SOURCE = """CREATE OR REPLACE PACKAGE BODY pkg_customer AS
  TYPE address_rec IS RECORD (city VARCHAR2(30));
  c_default_country CONSTANT VARCHAR2(2) := 'MY';
  g_batch_size NUMBER := 100;
  CURSOR c_customer IS SELECT customer_id FROM customers;
  c_unused CONSTANT NUMBER := 7;

  PROCEDURE update_address(p_address address_rec) IS
  BEGIN
    IF g_batch_size > 0 AND c_default_country = 'MY' THEN
      OPEN c_customer;
    END IF;
  END update_address;
END pkg_customer;
/
"""


def _parse(source: str = PACKAGE_SOURCE):
    return parse_plsql_source(
        source,
        snapshot_id="snapshot-1",
        source_path="packages/pkg_customer.sql",
        source_sha256=hashlib.sha256(source.encode("utf-8")).hexdigest(),
    )


def test_routine_links_only_referenced_package_declarations() -> None:
    artifact = build_code_retrieval_artifact(_parse(), PACKAGE_SOURCE)
    procedure = next(unit for unit in artifact.units if unit.source_kind == "procedure")
    declarations = {unit.display_name: unit for unit in artifact.units if unit.source_kind != "procedure"}

    assert procedure.derived_context is not None
    assert procedure.derived_context.referenced_types == ("address_rec",)
    assert procedure.derived_context.referenced_constants == ("c_default_country",)
    assert procedure.derived_context.referenced_globals == ("g_batch_size",)
    assert procedure.derived_context.referenced_cursors == ("c_customer",)
    assert "c_unused" not in procedure.retrieval_text
    assert set(procedure.related_unit_ids) == {
        declarations[name].unit_id
        for name in ("address_rec", "c_default_country", "g_batch_size", "c_customer")
    }


def test_original_text_is_citeable_and_derived_header_is_separate() -> None:
    artifact = build_code_retrieval_artifact(_parse(), PACKAGE_SOURCE)
    procedure = next(unit for unit in artifact.units if unit.source_kind == "procedure")

    expected = PACKAGE_SOURCE[
        procedure.source_map.start_offset : procedure.source_map.end_offset
    ]
    assert procedure.text == expected
    assert procedure.retrieval_text.startswith(DERIVED_CONTEXT_MARKER)
    assert procedure.retrieval_text.endswith(expected)
    assert procedure.text not in DERIVED_CONTEXT_MARKER


def test_retrieval_artifact_is_deterministic() -> None:
    parsed = _parse()

    first = build_code_retrieval_artifact(parsed, PACKAGE_SOURCE)
    second = build_code_retrieval_artifact(parsed, PACKAGE_SOURCE)

    assert first == second
    assert len({unit.unit_id for unit in first.units}) == first.total_units


def test_fallback_segments_remain_exact_original_source() -> None:
    source = "this is not PL/SQL\n" * 230
    parsed = _parse(source)

    artifact = build_code_retrieval_artifact(parsed, source)

    assert parsed.parser_state == "fallback_parse"
    assert all(unit.source_kind == "fallback_chunk" for unit in artifact.units)
    for unit in artifact.units:
        assert unit.text == source[unit.source_map.start_offset : unit.source_map.end_offset]
        assert unit.retrieval_text == unit.text


def test_successfully_parsed_ddl_is_retained_before_step_158_structure_extraction() -> None:
    source = "CREATE TABLE customer_stage (customer_id NUMBER);\n"
    parsed = _parse(source)

    artifact = build_code_retrieval_artifact(parsed, source)

    assert parsed.parser_state == "full_parse"
    assert artifact.total_units == 1
    assert artifact.units[0].source_kind == "source_chunk"
    assert artifact.units[0].text == source


def test_source_hash_mismatch_is_rejected() -> None:
    with pytest.raises(ValueError, match="source_text does not match"):
        build_code_retrieval_artifact(_parse(), PACKAGE_SOURCE + "-- changed")


def test_persisted_total_cannot_disagree_with_units() -> None:
    artifact = build_code_retrieval_artifact(_parse(), PACKAGE_SOURCE)

    with pytest.raises(ValueError, match="total_units"):
        CodeRetrievalArtifact.model_validate(
            {**artifact.model_dump(mode="json"), "total_units": artifact.total_units + 1}
        )


def test_conditional_state_is_retained_in_context() -> None:
    source = """CREATE OR REPLACE PACKAGE BODY pkg_conditional AS
$IF $$feature_enabled $THEN
  PROCEDURE enabled_path IS BEGIN NULL; END;
$END
END pkg_conditional;
/
"""
    parsed = _parse(source)

    artifact = build_code_retrieval_artifact(parsed, source)
    procedure = next(unit for unit in artifact.units if unit.source_kind == "procedure")

    assert procedure.conditional_state == "conditional_unknown"
    assert procedure.derived_context is not None
    assert procedure.derived_context.conditional_state == "conditional_unknown"


def test_derived_context_is_deduplicated_and_bounded() -> None:
    declarations = "\n".join(
        f"  c_value_{index} CONSTANT NUMBER := {index};" for index in range(25)
    )
    references = " + ".join(f"c_value_{index}" for index in range(25))
    source = f"""CREATE OR REPLACE PACKAGE BODY pkg_limits AS
{declarations}
  PROCEDURE use_values IS
    total NUMBER;
  BEGIN
    total := {references};
  END use_values;
END pkg_limits;
/
"""
    parsed = _parse(source)

    artifact = build_code_retrieval_artifact(parsed, source)
    procedure = next(unit for unit in artifact.units if unit.source_kind == "procedure")

    assert procedure.derived_context is not None
    assert len(procedure.derived_context.referenced_constants) == 20
    assert len(set(procedure.derived_context.referenced_constants)) == 20
    header = procedure.retrieval_text.split("\n\nORIGINAL CITATION SOURCE:", 1)[0]
    assert len(header) <= 2_000
