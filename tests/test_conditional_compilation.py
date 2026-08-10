from __future__ import annotations

from antlr4 import CommonTokenStream, InputStream

from app.code_ingestion.conditional_compilation import build_conditional_parse_view
from app.code_ingestion.generated.plsql.PlSqlLexer import PlSqlLexer
from app.code_ingestion.generated.plsql.PlSqlParser import PlSqlParser
from app.code_ingestion.snapshot_models import CompilerContext


CONDITIONAL_PACKAGE = """CREATE OR REPLACE PACKAGE BODY pkg_demo AS
$IF $$DEBUG $THEN
  PROCEDURE run_mode IS BEGIN NULL; END;
$ELSIF DBMS_DB_VERSION.VERSION >= 19 $THEN
  PROCEDURE run_mode IS BEGIN NULL; END;
$ELSE
  PROCEDURE run_mode IS BEGIN NULL; END;
$END
END pkg_demo;
/
"""


def _syntax_error_count(source: str) -> int:
    lexer = PlSqlLexer(InputStream(source))
    parser = PlSqlParser(CommonTokenStream(lexer))
    lexer.removeErrorListeners()
    parser.removeErrorListeners()
    parser.sql_script()
    return parser.getNumberOfSyntaxErrors()


def test_conditional_parse_view_preserves_offsets_and_enables_full_parse() -> None:
    view = build_conditional_parse_view(
        CONDITIONAL_PACKAGE,
        source_path="packages/pkg_demo.prc",
    )

    assert _syntax_error_count(CONDITIONAL_PACKAGE) == 0
    assert _syntax_error_count(view.text) == 0
    assert len(view.text) == len(CONDITIONAL_PACKAGE)
    assert [index for index, value in enumerate(view.text) if value == "\n"] == [
        index for index, value in enumerate(CONDITIONAL_PACKAGE) if value == "\n"
    ]
    assert view.original_sha256 != view.parse_view_sha256
    assert len(view.regions) == 1
    assert [branch.state for branch in view.regions[0].branches] == [
        "conditional_unknown",
        "conditional_unknown",
        "conditional_unknown",
    ]


def test_parse_view_closes_real_grammar_gap_for_conditional_type_declaration() -> None:
    source = """CREATE OR REPLACE PACKAGE pkg_types AS
$IF TRUE $THEN
  TYPE number_list IS TABLE OF NUMBER;
$END
END pkg_types;
/
"""
    view = build_conditional_parse_view(source, source_path="pkg_types.sql")

    assert _syntax_error_count(source) > 0
    assert _syntax_error_count(view.text) == 0
    assert source[view.regions[0].branches[0].body_source_map.start_offset :].lstrip().startswith(
        "TYPE number_list"
    )


def test_known_compiler_context_marks_one_branch_active() -> None:
    view = build_conditional_parse_view(
        CONDITIONAL_PACKAGE,
        source_path="packages/pkg_demo.prc",
        compiler_context=CompilerContext(
            oracle_version="19.22",
            plsql_ccflags="debug:false",
        ),
    )

    assert [branch.state for branch in view.regions[0].branches] == [
        "inactive",
        "active",
        "inactive",
    ]
    assert view.regions[0].branches[1].body_source_map.start_line == 4
    assert view.regions[0].branches[1].body_source_map.end_line == 5


def test_nested_directives_keep_parent_region_identity() -> None:
    source = """BEGIN
$IF TRUE $THEN
  $IF FALSE $THEN
    NULL;
  $ELSE
    NULL;
  $END
$END
END;
/"""
    view = build_conditional_parse_view(source, source_path="nested.sql")

    assert len(view.regions) == 2
    assert view.regions[0].parent_region_id is None
    assert view.regions[1].parent_region_id == view.regions[0].region_id
    assert _syntax_error_count(view.text) == 0


def test_directive_like_text_in_comments_and_strings_is_not_masked() -> None:
    source = """BEGIN
  -- $IF TRUE $THEN
  dbms_output.put_line('$END');
  NULL;
END;
/"""
    view = build_conditional_parse_view(source, source_path="literal.sql")

    assert view.regions == ()
    assert view.text == source


def test_error_directive_is_preserved_as_metadata_and_masked_in_parse_view() -> None:
    source = """BEGIN
$ERROR 'unsupported build' $END
NULL;
END;
/"""
    view = build_conditional_parse_view(source, source_path="error.sql")

    assert len(view.error_directives) == 1
    assert view.error_directives[0].expression == "'unsupported build'"
    assert "unsupported build" not in view.text
    assert _syntax_error_count(view.text) == 0


def test_unclosed_directive_produces_explicit_diagnostic() -> None:
    view = build_conditional_parse_view(
        "$IF TRUE $THEN\nNULL;\n",
        source_path="broken.sql",
    )

    assert [diagnostic.code for diagnostic in view.diagnostics] == [
        "unclosed_conditional_directive"
    ]
