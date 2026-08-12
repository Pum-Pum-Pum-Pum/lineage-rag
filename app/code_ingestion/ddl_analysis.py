from __future__ import annotations

from collections.abc import Iterable

from antlr4 import CommonTokenStream, InputStream, ParserRuleContext, Token

from app.code_ingestion.code_analysis_models import (
    AnalysisDiagnostic,
    ColumnDefinition,
    ConstraintDefinition,
    SchemaObject,
    SynonymDefinition,
)
from app.code_ingestion.generated.plsql.PlSqlLexer import PlSqlLexer
from app.code_ingestion.generated.plsql.PlSqlParser import PlSqlParser
from app.code_ingestion.oracle_identifiers import (
    canonical_json_hash,
    canonical_qualified_name,
    default_tokens,
    oracle_identifier,
    qualified_parts,
    source_map_for_tokens,
)
from app.code_ingestion.plsql_models import PlSqlFileParseArtifact, SourceMap


def extract_ddl_structures(
    source_text: str,
    parse_artifact: PlSqlFileParseArtifact,
) -> tuple[tuple[SchemaObject, ...], tuple[SynonymDefinition, ...], tuple[AnalysisDiagnostic, ...]]:
    if parse_artifact.parser_state != "full_parse":
        return (
            (),
            (),
            (
                AnalysisDiagnostic(
                    stage="ddl",
                    severity="warning",
                    code="ddl_extraction_skipped_for_degraded_parse",
                    message=(
                        "DDL structure extraction requires a trustworthy full parse; original "
                        "source remains available through degraded retrieval units."
                    ),
                    source_path=parse_artifact.source_path,
                ),
            ),
        )
    lexer = PlSqlLexer(InputStream(source_text))
    lexer.removeErrorListeners()
    stream = CommonTokenStream(lexer)
    parser = PlSqlParser(stream)
    parser.removeErrorListeners()
    tree = parser.sql_script()
    if parser.getNumberOfSyntaxErrors():
        return (
            (),
            (),
            (
                AnalysisDiagnostic(
                    stage="ddl",
                    severity="error",
                    code="ddl_reparse_failed_closed",
                    message="DDL analysis reparse was not trustworthy; no schema claims were emitted.",
                    source_path=parse_artifact.source_path,
                ),
            ),
        )

    objects: list[SchemaObject] = []
    synonyms: list[SynonymDefinition] = []
    for context in _walk(tree):
        if isinstance(context, PlSqlParser.Create_tableContext):
            objects.append(_table_object(context, source_text, parse_artifact.source_path))
        elif isinstance(context, PlSqlParser.Create_viewContext):
            objects.append(_simple_object(context, source_text, parse_artifact.source_path, "view"))
        elif isinstance(context, PlSqlParser.Create_sequenceContext):
            objects.append(_simple_object(context, source_text, parse_artifact.source_path, "sequence"))
        elif isinstance(context, PlSqlParser.Create_indexContext):
            objects.append(_index_object(context, source_text, parse_artifact.source_path))
        elif isinstance(context, PlSqlParser.Create_typeContext):
            objects.append(_type_object(context, source_text, parse_artifact.source_path))
        elif isinstance(context, PlSqlParser.Create_synonymContext):
            synonyms.append(_synonym(context, source_text, parse_artifact.source_path))
    return (
        tuple(sorted(objects, key=lambda item: item.source_map.start_offset)),
        tuple(sorted(synonyms, key=lambda item: item.source_map.start_offset)),
        (),
    )


def resolve_synonyms(
    schema_objects: tuple[SchemaObject, ...],
    synonyms: tuple[SynonymDefinition, ...],
) -> tuple[SynonymDefinition, ...]:
    by_name: dict[str, list[SynonymDefinition]] = {}
    for synonym in synonyms:
        by_name.setdefault(synonym.canonical_qualified_name, []).append(synonym)

    def object_candidates(target: str) -> tuple[SchemaObject, ...]:
        if "." in target:
            return tuple(item for item in schema_objects if item.canonical_qualified_name == target)
        return tuple(
            item
            for item in schema_objects
            if item.canonical_qualified_name.rsplit(".", 1)[-1] == target
        )

    def synonym_candidates(target: str) -> tuple[SynonymDefinition, ...]:
        if "." in target:
            return tuple(by_name.get(target, ()))
        return tuple(
            item
            for item in synonyms
            if item.canonical_qualified_name.rsplit(".", 1)[-1] == target
        )

    def resolve(
        synonym: SynonymDefinition,
        visiting: frozenset[str],
    ) -> tuple[str, str | None]:
        if synonym.database_link:
            return "database_link", None
        if synonym.synonym_id in visiting:
            return "cyclic", None
        objects = object_candidates(synonym.canonical_declared_target)
        if len(objects) == 1:
            return "resolved_in_snapshot", objects[0].object_id
        if len(objects) > 1:
            return "ambiguous", None
        targets = synonym_candidates(synonym.canonical_declared_target)
        if len(targets) == 1:
            return resolve(targets[0], visiting | {synonym.synonym_id})
        if len(targets) > 1:
            return "ambiguous", None
        if "." in synonym.canonical_declared_target:
            return "external_schema", None
        return "ambiguous", None

    resolved = []
    for synonym in synonyms:
        state, object_id = resolve(synonym, frozenset())
        resolved.append(
            synonym.model_copy(
                update={"resolution_state": state, "resolved_object_id": object_id}
            )
        )
    return tuple(resolved)


def _table_object(
    context: PlSqlParser.Create_tableContext,
    source_text: str,
    source_path: str,
) -> SchemaObject:
    schema_context = context.schema_name()
    schema = oracle_identifier(schema_context.getText()) if schema_context else None
    name = oracle_identifier(context.table_name().getText())
    columns: list[ColumnDefinition] = []
    constraints: list[ConstraintDefinition] = []
    for child in _walk(context):
        if isinstance(child, PlSqlParser.Column_definitionContext):
            column, inline = _column_definition(child, source_text, source_path)
            columns.append(column)
            constraints.extend(inline)
        elif isinstance(child, (PlSqlParser.Out_of_line_constraintContext, PlSqlParser.Out_of_line_ref_constraintContext)):
            constraints.append(_constraint(child, source_text, source_path, column_name=None))
    return SchemaObject(
        object_id=_object_id("table", source_path, context.start.start, context.stop.stop + 1),
        object_kind="table",
        name=name,
        schema_name=schema,
        canonical_qualified_name=_qualified(schema, name),
        source_path=source_path,
        source_map=_context_map(context, source_text, source_path),
        columns=tuple(columns),
        constraints=tuple(constraints),
    )


def _column_definition(
    context: PlSqlParser.Column_definitionContext,
    source_text: str,
    source_path: str,
) -> tuple[ColumnDefinition, tuple[ConstraintDefinition, ...]]:
    name = oracle_identifier(context.column_name().getText())
    type_context = context.datatype() or context.type_name()
    declared_type = type_context.getText() if type_context else None
    inline_contexts = tuple(
        item
        for item in context.inline_constraint()
        if item.getText().upper() != "NULL"
    )
    inline_constraints = tuple(
        _constraint(item, source_text, source_path, column_name=name.display_name)
        for item in inline_contexts
    )
    nullable = None
    if any(item.constraint_kind in {"not_null", "primary_key"} for item in inline_constraints):
        nullable = False
    elif any("NULL" in item.getText().upper() for item in inline_contexts):
        nullable = True
    expression = context.expression()
    return (
        ColumnDefinition(
            name=name,
            declared_type=declared_type,
            canonical_declared_type=declared_type.upper() if declared_type else None,
            nullable=nullable,
            default_expression=expression.getText() if expression else None,
            source_map=_context_map(context, source_text, source_path),
        ),
        inline_constraints,
    )


def _constraint(
    context: ParserRuleContext,
    source_text: str,
    source_path: str,
    *,
    column_name: str | None,
) -> ConstraintDefinition:
    tokens = _context_tokens(source_text, context)
    token_types = {token.type for token in tokens}
    if PlSqlLexer.PRIMARY in token_types:
        kind = "primary_key"
    elif PlSqlLexer.FOREIGN in token_types or PlSqlLexer.REFERENCES in token_types:
        kind = "foreign_key"
    elif PlSqlLexer.UNIQUE in token_types:
        kind = "unique"
    elif PlSqlLexer.CHECK in token_types:
        kind = "check"
    else:
        kind = "not_null"
    constraint_name = None
    if PlSqlLexer.CONSTRAINT in token_types or PlSqlLexer.CONSTRAINTS in token_types:
        marker = next(
            index
            for index, token in enumerate(tokens)
            if token.type in {PlSqlLexer.CONSTRAINT, PlSqlLexer.CONSTRAINTS}
        )
        if marker + 1 < len(tokens):
            constraint_name = oracle_identifier(tokens[marker + 1].text)
    columns = (column_name,) if column_name else _constraint_columns(tokens)
    referenced = None
    reference_index = next(
        (index for index, token in enumerate(tokens) if token.type == PlSqlLexer.REFERENCES),
        None,
    )
    if reference_index is not None:
        target_tokens = []
        for token in tokens[reference_index + 1 :]:
            if token.type == PlSqlLexer.LEFT_PAREN:
                break
            target_tokens.append(token)
        referenced = "".join(token.text for token in target_tokens) or None
    return ConstraintDefinition(
        name=constraint_name,
        constraint_kind=kind,  # type: ignore[arg-type]
        columns=columns,
        referenced_object=referenced,
        source_map=_context_map(context, source_text, source_path),
    )


def _constraint_columns(tokens: tuple[Token, ...]) -> tuple[str, ...]:
    start = next(
        (index for index, token in enumerate(tokens) if token.type == PlSqlLexer.LEFT_PAREN),
        None,
    )
    if start is None:
        return ()
    names = []
    for token in tokens[start + 1 :]:
        if token.type == PlSqlLexer.RIGHT_PAREN:
            break
        if token.type not in {PlSqlLexer.COMMA, PlSqlLexer.PERIOD} and token.text:
            names.append(token.text)
    return tuple(names)


def _simple_object(
    context: ParserRuleContext,
    source_text: str,
    source_path: str,
    kind: str,
) -> SchemaObject:
    if kind == "view":
        schema_context = context.schema_name()
        name_text = context.v.getText()
    else:
        parts = qualified_parts(context.sequence_name().getText())
        schema_context = None
        name_text = parts[-1].display_name
        if len(parts) > 1:
            schema_context = parts[-2]
    schema = (
        schema_context
        if hasattr(schema_context, "canonical_name")
        else oracle_identifier(schema_context.getText()) if schema_context else None
    )
    name = oracle_identifier(name_text)
    return SchemaObject(
        object_id=_object_id(kind, source_path, context.start.start, context.stop.stop + 1),
        object_kind=kind,  # type: ignore[arg-type]
        name=name,
        schema_name=schema,
        canonical_qualified_name=_qualified(schema, name),
        source_path=source_path,
        source_map=_context_map(context, source_text, source_path),
    )


def _index_object(
    context: PlSqlParser.Create_indexContext,
    source_text: str,
    source_path: str,
) -> SchemaObject:
    parts = qualified_parts(context.index_name().getText())
    schema = parts[-2] if len(parts) > 1 else None
    name = parts[-1]
    referenced = tuple(
        sorted(
            {
                child.getText().upper()
                for child in _walk(context)
                if isinstance(child, PlSqlParser.Tableview_nameContext)
            }
        )
    )
    return SchemaObject(
        object_id=_object_id("index", source_path, context.start.start, context.stop.stop + 1),
        object_kind="index",
        name=name,
        schema_name=schema,
        canonical_qualified_name=_qualified(schema, name),
        source_path=source_path,
        source_map=_context_map(context, source_text, source_path),
        referenced_objects=referenced,
    )


def _type_object(
    context: PlSqlParser.Create_typeContext,
    source_text: str,
    source_path: str,
) -> SchemaObject:
    definition = context.type_definition()
    body = context.type_body()
    type_name = definition.type_name().getText() if definition else body.type_name().getText()
    parts = qualified_parts(type_name)
    schema = parts[-2] if len(parts) > 1 else None
    name = parts[-1]
    normalized = context.getText().upper()
    kind = "collection_type" if "VARRAY" in normalized or "TABLEOF" in normalized else "object_type"
    return SchemaObject(
        object_id=_object_id(kind, source_path, context.start.start, context.stop.stop + 1),
        object_kind=kind,
        name=name,
        schema_name=schema,
        canonical_qualified_name=_qualified(schema, name),
        source_path=source_path,
        source_map=_context_map(context, source_text, source_path),
    )


def _synonym(
    context: PlSqlParser.Create_synonymContext,
    source_text: str,
    source_path: str,
) -> SynonymDefinition:
    tokens = _context_tokens(source_text, context)
    synonym_index = next(index for index, token in enumerate(tokens) if token.type == PlSqlLexer.SYNONYM)
    for_index = next(index for index, token in enumerate(tokens) if token.type == PlSqlLexer.FOR)
    at_index = next(
        (index for index, token in enumerate(tokens) if token.type == PlSqlLexer.AT_SIGN),
        None,
    )
    name_parts = _name_parts(tokens[synonym_index + 1 : for_index])
    target_end = at_index if at_index is not None else len(tokens)
    target_parts = _name_parts(tokens[for_index + 1 : target_end])
    is_public = any(token.type == PlSqlLexer.PUBLIC for token in tokens[:synonym_index])
    name = name_parts[-1]
    schema = name_parts[-2] if len(name_parts) > 1 else None
    canonical_name = (
        f"PUBLIC.{name.canonical_name}" if is_public else _qualified(schema, name)
    )
    declared_target = ".".join(part.display_name for part in target_parts)
    canonical_target = canonical_qualified_name(target_parts)
    database_link = (
        "".join(token.text for token in tokens[at_index + 1 :]) if at_index is not None else None
    )
    return SynonymDefinition(
        synonym_id=_object_id("synonym", source_path, context.start.start, context.stop.stop + 1),
        name=name,
        schema_name=schema,
        is_public=is_public,
        canonical_qualified_name=canonical_name,
        declared_target=declared_target,
        canonical_declared_target=canonical_target,
        database_link=database_link,
        resolution_state="database_link" if database_link else "ambiguous",
        source_path=source_path,
        source_map=_context_map(context, source_text, source_path),
    )


def _name_parts(tokens: Iterable[Token]):
    return tuple(
        oracle_identifier(token.text)
        for token in tokens
        if token.type != PlSqlLexer.PERIOD and token.text
    )


def _qualified(schema, name) -> str:
    return canonical_qualified_name(part for part in (schema, name) if part is not None)


def _walk(context: ParserRuleContext):
    yield context
    for child in getattr(context, "children", None) or []:
        if isinstance(child, ParserRuleContext):
            yield from _walk(child)


def _context_tokens(source_text: str, context: ParserRuleContext) -> tuple[Token, ...]:
    snippet = source_text[context.start.start : context.stop.stop + 1]
    return default_tokens(snippet)


def _context_map(context: ParserRuleContext, source_text: str, source_path: str) -> SourceMap:
    snippet_tokens = _context_tokens(source_text, context)
    return source_map_for_tokens(
        source_text,
        source_path,
        snippet_tokens,
        offset_adjustment=context.start.start,
    )


def _object_id(kind: str, source_path: str, start: int, end: int) -> str:
    return canonical_json_hash(
        {"kind": kind, "source_path": source_path, "start": start, "end": end}
    )
