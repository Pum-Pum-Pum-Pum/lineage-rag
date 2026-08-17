from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass

from antlr4 import Token

from app.code_ingestion.analysis_policy import CodeAnalysisPolicy
from app.code_ingestion.code_analysis_models import (
    CodeSymbol,
    DependencyEdge,
    SchemaObject,
)
from app.code_ingestion.generated.plsql.PlSqlLexer import PlSqlLexer
from app.code_ingestion.oracle_identifiers import (
    canonical_json_hash,
    canonical_qualified_name,
    default_tokens,
    matching_right_parenthesis,
    normalized_token_text,
    oracle_identifier,
    source_map_for_tokens,
    split_top_level,
)
from app.code_ingestion.plsql_models import ExtractedCodeNode, PlSqlFileParseArtifact


_DECLARATION_EDGE_KINDS = {
    "type": "type_reference",
    "constant": "constant_reference",
    "global_variable": "global_reference",
    "cursor": "cursor_reference",
}
_BUILTIN_TYPE_NAMES = {
    "BINARY_DOUBLE",
    "BINARY_FLOAT",
    "BLOB",
    "BOOLEAN",
    "CHAR",
    "CLOB",
    "DATE",
    "INTEGER",
    "NCHAR",
    "NCLOB",
    "NUMBER",
    "NVARCHAR2",
    "PLS_INTEGER",
    "RAW",
    "TIMESTAMP",
    "VARCHAR",
    "VARCHAR2",
}
_NON_CALL_PREFIX_TYPES = {
    PlSqlLexer.IF,
    PlSqlLexer.ELSIF,
    PlSqlLexer.WHILE,
    PlSqlLexer.FOR,
    PlSqlLexer.CASE,
    PlSqlLexer.RETURN,
    PlSqlLexer.PROCEDURE,
    PlSqlLexer.FUNCTION,
}


@dataclass(frozen=True)
class SymbolLookup:
    by_canonical_name: dict[str, tuple[CodeSymbol, ...]]


def build_symbol_lookup(symbols: tuple[CodeSymbol, ...]) -> SymbolLookup:
    grouped: dict[str, list[CodeSymbol]] = defaultdict(list)
    for symbol in symbols:
        grouped[symbol.canonical_qualified_name].append(symbol)
    return SymbolLookup(
        by_canonical_name={
            name: tuple(sorted(items, key=lambda item: item.occurrence_id))
            for name, items in grouped.items()
        }
    )


def extract_dependencies(
    source_text: str,
    parse_artifact: PlSqlFileParseArtifact,
    *,
    file_symbols: tuple[CodeSymbol, ...],
    all_symbols: tuple[CodeSymbol, ...],
    schema_objects: tuple[SchemaObject, ...],
    policy: CodeAnalysisPolicy,
    symbol_lookup: SymbolLookup | None = None,
) -> tuple[DependencyEdge, ...]:
    declarations = tuple(
        node
        for node in parse_artifact.extracted_nodes
        if node.node_kind in _DECLARATION_EDGE_KINDS
    )
    edges: list[DependencyEdge] = []
    for symbol in file_symbols:
        snippet = source_text[symbol.source_map.start_offset : symbol.source_map.end_offset]
        tokens = _without_nested_symbol_tokens(
            default_tokens(snippet),
            symbol=symbol,
            all_file_symbols=file_symbols,
        )
        edges.extend(
            _routine_call_edges(
                tokens,
                source_text=source_text,
                symbol=symbol,
                all_symbols=all_symbols,
                declarations=declarations,
                policy=policy,
                symbol_lookup=symbol_lookup,
            )
        )
        edges.extend(
            _table_edges(
                tokens,
                source_text=source_text,
                symbol=symbol,
                schema_objects=schema_objects,
            )
        )
        edges.extend(
            _dynamic_sql_edges(tokens, source_text=source_text, symbol=symbol)
        )
        edges.extend(
            _declaration_reference_edges(
                tokens,
                source_text=source_text,
                symbol=symbol,
                declarations=declarations,
            )
        )
    for cursor in (node for node in declarations if node.node_kind == "cursor"):
        cursor_text = source_text[cursor.source_map.start_offset : cursor.source_map.end_offset]
        edges.extend(
            _cursor_table_edges(
                default_tokens(cursor_text),
                source_text=source_text,
                source_path=parse_artifact.source_path,
                base_offset=cursor.source_map.start_offset,
                schema_objects=schema_objects,
                cursor_name=cursor.display_name,
            )
        )
    unique = {edge.edge_id: edge for edge in edges}
    return tuple(
        sorted(
            unique.values(),
            key=lambda edge: (
                edge.source_map.start_offset,
                edge.dependency_kind,
                edge.target_canonical_name,
            ),
        )
    )


def _without_nested_symbol_tokens(
    tokens: tuple[Token, ...],
    *,
    symbol: CodeSymbol,
    all_file_symbols: tuple[CodeSymbol, ...],
) -> tuple[Token, ...]:
    nested_ranges = [
        (
            item.source_map.start_offset - symbol.source_map.start_offset,
            item.source_map.end_offset - symbol.source_map.start_offset,
        )
        for item in all_file_symbols
        if item.occurrence_id != symbol.occurrence_id
        and symbol.source_map.start_offset < item.source_map.start_offset
        and item.source_map.end_offset <= symbol.source_map.end_offset
    ]
    return tuple(
        token
        for token in tokens
        if not any(start <= token.start < end for start, end in nested_ranges)
    )


def _routine_call_edges(
    tokens: tuple[Token, ...],
    *,
    source_text: str,
    symbol: CodeSymbol,
    all_symbols: tuple[CodeSymbol, ...],
    declarations: tuple[ExtractedCodeNode, ...],
    policy: CodeAnalysisPolicy,
    symbol_lookup: SymbolLookup | None,
) -> list[DependencyEdge]:
    edges: list[DependencyEdge] = []
    declared_types = {
        oracle_identifier(node.display_name).canonical_name
        for node in declarations
        if node.node_kind == "type"
    }
    indexed_access_names = _indexed_access_names(tokens)
    for left_index, token in enumerate(tokens):
        if token.type != PlSqlLexer.LEFT_PAREN or left_index == 0:
            continue
        name_tokens = _qualified_name_before(tokens, left_index)
        if not name_tokens:
            continue
        start_index = left_index - len(name_tokens)
        previous_type = tokens[start_index - 1].type if start_index else None
        if previous_type in _NON_CALL_PREFIX_TYPES:
            continue
        right_index = matching_right_parenthesis(tokens, left_index)
        if right_index is None:
            continue
        if _is_insert_column_list(tokens, start_index) or _is_collection_assignment(
            tokens, right_index
        ):
            continue
        canonical_target = _canonical_name_tokens(name_tokens)
        if canonical_target in indexed_access_names:
            continue
        final_name = canonical_target.rsplit(".", 1)[-1]
        if (
            final_name in policy.boundaries.ignored_builtin_calls
            or final_name in _BUILTIN_TYPE_NAMES
            or final_name in declared_types
        ):
            continue
        argument_tokens = tokens[left_index + 1 : right_index]
        argument_count = 0 if not argument_tokens else len(split_top_level(argument_tokens))
        candidates = _call_candidates(
            canonical_target,
            argument_count=argument_count,
            source_symbol=symbol,
            all_symbols=all_symbols,
            symbol_lookup=symbol_lookup,
        )
        target_components = canonical_target.split(".")
        owner_component = target_components[-2] if len(target_components) >= 2 else None
        if candidates:
            distinct_keys = {candidate.symbol_key for candidate in candidates}
            state = "resolved_in_snapshot" if len(distinct_keys) == 1 else "ambiguous"
            kind = "routine_call"
            confidence = "high"
        elif _is_missing_custom_program_unit(
            target_components,
            policy=policy,
        ):
            state, kind, confidence = "custom_source_missing", "routine_call", "high"
        elif owner_component is not None and (
            owner_component in policy.boundaries.kernel_package_names
            or any(
                owner_component.startswith(prefix)
                for prefix in policy.boundaries.kernel_package_prefixes
            )
        ):
            state, kind, confidence = "kernel_unavailable", "kernel_boundary", "high"
        elif owner_component is not None and any(
            owner_component.startswith(prefix)
            for prefix in policy.boundaries.external_package_prefixes
        ):
            state, kind, confidence = "external_schema", "external_package", "high"
        elif _is_inferred_kernel_package_call(
            target_components,
            policy=policy,
        ):
            state, kind, confidence = "kernel_unavailable", "kernel_boundary", "medium"
        else:
            state, kind, confidence = "unresolved", "routine_call", "medium"
        source_map = source_map_for_tokens(
            source_text,
            symbol.source_path,
            name_tokens,
            offset_adjustment=symbol.source_map.start_offset,
        )
        edges.append(
            _edge(
                kind=kind,
                symbol=symbol,
                source_map=source_map,
                target_display="".join(item.text for item in name_tokens),
                target_canonical=canonical_target,
                state=state,
                candidates=tuple(item.occurrence_id for item in candidates),
                confidence=confidence,
            )
        )
    return edges


def _is_inferred_kernel_package_call(
    target_components: list[str],
    *,
    policy: CodeAnalysisPolicy,
) -> bool:
    """Apply the approved owner-package suffix convention conservatively.

    For ``SCHEMA.PACKAGE.ROUTINE`` and ``PACKAGE.ROUTINE``, the owner package
    is the component immediately before the routine. Unqualified calls remain
    unresolved because tokens alone cannot prove whether they are functions,
    procedures, built-ins, or local declarations absent from the snapshot.
    """
    if not policy.boundaries.infer_noncustom_qualified_packages_as_kernel:
        return False
    if len(target_components) < 2:
        return False
    owner_package = target_components[-2]
    return not any(
        owner_package.endswith(suffix)
        for suffix in policy.boundaries.custom_program_unit_suffixes
    )


def _is_missing_custom_program_unit(
    target_components: list[str],
    *,
    policy: CodeAnalysisPolicy,
) -> bool:
    suffixes = policy.boundaries.custom_program_unit_suffixes
    called_unit = target_components[-1]
    if called_unit.endswith(suffixes):
        return True
    if len(target_components) < 2:
        return False
    owner_package = target_components[-2]
    return owner_package.endswith(suffixes)


def _table_edges(
    tokens: tuple[Token, ...],
    *,
    source_text: str,
    symbol: CodeSymbol,
    schema_objects: tuple[SchemaObject, ...],
) -> list[DependencyEdge]:
    edges: list[DependencyEdge] = []
    for kind, name_tokens in _table_references(tokens):
        canonical_target = _canonical_name_tokens(name_tokens)
        candidates = _schema_candidates(canonical_target, schema_objects)
        if len(candidates) == 1:
            state = "resolved_in_snapshot"
        elif len(candidates) > 1:
            state = "ambiguous"
        elif "." in canonical_target:
            state = "external_schema"
        else:
            state = "unresolved"
        source_map = source_map_for_tokens(
            source_text,
            symbol.source_path,
            name_tokens,
            offset_adjustment=symbol.source_map.start_offset,
        )
        edges.append(
            _edge(
                kind=kind,
                symbol=symbol,
                source_map=source_map,
                target_display="".join(item.text for item in name_tokens),
                target_canonical=canonical_target,
                state=state,
                confidence="high",
            )
        )
    return edges


def _cursor_table_edges(
    tokens: tuple[Token, ...],
    *,
    source_text: str,
    source_path: str,
    base_offset: int,
    schema_objects: tuple[SchemaObject, ...],
    cursor_name: str,
) -> list[DependencyEdge]:
    edges = []
    for kind, name_tokens in _table_references(tokens):
        canonical_target = _canonical_name_tokens(name_tokens)
        candidates = _schema_candidates(canonical_target, schema_objects)
        state = (
            "resolved_in_snapshot"
            if len(candidates) == 1
            else "ambiguous"
            if len(candidates) > 1
            else "external_schema"
            if "." in canonical_target
            else "unresolved"
        )
        source_map = source_map_for_tokens(
            source_text,
            source_path,
            name_tokens,
            offset_adjustment=base_offset,
        )
        edge_id = canonical_json_hash(
            {
                "source_path": source_path,
                "cursor": cursor_name,
                "kind": kind,
                "target": canonical_target,
                "start": source_map.start_offset,
            }
        )
        edges.append(
            DependencyEdge(
                edge_id=edge_id,
                dependency_kind=kind,  # type: ignore[arg-type]
                source_path=source_path,
                source_map=source_map,
                target_display_name="".join(token.text for token in name_tokens),
                target_canonical_name=canonical_target,
                resolution_state=state,  # type: ignore[arg-type]
                extraction_method="antlr_tokens",
                confidence="high",
            )
        )
    return edges


def _table_references(tokens: tuple[Token, ...]) -> tuple[tuple[str, tuple[Token, ...]], ...]:
    references: list[tuple[str, tuple[Token, ...]]] = []
    parenthesis_depth = 0
    from_clause_depth: int | None = None
    stop_types = {
        PlSqlLexer.WHERE,
        PlSqlLexer.GROUP,
        PlSqlLexer.HAVING,
        PlSqlLexer.ORDER,
        PlSqlLexer.CONNECT,
        PlSqlLexer.START,
        PlSqlLexer.UNION,
        PlSqlLexer.SEMICOLON,
    }
    for index, token in enumerate(tokens):
        if token.type == PlSqlLexer.LEFT_PAREN:
            parenthesis_depth += 1
        elif token.type == PlSqlLexer.RIGHT_PAREN:
            parenthesis_depth = max(0, parenthesis_depth - 1)
        if (
            from_clause_depth is not None
            and parenthesis_depth == from_clause_depth
            and token.type in stop_types
        ):
            from_clause_depth = None
        kind: str | None = None
        target_index: int | None = None
        if token.type == PlSqlLexer.FROM:
            kind, target_index = "table_read", index + 1
            from_clause_depth = parenthesis_depth
        elif token.type == PlSqlLexer.JOIN:
            kind, target_index = "table_read", index + 1
        elif (
            token.type == PlSqlLexer.COMMA
            and from_clause_depth is not None
            and parenthesis_depth == from_clause_depth
        ):
            kind, target_index = "table_read", index + 1
        elif token.type == PlSqlLexer.UPDATE:
            kind, target_index = "table_write", index + 1
        elif token.type in {PlSqlLexer.INSERT, PlSqlLexer.MERGE}:
            if index + 1 < len(tokens) and tokens[index + 1].type == PlSqlLexer.INTO:
                kind, target_index = "table_write", index + 2
        elif token.type == PlSqlLexer.DELETE:
            if index + 1 < len(tokens) and tokens[index + 1].type == PlSqlLexer.FROM:
                kind, target_index = "table_write", index + 2
        if kind is None or target_index is None:
            continue
        name_tokens = _qualified_name_at(tokens, target_index)
        if name_tokens:
            references.append((kind, name_tokens))
    return tuple(references)


def _dynamic_sql_edges(
    tokens: tuple[Token, ...],
    *,
    source_text: str,
    symbol: CodeSymbol,
) -> list[DependencyEdge]:
    edges: list[DependencyEdge] = []
    index = 0
    while index < len(tokens):
        start = None
        expression_start = None
        if (
            tokens[index].type == PlSqlLexer.EXECUTE
            and index + 1 < len(tokens)
            and tokens[index + 1].type == PlSqlLexer.IMMEDIATE
        ):
            start, expression_start = index, index + 2
        elif tokens[index].type == PlSqlLexer.OPEN:
            for_index = next(
                (
                    candidate
                    for candidate in range(index + 1, min(len(tokens), index + 5))
                    if tokens[candidate].type == PlSqlLexer.FOR
                ),
                None,
            )
            if for_index is not None and for_index + 1 < len(tokens):
                if tokens[for_index + 1].type != PlSqlLexer.SELECT:
                    start, expression_start = index, for_index + 1
        if start is None or expression_start is None:
            index += 1
            continue
        end = expression_start
        while end < len(tokens) and tokens[end].type not in {
            PlSqlLexer.SEMICOLON,
            PlSqlLexer.USING,
            PlSqlLexer.INTO,
            PlSqlLexer.RETURNING,
        }:
            end += 1
        mapped_tokens = tokens[start : max(expression_start + 1, end)]
        expression_tokens = tokens[expression_start:end]
        source_map = source_map_for_tokens(
            source_text,
            symbol.source_path,
            mapped_tokens,
            offset_adjustment=symbol.source_map.start_offset,
        )
        expression = normalized_token_text(expression_tokens)[:500] or "<empty>"
        edges.append(
            _edge(
                kind="dynamic_sql",
                symbol=symbol,
                source_map=source_map,
                target_display=expression,
                target_canonical=expression,
                state="dynamic_unknown",
                confidence="high",
            )
        )
        index = max(index + 1, end)
    return edges


def _declaration_reference_edges(
    tokens: tuple[Token, ...],
    *,
    source_text: str,
    symbol: CodeSymbol,
    declarations: tuple[ExtractedCodeNode, ...],
) -> list[DependencyEdge]:
    positions: dict[str, Token] = {}
    for token in tokens:
        if _is_name_token(token):
            positions.setdefault(oracle_identifier(token.text).canonical_name, token)
    edges: list[DependencyEdge] = []
    for declaration in declarations:
        canonical = oracle_identifier(declaration.display_name).canonical_name
        token = positions.get(canonical)
        if token is None:
            continue
        source_map = source_map_for_tokens(
            source_text,
            symbol.source_path,
            (token,),
            offset_adjustment=symbol.source_map.start_offset,
        )
        edges.append(
            _edge(
                kind=_DECLARATION_EDGE_KINDS[declaration.node_kind],
                symbol=symbol,
                source_map=source_map,
                target_display=declaration.display_name,
                target_canonical=canonical,
                state="resolved_in_snapshot",
                confidence="high",
            )
        )
    return edges


def _call_candidates(
    canonical_target: str,
    *,
    argument_count: int,
    source_symbol: CodeSymbol,
    all_symbols: tuple[CodeSymbol, ...],
    symbol_lookup: SymbolLookup | None,
) -> tuple[CodeSymbol, ...]:
    possible_names = [canonical_target]
    if "." not in canonical_target:
        source_parts = source_symbol.canonical_qualified_name.split(".")[:-1]
        for length in range(len(source_parts), 0, -1):
            possible_names.append(".".join((*source_parts[:length], canonical_target)))
    if symbol_lookup is None:
        pool = all_symbols
    else:
        pool = tuple(
            symbol
            for name in possible_names
            for symbol in symbol_lookup.by_canonical_name.get(name, ())
        )
    candidates = [
        symbol
        for symbol in pool
        if symbol.canonical_qualified_name in possible_names
        and len(symbol.parameters) == argument_count
    ]
    return tuple(sorted(candidates, key=lambda item: item.occurrence_id))


def _schema_candidates(
    canonical_target: str,
    schema_objects: tuple[SchemaObject, ...],
) -> tuple[SchemaObject, ...]:
    if "." in canonical_target:
        return tuple(
            item for item in schema_objects if item.canonical_qualified_name == canonical_target
        )
    return tuple(
        item
        for item in schema_objects
        if item.canonical_qualified_name.rsplit(".", 1)[-1] == canonical_target
    )


def _qualified_name_before(tokens: tuple[Token, ...], left_index: int) -> tuple[Token, ...]:
    end = left_index - 1
    if end < 0 or not _is_name_token(tokens[end]):
        return ()
    start = end
    while start >= 2 and tokens[start - 1].type == PlSqlLexer.PERIOD and _is_name_token(tokens[start - 2]):
        start -= 2
    return tokens[start : end + 1]


def _qualified_name_at(tokens: tuple[Token, ...], start: int) -> tuple[Token, ...]:
    if start >= len(tokens) or not _is_name_token(tokens[start]):
        return ()
    end = start + 1
    while end + 1 < len(tokens) and tokens[end].type == PlSqlLexer.PERIOD and _is_name_token(tokens[end + 1]):
        end += 2
    return tokens[start:end]


def _canonical_name_tokens(tokens: Sequence[Token]) -> str:
    return canonical_qualified_name(
        oracle_identifier(token.text)
        for token in tokens
        if token.type != PlSqlLexer.PERIOD
    )


def _is_name_token(token: Token) -> bool:
    return token.type in {PlSqlLexer.REGULAR_ID, PlSqlLexer.DELIMITED_ID}


def _is_insert_column_list(tokens: tuple[Token, ...], name_start_index: int) -> bool:
    """Reject `INSERT INTO table (...)` without hiding ordinary calls."""

    return (
        name_start_index > 0
        and tokens[name_start_index - 1].type == PlSqlLexer.INTO
        and any(
            token.type == PlSqlLexer.INSERT
            for token in tokens[max(0, name_start_index - 4) : name_start_index - 1]
        )
    )


def _is_collection_assignment(tokens: tuple[Token, ...], right_index: int) -> bool:
    """Reject lexer-visible collection/record assignment targets as calls."""

    cursor = right_index + 1
    if cursor < len(tokens) and tokens[cursor].type == PlSqlLexer.ASSIGN_OP:
        return True
    while (
        cursor + 1 < len(tokens)
        and tokens[cursor].type == PlSqlLexer.PERIOD
        and _is_selector_token(tokens[cursor + 1])
    ):
        cursor += 2
    return cursor < len(tokens) and tokens[cursor].type == PlSqlLexer.ASSIGN_OP


def _indexed_access_names(tokens: tuple[Token, ...]) -> set[str]:
    """Find collection/record index targets proven by a following selector."""

    names: set[str] = set()
    for index, token in enumerate(tokens):
        if token.type != PlSqlLexer.LEFT_PAREN or index == 0:
            continue
        name_tokens = _qualified_name_before(tokens, index)
        if not name_tokens:
            continue
        right = matching_right_parenthesis(tokens, index)
        if (
            right is not None
            and right + 2 < len(tokens)
            and tokens[right + 1].type == PlSqlLexer.PERIOD
            and _is_selector_token(tokens[right + 2])
        ):
            names.add(_canonical_name_tokens(name_tokens))
    return names


def _is_selector_token(token: Token) -> bool:
    text = token.text or ""
    return bool(text) and (text[0].isalpha() or text[0] in {'"', "_", "$", "#"})


def _edge(
    *,
    kind: str,
    symbol: CodeSymbol,
    source_map,
    target_display: str,
    target_canonical: str,
    state: str,
    confidence: str,
    candidates: tuple[str, ...] = (),
) -> DependencyEdge:
    edge_id = canonical_json_hash(
        {
            "source": symbol.occurrence_id,
            "kind": kind,
            "target": target_canonical,
            "start": source_map.start_offset,
            "end": source_map.end_offset,
        }
    )
    return DependencyEdge(
        edge_id=edge_id,
        dependency_kind=kind,  # type: ignore[arg-type]
        source_symbol_occurrence_id=symbol.occurrence_id,
        source_path=symbol.source_path,
        source_map=source_map,
        target_display_name=target_display,
        target_canonical_name=target_canonical,
        resolution_state=state,  # type: ignore[arg-type]
        candidate_symbol_occurrence_ids=candidates,
        extraction_method="antlr_tokens",
        confidence=confidence,  # type: ignore[arg-type]
    )
