from __future__ import annotations

from collections import defaultdict

from antlr4 import Token

from app.code_ingestion.code_analysis_models import (
    AnalysisDiagnostic,
    CodeSymbol,
    ParameterContract,
)
from app.code_ingestion.generated.plsql.PlSqlLexer import PlSqlLexer
from app.code_ingestion.oracle_identifiers import (
    canonical_json_hash,
    canonical_qualified_name,
    default_tokens,
    display_qualified_name,
    matching_right_parenthesis,
    normalized_token_text,
    oracle_identifier,
    qualified_parts,
    split_top_level,
)
from app.code_ingestion.plsql_models import ExtractedCodeNode, PlSqlFileParseArtifact


_ROUTINE_KINDS = {"procedure", "procedure_spec", "function", "function_spec"}
_SIGNATURE_STOP_TOKENS = {
    PlSqlLexer.IS,
    PlSqlLexer.AS,
    PlSqlLexer.SEMICOLON,
}


def extract_symbols(
    parse_artifact: PlSqlFileParseArtifact,
    *,
    module_id: str,
) -> tuple[CodeSymbol, ...]:
    symbols = [
        _symbol_from_node(parse_artifact, node, module_id=module_id)
        for node in parse_artifact.extracted_nodes
        if node.node_kind in _ROUTINE_KINDS
    ]
    return tuple(
        sorted(
            symbols,
            key=lambda symbol: (
                symbol.source_map.start_offset,
                symbol.occurrence_role,
                symbol.occurrence_id,
            ),
        )
    )


def diagnose_symbol_groups(symbols: tuple[CodeSymbol, ...]) -> tuple[AnalysisDiagnostic, ...]:
    grouped: dict[str, list[CodeSymbol]] = defaultdict(list)
    for symbol in symbols:
        grouped[symbol.symbol_key].append(symbol)

    diagnostics: list[AnalysisDiagnostic] = []
    for symbol_key, occurrences in sorted(grouped.items()):
        if len(occurrences) < 2:
            continue
        by_role: dict[str, list[CodeSymbol]] = defaultdict(list)
        for occurrence in occurrences:
            by_role[occurrence.occurrence_role].append(occurrence)
        duplicate_role = any(len(items) > 1 for items in by_role.values())
        if duplicate_role:
            diagnostics.append(
                AnalysisDiagnostic(
                    stage="symbol",
                    severity="error",
                    code="overload_symbol_collision",
                    message=(
                        "Multiple same-role routine occurrences share one overload-safe symbol "
                        f"key ({symbol_key[:12]}); all occurrences were retained."
                    ),
                    related_occurrence_ids=tuple(item.occurrence_id for item in occurrences),
                )
            )
        elif len({_compatibility_hash(item) for item in occurrences}) > 1:
            diagnostics.append(
                AnalysisDiagnostic(
                    stage="symbol",
                    severity="error",
                    code="declaration_implementation_signature_mismatch",
                    message=(
                        "A declaration and implementation share an overload discriminator but "
                        "their full signatures differ; neither occurrence was overwritten."
                    ),
                    related_occurrence_ids=tuple(item.occurrence_id for item in occurrences),
                )
            )
    return tuple(diagnostics)


def _compatibility_hash(symbol: CodeSymbol) -> str:
    return canonical_json_hash(
        {
            "parameters": [
                {
                    "name": parameter.name.canonical_name,
                    "quoted": parameter.name.is_quoted,
                    "type": parameter.canonical_declared_type,
                    "mode": parameter.mode,
                    "nocopy": parameter.nocopy,
                }
                for parameter in symbol.parameters
            ],
            "return_type": symbol.canonical_return_type,
        }
    )


def _symbol_from_node(
    parse_artifact: PlSqlFileParseArtifact,
    node: ExtractedCodeNode,
    *,
    module_id: str,
) -> CodeSymbol:
    signature = node.signature_text or ""
    tokens = default_tokens(signature)
    symbol_kind = "function" if node.node_kind.startswith("function") else "procedure"
    parameters, closing_parenthesis = _parse_parameters(tokens)
    return_type, canonical_return_type = _parse_return_type(
        tokens,
        symbol_kind=symbol_kind,
        start_index=(closing_parenthesis + 1 if closing_parenthesis is not None else 0),
    )

    name = oracle_identifier(qualified_parts(node.display_name)[-1].display_name)
    qualified = []
    if node.package_name:
        qualified.extend(qualified_parts(node.package_name))
    qualified.extend(oracle_identifier(value) for value in node.enclosing_routines)
    display_parts = qualified_parts(node.display_name)
    if not node.package_name and len(display_parts) > 1:
        qualified.extend(display_parts)
    else:
        qualified.append(name)
    canonical_name = canonical_qualified_name(qualified)
    display_name = display_qualified_name(qualified)

    overload_payload = {
        "parameter_count": len(parameters),
        "parameters": [
            {
                "name": parameter.name.canonical_name,
                "quoted": parameter.name.is_quoted,
                "declared_type": parameter.canonical_declared_type,
                "type_family": parameter.type_family,
            }
            for parameter in parameters
        ],
    }
    overload_hash = canonical_json_hash(overload_payload)
    declaration_payload = {
        **overload_payload,
        "parameters": [
            {
                **overload_payload["parameters"][index],
                "mode": parameter.mode,
                "nocopy": parameter.nocopy,
                "has_default": parameter.has_default,
                "default": parameter.normalized_default,
            }
            for index, parameter in enumerate(parameters)
        ],
        "return_type": canonical_return_type,
        "conditional_state": node.conditional_state,
    }
    declaration_hash = canonical_json_hash(declaration_payload)
    symbol_key = canonical_json_hash(
        {
            "language": "plsql",
            "module_id": module_id,
            "canonical_qualified_name": canonical_name,
            "symbol_kind": symbol_kind,
            "overload_discriminator_hash": overload_hash,
        }
    )
    role = "declaration" if node.node_kind.endswith("_spec") else "implementation"
    occurrence_id = canonical_json_hash(
        {
            "snapshot_id": parse_artifact.snapshot_id,
            "source_path": parse_artifact.source_path,
            "source_node_id": node.node_id,
            "role": role,
        }
    )
    return CodeSymbol(
        occurrence_id=occurrence_id,
        symbol_key=symbol_key,
        source_node_id=node.node_id,
        module_id=module_id,
        snapshot_id=parse_artifact.snapshot_id,
        source_path=parse_artifact.source_path,
        source_map=node.source_map,
        occurrence_role=role,
        symbol_kind=symbol_kind,
        name=name,
        qualified_display_name=display_name,
        canonical_qualified_name=canonical_name,
        parameters=parameters,
        return_type=return_type,
        canonical_return_type=canonical_return_type,
        overload_discriminator_hash=overload_hash,
        declaration_signature_hash=declaration_hash,
        conditional_state=node.conditional_state,
    )


def _parse_parameters(tokens: tuple[Token, ...]) -> tuple[tuple[ParameterContract, ...], int | None]:
    routine_index = next(
        (index for index, token in enumerate(tokens) if token.type in {PlSqlLexer.PROCEDURE, PlSqlLexer.FUNCTION}),
        None,
    )
    if routine_index is None:
        return (), None
    left = next(
        (
            index
            for index in range(routine_index + 1, len(tokens))
            if tokens[index].type == PlSqlLexer.LEFT_PAREN
        ),
        None,
    )
    if left is None:
        return (), None
    right = matching_right_parenthesis(tokens, left)
    if right is None:
        return (), None
    groups = split_top_level(tokens[left + 1 : right])
    return tuple(_parse_parameter(group, index + 1) for index, group in enumerate(groups)), right


def _parse_parameter(tokens: tuple[Token, ...], position: int) -> ParameterContract:
    if len(tokens) < 2:
        raise ValueError("PL/SQL parameter must contain a name and declared type")
    name = oracle_identifier(tokens[0].text)
    default_index = next(
        (
            index
            for index, token in enumerate(tokens)
            if token.type in {PlSqlLexer.ASSIGN_OP, PlSqlLexer.DEFAULT}
        ),
        None,
    )
    declaration_end = default_index if default_index is not None else len(tokens)
    mode_tokens: list[str] = []
    nocopy = False
    type_tokens: list[Token] = []
    for token in tokens[1:declaration_end]:
        if token.type in {PlSqlLexer.IN, PlSqlLexer.OUT} and not type_tokens:
            mode_tokens.append(token.text.upper())
        elif token.type == PlSqlLexer.NOCOPY and not type_tokens:
            nocopy = True
        else:
            type_tokens.append(token)
    if not type_tokens:
        raise ValueError(f"Parameter {name.display_name!r} has no declared type")
    mode = "IN OUT" if mode_tokens == ["IN", "OUT"] else (mode_tokens[0] if mode_tokens else "IN")
    declared_type = "".join(token.text for token in type_tokens)
    canonical_type = normalized_token_text(type_tokens)
    default_tokens_value = tokens[default_index + 1 :] if default_index is not None else ()
    return ParameterContract(
        position=position,
        name=name,
        declared_type=declared_type,
        canonical_declared_type=canonical_type,
        type_family=_type_family(canonical_type),
        mode=mode,  # type: ignore[arg-type]
        nocopy=nocopy,
        has_default=default_index is not None,
        normalized_default=(
            normalized_token_text(default_tokens_value) if default_tokens_value else None
        ),
    )


def _parse_return_type(
    tokens: tuple[Token, ...],
    *,
    symbol_kind: str,
    start_index: int,
) -> tuple[str | None, str | None]:
    if symbol_kind != "function":
        return None, None
    return_index = next(
        (index for index in range(start_index, len(tokens)) if tokens[index].type == PlSqlLexer.RETURN),
        None,
    )
    if return_index is None:
        return None, None
    end = next(
        (
            index
            for index in range(return_index + 1, len(tokens))
            if tokens[index].type in _SIGNATURE_STOP_TOKENS
        ),
        len(tokens),
    )
    value_tokens = tokens[return_index + 1 : end]
    if not value_tokens:
        return None, None
    return "".join(token.text for token in value_tokens), normalized_token_text(value_tokens)


def _type_family(canonical_type: str) -> str:
    compact = canonical_type.replace(" ", "")
    if any(compact.startswith(name) for name in ("NUMBER", "INTEGER", "PLS_INTEGER", "BINARY_FLOAT", "BINARY_DOUBLE", "DECIMAL", "NUMERIC")):
        return "numeric"
    if any(compact.startswith(name) for name in ("VARCHAR", "VARCHAR2", "CHAR", "NCHAR", "NVARCHAR2", "CLOB", "NCLOB")):
        return "character"
    if any(compact.startswith(name) for name in ("DATE", "TIMESTAMP", "INTERVAL")):
        return "datetime"
    if any(compact.startswith(name) for name in ("RAW", "LONGRAW", "BLOB")):
        return "binary"
    if compact.startswith("BOOLEAN"):
        return "boolean"
    if "%TYPE" in compact:
        return "anchored_type"
    if "%ROWTYPE" in compact:
        return "anchored_rowtype"
    return "user_defined"
