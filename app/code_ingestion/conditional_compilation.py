from __future__ import annotations

import bisect
import hashlib
import re
from dataclasses import dataclass, field

from antlr4 import CommonTokenStream, InputStream, Token

from app.code_ingestion.generated.plsql.PlSqlLexer import PlSqlLexer
from app.code_ingestion.plsql_models import (
    ConditionalBranch,
    ConditionalErrorDirective,
    ConditionalParseView,
    ConditionalRegion,
    ParseDiagnostic,
    SourceMap,
)
from app.code_ingestion.snapshot_models import CompilerContext


_DIRECTIVE_TYPES = {
    PlSqlLexer.DOLLAR_IF,
    PlSqlLexer.DOLLAR_THEN,
    PlSqlLexer.DOLLAR_ELSIF,
    PlSqlLexer.DOLLAR_ELSE,
    PlSqlLexer.DOLLAR_END,
    PlSqlLexer.DOLLAR_ERROR,
}


@dataclass
class _BranchBuilder:
    branch_kind: str
    directive_line: int
    directive_start: int
    expression_start: int | None
    expression_end: int | None = None
    body_start: int | None = None
    body_end: int | None = None


@dataclass
class _RegionBuilder:
    region_id: str
    parent_region_id: str | None
    start_offset: int
    start_line: int
    branches: list[_BranchBuilder] = field(default_factory=list)
    end_offset: int | None = None


@dataclass
class _ErrorBuilder:
    start_offset: int
    start_line: int
    expression_start: int


def build_conditional_parse_view(
    source_text: str,
    *,
    source_path: str,
    compiler_context: CompilerContext | None = None,
) -> ConditionalParseView:
    """Mask compiler directives without changing source offsets or line numbers."""

    lexer = PlSqlLexer(InputStream(source_text))
    token_stream = CommonTokenStream(lexer)
    token_stream.fill()
    tokens = [token for token in token_stream.tokens if token.type in _DIRECTIVE_TYPES]
    line_starts = _line_starts(source_text)
    selection_stack: list[_RegionBuilder] = []
    control_stack: list[tuple[str, object]] = []
    completed_regions: list[_RegionBuilder] = []
    completed_errors: list[tuple[_ErrorBuilder, int]] = []
    mask_ranges: list[tuple[int, int]] = []
    diagnostics: list[ParseDiagnostic] = []

    for token in tokens:
        if token.type == PlSqlLexer.DOLLAR_IF:
            parent_id = selection_stack[-1].region_id if selection_stack else None
            region = _RegionBuilder(
                region_id=f"conditional::{token.start}",
                parent_region_id=parent_id,
                start_offset=token.start,
                start_line=token.line,
            )
            region.branches.append(
                _BranchBuilder(
                    branch_kind="if",
                    directive_line=token.line,
                    directive_start=token.start,
                    expression_start=token.stop + 1,
                )
            )
            selection_stack.append(region)
            control_stack.append(("selection", region))
        elif token.type == PlSqlLexer.DOLLAR_THEN:
            if not selection_stack or not selection_stack[-1].branches:
                diagnostics.append(_malformed(token, "$THEN without an open $IF"))
                mask_ranges.append((token.start, token.stop + 1))
                continue
            branch = selection_stack[-1].branches[-1]
            branch.expression_end = token.start
            branch.body_start = token.stop + 1
            assert branch.expression_start is not None
            mask_ranges.append((branch.directive_start, token.stop + 1))
        elif token.type == PlSqlLexer.DOLLAR_ELSIF:
            if not selection_stack:
                diagnostics.append(_malformed(token, "$ELSIF without an open $IF"))
                mask_ranges.append((token.start, token.stop + 1))
                continue
            current = selection_stack[-1].branches[-1]
            current.body_end = token.start
            selection_stack[-1].branches.append(
                _BranchBuilder(
                    branch_kind="elsif",
                    directive_line=token.line,
                    directive_start=token.start,
                    expression_start=token.stop + 1,
                )
            )
        elif token.type == PlSqlLexer.DOLLAR_ELSE:
            if not selection_stack:
                diagnostics.append(_malformed(token, "$ELSE without an open $IF"))
            else:
                selection_stack[-1].branches[-1].body_end = token.start
                selection_stack[-1].branches.append(
                    _BranchBuilder(
                        branch_kind="else",
                        directive_line=token.line,
                        directive_start=token.start,
                        expression_start=None,
                        body_start=token.stop + 1,
                    )
                )
            mask_ranges.append((token.start, token.stop + 1))
        elif token.type == PlSqlLexer.DOLLAR_ERROR:
            error = _ErrorBuilder(
                start_offset=token.start,
                start_line=token.line,
                expression_start=token.stop + 1,
            )
            control_stack.append(("error", error))
        elif token.type == PlSqlLexer.DOLLAR_END:
            if not control_stack:
                diagnostics.append(_malformed(token, "$END without an open directive"))
                mask_ranges.append((token.start, token.stop + 1))
                continue
            control_kind, builder = control_stack.pop()
            if control_kind == "error":
                assert isinstance(builder, _ErrorBuilder)
                completed_errors.append((builder, token.stop + 1))
                mask_ranges.append((builder.start_offset, token.stop + 1))
                continue
            assert isinstance(builder, _RegionBuilder)
            builder.branches[-1].body_end = token.start
            builder.end_offset = token.stop + 1
            completed_regions.append(builder)
            if selection_stack and selection_stack[-1] is builder:
                selection_stack.pop()
            mask_ranges.append((token.start, token.stop + 1))

    for control_kind, builder in control_stack:
        start_line = builder.start_line if isinstance(builder, (_RegionBuilder, _ErrorBuilder)) else None
        diagnostics.append(
            ParseDiagnostic(
                stage="conditional_scan",
                severity="error",
                code="unclosed_conditional_directive",
                message=f"Unclosed compiler directive of type {control_kind}.",
                line=start_line,
            )
        )

    parse_characters = list(source_text)
    for start, end in _merge_ranges(mask_ranges):
        for index in range(max(0, start), min(len(parse_characters), end)):
            if parse_characters[index] not in {"\r", "\n"}:
                parse_characters[index] = " "
    parse_text = "".join(parse_characters)

    regions = tuple(
        _finalize_region(
            builder,
            source_text=source_text,
            source_path=source_path,
            line_starts=line_starts,
            compiler_context=compiler_context,
        )
        for builder in sorted(completed_regions, key=lambda item: item.start_offset)
        if builder.end_offset is not None
    )
    errors = tuple(
        ConditionalErrorDirective(
            source_map=_source_map(
                source_path,
                builder.start_offset,
                end_offset,
                line_starts,
            ),
            expression=source_text[builder.expression_start : end_offset - len("$END")].strip(),
            state="conditional_unknown" if compiler_context is None else "unresolved",
        )
        for builder, end_offset in completed_errors
    )
    return ConditionalParseView(
        original_sha256=hashlib.sha256(source_text.encode("utf-8")).hexdigest(),
        parse_view_sha256=hashlib.sha256(parse_text.encode("utf-8")).hexdigest(),
        text=parse_text,
        regions=regions,
        error_directives=errors,
        diagnostics=tuple(diagnostics),
    )


def conditional_state_for_range(
    regions: tuple[ConditionalRegion, ...],
    *,
    start_offset: int,
    end_offset: int,
) -> str:
    containing = [
        branch.state
        for region in regions
        for branch in region.branches
        if branch.body_source_map.start_offset <= start_offset
        and end_offset <= branch.body_source_map.end_offset
    ]
    if not containing:
        return "unconditional"
    for state in ("inactive", "unresolved", "conditional_unknown", "active"):
        if state in containing:
            return state
    return "unconditional"


def _finalize_region(
    builder: _RegionBuilder,
    *,
    source_text: str,
    source_path: str,
    line_starts: list[int],
    compiler_context: CompilerContext | None,
) -> ConditionalRegion:
    expressions = [
        (
            source_text[branch.expression_start : branch.expression_end].strip()
            if branch.expression_start is not None and branch.expression_end is not None
            else None
        )
        for branch in builder.branches
    ]
    states = _resolve_branch_states(expressions, compiler_context)
    branches: list[ConditionalBranch] = []
    for branch, expression, state in zip(builder.branches, expressions, states, strict=True):
        body_start = branch.body_start if branch.body_start is not None else branch.expression_end
        body_end = branch.body_end if branch.body_end is not None else body_start
        assert body_start is not None and body_end is not None
        branches.append(
            ConditionalBranch(
                branch_kind=branch.branch_kind,  # type: ignore[arg-type]
                expression=expression,
                state=state,  # type: ignore[arg-type]
                directive_line=branch.directive_line,
                body_source_map=_source_map(source_path, body_start, body_end, line_starts),
            )
        )
    assert builder.end_offset is not None
    return ConditionalRegion(
        region_id=builder.region_id,
        parent_region_id=builder.parent_region_id,
        source_map=_source_map(source_path, builder.start_offset, builder.end_offset, line_starts),
        branches=tuple(branches),
    )


def _resolve_branch_states(
    expressions: list[str | None],
    compiler_context: CompilerContext | None,
) -> list[str]:
    if compiler_context is None or (
        compiler_context.oracle_version is None and compiler_context.plsql_ccflags is None
    ):
        return ["conditional_unknown"] * len(expressions)

    active_found = False
    unresolved_before = False
    states: list[str] = []
    flags = _parse_ccflags(compiler_context.plsql_ccflags)
    for expression in expressions:
        if expression is None:
            if active_found:
                states.append("inactive")
            elif unresolved_before:
                states.append("unresolved")
            else:
                states.append("active")
            continue
        result = _evaluate_expression(expression, flags, compiler_context.oracle_version)
        if active_found:
            states.append("inactive")
        elif result is True and not unresolved_before:
            states.append("active")
            active_found = True
        elif result is False:
            states.append("inactive")
        else:
            states.append("unresolved")
            unresolved_before = True
    return states


def _parse_ccflags(value: str | None) -> dict[str, object]:
    if not value:
        return {}
    result: dict[str, object] = {}
    for item in value.split(","):
        if ":" not in item:
            continue
        name, raw_value = item.split(":", 1)
        normalized = raw_value.strip().strip("'\"")
        if normalized.upper() in {"TRUE", "FALSE"}:
            parsed: object = normalized.upper() == "TRUE"
        elif re.fullmatch(r"[+-]?\d+", normalized):
            parsed = int(normalized)
        else:
            parsed = normalized
        result[name.strip().upper()] = parsed
    return result


def _evaluate_expression(
    expression: str,
    flags: dict[str, object],
    oracle_version: str | None,
) -> bool | None:
    normalized = " ".join(expression.strip().split())
    upper = normalized.upper()
    if upper == "TRUE":
        return True
    if upper == "FALSE":
        return False
    if upper.startswith("NOT "):
        inner = _evaluate_expression(normalized[4:], flags, oracle_version)
        return None if inner is None else not inner
    flag_match = re.fullmatch(r"\$\$([A-Za-z][A-Za-z0-9_$#]*)", normalized)
    if flag_match:
        value = flags.get(flag_match.group(1).upper())
        return value if isinstance(value, bool) else None
    comparison = re.fullmatch(
        r"(\$\$[A-Za-z][A-Za-z0-9_$#]*|DBMS_DB_VERSION\.(?:VERSION|RELEASE))\s*"
        r"(=|<>|!=|<=|>=|<|>)\s*(TRUE|FALSE|[+-]?\d+|'[^']*'|\"[^\"]*\")",
        normalized,
        re.IGNORECASE,
    )
    if not comparison:
        return None
    left_name, operator, raw_right = comparison.groups()
    if left_name.upper().startswith("$$"):
        left = flags.get(left_name[2:].upper())
    else:
        version_parts = [int(part) for part in re.findall(r"\d+", oracle_version or "")]
        if not version_parts:
            return None
        left = version_parts[0] if left_name.upper().endswith(".VERSION") else (
            version_parts[1] if len(version_parts) > 1 else 0
        )
    right: object
    if raw_right.upper() in {"TRUE", "FALSE"}:
        right = raw_right.upper() == "TRUE"
    elif re.fullmatch(r"[+-]?\d+", raw_right):
        right = int(raw_right)
    else:
        right = raw_right[1:-1]
    if left is None or type(left) is not type(right):
        return None
    operations = {
        "=": lambda: left == right,
        "<>": lambda: left != right,
        "!=": lambda: left != right,
        "<": lambda: left < right,  # type: ignore[operator]
        ">": lambda: left > right,  # type: ignore[operator]
        "<=": lambda: left <= right,  # type: ignore[operator]
        ">=": lambda: left >= right,  # type: ignore[operator]
    }
    return operations[operator]()


def _malformed(token: Token, message: str) -> ParseDiagnostic:
    return ParseDiagnostic(
        stage="conditional_scan",
        severity="error",
        code="malformed_conditional_directive",
        message=message,
        line=token.line,
        column=token.column,
    )


def _line_starts(text: str) -> list[int]:
    starts = [0]
    starts.extend(index + 1 for index, character in enumerate(text) if character == "\n")
    return starts


def _source_map(source_path: str, start: int, end: int, line_starts: list[int]) -> SourceMap:
    bounded_end = max(start, end)
    start_line = bisect.bisect_right(line_starts, start)
    line_reference = max(start, bounded_end - 1)
    end_line = bisect.bisect_right(line_starts, line_reference)
    return SourceMap(
        source_path=source_path,
        start_line=max(1, start_line),
        end_line=max(1, end_line),
        start_offset=start,
        end_offset=bounded_end,
    )


def _merge_ranges(ranges: list[tuple[int, int]]) -> list[tuple[int, int]]:
    merged: list[list[int]] = []
    for start, end in sorted(ranges):
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return [(start, end) for start, end in merged]
