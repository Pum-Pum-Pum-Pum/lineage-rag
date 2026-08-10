from __future__ import annotations

import hashlib

from antlr4 import CommonTokenStream, InputStream, Token

from app.code_ingestion.generated.plsql.PlSqlLexer import PlSqlLexer
from app.code_ingestion.plsql_models import ParsedSegment, SourceMap


def find_routine_segments(source_text: str, *, source_path: str) -> tuple[ParsedSegment, ...]:
    """Conservatively split routines using lexer tokens, never source regexes."""

    lexer = PlSqlLexer(InputStream(source_text))
    lexer.removeErrorListeners()
    stream = CommonTokenStream(lexer)
    stream.fill()
    tokens = [
        token
        for token in stream.tokens
        if token.type != Token.EOF and token.channel == Token.DEFAULT_CHANNEL
    ]
    segments: list[ParsedSegment] = []
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if token.type not in {PlSqlLexer.PROCEDURE, PlSqlLexer.FUNCTION}:
            index += 1
            continue
        result = _find_routine_end(tokens, index)
        if result is None:
            index += 1
            continue
        end_index, is_spec = result
        end_token = tokens[end_index]
        name = _next_identifier_text(tokens, index + 1)
        kind_prefix = "procedure" if token.type == PlSqlLexer.PROCEDURE else "function"
        segment_kind = f"{kind_prefix}_spec" if is_spec else kind_prefix
        segment_id = hashlib.sha256(
            f"{source_path}|{token.start}|{end_token.stop + 1}|{segment_kind}".encode("utf-8")
        ).hexdigest()
        segments.append(
            ParsedSegment(
                segment_id=segment_id,
                segment_kind=segment_kind,  # type: ignore[arg-type]
                display_name=name,
                source_map=SourceMap(
                    source_path=source_path,
                    start_line=token.line,
                    end_line=end_token.line,
                    start_offset=token.start,
                    end_offset=end_token.stop + 1,
                ),
                parse_succeeded=False,
                syntax_error_count=0,
            )
        )
        index = end_index + 1
    return tuple(_remove_contained_duplicates(segments))


def build_fallback_segments(
    source_text: str,
    *,
    source_path: str,
    max_lines: int = 200,
    overlap_lines: int = 20,
) -> tuple[ParsedSegment, ...]:
    if max_lines <= 0 or overlap_lines < 0 or overlap_lines >= max_lines:
        raise ValueError("Fallback line bounds are invalid")
    lines = source_text.splitlines(keepends=True)
    if not lines:
        return ()
    offsets = [0]
    for line in lines:
        offsets.append(offsets[-1] + len(line))
    result: list[ParsedSegment] = []
    start_line_index = 0
    while start_line_index < len(lines):
        end_line_index = min(len(lines), start_line_index + max_lines)
        start_offset = offsets[start_line_index]
        end_offset = offsets[end_line_index]
        segment_id = hashlib.sha256(
            f"{source_path}|fallback|{start_offset}|{end_offset}".encode("utf-8")
        ).hexdigest()
        result.append(
            ParsedSegment(
                segment_id=segment_id,
                segment_kind="fallback_chunk",
                display_name=f"lines_{start_line_index + 1}_{end_line_index}",
                source_map=SourceMap(
                    source_path=source_path,
                    start_line=start_line_index + 1,
                    end_line=end_line_index,
                    start_offset=start_offset,
                    end_offset=end_offset,
                ),
                parse_succeeded=False,
                syntax_error_count=0,
            )
        )
        if end_line_index == len(lines):
            break
        start_line_index = end_line_index - overlap_lines
    return tuple(result)


def detect_package_name(source_text: str) -> str | None:
    lexer = PlSqlLexer(InputStream(source_text))
    lexer.removeErrorListeners()
    stream = CommonTokenStream(lexer)
    stream.fill()
    tokens = [token for token in stream.tokens if token.channel == Token.DEFAULT_CHANNEL]
    for index, token in enumerate(tokens):
        if token.type != PlSqlLexer.PACKAGE:
            continue
        cursor = index + 1
        if cursor < len(tokens) and tokens[cursor].type == PlSqlLexer.BODY:
            cursor += 1
        while cursor < len(tokens) and tokens[cursor].type in {
            PlSqlLexer.EDITIONABLE,
            PlSqlLexer.NONEDITIONABLE,
        }:
            cursor += 1
        if cursor < len(tokens):
            return tokens[cursor].text
    return None


def _find_routine_end(tokens: list[Token], start_index: int) -> tuple[int, bool] | None:
    paren_depth = 0
    body_started = False
    begin_depth = 0
    declaration_keyword_seen = False
    for index in range(start_index + 1, len(tokens)):
        token = tokens[index]
        if token.type == PlSqlLexer.LEFT_PAREN:
            paren_depth += 1
        elif token.type == PlSqlLexer.RIGHT_PAREN:
            paren_depth = max(0, paren_depth - 1)
        elif paren_depth == 0 and token.type in {PlSqlLexer.IS, PlSqlLexer.AS}:
            declaration_keyword_seen = True
        elif paren_depth == 0 and token.type == PlSqlLexer.BEGIN:
            body_started = True
            begin_depth += 1
        elif body_started and token.type == PlSqlLexer.END:
            next_type = tokens[index + 1].type if index + 1 < len(tokens) else Token.EOF
            if next_type not in {PlSqlLexer.IF, PlSqlLexer.LOOP, PlSqlLexer.CASE}:
                begin_depth = max(0, begin_depth - 1)
                if begin_depth == 0:
                    semicolon = _next_token_type(tokens, index + 1, PlSqlLexer.SEMICOLON)
                    return (semicolon, False) if semicolon is not None else None
        elif paren_depth == 0 and token.type == PlSqlLexer.SEMICOLON:
            if not declaration_keyword_seen:
                return index, True
            if not body_started and _contains_external(tokens, start_index, index):
                return index, False
    return None


def _next_identifier_text(tokens: list[Token], start_index: int) -> str | None:
    if start_index >= len(tokens):
        return None
    return tokens[start_index].text


def _next_token_type(tokens: list[Token], start_index: int, token_type: int) -> int | None:
    for index in range(start_index, min(len(tokens), start_index + 4)):
        if tokens[index].type == token_type:
            return index
    return None


def _contains_external(tokens: list[Token], start: int, end: int) -> bool:
    return any(token.type == PlSqlLexer.EXTERNAL for token in tokens[start:end])


def _remove_contained_duplicates(segments: list[ParsedSegment]) -> list[ParsedSegment]:
    result: list[ParsedSegment] = []
    for segment in segments:
        if any(
            existing.source_map.start_offset <= segment.source_map.start_offset
            and segment.source_map.end_offset <= existing.source_map.end_offset
            for existing in result
        ):
            continue
        result.append(segment)
    return result

