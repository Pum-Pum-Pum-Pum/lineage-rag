from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Sequence

from antlr4 import CommonTokenStream, InputStream, Token

from app.code_ingestion.code_analysis_models import OracleIdentifier
from app.code_ingestion.generated.plsql.PlSqlLexer import PlSqlLexer
from app.code_ingestion.plsql_models import SourceMap


_PUNCTUATION = {
    PlSqlLexer.LEFT_PAREN,
    PlSqlLexer.RIGHT_PAREN,
    PlSqlLexer.COMMA,
    PlSqlLexer.PERIOD,
    PlSqlLexer.SEMICOLON,
    PlSqlLexer.AT_SIGN,
}


def default_tokens(source_text: str) -> tuple[Token, ...]:
    lexer = PlSqlLexer(InputStream(source_text))
    lexer.removeErrorListeners()
    stream = CommonTokenStream(lexer)
    stream.fill()
    return tuple(
        token
        for token in stream.tokens
        if token.type != Token.EOF and token.channel == Token.DEFAULT_CHANNEL
    )


def oracle_identifier(value: str) -> OracleIdentifier:
    display = value.strip()
    quoted = len(display) >= 2 and display.startswith('"') and display.endswith('"')
    canonical = display if quoted else display.upper()
    return OracleIdentifier(
        display_name=display,
        canonical_name=canonical,
        is_quoted=quoted,
    )


def qualified_parts(value: str) -> tuple[OracleIdentifier, ...]:
    tokens = default_tokens(value)
    return tuple(
        oracle_identifier(token.text)
        for token in tokens
        if token.type not in _PUNCTUATION and token.text
    )


def canonical_qualified_name(parts: Iterable[OracleIdentifier]) -> str:
    return ".".join(part.canonical_name for part in parts)


def display_qualified_name(parts: Iterable[OracleIdentifier]) -> str:
    return ".".join(part.display_name for part in parts)


def normalized_token_text(tokens: Sequence[Token]) -> str:
    pieces: list[str] = []
    for token in tokens:
        text = token.text or ""
        if token.type == PlSqlLexer.DELIMITED_ID:
            pieces.append(text)
        elif token.type in {
            PlSqlLexer.CHAR_STRING,
            PlSqlLexer.NATIONAL_CHAR_STRING_LIT,
            PlSqlLexer.BIT_STRING_LIT,
            PlSqlLexer.HEX_STRING_LIT,
        }:
            pieces.append(text)
        elif text and (text[0].isalpha() or text[0] in {'"', '_', '$', '#'}):
            pieces.append(text.upper())
        else:
            pieces.append(text)
    return " ".join(pieces)


def canonical_json_hash(payload: object) -> str:
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def matching_right_parenthesis(tokens: Sequence[Token], left_index: int) -> int | None:
    depth = 0
    for index in range(left_index, len(tokens)):
        if tokens[index].type == PlSqlLexer.LEFT_PAREN:
            depth += 1
        elif tokens[index].type == PlSqlLexer.RIGHT_PAREN:
            depth -= 1
            if depth == 0:
                return index
    return None


def split_top_level(
    tokens: Sequence[Token],
    *,
    separator_type: int = PlSqlLexer.COMMA,
) -> tuple[tuple[Token, ...], ...]:
    groups: list[tuple[Token, ...]] = []
    current: list[Token] = []
    depth = 0
    for token in tokens:
        if token.type == PlSqlLexer.LEFT_PAREN:
            depth += 1
        elif token.type == PlSqlLexer.RIGHT_PAREN:
            depth = max(0, depth - 1)
        if token.type == separator_type and depth == 0:
            if current:
                groups.append(tuple(current))
            current = []
            continue
        current.append(token)
    if current:
        groups.append(tuple(current))
    return tuple(groups)


def source_map_for_tokens(
    source_text: str,
    source_path: str,
    tokens: Sequence[Token],
    *,
    offset_adjustment: int = 0,
) -> SourceMap:
    if not tokens:
        raise ValueError("Cannot create a source map for an empty token sequence")
    start = tokens[0].start + offset_adjustment
    end = tokens[-1].stop + 1 + offset_adjustment
    start_line = source_text.count("\n", 0, start) + 1
    end_line = source_text.count("\n", 0, max(start, end - 1)) + 1
    return SourceMap(
        source_path=source_path,
        start_line=start_line,
        end_line=end_line,
        start_offset=start,
        end_offset=end,
    )


def token_slice_text(source_text: str, tokens: Sequence[Token], *, offset: int = 0) -> str:
    if not tokens:
        return ""
    return source_text[tokens[0].start + offset : tokens[-1].stop + 1 + offset]
