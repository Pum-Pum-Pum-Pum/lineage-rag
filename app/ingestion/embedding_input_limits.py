from __future__ import annotations

"""Deterministic local bounds for FDD embedding inputs.

The OpenAI embeddings endpoint rejects a single input above 8,192 tokens.  We
use a conservative UTF-8-byte budget rather than guessing token counts without
the provider tokenizer: a byte-level tokenizer cannot emit more tokens than
input bytes, and 6,000 leaves headroom below the provider limit.  The smaller
source budget reserves space for derived retrieval context.
"""


MAX_EMBEDDING_INPUT_BYTES = 6_000
MAX_SOURCE_UNIT_BYTES = 4_500
MAX_DERIVED_CONTEXT_BYTES = 1_000


def utf8_byte_length(text: str) -> int:
    return len(text.encode("utf-8"))


def truncate_utf8(text: str, max_bytes: int) -> str:
    """Return a deterministic UTF-8-safe prefix within ``max_bytes``."""

    if max_bytes <= 0:
        raise ValueError("max_bytes must be greater than zero")
    if utf8_byte_length(text) <= max_bytes:
        return text

    result: list[str] = []
    used = 0
    for character in text:
        character_bytes = utf8_byte_length(character)
        if used + character_bytes > max_bytes:
            break
        result.append(character)
        used += character_bytes
    return "".join(result)


def split_text_by_utf8_bytes(text: str, max_bytes: int) -> tuple[str, ...]:
    """Split losslessly at whitespace where possible and never split UTF-8."""

    if max_bytes <= 0:
        raise ValueError("max_bytes must be greater than zero")
    if not text:
        return ()
    if utf8_byte_length(text) <= max_bytes:
        return (text,)

    pieces: list[str] = []
    start = 0
    text_length = len(text)
    while start < text_length:
        used = 0
        index = start
        last_boundary: int | None = None
        while index < text_length:
            character_bytes = utf8_byte_length(text[index])
            if used + character_bytes > max_bytes:
                break
            used += character_bytes
            index += 1
            if text[index - 1].isspace():
                last_boundary = index

        if index == start:
            raise ValueError("A single character exceeds the configured UTF-8 input limit")
        cut = last_boundary if last_boundary is not None and last_boundary > start else index
        pieces.append(text[start:cut])
        start = cut

    return tuple(pieces)
