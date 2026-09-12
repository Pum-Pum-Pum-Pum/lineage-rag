from __future__ import annotations

import re


_IDENTIFIER_TOKEN_PATTERN = re.compile(
    r"[A-Z]+(?=[A-Z][a-z]|\d|$)|[A-Z]?[a-z]+|\d+"
)
_IDENTIFIER_ALIASES = {
    "txn": "transaction",
    "txns": "transaction",
    "sent": "send",
    "sending": "send",
}


def normalized_identifier_tokens(value: str) -> set[str]:
    """Return bounded normalized tokens for an identifier or caller query."""

    expanded = value.replace("_", " ").replace("$", " ")
    raw = _IDENTIFIER_TOKEN_PATTERN.findall(expanded)
    return {
        _IDENTIFIER_ALIASES.get(token.casefold(), token.casefold()) for token in raw
    }


def identifier_affinity(query: str, identifier: str) -> int:
    """Count normalized query tokens shared with a concrete identifier."""

    return len(normalized_identifier_tokens(query) & normalized_identifier_tokens(identifier))
