from __future__ import annotations

from dataclasses import replace
import re

from app.code_indexing.models import CodeIndexArtifact
from app.retrieval.lexical_search import (
    LexicalSearchDocument,
    LexicalSearchResult,
    search_lexical_documents,
    tokenize,
)


# A code identifier explicitly named by the caller is stronger evidence than
# generic words such as "visible", "custom", or "behavior".  This is a
# code-lane-only ranking adjustment; it does not alter FDD lexical retrieval.
EXACT_CODE_SYMBOL_BONUS = 100.0
EXACT_CODE_SOURCE_BONUS = 100.0

# Natural-language scaffolding is useful to an answer writer but not a useful
# discriminator between PL/SQL routines.  In a larger corpus it can otherwise
# overwhelm the actual product and integration names in a code-only lexical
# search.  This is deliberately code-lane-only: FDD search retains its current
# wording-sensitive behaviour.
CODE_QUERY_STOPWORDS = frozenset(
    {
        "behavior",
        "code",
        "custom",
        "details",
        "flow",
        "handles",
        "logic",
        "routine",
        "routines",
        "system",
        "that",
        "visible",
        "where",
    }
)

# A small, explicit vocabulary of source-code abbreviations.  These are
# retrieval hints only: they do not change cited text, infer schema semantics,
# or establish an FDD-to-code mapping.
CODE_IDENTIFIER_ALIASES = {
    "uh": ("unitholder",),
    "txn": ("transaction",),
}

IDENTIFIER_TERM_BONUS = 3.0
TECHNICAL_ENTITY_COOCCURRENCE_BONUS = 10.0


def _normalise_word(token: str) -> str:
    """Return a conservative singular/verb form for identifier matching."""

    token = token.casefold()
    if token == "sent":
        return "send"
    if len(token) > 4 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
        return token[:-1]
    return token


def _identifier_tokens(value: str) -> set[str]:
    """Split a PL/SQL identifier into stable searchable words and aliases."""

    split = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", value)
    split = re.sub(r"(?<=[A-Z])(?=[A-Z][a-z])", " ", split)
    tokens = {_normalise_word(token) for token in tokenize(split)}
    for token in tuple(tokens):
        tokens.update(CODE_IDENTIFIER_ALIASES.get(token, ()))
    return tokens


def _rankable_query_terms(query: str) -> list[str]:
    terms = [
        _normalise_word(token)
        for token in tokenize(query)
        if token not in CODE_QUERY_STOPWORDS
    ]
    return terms or tokenize(query)


def _explicit_technical_entities(query: str) -> set[str]:
    """Find caller-supplied acronyms or mixed-case product/integration names."""

    return {
        match.group(0).casefold()
        for match in re.finditer(r"\b[A-Za-z][A-Za-z0-9]*\b", query)
        if len(match.group(0)) >= 2
        and (
            match.group(0).isupper()
            or (
                match.group(0)[0].isupper()
                and any(character.isupper() for character in match.group(0)[1:])
            )
        )
    }


def _code_rank_bonus(
    *,
    record: object,
    query_identifier_terms: set[str],
    technical_entities: set[str],
) -> float:
    """Compute bounded code-lane ranking hints from immutable record metadata."""

    display_name = str(getattr(record, "display_name"))
    package_name = str(getattr(record, "package_name"))
    source_path = str(getattr(record, "source_path"))
    identifier_terms = (
        _identifier_tokens(display_name)
        | _identifier_tokens(package_name)
        | _identifier_tokens(source_path)
    )
    identifier_matches = query_identifier_terms & identifier_terms
    bonus = IDENTIFIER_TERM_BONUS * len(identifier_matches)

    # Two caller-supplied technical names in one cited unit are substantially
    # more specific than generic prose containing only one of them.  A single
    # acronym (for example, AML) is intentionally not enough to activate this.
    if len(technical_entities) >= 2:
        document_terms = set(tokenize(str(getattr(record, "embedding_text"))))
        if technical_entities.issubset(document_terms):
            bonus += TECHNICAL_ENTITY_COOCCURRENCE_BONUS
    return bonus


def _derived_identifier_search_text(record: object) -> str:
    """Return non-citation identifier hints so exact filenames enter ranking."""

    values = (
        str(getattr(record, "source_path")),
        str(getattr(record, "display_name")),
        str(getattr(record, "package_name")),
    )
    raw_tokens = [token for value in values for token in tokenize(value)]
    expanded_tokens = [
        token
        for value in values
        for token in sorted(_identifier_tokens(value))
    ]
    return "DERIVED IDENTIFIER SEARCH HINTS: " + " ".join(
        dict.fromkeys(raw_tokens + expanded_tokens)
    )


def search_code_lexical_artifact(
    artifact: CodeIndexArtifact,
    query: str,
    *,
    limit: int = 10,
    source_kind: str | None = None,
    allowed_unit_ids: set[str] | None = None,
) -> list[LexicalSearchResult]:
    rankable_terms = _rankable_query_terms(query)
    lexical_query = " ".join(rankable_terms)
    documents = [
        LexicalSearchDocument(
            document_name=record.source_path,
            document_id=record.snapshot_id,
            unit_id=record.unit_id,
            unit_index=record.unit_index,
            source_kind=record.source_kind,
            document_family=record.module_id,
            release_label=record.snapshot_id,
            text=record.citation_text,
            # This local derived context is ranking-only and is never exposed
            # as a citation.  It admits exact filenames and identifier words
            # into lexical candidate selection without rebuilding embeddings.
            retrieval_text=(
                f"{record.embedding_text}\n{_derived_identifier_search_text(record)}"
            ),
            parent_unit_id=record.parent_unit_id,
        )
        for record in artifact.records
        if allowed_unit_ids is None or record.unit_id in allowed_unit_ids
    ]
    # The lexical engine already scans every local record. Request the full
    # ranked set here so an exact symbol that initially loses to generic prose
    # can be deterministically promoted before the code candidate budget is
    # applied.
    ranked = search_lexical_documents(
        documents,
        lexical_query,
        limit=len(documents),
        source_kind=source_kind,
    )
    exact_symbols = set(tokenize(query))
    query_identifier_terms = set(rankable_terms)
    technical_entities = _explicit_technical_entities(query)
    records_by_unit_id = {record.unit_id: record for record in artifact.records}
    boosted = []
    for result in ranked:
        record = records_by_unit_id[str(result.payload["unit_id"])]
        score = result.score
        if record.display_name.casefold() in exact_symbols:
            score += EXACT_CODE_SYMBOL_BONUS
        source_stem = record.source_path.rsplit(".", 1)[0].casefold()
        if source_stem in exact_symbols:
            # A logical filename is an explicit user reference, just like a
            # routine name.  This is especially important for the parser
            # inventory contract, which is attached only to returned evidence.
            score += EXACT_CODE_SOURCE_BONUS
        score += _code_rank_bonus(
            record=record,
            query_identifier_terms=query_identifier_terms,
            technical_entities=technical_entities,
        )
        boosted.append(replace(result, score=score))
    return sorted(
        boosted,
        key=lambda result: (
            -result.score,
            result.payload["document_name"],
            result.payload["unit_index"],
            result.payload["unit_id"],
        ),
    )[:limit]
