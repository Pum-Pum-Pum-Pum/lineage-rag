from __future__ import annotations

import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any


TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]+(?:[-_][A-Za-z0-9]+)*")

LEXICAL_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "does",
    "for",
    "from",
    "how",
    "in",
    "is",
    "of",
    "on",
    "or",
    "the",
    "to",
    "what",
    "which",
    "with",
}


# This applies only to an embedded Validation sheet where the user's question
# explicitly asks about validation and includes a stable identifier such as
# ``Re-Query`` or ``B-01``. It is deliberately not a corpus-wide weight change.
VALIDATION_SHEET_IDENTIFIER_AFFINITY_BONUS = 40.0


@dataclass(frozen=True)
class LexicalSearchDocument:
    document_name: str
    unit_id: str
    unit_index: int
    source_kind: str
    document_family: str
    release_label: str
    text: str
    document_id: str = ""
    retrieval_text: str = ""
    parent_unit_id: str | None = None
    document_lineage_key: str = ""
    document_revision: str | None = None
    attachment_path: str | None = None
    attachment_sha256: str | None = None
    sheet_name: str | None = None
    sheet_role: str | None = None
    source_range: str | None = None


@dataclass(frozen=True)
class LexicalSearchResult:
    point_id: str
    score: float
    payload: dict[str, Any]


def tokenize(text: str) -> list[str]:
    """Tokenize text for the dependency-free lexical retrieval baseline.

    The tokenizer intentionally preserves simple enterprise identifiers such as
    `B-01`, `T-1`, and underscore-connected tokens because those exact strings
    often matter in functional specification retrieval.
    """

    return [match.group(0).lower() for match in TOKEN_PATTERN.finditer(text)]


def build_query_terms(query_text: str) -> list[str]:
    """Build searchable query terms while dropping common stopwords."""

    tokens = tokenize(query_text.strip())
    if not tokens:
        raise ValueError("Query text must contain at least one searchable token")

    terms = [token for token in tokens if token not in LEXICAL_STOPWORDS]
    return terms or tokens


def load_retrieval_ready_documents(
    artifact_directory: str | Path,
) -> list[LexicalSearchDocument]:
    """Load units from persisted `.retrieval_ready.json` artifacts."""

    directory = Path(artifact_directory)
    if not directory.exists():
        return []

    documents: list[LexicalSearchDocument] = []

    for artifact_file in sorted(directory.glob("*.retrieval_ready.json")):
        payload = json.loads(artifact_file.read_text(encoding="utf-8"))
        document_name = str(payload.get("document_name", artifact_file.name))
        fallback_document_id = str(payload.get("document_id", ""))
        fallback_family = str(payload.get("document_family", ""))
        fallback_release = str(payload.get("release_label", ""))

        for unit in payload.get("units", []):
            documents.append(
                LexicalSearchDocument(
                    document_name=document_name,
                    unit_id=str(unit["unit_id"]),
                    unit_index=int(unit["unit_index"]),
                    source_kind=str(unit["source_kind"]),
                    document_family=str(unit.get("document_family", fallback_family)),
                    release_label=str(unit.get("release_label", fallback_release)),
                    text=str(unit.get("text", "")),
                    document_id=str(unit.get("document_id") or fallback_document_id),
                    retrieval_text=str(unit.get("retrieval_text") or unit.get("text", "")),
                    parent_unit_id=unit.get("parent_unit_id"),
                    document_lineage_key=str(unit.get("document_lineage_key", "")),
                    document_revision=unit.get("document_revision"),
                    attachment_path=unit.get("attachment_path"),
                    attachment_sha256=unit.get("attachment_sha256"),
                    sheet_name=unit.get("sheet_name"),
                    sheet_role=unit.get("sheet_role"),
                    source_range=unit.get("source_range"),
                )
            )

    return documents


def search_lexical_documents(
    documents: list[LexicalSearchDocument],
    query_text: str,
    limit: int = 5,
    document_family: str | None = None,
    release_label: str | None = None,
    source_kind: str | None = None,
) -> list[LexicalSearchResult]:
    """Run dependency-free lexical search over retrieval-ready documents."""

    if limit <= 0:
        raise ValueError("Search limit must be greater than 0")

    query_terms = build_query_terms(query_text)
    candidates = _filter_documents(
        documents,
        document_family=document_family,
        release_label=release_label,
        source_kind=source_kind,
    )
    if not candidates:
        return []

    idf_by_term = _build_query_term_idf(query_terms, candidates)
    scored_results: list[LexicalSearchResult] = []

    for document in candidates:
        retrieval_text = document.retrieval_text or document.text
        document_tokens = tokenize(retrieval_text)
        score, matched_terms = _score_document(
            query_terms=query_terms,
            document_tokens=document_tokens,
            idf_by_term=idf_by_term,
        )
        affinity_bonus, affinity_terms = _validation_sheet_identifier_affinity(
            query_terms=query_terms,
            document_tokens=document_tokens,
            document=document,
        )
        score += affinity_bonus
        if score <= 0:
            continue

        scored_results.append(
            LexicalSearchResult(
                point_id=document.unit_id,
                score=score,
                payload={
                    "document_name": document.document_name,
                    "document_id": document.document_id,
                    "unit_id": document.unit_id,
                    "unit_index": document.unit_index,
                    "source_kind": document.source_kind,
                    "document_family": document.document_family,
                    "release_label": document.release_label,
                    "text": document.text,
                    "retrieval_text": retrieval_text,
                    "parent_unit_id": document.parent_unit_id,
                    "document_lineage_key": document.document_lineage_key,
                    "document_revision": document.document_revision,
                    "attachment_path": document.attachment_path,
                    "attachment_sha256": document.attachment_sha256,
                    "sheet_name": document.sheet_name,
                    "sheet_role": document.sheet_role,
                    "source_range": document.source_range,
                    "retrieval_method": "lexical",
                    "matched_query_terms": matched_terms,
                    "validation_sheet_identifier_affinity_terms": affinity_terms,
                    "validation_sheet_identifier_affinity_bonus": affinity_bonus,
                },
            )
        )

    return sorted(
        scored_results,
        key=lambda result: (
            -result.score,
            result.payload["document_name"],
            result.payload["unit_index"],
            result.payload["unit_id"],
        ),
    )[:limit]


def search_lexical_artifacts(
    artifact_directory: str | Path,
    query_text: str,
    limit: int = 5,
    document_family: str | None = None,
    release_label: str | None = None,
    source_kind: str | None = None,
) -> list[LexicalSearchResult]:
    """Load retrieval-ready artifacts from disk and run lexical search."""

    documents = load_retrieval_ready_documents(artifact_directory)
    return search_lexical_documents(
        documents=documents,
        query_text=query_text,
        limit=limit,
        document_family=document_family,
        release_label=release_label,
        source_kind=source_kind,
    )


def _filter_documents(
    documents: list[LexicalSearchDocument],
    document_family: str | None = None,
    release_label: str | None = None,
    source_kind: str | None = None,
) -> list[LexicalSearchDocument]:
    return [
        document
        for document in documents
        if (document_family is None or document.document_family == document_family)
        and (release_label is None or document.release_label == release_label)
        and (source_kind is None or document.source_kind == source_kind)
    ]


def _build_query_term_idf(
    query_terms: list[str],
    documents: list[LexicalSearchDocument],
) -> dict[str, float]:
    document_count = len(documents)
    document_frequencies: Counter[str] = Counter()

    for document in documents:
        document_token_set = set(tokenize(document.text))
        for term in set(query_terms):
            if term in document_token_set:
                document_frequencies[term] += 1

    return {
        term: math.log((document_count + 1) / (document_frequencies[term] + 1)) + 1.0
        for term in set(query_terms)
    }


def _score_document(
    query_terms: list[str],
    document_tokens: list[str],
    idf_by_term: dict[str, float],
) -> tuple[float, list[str]]:
    token_counts = Counter(document_tokens)
    unique_query_terms = set(query_terms)
    matched_terms = sorted(term for term in unique_query_terms if token_counts[term] > 0)

    if not matched_terms:
        return 0.0, []

    idf_score = sum(idf_by_term[term] for term in matched_terms)
    term_frequency_bonus = sum(min(token_counts[term] - 1, 2) * 0.10 for term in matched_terms)
    coverage_multiplier = 1.0 + (len(matched_terms) / len(unique_query_terms))

    return (idf_score + term_frequency_bonus) * coverage_multiplier, matched_terms


def _validation_sheet_identifier_affinity(
    *,
    query_terms: list[str],
    document_tokens: list[str],
    document: LexicalSearchDocument,
) -> tuple[float, list[str]]:
    """Return a bounded validation-sheet boost for explicit query identifiers.

    Generic words such as ``validation`` and ``service`` occur widely in FDDs.
    This narrow signal lets an exact enterprise identifier in a validation-focused
    query keep its matching workbook row inside the normal candidate/evidence
    budget. It never applies to paragraphs, tables, request/response sheets, or
    queries that lack a stable identifier.
    """

    if document.source_kind != "embedded_workbook":
        return 0.0, []
    if (document.sheet_role or "").casefold() != "validation":
        return 0.0, []
    if not any(term.rstrip("s") == "validation" for term in query_terms):
        return 0.0, []

    document_token_set = set(document_tokens)
    identifier_terms = [
        term
        for term in query_terms
        if ("-" in term or "_" in term or any(character.isdigit() for character in term))
        and term in document_token_set
    ]
    if not identifier_terms:
        return 0.0, []
    return VALIDATION_SHEET_IDENTIFIER_AFFINITY_BONUS, sorted(set(identifier_terms))
