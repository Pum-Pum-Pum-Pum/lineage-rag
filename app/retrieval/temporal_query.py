from __future__ import annotations

import re
from dataclasses import dataclass, replace
from typing import Sequence

from app.vectorstore.qdrant_search import QdrantSearchResult


CURRENT_STATE_PATTERN = re.compile(
    r"\b(current|currently|latest|now|today|as[- ]of)\b",
    re.IGNORECASE,
)
NON_TEMPORAL_CURRENT_PATTERN = re.compile(
    r"\bcurrent\s+(application|business|system|report)\s+date\b",
    re.IGNORECASE,
)
REFERENTIAL_PATTERN = re.compile(
    r"\b(it|its|this|that|these|those|same|there|them)\b",
    re.IGNORECASE,
)
HISTORICAL_CONTEXT_PATTERN = re.compile(
    r"\b(original|previous|previously|prior|before|historical|formerly)\b",
    re.IGNORECASE,
)
RELEASE_PATTERN = re.compile(r"\bR(\d+)\b", re.IGNORECASE)

CURRENT_STATE_RETRIEVAL_EXPANSION = (
    "Interpret current as the resulting production state after the latest "
    "deployed relevant release. Retrieve change details for retained, "
    "consolidated, renamed, and removed items; treat Existing Functionality "
    "as the pre-change baseline."
)


@dataclass(frozen=True)
class TemporalQueryPlan:
    original_query: str
    retrieval_query: str
    is_current_state: bool
    effective_release_label: str | None
    release_source: str | None
    referenced_release_labels: tuple[str, ...] = ()
    historical_context_requested: bool = False


def build_temporal_query_plan(
    query: str,
    *,
    requested_release_label: str | None = None,
    conversation_context: str | None = None,
) -> TemporalQueryPlan:
    """Build a deterministic retrieval plan without treating memory as proof."""

    original_query = query.strip()
    if not original_query:
        raise ValueError("query must not be blank")

    is_current_state = (
        CURRENT_STATE_PATTERN.search(original_query) is not None
        and NON_TEMPORAL_CURRENT_PATTERN.search(original_query) is None
    )
    historical_context_requested = (
        is_current_state
        and HISTORICAL_CONTEXT_PATTERN.search(original_query) is not None
    )
    effective_release_label: str | None = None
    release_source: str | None = None
    referenced_release_labels: tuple[str, ...] = ()

    if requested_release_label:
        effective_release_label = normalize_release_label(requested_release_label)
        release_source = "request_filter"
    elif not is_current_state:
        query_releases = extract_release_labels(original_query)
        if len(query_releases) == 1:
            effective_release_label = query_releases[0]
            release_source = "query"
        elif conversation_context and REFERENTIAL_PATTERN.search(original_query):
            context_releases = extract_release_labels(conversation_context)
            if context_releases:
                effective_release_label = latest_release_label(context_releases)
                release_source = "conversation_context"
    elif conversation_context and REFERENTIAL_PATTERN.search(original_query):
        context_releases = extract_release_labels(conversation_context)
        if context_releases:
            effective_release_label = latest_release_label(context_releases)
            release_source = "conversation_context"

    if is_current_state and effective_release_label is None:
        referenced_release_labels = tuple(extract_release_labels(original_query))

    retrieval_parts = [original_query]
    if effective_release_label and release_source == "conversation_context":
        retrieval_parts.append(
            f"Resolved conversation release: {effective_release_label}."
        )
    if is_current_state:
        retrieval_parts.append(CURRENT_STATE_RETRIEVAL_EXPANSION)

    return TemporalQueryPlan(
        original_query=original_query,
        retrieval_query="\n".join(retrieval_parts),
        is_current_state=is_current_state,
        effective_release_label=effective_release_label,
        release_source=release_source,
        referenced_release_labels=referenced_release_labels,
        historical_context_requested=historical_context_requested,
    )


def scope_results_to_temporal_plan(
    results: Sequence[QdrantSearchResult],
    plan: TemporalQueryPlan,
    *,
    limit: int,
) -> tuple[list[QdrantSearchResult], TemporalQueryPlan]:
    """Apply latest-deployed-release semantics after broad candidate retrieval."""

    if limit <= 0:
        raise ValueError("limit must be greater than 0")

    effective_release = plan.effective_release_label
    release_source = plan.release_source

    if plan.is_current_state and effective_release is None:
        candidate_releases = _relevance_bounded_release_labels(results)
        if candidate_releases:
            effective_release = latest_release_label(candidate_releases)
            release_source = "retrieved_candidates"

    if effective_release is None:
        return list(results[:limit]), plan

    permitted_releases = {effective_release, *plan.referenced_release_labels}
    if (
        plan.is_current_state
        and plan.historical_context_requested
        and not plan.referenced_release_labels
    ):
        permitted_releases.update(_relevance_bounded_release_labels(results))
    scoped = [
        result
        for result in results
        if _normalized_payload_release(result) in permitted_releases
    ]
    updated_plan = replace(
        plan,
        effective_release_label=effective_release,
        release_source=release_source,
    )
    return scoped[:limit], updated_plan


def extract_release_labels(text: str) -> list[str]:
    labels: list[str] = []
    for match in RELEASE_PATTERN.finditer(text):
        label = f"R{int(match.group(1))}"
        if label not in labels:
            labels.append(label)
    return labels


def latest_release_label(labels: Sequence[str]) -> str:
    normalized = [normalize_release_label(label) for label in labels]
    if not normalized:
        raise ValueError("at least one release label is required")
    return max(normalized, key=release_number)


def normalize_release_label(label: str) -> str:
    match = RELEASE_PATTERN.fullmatch(label.strip())
    if match is None:
        raise ValueError(f"invalid release label: {label}")
    return f"R{int(match.group(1))}"


def release_number(label: str) -> int:
    normalized = normalize_release_label(label)
    return int(normalized[1:])


def _normalized_payload_release(
    result: QdrantSearchResult,
) -> str | None:
    raw_label = str(result.payload.get("release_label", "")).strip()
    if not raw_label:
        return None
    try:
        return normalize_release_label(raw_label)
    except ValueError:
        return None


def _relevance_bounded_release_labels(
    results: Sequence[QdrantSearchResult],
) -> list[str]:
    """Keep releases with repeated rank support or a top-three result.

    A numerically newer release must not become the current functional state
    merely because one weak, unrelated unit entered a broad candidate set.
    Rank support is intentionally computed after fusion, so dense and lexical
    evidence contribute through the configured weighted-RRF contract.
    """

    support_by_release: dict[str, float] = {}
    best_rank_by_release: dict[str, int] = {}
    for rank, result in enumerate(results, start=1):
        release_label = _normalized_payload_release(result)
        if release_label is None:
            continue
        support_by_release[release_label] = (
            support_by_release.get(release_label, 0.0) + (1.0 / rank)
        )
        best_rank_by_release.setdefault(release_label, rank)

    if not support_by_release:
        return []
    strongest_support = max(support_by_release.values())
    return [
        release_label
        for release_label, support in support_by_release.items()
        if best_rank_by_release[release_label] <= 3
        or support >= strongest_support * 0.25
    ]
