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
    effective_release_label: str | None = None
    release_source: str | None = None

    if requested_release_label:
        effective_release_label = normalize_release_label(requested_release_label)
        release_source = "request_filter"
    else:
        query_releases = extract_release_labels(original_query)
        if len(query_releases) == 1:
            effective_release_label = query_releases[0]
            release_source = "query"
        elif conversation_context and REFERENTIAL_PATTERN.search(original_query):
            context_releases = extract_release_labels(conversation_context)
            if context_releases:
                effective_release_label = latest_release_label(context_releases)
                release_source = "conversation_context"

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
        candidate_releases = [
            str(result.payload.get("release_label", ""))
            for result in results
            if str(result.payload.get("release_label", "")).strip()
        ]
        if candidate_releases:
            effective_release = latest_release_label(candidate_releases)
            release_source = "retrieved_candidates"

    if effective_release is None:
        return list(results[:limit]), plan

    scoped = [
        result
        for result in results
        if _normalized_payload_release(result) == effective_release
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
