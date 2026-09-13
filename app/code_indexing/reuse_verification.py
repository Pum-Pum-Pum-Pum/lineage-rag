"""Fail-closed verification of embedding reuse against an immutable base snapshot."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Iterable

from app.code_indexing.models import CodeIndexArtifact, CodeIndexRecord
from app.code_ingestion.snapshot_models import CodeSnapshotManifest


@dataclass(frozen=True)
class SourceReuseSummary:
    source_path: str
    change_state: str
    total_records: int
    cached_records: int
    embedded_records: int


def verify_embedding_reuse(
    *,
    snapshot: CodeSnapshotManifest,
    base_artifact: CodeIndexArtifact,
    embedded_artifact: CodeIndexArtifact,
) -> dict[str, object]:
    """Verify that a candidate embeds only source paths changed from its base.

    The test is intentionally path-aware after the vector-bearing artifact has
    been created.  It cannot inspect source text and emits only identities and
    count metadata suitable for an audit report.
    """

    if snapshot.diff.base_snapshot_id is None:
        raise ValueError("Reuse verification requires a snapshot with a base_snapshot_id")
    if base_artifact.status != "embedded" or embedded_artifact.status != "embedded":
        raise ValueError("Reuse verification requires embedded base and candidate artifacts")
    if base_artifact.snapshot_id != snapshot.diff.base_snapshot_id:
        raise ValueError("Base artifact does not match the snapshot base_snapshot_id")
    if embedded_artifact.snapshot_id != snapshot.snapshot_id:
        raise ValueError("Embedded artifact does not match the snapshot manifest")
    if base_artifact.embedding_model != embedded_artifact.embedding_model:
        raise ValueError("Base and candidate embedding models do not match")
    if base_artifact.embedding_input_version != embedded_artifact.embedding_input_version:
        raise ValueError("Base and candidate embedding input versions do not match")

    states = _source_change_states(snapshot)
    candidate_paths = {record.source_path for record in embedded_artifact.records}
    unknown_paths = sorted(candidate_paths.difference(states))
    if unknown_paths:
        raise ValueError(f"Embedded artifact contains paths absent from snapshot: {unknown_paths}")

    summaries = _summaries(embedded_artifact.records, states)
    unexpected_cache_miss_paths = sorted(
        item.source_path for item in summaries
        if item.change_state == "unchanged" and item.embedded_records > 0
    )
    if unexpected_cache_miss_paths:
        raise RuntimeError(
            "Unchanged source paths contain newly embedded records: "
            f"{unexpected_cache_miss_paths}"
        )

    base_cache_keys = {record.cache_key for record in base_artifact.records}
    invalid_cached_records = sorted(
        record.unit_id
        for record in embedded_artifact.records
        if record.embedding_status == "cached" and record.cache_key not in base_cache_keys
    )
    if invalid_cached_records:
        raise RuntimeError("Cached records are not present in the declared base artifact")

    return {
        "schema_version": "code_embedding_reuse_verification_v1",
        "status": "pass",
        "snapshot_id": snapshot.snapshot_id,
        "base_snapshot_id": base_artifact.snapshot_id,
        "base_artifact_identity_sha256": base_artifact.artifact_identity_sha256,
        "embedded_artifact_identity_sha256": embedded_artifact.artifact_identity_sha256,
        "embedding_model": embedded_artifact.embedding_model,
        "snapshot_delta": {
            "added": list(snapshot.diff.added),
            "modified": list(snapshot.diff.modified),
            "formatting_only_modified": list(snapshot.diff.formatting_only_modified),
            "deleted": list(snapshot.diff.deleted),
            "unchanged": list(snapshot.diff.unchanged),
        },
        "cached_records": sum(item.cached_records for item in summaries),
        "embedded_records": sum(item.embedded_records for item in summaries),
        "unexpected_cache_miss_paths": unexpected_cache_miss_paths,
        "source_paths": [item.__dict__ for item in summaries],
        "external_calls_performed": False,
    }


def _source_change_states(snapshot: CodeSnapshotManifest) -> dict[str, str]:
    states: dict[str, str] = {}
    for state, paths in (
        ("added", snapshot.diff.added),
        ("modified", snapshot.diff.modified),
        ("formatting_only_modified", snapshot.diff.formatting_only_modified),
        ("unchanged", snapshot.diff.unchanged),
    ):
        for path in paths:
            if path in states:
                raise ValueError(f"Snapshot path appears in multiple diff states: {path}")
            states[path] = state
    return states


def _summaries(
    records: Iterable[CodeIndexRecord], states: dict[str, str]
) -> tuple[SourceReuseSummary, ...]:
    grouped: dict[str, list[CodeIndexRecord]] = defaultdict(list)
    for record in records:
        grouped[record.source_path].append(record)
    return tuple(
        SourceReuseSummary(
            source_path=path,
            change_state=states[path],
            total_records=len(items),
            cached_records=sum(item.embedding_status == "cached" for item in items),
            embedded_records=sum(item.embedding_status == "embedded" for item in items),
        )
        for path, items in sorted(grouped.items(), key=lambda item: item[0].casefold())
    )
