from __future__ import annotations

import hashlib
import json
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from app.ingestion.docx_loader import DiscoveredDocxFile
from app.ingestion.filename_parser import ParsedDocumentName, parse_document_filename


ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_POLICY_PATH = ROOT_DIR / "config" / "fdd_document_lineage.toml"


@dataclass(frozen=True)
class FddDocumentLineagePolicy:
    schema_version: str
    full_replacement_keys: frozenset[str]
    sha256: str


@dataclass(frozen=True)
class FddDocumentSelection:
    current_sources: tuple[DiscoveredDocxFile, ...]
    superseded_sources: tuple[DiscoveredDocxFile, ...]
    policy_sha256: str


def load_fdd_document_lineage_policy(
    path: str | Path = DEFAULT_POLICY_PATH,
) -> FddDocumentLineagePolicy:
    raw = Path(path).read_bytes()
    payload = tomllib.loads(raw.decode("utf-8"))
    keys = payload.get("full_replacement", {}).get("document_lineage_keys", [])
    if not isinstance(keys, list) or not all(isinstance(item, str) and item.strip() for item in keys):
        raise ValueError("full_replacement.document_lineage_keys must contain non-empty strings")
    return FddDocumentLineagePolicy(
        schema_version=str(payload.get("schema_version", "")),
        full_replacement_keys=frozenset(item.casefold() for item in keys),
        sha256=hashlib.sha256(_canonical_policy_bytes(payload)).hexdigest(),
    )


def select_current_document_sources(
    sources: Sequence[DiscoveredDocxFile],
    *,
    policy: FddDocumentLineagePolicy | None = None,
) -> FddDocumentSelection:
    """Select current sources without deleting superseded archive documents.

    Only explicitly configured full-replacement streams are collapsed by
    filename revision. Unconfigured documents remain independent evidence.
    """

    resolved_policy = policy or load_fdd_document_lineage_policy()
    parsed = [(source, parse_document_filename(source.file_path)) for source in sources]
    grouped: dict[str, list[tuple[DiscoveredDocxFile, ParsedDocumentName]]] = {}
    independent: list[DiscoveredDocxFile] = []
    for source, name in parsed:
        if (
            name.document_revision is None
            or name.document_lineage_key.casefold() not in resolved_policy.full_replacement_keys
        ):
            independent.append(source)
            continue
        grouped.setdefault(name.document_lineage_key.casefold(), []).append((source, name))

    current = list(independent)
    superseded: list[DiscoveredDocxFile] = []
    for items in grouped.values():
        latest = max(items, key=lambda item: item[1].document_revision_sort_key or ())
        current.append(latest[0])
        superseded.extend(source for source, _ in items if source != latest[0])

    return FddDocumentSelection(
        current_sources=tuple(sorted(current, key=lambda item: item.file_name.casefold())),
        superseded_sources=tuple(sorted(superseded, key=lambda item: item.file_name.casefold())),
        policy_sha256=resolved_policy.sha256,
    )


def _canonical_policy_bytes(payload: dict) -> bytes:
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
