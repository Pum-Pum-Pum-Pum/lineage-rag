from __future__ import annotations

import hashlib
import json
import re
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping


ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_INGESTION_POLICY_PATH = ROOT_DIR / "config" / "ingestion_sources.toml"
POLICY_SCHEMA_VERSION = "ingestion_source_policy_v1"
SUPPORTED_HANDLERS = {
    "fdd": frozenset({"docx"}),
    "code": frozenset({"plsql", "ddl"}),
}
_EXTENSION_PATTERN = re.compile(r"^\.[a-z0-9][a-z0-9_+-]{0,15}$")


@dataclass(frozen=True)
class ExtensionRule:
    extension: str
    handler: str


@dataclass(frozen=True)
class IngestionSourcePolicy:
    schema_version: str
    fdd_extensions: tuple[ExtensionRule, ...]
    code_extensions: tuple[ExtensionRule, ...]
    policy_sha256: str
    policy_path: Path

    def extension_map(self, lane: str) -> dict[str, str]:
        if lane == "fdd":
            rules = self.fdd_extensions
        elif lane == "code":
            rules = self.code_extensions
        else:
            raise ValueError(f"Unknown ingestion lane: {lane!r}")
        return {rule.extension: rule.handler for rule in rules}

    def extensions_for(self, lane: str, *, handler: str) -> frozenset[str]:
        return frozenset(
            extension
            for extension, configured_handler in self.extension_map(lane).items()
            if configured_handler == handler
        )


def load_ingestion_source_policy(
    path: str | Path = DEFAULT_INGESTION_POLICY_PATH,
) -> IngestionSourcePolicy:
    policy_path = Path(path)
    if not policy_path.is_file():
        raise FileNotFoundError(f"Ingestion source policy not found: {policy_path}")
    with policy_path.open("rb") as handle:
        payload = tomllib.load(handle)

    allowed_top_level = {"schema_version", "fdd", "code"}
    unknown_top_level = set(payload) - allowed_top_level
    if unknown_top_level:
        raise ValueError(
            "Unknown ingestion policy fields: " + ", ".join(sorted(unknown_top_level))
        )
    schema_version = payload.get("schema_version")
    if schema_version != POLICY_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported ingestion policy schema_version: {schema_version!r}"
        )

    fdd_rules = _parse_lane(payload, lane="fdd")
    code_rules = _parse_lane(payload, lane="code")
    canonical_payload = {
        "schema_version": schema_version,
        "fdd": {rule.extension: rule.handler for rule in fdd_rules},
        "code": {rule.extension: rule.handler for rule in code_rules},
    }
    policy_sha256 = hashlib.sha256(
        json.dumps(
            canonical_payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    return IngestionSourcePolicy(
        schema_version=schema_version,
        fdd_extensions=fdd_rules,
        code_extensions=code_rules,
        policy_sha256=policy_sha256,
        policy_path=policy_path.resolve(),
    )


def _parse_lane(payload: Mapping[str, object], *, lane: str) -> tuple[ExtensionRule, ...]:
    lane_payload = payload.get(lane)
    if not isinstance(lane_payload, dict) or set(lane_payload) != {"extensions"}:
        raise ValueError(f"Policy lane {lane!r} must contain only an extensions table")
    extensions = lane_payload.get("extensions")
    if not isinstance(extensions, dict) or not extensions:
        raise ValueError(f"Policy lane {lane!r} must enable at least one extension")

    rules: list[ExtensionRule] = []
    seen: set[str] = set()
    for raw_extension, raw_handler in extensions.items():
        if not isinstance(raw_extension, str) or not isinstance(raw_handler, str):
            raise ValueError(f"Policy lane {lane!r} extensions and handlers must be strings")
        extension = raw_extension.strip().lower()
        handler = raw_handler.strip().lower()
        if not _EXTENSION_PATTERN.fullmatch(extension):
            raise ValueError(f"Invalid configured extension for {lane}: {raw_extension!r}")
        if extension in seen:
            raise ValueError(f"Duplicate configured extension for {lane}: {extension}")
        if handler not in SUPPORTED_HANDLERS[lane]:
            raise ValueError(
                f"Configured handler {handler!r} is not implemented for lane {lane!r}"
            )
        seen.add(extension)
        rules.append(ExtensionRule(extension=extension, handler=handler))
    return tuple(sorted(rules, key=lambda rule: rule.extension))

