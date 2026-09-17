"""Immutable, non-secret contracts for a coordinated knowledge release."""
from __future__ import annotations

import hashlib
import json
from pathlib import PurePosixPath
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Frozen(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


def _identity(value: dict) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


class SourceIdentity(Frozen):
    path: str = Field(min_length=1)
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    size_bytes: int = Field(ge=0)
    logical_key: str = Field(min_length=1)
    revision: str | None = None

    @model_validator(mode="after")
    def safe_path(self):
        normalized = self.path.replace("\\", "/")
        if normalized.startswith("/") or ".." in PurePosixPath(normalized).parts:
            raise ValueError("Source path must be a safe relative path")
        return self


class ChangeSet(Frozen):
    added: tuple[str, ...] = ()
    modified: tuple[str, ...] = ()
    unchanged: tuple[str, ...] = ()
    moved: tuple[tuple[str, str], ...] = ()
    removed: tuple[str, ...] = ()
    superseded: tuple[str, ...] = ()
    withdrawn: tuple[str, ...] = ()

    @model_validator(mode="after")
    def no_duplicate_membership(self):
        categories = (self.added, self.modified, self.unchanged, self.removed, self.superseded, self.withdrawn)
        members = [item.casefold() for group in categories for item in group]
        if len(members) != len(set(members)):
            raise ValueError("A source cannot belong to multiple change classes")
        return self


class FddUpdatePlan(Frozen):
    schema_version: Literal["knowledge_fdd_update_plan_v1"] = "knowledge_fdd_update_plan_v1"
    base_generation: str
    target_generation: str
    source_policy_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    baseline_sources: tuple[SourceIdentity, ...]
    update_sources: tuple[SourceIdentity, ...]
    target_sources: tuple[SourceIdentity, ...]
    changes: ChangeSet
    identity_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def identity_matches(self):
        if _identity(self.model_dump(mode="json", exclude={"identity_sha256"})) != self.identity_sha256:
            raise ValueError("FDD update-plan identity mismatch")
        return self


class KnowledgeReleaseManifest(Frozen):
    """The one combination that an MCP process is allowed to serve."""
    schema_version: Literal["knowledge_release_manifest_v1"] = "knowledge_release_manifest_v1"
    run_id: str
    mode: Literal["fdd", "code", "both", "review"]
    fdd_generation: str
    fdd_collection: str
    fdd_processed_directory: str
    fdd_stage_directory: str | None = None
    fdd_stage_manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    code_snapshot_id: str
    code_collection: str
    code_artifact_path: str
    code_artifact_identity_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    code_analysis_directory: str
    lineage_path: str
    lineage_identity_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    deferred_lineage_path: str | None = None
    review_identity_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    evaluation_report_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    runtime_files_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    manifest_identity_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def identity_matches(self):
        if _identity(self.model_dump(mode="json", exclude={"manifest_identity_sha256"})) != self.manifest_identity_sha256:
            raise ValueError("Knowledge release-manifest identity mismatch")
        return self


def make_fdd_plan(**values) -> FddUpdatePlan:
    provisional = {"schema_version": "knowledge_fdd_update_plan_v1", **values}
    # Callers naturally pass frozen Pydantic source/change contracts. Convert
    # them before hashing so the identity is deterministic and serializable.
    serializable = {
        key: (value.model_dump(mode="json") if isinstance(value, BaseModel) else
              [item.model_dump(mode="json") if isinstance(item, BaseModel) else item for item in value]
              if isinstance(value, (list, tuple)) else value)
        for key, value in provisional.items()
    }
    return FddUpdatePlan(**provisional, identity_sha256=_identity(serializable))


def make_release_manifest(**values) -> KnowledgeReleaseManifest:
    provisional = {"schema_version": "knowledge_release_manifest_v1", **values}
    return KnowledgeReleaseManifest(**provisional, manifest_identity_sha256=_identity(provisional))
