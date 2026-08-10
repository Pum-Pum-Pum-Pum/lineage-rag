from __future__ import annotations

from datetime import datetime
from pathlib import PurePosixPath
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


REQUEST_SCHEMA_VERSION = "code_snapshot_request_v1"
MANIFEST_SCHEMA_VERSION = "code_snapshot_manifest_v1"


def normalize_relative_path(value: str) -> str:
    """Return a stable POSIX relative path or reject an unsafe path."""

    normalized = value.strip().replace("\\", "/")
    path = PurePosixPath(normalized)
    if (
        not normalized
        or path.is_absolute()
        or any(part in {"", ".", ".."} for part in path.parts)
        or ":" in path.parts[0]
    ):
        raise ValueError(f"Expected a safe relative path, got {value!r}")
    return path.as_posix()


class FrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)


class CompilerContext(FrozenModel):
    oracle_version: str | None = None
    plsql_ccflags: str | None = None


class SnapshotRequest(FrozenModel):
    schema_version: Literal[REQUEST_SCHEMA_VERSION] = REQUEST_SCHEMA_VERSION
    module_set: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
    svn_revision: str = Field(pattern=r"^[1-9][0-9]*$")
    application_build: str = Field(min_length=1, max_length=128)
    reviewer: str = Field(min_length=1, max_length=256)
    base_snapshot_id: str | None = Field(
        default=None,
        pattern=r"^[A-Za-z0-9][A-Za-z0-9._-]{0,255}$",
    )
    expected_changed_packages: tuple[str, ...] = ()
    compiler_context: CompilerContext = Field(default_factory=CompilerContext)

    @field_validator("expected_changed_packages")
    @classmethod
    def validate_expected_changed_packages(cls, values: tuple[str, ...]) -> tuple[str, ...]:
        normalized = tuple(normalize_relative_path(value) for value in values)
        if len({value.casefold() for value in normalized}) != len(normalized):
            raise ValueError("expected_changed_packages contains duplicate paths")
        return normalized


class ValidationIssue(FrozenModel):
    severity: Literal["warning", "error"]
    code: str
    path: str | None = None
    message: str


class CodeFileManifestEntry(FrozenModel):
    path: str
    extension: str
    source_handler: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    normalized_text_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    size_bytes: int = Field(ge=0)
    encoding: str
    line_count: int = Field(ge=0)
    is_large_file: bool = False
    warnings: tuple[str, ...] = ()

    @field_validator("path")
    @classmethod
    def validate_path(cls, value: str) -> str:
        return normalize_relative_path(value)


class IntakeValidationReport(FrozenModel):
    schema_version: Literal["code_intake_validation_v1"] = "code_intake_validation_v1"
    source_directory: str
    ingestion_policy_schema_version: str
    ingestion_policy_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    files: tuple[CodeFileManifestEntry, ...]
    warnings: tuple[ValidationIssue, ...] = ()


class ExactRename(FrozenModel):
    old_path: str
    new_path: str
    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


class SnapshotDiff(FrozenModel):
    schema_version: Literal["code_snapshot_diff_v1"] = "code_snapshot_diff_v1"
    base_snapshot_id: str | None = None
    ingestion_policy_changed_from_base: bool = False
    added: tuple[str, ...] = ()
    modified: tuple[str, ...] = ()
    deleted: tuple[str, ...] = ()
    unchanged: tuple[str, ...] = ()
    formatting_only_modified: tuple[str, ...] = ()
    exact_renames: tuple[ExactRename, ...] = ()
    ambiguous_rename_hashes: tuple[str, ...] = ()
    expected_changed_packages: tuple[str, ...] = ()
    missing_expected_changes: tuple[str, ...] = ()
    unexpected_changed_files: tuple[str, ...] = ()


class CodeSnapshotManifest(FrozenModel):
    schema_version: Literal[MANIFEST_SCHEMA_VERSION] = MANIFEST_SCHEMA_VERSION
    snapshot_id: str
    snapshot_content_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    created_at_utc: datetime
    immutable: Literal[True] = True
    ingestion_policy_schema_version: str
    ingestion_policy_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    request: SnapshotRequest
    source_directory_name: Literal["source"] = "source"
    files: tuple[CodeFileManifestEntry, ...]
    diff: SnapshotDiff
