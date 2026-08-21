from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal, Sequence

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.code_indexing.models import CodeIndexArtifact
from app.code_ingestion.code_analysis_models import CodeStaticAnalysisArtifact


class FrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class FddCodeTarget(FrozenModel):
    module_id: str
    path: str
    qualified_name: str | None = None
    symbol_kind: Literal["procedure", "function"] | None = None
    overload_discriminator_hash: str | None = Field(
        default=None, pattern=r"^[0-9a-f]{64}$"
    )
    selector_scope: Literal["file", "all_overloads", "overload"]
    rationale: str = Field(min_length=10)

    @model_validator(mode="after")
    def validate_selector(self) -> "FddCodeTarget":
        if self.selector_scope == "file":
            if any(
                value is not None
                for value in (
                    self.qualified_name,
                    self.symbol_kind,
                    self.overload_discriminator_hash,
                )
            ):
                raise ValueError("File selectors cannot contain symbol fields")
        elif not self.qualified_name or not self.symbol_kind:
            raise ValueError("Symbol selectors require qualified_name and symbol_kind")
        if self.selector_scope == "overload" and not self.overload_discriminator_hash:
            raise ValueError("Overload selectors require overload_discriminator_hash")
        if self.selector_scope == "all_overloads" and self.overload_discriminator_hash:
            raise ValueError("All-overloads selectors must omit overload hash")
        return self


class FddCodeMapping(FrozenModel):
    mapping_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    fdd_document_id: str
    fdd_release_label: str = Field(pattern=r"^R\d+$")
    code_snapshot_id: str
    targets: tuple[FddCodeTarget, ...]
    mapping_status: Literal["candidate", "reviewed"] = "candidate"
    rationale: str = Field(min_length=10)
    reviewer: str | None = None

    @model_validator(mode="after")
    def validate_review(self) -> "FddCodeMapping":
        if not self.targets:
            raise ValueError("FDD/code mappings require at least one target")
        if self.mapping_status == "reviewed" and not (self.reviewer or "").strip():
            raise ValueError("Reviewed mappings require a reviewer")
        if self.mapping_status == "candidate" and self.reviewer is not None:
            raise ValueError("Candidate mappings must not claim a reviewer")
        return self


class FddCodeLineageArtifact(FrozenModel):
    schema_version: Literal["fdd_code_lineage_v1"] = "fdd_code_lineage_v1"
    status: Literal["candidate", "reviewed"]
    fdd_generation: str
    code_snapshot_id: str
    code_artifact_identity_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    mappings: tuple[FddCodeMapping, ...]
    source_candidate_artifact_identity_sha256: str | None = Field(
        default=None, pattern=r"^[0-9a-f]{64}$"
    )
    review_packet_sha256: str | None = Field(
        default=None, pattern=r"^[0-9a-f]{64}$"
    )
    reviewer: str | None = None
    artifact_identity_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_status(self) -> "FddCodeLineageArtifact":
        if len({item.mapping_id for item in self.mappings}) != len(self.mappings):
            raise ValueError("Mapping IDs must be unique")
        expected = "reviewed" if self.mappings and all(
            item.mapping_status == "reviewed" for item in self.mappings
        ) else "candidate"
        if self.status != expected:
            raise ValueError("Artifact status does not match mapping review states")
        review_fields = (
            self.source_candidate_artifact_identity_sha256,
            self.review_packet_sha256,
            self.reviewer,
        )
        if self.status == "reviewed" and not all(review_fields):
            raise ValueError("Reviewed lineage artifacts require candidate, packet, and reviewer bindings")
        if self.status == "candidate" and any(review_fields):
            raise ValueError("Candidate lineage artifacts must not claim review bindings")
        return self


def create_mapping(
    *,
    fdd_document_id: str,
    fdd_release_label: str,
    code_snapshot_id: str,
    targets: Sequence[FddCodeTarget],
    rationale: str,
    mapping_status: Literal["candidate", "reviewed"] = "candidate",
    reviewer: str | None = None,
) -> FddCodeMapping:
    values = {
        "fdd_document_id": fdd_document_id,
        "fdd_release_label": fdd_release_label,
        "code_snapshot_id": code_snapshot_id,
        "targets": [item.model_dump(mode="json") for item in targets],
        "mapping_status": mapping_status,
        "rationale": rationale,
        "reviewer": reviewer,
    }
    mapping_id = _identity(values)
    return FddCodeMapping(mapping_id=mapping_id, **values)


def build_lineage_artifact(
    *,
    fdd_generation: str,
    code_artifact: CodeIndexArtifact,
    mappings: Sequence[FddCodeMapping],
    source_candidate_artifact_identity_sha256: str | None = None,
    review_packet_sha256: str | None = None,
    reviewer: str | None = None,
) -> FddCodeLineageArtifact:
    ordered = tuple(sorted(mappings, key=lambda item: item.mapping_id))
    status = "reviewed" if ordered and all(
        item.mapping_status == "reviewed" for item in ordered
    ) else "candidate"
    values = {
        "status": status,
        "fdd_generation": fdd_generation,
        "code_snapshot_id": code_artifact.snapshot_id,
        "code_artifact_identity_sha256": code_artifact.artifact_identity_sha256,
        "mappings": [item.model_dump(mode="json") for item in ordered],
        "source_candidate_artifact_identity_sha256": source_candidate_artifact_identity_sha256,
        "review_packet_sha256": review_packet_sha256,
        "reviewer": reviewer,
    }
    return FddCodeLineageArtifact(
        **values,
        artifact_identity_sha256=_identity(values),
    )


def validate_lineage_artifact(
    artifact: FddCodeLineageArtifact,
    *,
    fdd_document_ids: set[str],
    code_artifact: CodeIndexArtifact,
    analysis_directory: Path,
) -> dict[str, int | str]:
    if artifact.code_snapshot_id != code_artifact.snapshot_id:
        raise ValueError("Lineage artifact code snapshot does not match code artifact")
    if artifact.code_artifact_identity_sha256 != code_artifact.artifact_identity_sha256:
        raise ValueError("Lineage artifact is bound to a different code artifact")
    paths = {record.source_path for record in code_artifact.records}
    analyses = _load_analysis(analysis_directory)
    for mapping in artifact.mappings:
        if mapping.fdd_document_id not in fdd_document_ids:
            raise ValueError(f"Unknown FDD document ID: {mapping.fdd_document_id}")
        if mapping.code_snapshot_id != code_artifact.snapshot_id:
            raise ValueError(f"Mapping snapshot mismatch: {mapping.mapping_id}")
        for target in mapping.targets:
            if target.module_id != code_artifact.module_id:
                raise ValueError(f"Unknown code module: {target.module_id}")
            if target.path not in paths:
                raise ValueError(f"Unknown code path: {target.path}")
            if target.selector_scope != "file":
                symbols = analyses.get(target.path, ())
                matches = [
                    symbol
                    for symbol in symbols
                    if symbol.canonical_qualified_name == target.qualified_name
                    and symbol.symbol_kind == target.symbol_kind
                ]
                if target.selector_scope == "overload":
                    matches = [
                        symbol
                        for symbol in matches
                        if symbol.overload_discriminator_hash
                        == target.overload_discriminator_hash
                    ]
                if not matches:
                    raise ValueError(
                        f"Symbol selector resolved to no code symbol: {target.qualified_name}"
                    )
    return {
        "status": artifact.status,
        "mappings": len(artifact.mappings),
        "targets": sum(len(item.targets) for item in artifact.mappings),
    }


def resolve_target_unit_ids(
    artifact: FddCodeLineageArtifact,
    *,
    known_fdd_document_ids: set[str],
    selected_fdd_document_ids: set[str],
    code_artifact: CodeIndexArtifact,
    analysis_directory: Path,
) -> tuple[set[str], tuple[str, ...]]:
    validate_lineage_artifact(
        artifact,
        fdd_document_ids=known_fdd_document_ids,
        code_artifact=code_artifact,
        analysis_directory=analysis_directory,
    )
    unit_ids: set[str] = set()
    mapping_ids: list[str] = []
    analyses = _load_analysis(analysis_directory)
    for mapping in artifact.mappings:
        if (
            mapping.mapping_status != "reviewed"
            or mapping.fdd_document_id not in selected_fdd_document_ids
        ):
            continue
        mapping_ids.append(mapping.mapping_id)
        for target in mapping.targets:
            if target.selector_scope == "file":
                unit_ids.update(
                    record.unit_id
                    for record in code_artifact.records
                    if record.source_path == target.path
                )
                continue
            matches = [
                symbol
                for symbol in analyses.get(target.path, ())
                if symbol.canonical_qualified_name == target.qualified_name
                and symbol.symbol_kind == target.symbol_kind
                and (
                    target.selector_scope == "all_overloads"
                    or symbol.overload_discriminator_hash
                    == target.overload_discriminator_hash
                )
            ]
            for symbol in matches:
                unit_ids.update(
                    record.unit_id
                    for record in code_artifact.records
                    if record.source_path == target.path
                    and _ranges_overlap(record, symbol.source_map)
                )
    return unit_ids, tuple(sorted(mapping_ids))


def write_lineage_artifact_no_overwrite(
    artifact: FddCodeLineageArtifact, path: Path
) -> Path:
    if path.exists():
        raise FileExistsError(f"Lineage artifact already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = artifact.model_dump(mode="json")
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
    observed = FddCodeLineageArtifact.model_validate_json(path.read_text(encoding="utf-8"))
    if observed != artifact:
        raise RuntimeError("Persisted lineage artifact failed round-trip validation")
    return path


def _load_analysis(directory: Path) -> dict[str, tuple]:
    loaded: dict[str, tuple] = {}
    artifact_directory = directory / "analysis" if (directory / "analysis").is_dir() else directory
    for path in sorted(artifact_directory.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("schema_version") != "code_static_analysis_v1":
            continue
        item = CodeStaticAnalysisArtifact.model_validate(payload)
        if item.source_path in loaded:
            raise ValueError(
                f"Duplicate static-analysis artifact for source path: {item.source_path}"
            )
        loaded[item.source_path] = item.symbols
    return loaded


def _ranges_overlap(record, source_map) -> bool:
    record_map = record.parent_source_map or record.source_map
    return (
        record_map.start_offset < source_map.end_offset
        and source_map.start_offset < record_map.end_offset
    )


def _identity(values: dict) -> str:
    encoded = json.dumps(values, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()
