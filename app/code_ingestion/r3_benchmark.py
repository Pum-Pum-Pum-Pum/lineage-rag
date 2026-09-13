"""Reviewed input contract for the controlled R3 seven-package expansion."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from app.code_ingestion.snapshot_models import CodeSnapshotManifest, normalize_relative_path


class FrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)


class R3PackagePair(FrozenModel):
    pair_id: str = Field(pattern=r"^[a-z0-9][a-z0-9-]{2,80}$")
    category: Literal[
        "fdd_enhancement",
        "multi_routine_validation",
        "table_heavy",
        "no_approved_fdd_mapping",
        "cross_package_dependency",
    ]
    spec_path: str
    body_path: str
    key_routines: tuple[str, ...] = Field(min_length=1)
    expected_callers: tuple[str, ...] = ()
    expected_callees: tuple[str, ...] = ()
    fdd_document_ids: tuple[str, ...] = ()
    enhancement_markers: tuple[str, ...] = ()
    expected_outcome: Literal[
        "reviewed_lineage_candidate",
        "code_workflow_evidence",
        "table_operation_evidence",
        "no_reviewed_fdd_lineage",
        "cross_package_workflow_evidence",
    ]
    rationale: str = Field(min_length=10)

    @field_validator("spec_path", "body_path")
    @classmethod
    def normalize_path(cls, value: str) -> str:
        return normalize_relative_path(value)

    @model_validator(mode="after")
    def validate_category_contract(self) -> "R3PackagePair":
        if self.spec_path.casefold() == self.body_path.casefold():
            raise ValueError("A package pair requires different spec_path and body_path")
        if self.category == "fdd_enhancement":
            if not self.fdd_document_ids or not self.enhancement_markers:
                raise ValueError("FDD enhancement pairs require FDD IDs and exact markers")
            if self.expected_outcome != "reviewed_lineage_candidate":
                raise ValueError("FDD enhancement pairs require reviewed_lineage_candidate")
        if self.category == "cross_package_dependency":
            if not self.expected_callers or not self.expected_callees:
                raise ValueError("Cross-package pairs require expected callers and callees")
            if self.expected_outcome != "cross_package_workflow_evidence":
                raise ValueError("Cross-package pairs require cross_package_workflow_evidence")
        if self.category == "no_approved_fdd_mapping":
            if self.fdd_document_ids or self.enhancement_markers:
                raise ValueError("No-approved-FDD pairs must not name an FDD or enhancement marker")
            if self.expected_outcome != "no_reviewed_fdd_lineage":
                raise ValueError("No-approved-FDD pairs require no_reviewed_fdd_lineage")
        return self


class R3BenchmarkManifest(FrozenModel):
    schema_version: Literal["code_r3_benchmark_manifest_v1"] = (
        "code_r3_benchmark_manifest_v1"
    )
    snapshot_request: Literal["fci-custom-r3"] = "fci-custom-r3"
    base_snapshot_id: Literal["fci-custom-r2-ffd9732906d4"] = (
        "fci-custom-r2-ffd9732906d4"
    )
    fdd_generation: Literal["functional_specs_v9"] = "functional_specs_v9"
    modified_base_source_path: str
    package_pairs: tuple[R3PackagePair, ...] = Field(min_length=7, max_length=7)
    sme_reviewed: bool = False
    review_status: Literal["draft", "reviewed"] = "draft"
    reviewer: str | None = None

    @field_validator("modified_base_source_path")
    @classmethod
    def normalize_modified_path(cls, value: str) -> str:
        return normalize_relative_path(value)

    @model_validator(mode="after")
    def validate_contract(self) -> "R3BenchmarkManifest":
        if self.review_status == "reviewed" and (not self.sme_reviewed or not self.reviewer):
            raise ValueError("Reviewed R3 benchmark requires SME review and reviewer")
        if self.review_status == "draft" and (self.sme_reviewed or self.reviewer is not None):
            raise ValueError("Draft R3 benchmark must not claim SME review or a reviewer")
        ids = [item.pair_id for item in self.package_pairs]
        if len(ids) != len(set(ids)):
            raise ValueError("R3 benchmark pair IDs must be unique")
        source_paths = [
            path
            for item in self.package_pairs
            for path in (item.spec_path, item.body_path)
        ]
        if len(source_paths) != 14 or len({path.casefold() for path in source_paths}) != 14:
            raise ValueError("R3 benchmark must contain exactly 14 unique spec/body source paths")
        if self.modified_base_source_path.casefold() in {
            path.casefold() for path in source_paths
        }:
            raise ValueError("The modified base source is not one of the seven new package pairs")
        categories = [item.category for item in self.package_pairs]
        expected = {
            "fdd_enhancement": 2,
            "multi_routine_validation": 2,
            "table_heavy": 1,
            "no_approved_fdd_mapping": 1,
            "cross_package_dependency": 1,
        }
        observed = {category: categories.count(category) for category in expected}
        if observed != expected:
            raise ValueError(f"R3 benchmark category counts must be {expected}, got {observed}")
        return self


def load_r3_benchmark_manifest(path: Path) -> R3BenchmarkManifest:
    return R3BenchmarkManifest.model_validate_json(path.read_text(encoding="utf-8"))


def verify_r3_benchmark_against_snapshot(
    *, benchmark: R3BenchmarkManifest, snapshot: CodeSnapshotManifest
) -> dict[str, object]:
    if snapshot.request.module_set != "fci-custom" or snapshot.request.svn_revision != "3":
        raise ValueError("R3 benchmark is only valid for the fci-custom-r3 snapshot request")
    if snapshot.diff.base_snapshot_id != benchmark.base_snapshot_id:
        raise ValueError("R3 snapshot base does not match reviewed benchmark base")
    expected_added = {
        path
        for pair in benchmark.package_pairs
        for path in (pair.spec_path, pair.body_path)
    }
    actual_added = set(snapshot.diff.added)
    if actual_added != expected_added:
        raise ValueError(
            "R3 added paths do not match benchmark package pairs: "
            f"missing={sorted(expected_added - actual_added)}, "
            f"unexpected={sorted(actual_added - expected_added)}"
        )
    if set(snapshot.diff.modified) != {benchmark.modified_base_source_path}:
        raise ValueError(
            "R3 modified paths must contain exactly the reviewed base change: "
            f"{benchmark.modified_base_source_path}"
        )
    if snapshot.diff.deleted or snapshot.diff.formatting_only_modified:
        raise ValueError("R3 benchmark rejects deletions and formatting-only modifications")
    if snapshot.diff.missing_expected_changes or snapshot.diff.unexpected_changed_files:
        raise ValueError("R3 snapshot diff does not match its expected_changed_packages assertion")
    return {
        "schema_version": "code_r3_benchmark_verification_v1",
        "status": "pass",
        "snapshot_id": snapshot.snapshot_id,
        "base_snapshot_id": snapshot.diff.base_snapshot_id,
        "fdd_generation": benchmark.fdd_generation,
        "package_pairs": len(benchmark.package_pairs),
        "new_source_files": len(expected_added),
        "modified_base_source_path": benchmark.modified_base_source_path,
        "external_calls_performed": False,
    }


def verify_r3_benchmark_fdd_coverage(
    *, benchmark: R3BenchmarkManifest, fdd_document_ids: set[str]
) -> None:
    expected = {
        document_id
        for pair in benchmark.package_pairs
        for document_id in pair.fdd_document_ids
    }
    missing = sorted(expected.difference(fdd_document_ids))
    if missing:
        raise ValueError(
            "R3 benchmark references FDD documents absent from functional_specs_v9: "
            f"{missing}"
        )


def write_r3_benchmark_template() -> str:
    """Return a documentation template; it is intentionally not reviewable as-is."""

    return json.dumps(
        {
            "schema_version": "code_r3_benchmark_manifest_v1",
            "snapshot_request": "fci-custom-r3",
            "base_snapshot_id": "fci-custom-r2-ffd9732906d4",
            "fdd_generation": "functional_specs_v9",
            "modified_base_source_path": "REPLACE/modified_r2_package.sql",
            "package_pairs": [],
            "sme_reviewed": False,
            "review_status": "draft",
            "reviewer": None,
        },
        indent=2,
    )
