from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.code_ingestion.plsql_models import SourceMap


class FrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class CodeIndexRecord(FrozenModel):
    schema_version: Literal["code_index_record_v1"] = "code_index_record_v1"
    unit_id: str
    point_id: str
    unit_index: int = Field(ge=0)
    snapshot_id: str
    module_id: str
    source_path: str
    source_kind: str
    display_name: str
    package_name: str | None = None
    source_map: SourceMap
    parent_unit_id: str | None = None
    parent_source_map: SourceMap | None = None
    chunk_index: int | None = Field(default=None, ge=0)
    chunk_count: int | None = Field(default=None, ge=1)
    parser_state: str
    conditional_state: str
    citation_text: str
    embedding_text: str
    embedding_input_version: Literal["code_embedding_input_v1"] = (
        "code_embedding_input_v1"
    )
    content_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    cache_key: str = Field(pattern=r"^[0-9a-f]{64}$")
    embedding_model: str
    embedding_status: Literal["pending", "cached", "embedded"] = "pending"
    vector: tuple[float, ...] | None = None

    @model_validator(mode="after")
    def validate_embedding_state(self) -> "CodeIndexRecord":
        if self.embedding_status == "pending" and self.vector is not None:
            raise ValueError("Pending record must not contain a vector")
        if self.embedding_status != "pending" and not self.vector:
            raise ValueError("Embedded/cached record must contain a vector")
        return self


class CodeIndexArtifact(FrozenModel):
    schema_version: Literal["code_index_artifact_v2"] = "code_index_artifact_v2"
    status: Literal["prepared", "embedded"]
    snapshot_id: str
    snapshot_content_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    parse_generation: str
    analysis_policy_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    dependency_review_status: Literal["draft", "reviewed"] = "draft"
    module_id: str
    embedding_model: str
    embedding_input_version: Literal["code_embedding_input_v1"] = (
        "code_embedding_input_v1"
    )
    vector_dimension: int | None = Field(default=None, gt=0)
    total_records: int = Field(ge=0)
    artifact_identity_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    records: tuple[CodeIndexRecord, ...]

    @model_validator(mode="after")
    def validate_records(self) -> "CodeIndexArtifact":
        if self.total_records != len(self.records):
            raise ValueError("total_records must match records")
        if len({record.unit_id for record in self.records}) != len(self.records):
            raise ValueError("Code index unit IDs must be unique")
        if len({record.point_id for record in self.records}) != len(self.records):
            raise ValueError("Code index point IDs must be unique")
        if self.status == "prepared" and any(
            record.embedding_status != "pending" for record in self.records
        ):
            raise ValueError("Prepared artifacts may contain only pending records")
        if self.status == "embedded":
            if self.vector_dimension is None:
                raise ValueError("Embedded artifact requires vector_dimension")
            if any(
                record.vector is None or len(record.vector) != self.vector_dimension
                for record in self.records
            ):
                raise ValueError("Every embedded vector must match vector_dimension")
        return self
