from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.code_ingestion.plsql_models import SourceMap


class FrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class OracleIdentifier(FrozenModel):
    display_name: str
    canonical_name: str
    is_quoted: bool


class ParameterContract(FrozenModel):
    position: int = Field(ge=1)
    name: OracleIdentifier
    declared_type: str
    canonical_declared_type: str
    type_family: str
    mode: Literal["IN", "OUT", "IN OUT"] = "IN"
    nocopy: bool = False
    has_default: bool = False
    normalized_default: str | None = None


class CodeSymbol(FrozenModel):
    occurrence_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    symbol_key: str = Field(pattern=r"^[0-9a-f]{64}$")
    source_node_id: str
    language: Literal["plsql"] = "plsql"
    module_id: str
    snapshot_id: str
    source_path: str
    source_map: SourceMap
    occurrence_role: Literal["declaration", "implementation"]
    symbol_kind: Literal["procedure", "function"]
    name: OracleIdentifier
    qualified_display_name: str
    canonical_qualified_name: str
    parameters: tuple[ParameterContract, ...] = ()
    return_type: str | None = None
    canonical_return_type: str | None = None
    overload_discriminator_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    declaration_signature_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    conditional_state: str


class AnalysisDiagnostic(FrozenModel):
    stage: Literal["symbol", "dependency", "ddl", "snapshot_resolution"]
    severity: Literal["info", "warning", "error"]
    code: str
    message: str
    source_path: str | None = None
    source_map: SourceMap | None = None
    related_occurrence_ids: tuple[str, ...] = ()


class DependencyEdge(FrozenModel):
    edge_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    dependency_kind: Literal[
        "routine_call",
        "table_read",
        "table_write",
        "type_reference",
        "constant_reference",
        "global_reference",
        "cursor_reference",
        "dynamic_sql",
        "kernel_boundary",
        "external_package",
    ]
    source_symbol_occurrence_id: str | None = None
    source_path: str
    source_map: SourceMap
    target_display_name: str
    target_canonical_name: str
    resolution_state: Literal[
        "resolved_in_snapshot",
        "unresolved",
        "ambiguous",
        "dynamic_unknown",
        "custom_source_missing",
        "kernel_unavailable",
        "external_schema",
    ]
    candidate_symbol_occurrence_ids: tuple[str, ...] = ()
    extraction_method: Literal["antlr_tokens", "antlr_tree"]
    confidence: Literal["high", "medium", "degraded"]


class ColumnDefinition(FrozenModel):
    name: OracleIdentifier
    declared_type: str | None = None
    canonical_declared_type: str | None = None
    nullable: bool | None = None
    default_expression: str | None = None
    source_map: SourceMap


class ConstraintDefinition(FrozenModel):
    name: OracleIdentifier | None = None
    constraint_kind: Literal["primary_key", "foreign_key", "unique", "check", "not_null"]
    columns: tuple[str, ...] = ()
    referenced_object: str | None = None
    source_map: SourceMap


class SchemaObject(FrozenModel):
    object_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    object_kind: Literal["table", "view", "sequence", "index", "object_type", "collection_type"]
    name: OracleIdentifier
    schema_name: OracleIdentifier | None = None
    canonical_qualified_name: str
    source_path: str
    source_map: SourceMap
    columns: tuple[ColumnDefinition, ...] = ()
    constraints: tuple[ConstraintDefinition, ...] = ()
    referenced_objects: tuple[str, ...] = ()


class SynonymDefinition(FrozenModel):
    synonym_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    name: OracleIdentifier
    schema_name: OracleIdentifier | None = None
    is_public: bool = False
    canonical_qualified_name: str
    declared_target: str
    canonical_declared_target: str
    database_link: str | None = None
    resolution_state: Literal[
        "resolved_in_snapshot",
        "external_schema",
        "database_link",
        "ambiguous",
        "cyclic",
    ]
    resolved_object_id: str | None = None
    source_path: str
    source_map: SourceMap


class CodeStaticAnalysisArtifact(FrozenModel):
    schema_version: Literal["code_static_analysis_v1"] = "code_static_analysis_v1"
    module_id: str
    snapshot_id: str
    source_path: str
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    analysis_policy_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    parser_state: str
    symbols: tuple[CodeSymbol, ...] = ()
    dependencies: tuple[DependencyEdge, ...] = ()
    schema_objects: tuple[SchemaObject, ...] = ()
    synonyms: tuple[SynonymDefinition, ...] = ()
    diagnostics: tuple[AnalysisDiagnostic, ...] = ()

    @model_validator(mode="after")
    def validate_occurrence_identity(self) -> "CodeStaticAnalysisArtifact":
        occurrence_ids = [symbol.occurrence_id for symbol in self.symbols]
        if len(set(occurrence_ids)) != len(occurrence_ids):
            raise ValueError("Symbol occurrence IDs must be unique within an artifact")
        edge_ids = [edge.edge_id for edge in self.dependencies]
        if len(set(edge_ids)) != len(edge_ids):
            raise ValueError("Dependency edge IDs must be unique within an artifact")
        return self
