from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.code_ingestion.snapshot_models import CompilerContext


class FrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class SourceMap(FrozenModel):
    source_path: str
    start_line: int = Field(ge=1)
    end_line: int = Field(ge=1)
    start_offset: int = Field(ge=0)
    end_offset: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_order(self) -> "SourceMap":
        if self.end_line < self.start_line or self.end_offset < self.start_offset:
            raise ValueError("Source-map end must not precede its start")
        return self


class ParseDiagnostic(FrozenModel):
    stage: Literal["conditional_scan", "full_parse", "segmented_parse", "fallback", "worker"]
    severity: Literal["info", "warning", "error"]
    code: str
    message: str
    line: int | None = Field(default=None, ge=1)
    column: int | None = Field(default=None, ge=0)


class ConditionalBranch(FrozenModel):
    branch_kind: Literal["if", "elsif", "else"]
    expression: str | None = None
    state: Literal["active", "inactive", "unresolved", "conditional_unknown"]
    directive_line: int = Field(ge=1)
    body_source_map: SourceMap


class ConditionalRegion(FrozenModel):
    region_id: str
    parent_region_id: str | None = None
    source_map: SourceMap
    branches: tuple[ConditionalBranch, ...]


class ConditionalErrorDirective(FrozenModel):
    source_map: SourceMap
    expression: str
    state: Literal["unresolved", "conditional_unknown"]


class ConditionalParseView(FrozenModel):
    schema_version: Literal["plsql_conditional_parse_view_v1"] = "plsql_conditional_parse_view_v1"
    original_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    parse_view_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    text: str
    regions: tuple[ConditionalRegion, ...] = ()
    error_directives: tuple[ConditionalErrorDirective, ...] = ()
    diagnostics: tuple[ParseDiagnostic, ...] = ()


class ParsedSegment(FrozenModel):
    segment_id: str
    segment_kind: Literal[
        "file",
        "package",
        "package_body",
        "procedure",
        "procedure_spec",
        "function",
        "function_spec",
        "fallback_chunk",
    ]
    display_name: str | None = None
    source_map: SourceMap
    parse_succeeded: bool
    syntax_error_count: int = Field(ge=0)
    degradation_reason: str | None = None


class ExtractedCodeNode(FrozenModel):
    node_id: str
    node_kind: Literal[
        "package",
        "package_body",
        "procedure",
        "procedure_spec",
        "function",
        "function_spec",
        "type",
        "constant",
        "global_variable",
        "cursor",
    ]
    display_name: str
    package_name: str | None = None
    enclosing_routines: tuple[str, ...] = ()
    extraction_method: Literal["antlr", "token_structural"] = "antlr"
    source_map: SourceMap
    signature_text: str | None = None
    conditional_state: Literal[
        "unconditional",
        "active",
        "inactive",
        "unresolved",
        "conditional_unknown",
    ] = "unconditional"


class PlSqlFileParseArtifact(FrozenModel):
    schema_version: Literal["plsql_file_parse_v1"] = "plsql_file_parse_v1"
    snapshot_id: str
    source_path: str
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    parser_state: Literal["full_parse", "segmented_parse", "fallback_parse", "failed"]
    antlr_tool_version: Literal["4.13.2"] = "4.13.2"
    antlr_runtime_version: Literal["4.13.2"] = "4.13.2"
    grammar_commit: Literal["a7704d4c029c33a89818ac103f758f7c72d8d16c"] = (
        "a7704d4c029c33a89818ac103f758f7c72d8d16c"
    )
    duration_ms: float = Field(ge=0)
    peak_memory_bytes: int = Field(ge=0)
    syntax_error_count: int = Field(ge=0)
    conditional_regions: tuple[ConditionalRegion, ...] = ()
    conditional_error_directives: tuple[ConditionalErrorDirective, ...] = ()
    segments: tuple[ParsedSegment, ...] = ()
    extracted_nodes: tuple[ExtractedCodeNode, ...] = ()
    diagnostics: tuple[ParseDiagnostic, ...] = ()


class PackageContextSummary(FrozenModel):
    package_name: str
    public_signature: str | None = None
    referenced_types: tuple[str, ...] = ()
    referenced_constants: tuple[str, ...] = ()
    referenced_globals: tuple[str, ...] = ()
    referenced_cursors: tuple[str, ...] = ()
    conditional_state: str = "unconditional"


class CodeRetrievalUnit(FrozenModel):
    unit_id: str
    source_kind: str
    snapshot_id: str
    source_path: str
    source_map: SourceMap
    display_name: str
    package_name: str | None = None
    text: str
    retrieval_text: str
    derived_context: PackageContextSummary | None = None
    related_unit_ids: tuple[str, ...] = ()
    parser_state: str
    conditional_state: str


class CodeRetrievalArtifact(FrozenModel):
    schema_version: Literal["code_retrieval_artifact_v1"] = "code_retrieval_artifact_v1"
    snapshot_id: str
    source_path: str
    total_units: int = Field(ge=0)
    units: tuple[CodeRetrievalUnit, ...]

    @model_validator(mode="after")
    def validate_unit_contract(self) -> "CodeRetrievalArtifact":
        if self.total_units != len(self.units):
            raise ValueError("total_units must equal the number of units")
        if len({unit.unit_id for unit in self.units}) != len(self.units):
            raise ValueError("Retrieval unit IDs must be unique")
        return self


class ParserWorkerRequest(FrozenModel):
    schema_version: Literal["plsql_parser_worker_request_v1"] = "plsql_parser_worker_request_v1"
    input_file: str
    snapshot_id: str
    source_path: str
    source_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    encoding: str
    compiler_context: CompilerContext = Field(default_factory=CompilerContext)
    parse_mode: Literal["full", "segmented"] = "full"
    max_segment_characters: int = Field(default=1_000, gt=0)


class CodeParseStageManifest(FrozenModel):
    schema_version: Literal["code_parse_stage_v2"] = "code_parse_stage_v2"
    status: Literal["complete", "complete_with_degradation", "failed"]
    snapshot_id: str
    snapshot_content_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    parser_generation: Literal["plsql_antlr_4_13_2_analysis_v3"] = (
        "plsql_antlr_4_13_2_analysis_v3"
    )
    analysis_policy_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    file_count: int = Field(ge=0)
    state_counts: dict[str, int]
    parse_artifacts: tuple[str, ...]
    retrieval_artifacts: tuple[str, ...]
    analysis_artifacts: tuple[str, ...]
    timeout_seconds: float = Field(gt=0)
    memory_limit_bytes: int = Field(gt=0)
    max_segment_characters: int = Field(gt=0)

    @model_validator(mode="after")
    def validate_stage_counts(self) -> "CodeParseStageManifest":
        required_states = {"full_parse", "segmented_parse", "fallback_parse", "failed"}
        if set(self.state_counts) != required_states:
            raise ValueError("state_counts must contain exactly the declared parser states")
        if any(count < 0 for count in self.state_counts.values()):
            raise ValueError("Parser state counts must not be negative")
        if sum(self.state_counts.values()) != self.file_count:
            raise ValueError("Parser state counts must equal file_count")
        if len(self.parse_artifacts) != self.file_count:
            raise ValueError("Every source file must have one parse artifact")
        if len(self.retrieval_artifacts) != self.file_count:
            raise ValueError("Every source file must have one retrieval artifact")
        if len(self.analysis_artifacts) != self.file_count:
            raise ValueError("Every source file must have one static-analysis artifact")
        return self
