from __future__ import annotations

import hashlib
import json
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.code_retrieval.models import CodeEvidence
from app.fdd_code_lineage.combined_retrieval import FddEvidence


ToolName = Literal["fdd_search", "code_search", "impact_graph"]
KnowledgeMode = Literal["fdd", "code", "combined"]


class FrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)


class ToolInvocation(FrozenModel):
    invocation_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    tool_name: ToolName
    query: str = Field(min_length=1, max_length=4000)
    limit: int = Field(ge=1)


class ExplicitToolPlan(FrozenModel):
    schema_version: Literal["explicit_tool_plan_v1"] = "explicit_tool_plan_v1"
    knowledge_mode: KnowledgeMode
    invocations: tuple[ToolInvocation, ...]
    plan_identity_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")

    @model_validator(mode="after")
    def validate_identity_and_uniqueness(self) -> "ExplicitToolPlan":
        if not self.invocations:
            raise ValueError("An explicit tool plan requires at least one invocation")
        invocation_ids = [item.invocation_id for item in self.invocations]
        if len(set(invocation_ids)) != len(invocation_ids):
            raise ValueError("Tool invocation IDs must be unique")
        for position, item in enumerate(self.invocations):
            expected_invocation_id = _identity(
                {
                    "position": position,
                    "tool_name": item.tool_name,
                    "query": item.query,
                    "limit": item.limit,
                }
            )
            if item.invocation_id != expected_invocation_id:
                raise ValueError("Tool invocation identity does not match its contents")
        expected = _identity(
            {
                "schema_version": self.schema_version,
                "knowledge_mode": self.knowledge_mode,
                "invocations": [item.model_dump(mode="json") for item in self.invocations],
            }
        )
        if self.plan_identity_sha256 != expected:
            raise ValueError("Tool plan identity does not match its contents")
        return self


class FddSearchToolResult(FrozenModel):
    tool_name: Literal["fdd_search"] = "fdd_search"
    query: str
    evidence: tuple[FddEvidence, ...]
    truncated: bool = False


class CodeSearchToolResult(FrozenModel):
    tool_name: Literal["code_search"] = "code_search"
    query: str
    snapshot_id: str
    evidence: tuple[CodeEvidence, ...]
    truncated: bool = False


class ImpactGraphNode(FrozenModel):
    node_id: str
    node_kind: Literal["fdd_document", "code_unit", "static_dependency"]
    label: str
    source_identity: str


class ImpactGraphEdge(FrozenModel):
    edge_id: str
    edge_kind: str
    source_node_id: str
    target_node_id: str
    resolution_state: str
    evidence_identity: str


class ImpactGraphToolResult(FrozenModel):
    tool_name: Literal["impact_graph"] = "impact_graph"
    query: str
    nodes: tuple[ImpactGraphNode, ...]
    edges: tuple[ImpactGraphEdge, ...]
    truncated: bool = False
    unknowns: tuple[str, ...] = ()


ToolOutput = Annotated[
    FddSearchToolResult | CodeSearchToolResult | ImpactGraphToolResult,
    Field(discriminator="tool_name"),
]
class ToolCallTrace(FrozenModel):
    invocation_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    tool_name: ToolName
    status: Literal["completed", "failed", "blocked"]
    result_count: int = Field(ge=0)
    evidence_identities: tuple[str, ...] = ()
    error_type: str | None = None


class BoundedToolTrace(FrozenModel):
    schema_version: Literal["bounded_tool_trace_v1"] = "bounded_tool_trace_v1"
    plan_identity_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    policy_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    status: Literal["completed", "failed", "blocked"]
    calls: tuple[ToolCallTrace, ...]
    total_evidence_units: int = Field(ge=0)
    automatic_routing_used: Literal[False] = False


class BoundedToolExecution(FrozenModel):
    schema_version: Literal["bounded_tool_execution_v1"] = "bounded_tool_execution_v1"
    trace: BoundedToolTrace
    outputs: tuple[ToolOutput, ...]


def create_explicit_tool_plan(
    *, knowledge_mode: KnowledgeMode,
    invocations: list[tuple[ToolName, str, int]] | tuple[tuple[ToolName, str, int], ...],
) -> ExplicitToolPlan:
    built: list[ToolInvocation] = []
    for position, (tool_name, query, limit) in enumerate(invocations):
        values = {
            "position": position,
            "tool_name": tool_name,
            "query": query.strip(),
            "limit": limit,
        }
        built.append(
            ToolInvocation(
                invocation_id=_identity(values),
                tool_name=tool_name,
                query=query,
                limit=limit,
            )
        )
    identity_values = {
        "schema_version": "explicit_tool_plan_v1",
        "knowledge_mode": knowledge_mode,
        "invocations": [item.model_dump(mode="json") for item in built],
    }
    return ExplicitToolPlan(
        knowledge_mode=knowledge_mode,
        invocations=tuple(built),
        plan_identity_sha256=_identity(identity_values),
    )


def identity(value: object) -> str:
    return _identity(value)


def _identity(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
