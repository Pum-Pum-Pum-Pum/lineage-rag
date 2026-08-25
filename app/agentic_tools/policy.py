from __future__ import annotations

import hashlib
import json
import tomllib
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.agentic_tools.models import ExplicitToolPlan, ToolName


DEFAULT_AGENTIC_TOOLS_POLICY_PATH = (
    Path(__file__).resolve().parents[2] / "config" / "agentic_tools.toml"
)


class FrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ToolBudgets(FrozenModel):
    max_calls: int = Field(ge=1, le=10)
    max_results_per_call: int = Field(ge=1, le=50)
    max_total_evidence_units: int = Field(ge=1, le=100)
    max_impact_nodes: int = Field(ge=1, le=200)
    max_impact_edges: int = Field(ge=1, le=500)


class ToolControls(FrozenModel):
    automatic_routing: Literal[False] = False
    fdd_tools: tuple[ToolName, ...]
    code_tools: tuple[ToolName, ...]
    combined_tools: tuple[ToolName, ...]

    @field_validator("fdd_tools", "code_tools", "combined_tools")
    @classmethod
    def unique_tools(cls, values: tuple[ToolName, ...]) -> tuple[ToolName, ...]:
        if len(set(values)) != len(values):
            raise ValueError("Allowed tool lists must not contain duplicates")
        return values


class BoundedAgenticToolsPolicy(FrozenModel):
    schema_version: Literal["bounded_agentic_tools_policy_v1"]
    budgets: ToolBudgets
    controls: ToolControls

    @property
    def sha256(self) -> str:
        encoded = json.dumps(
            self.model_dump(mode="json"), sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def validate_plan(self, plan: ExplicitToolPlan) -> None:
        allowed = {
            "fdd": set(self.controls.fdd_tools),
            "code": set(self.controls.code_tools),
            "combined": set(self.controls.combined_tools),
        }[plan.knowledge_mode]
        if len(plan.invocations) > self.budgets.max_calls:
            raise ValueError("Explicit tool plan exceeds max_calls")
        if any(item.tool_name not in allowed for item in plan.invocations):
            raise ValueError("Explicit tool plan contains a tool not allowed for its mode")
        if any(item.limit > self.budgets.max_results_per_call for item in plan.invocations):
            raise ValueError("Tool invocation exceeds max_results_per_call")
        requested = sum(
            item.limit for item in plan.invocations if item.tool_name != "impact_graph"
        )
        if requested > self.budgets.max_total_evidence_units:
            raise ValueError("Explicit tool plan exceeds max_total_evidence_units")


def load_agentic_tools_policy(
    path: Path = DEFAULT_AGENTIC_TOOLS_POLICY_PATH,
) -> BoundedAgenticToolsPolicy:
    if not path.is_file():
        raise FileNotFoundError(f"Agentic tools policy not found: {path}")
    with path.open("rb") as handle:
        policy = BoundedAgenticToolsPolicy.model_validate(tomllib.load(handle))
    if policy.controls.fdd_tools != ("fdd_search",):
        raise ValueError("FDD mode must expose only fdd_search")
    if policy.controls.code_tools != ("code_search",):
        raise ValueError("Code mode must expose only code_search")
    if set(policy.controls.combined_tools) != {
        "fdd_search", "code_search", "impact_graph"
    }:
        raise ValueError("Combined mode must expose the three approved read-only tools")
    return policy
