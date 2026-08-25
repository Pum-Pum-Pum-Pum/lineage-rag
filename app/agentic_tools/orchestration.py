from __future__ import annotations

from collections.abc import Callable, Mapping

from app.agentic_tools.models import (
    BoundedToolExecution,
    BoundedToolTrace,
    ExplicitToolPlan,
    ToolCallTrace,
    ToolOutput,
)
from app.agentic_tools.policy import BoundedAgenticToolsPolicy


ToolHandler = Callable[[], ToolOutput]


def execute_explicit_tool_plan(
    *,
    plan: ExplicitToolPlan,
    policy: BoundedAgenticToolsPolicy,
    handlers: Mapping[str, ToolHandler],
) -> BoundedToolExecution:
    """Execute a caller-authored plan sequentially with no model-driven routing."""

    policy.validate_plan(plan)
    calls: list[ToolCallTrace] = []
    outputs: list[ToolOutput] = []
    total_evidence = 0
    status = "completed"
    for invocation in plan.invocations:
        handler = handlers.get(invocation.invocation_id)
        if handler is None:
            calls.append(
                ToolCallTrace(
                    invocation_id=invocation.invocation_id,
                    tool_name=invocation.tool_name,
                    status="blocked",
                    result_count=0,
                    error_type="MissingToolHandler",
                )
            )
            status = "blocked"
            break
        try:
            output = handler()
            if output.tool_name != invocation.tool_name:
                raise RuntimeError("Tool handler returned the wrong output type")
            result_count, evidence_ids = _result_summary(output)
            added_evidence = 0 if output.tool_name == "impact_graph" else result_count
            if total_evidence + added_evidence > policy.budgets.max_total_evidence_units:
                raise RuntimeError("Tool output exceeded the total evidence budget")
            total_evidence += added_evidence
            outputs.append(output)
            calls.append(
                ToolCallTrace(
                    invocation_id=invocation.invocation_id,
                    tool_name=invocation.tool_name,
                    status="completed",
                    result_count=result_count,
                    evidence_identities=evidence_ids,
                )
            )
        except Exception as exc:  # fail closed; trace type, never source or secrets
            calls.append(
                ToolCallTrace(
                    invocation_id=invocation.invocation_id,
                    tool_name=invocation.tool_name,
                    status="failed",
                    result_count=0,
                    error_type=type(exc).__name__,
                )
            )
            status = "failed"
            break
    trace = BoundedToolTrace(
        plan_identity_sha256=plan.plan_identity_sha256,
        policy_sha256=policy.sha256,
        status=status,
        calls=tuple(calls),
        total_evidence_units=total_evidence,
    )
    return BoundedToolExecution(trace=trace, outputs=tuple(outputs))


def _result_summary(output: ToolOutput) -> tuple[int, tuple[str, ...]]:
    if output.tool_name == "fdd_search":
        return len(output.evidence), tuple(item.unit_id for item in output.evidence)
    if output.tool_name == "code_search":
        return len(output.evidence), tuple(item.unit_id for item in output.evidence)
    return len(output.edges), tuple(item.edge_id for item in output.edges)
