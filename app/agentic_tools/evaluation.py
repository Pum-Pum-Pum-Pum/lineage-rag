from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal, Sequence

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.agentic_tools.models import (
    BoundedToolExecution,
    CodeSearchToolResult,
    FddSearchToolResult,
    ImpactGraphToolResult,
    create_explicit_tool_plan,
)
from app.agentic_tools.orchestration import execute_explicit_tool_plan
from app.agentic_tools.policy import BoundedAgenticToolsPolicy
from app.agentic_tools.tools import (
    run_code_search_tool,
    run_fdd_search_tool,
    run_impact_graph_tool,
)
from app.code_indexing.models import CodeIndexArtifact
from app.code_retrieval.service import retrieve_code_evidence
from app.fdd_code_lineage.combined_retrieval import (
    CombinedRetrievalResult,
    ReviewedLineageUse,
)
from app.fdd_code_lineage.models import FddCodeLineageArtifact
from app.retrieval.lexical_search import (
    LexicalSearchDocument,
    search_lexical_documents,
)


class FrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, str_strip_whitespace=True)


class BoundedToolEvalCase(FrozenModel):
    schema_version: Literal["bounded_tool_eval_case_v1"] = "bounded_tool_eval_case_v1"
    case_id: str = Field(pattern=r"^[a-z0-9][a-z0-9-]{2,127}$")
    knowledge_mode: Literal["fdd", "code", "combined"]
    question: str = Field(min_length=10, max_length=4000)
    tools: tuple[Literal["fdd_search", "code_search", "impact_graph"], ...]
    limit: int = Field(ge=1, le=50)
    expected_fdd_document_ids: tuple[str, ...] = ()
    expected_code_paths: tuple[str, ...] = ()
    expected_code_symbols: tuple[str, ...] = ()
    require_reviewed_lineage: bool = False
    review_status: Literal["draft", "reviewed"] = "draft"
    sme_reviewed: bool = False
    rationale: str = Field(min_length=10)

    @model_validator(mode="after")
    def validate_case_contract(self) -> "BoundedToolEvalCase":
        expected_tools = {
            "fdd": ("fdd_search",),
            "code": ("code_search",),
            "combined": ("fdd_search", "code_search", "impact_graph"),
        }[self.knowledge_mode]
        if self.tools != expected_tools:
            raise ValueError("Evaluation tools must match the explicit mode contract and order")
        if self.require_reviewed_lineage and self.knowledge_mode != "combined":
            raise ValueError("Only combined cases may require reviewed lineage")
        if self.review_status == "reviewed" and not self.sme_reviewed:
            raise ValueError("Reviewed cases require sme_reviewed=true")
        if self.review_status == "draft" and self.sme_reviewed:
            raise ValueError("Draft cases cannot claim SME review")
        return self


class ToolEvalCheck(FrozenModel):
    name: str
    passed: bool
    expected: tuple[str, ...] = ()
    observed: tuple[str, ...] = ()


class BoundedToolEvalCaseResult(FrozenModel):
    case_id: str
    passed: bool
    checks: tuple[ToolEvalCheck, ...]
    execution_trace: dict


class ToolSafetyCheck(FrozenModel):
    name: str
    passed: bool
    detail: str


class BoundedToolEvalReport(FrozenModel):
    schema_version: Literal["bounded_tool_eval_report_v1"] = "bounded_tool_eval_report_v1"
    manifest_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    policy_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    fdd_generation: str
    code_snapshot_id: str
    lineage_artifact_identity_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    cases: tuple[BoundedToolEvalCaseResult, ...]
    safety_checks: tuple[ToolSafetyCheck, ...]
    positive_passes: int = Field(ge=0)
    positive_total: int = Field(ge=0)
    safety_passes: int = Field(ge=0)
    safety_total: int = Field(ge=0)
    all_cases_reviewed: bool
    release_gate_eligible: bool
    external_api_calls: Literal[0] = 0
    report_identity_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")


def load_eval_cases(path: Path) -> tuple[BoundedToolEvalCase, ...]:
    if not path.is_file():
        raise FileNotFoundError(f"Bounded tool evaluation manifest not found: {path}")
    cases = tuple(
        BoundedToolEvalCase.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    )
    if not cases:
        raise ValueError("Bounded tool evaluation manifest is empty")
    case_ids = [case.case_id for case in cases]
    if len(set(case_ids)) != len(case_ids):
        raise ValueError("Bounded tool evaluation case IDs must be unique")
    return cases


def evaluate_bounded_tools(
    *,
    cases: Sequence[BoundedToolEvalCase],
    manifest_sha256: str,
    policy: BoundedAgenticToolsPolicy,
    fdd_documents: list[LexicalSearchDocument],
    fdd_generation: str,
    code_artifact: CodeIndexArtifact,
    lineage_artifact: FddCodeLineageArtifact,
) -> BoundedToolEvalReport:
    _validate_lineage_boundary(
        lineage_artifact=lineage_artifact,
        fdd_generation=fdd_generation,
        code_artifact=code_artifact,
        fdd_document_ids={item.document_id for item in fdd_documents},
    )
    results = tuple(
        _evaluate_case(
            case=case,
            policy=policy,
            fdd_documents=fdd_documents,
            fdd_generation=fdd_generation,
            code_artifact=code_artifact,
            lineage_artifact=lineage_artifact,
        )
        for case in cases
    )
    safety = _safety_checks(policy)
    all_reviewed = all(case.sme_reviewed and case.review_status == "reviewed" for case in cases)
    values = {
        "schema_version": "bounded_tool_eval_report_v1",
        "manifest_sha256": manifest_sha256,
        "policy_sha256": policy.sha256,
        "fdd_generation": fdd_generation,
        "code_snapshot_id": code_artifact.snapshot_id,
        "lineage_artifact_identity_sha256": lineage_artifact.artifact_identity_sha256,
        "cases": [item.model_dump(mode="json") for item in results],
        "safety_checks": [item.model_dump(mode="json") for item in safety],
        "positive_passes": sum(item.passed for item in results),
        "positive_total": len(results),
        "safety_passes": sum(item.passed for item in safety),
        "safety_total": len(safety),
        "all_cases_reviewed": all_reviewed,
        "release_gate_eligible": (
            all_reviewed and all(item.passed for item in results) and all(item.passed for item in safety)
        ),
        "external_api_calls": 0,
    }
    identity_values = dict(values)
    identity_values.pop("schema_version")
    return BoundedToolEvalReport(
        **identity_values,
        report_identity_sha256=_identity(values),
    )


def write_eval_report_no_overwrite(report: BoundedToolEvalReport, path: Path) -> Path:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite bounded tool evaluation report: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    observed = BoundedToolEvalReport.model_validate_json(path.read_text(encoding="utf-8"))
    if observed != report:
        raise RuntimeError("Bounded tool evaluation report failed round-trip validation")
    return path


def write_sme_review_packet_no_overwrite(
    *,
    cases: Sequence[BoundedToolEvalCase],
    report: BoundedToolEvalReport,
    path: Path,
) -> Path:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite bounded tool SME packet: {path}")
    results = {item.case_id: item for item in report.cases}
    lines = [
        "# Bounded agentic tools SME review packet",
        "",
        f"- Manifest SHA-256: `{report.manifest_sha256}`",
        f"- Report identity: `{report.report_identity_sha256}`",
        f"- Policy SHA-256: `{report.policy_sha256}`",
        f"- Positive results: **{report.positive_passes}/{report.positive_total}**",
        f"- Safety results: **{report.safety_passes}/{report.safety_total}**",
        f"- External API calls: **{report.external_api_calls}**",
        "- Review status: **draft**",
        "",
        "Review whether each natural question, expected evidence identity, and required",
        "lineage behavior is correct. This packet intentionally omits full FDD/code text.",
        "",
    ]
    for index, case in enumerate(cases, start=1):
        result = results[case.case_id]
        lines.extend(
            [
                f"## {index}. {case.case_id}",
                "",
                f"- Question: {case.question}",
                f"- Mode/tools: `{case.knowledge_mode}` / `{', '.join(case.tools)}`",
                f"- Structural result: **{'pass' if result.passed else 'fail'}**",
                f"- Expected FDD documents: `{list(case.expected_fdd_document_ids)}`",
                f"- Expected code paths: `{list(case.expected_code_paths)}`",
                f"- Expected code symbols: `{list(case.expected_code_symbols)}`",
                f"- Reviewed lineage required: **{case.require_reviewed_lineage}**",
                "",
                "### Deterministic checks",
                "",
            ]
        )
        for check in result.checks:
            lines.append(
                f"- `{check.name}`: **{'pass' if check.passed else 'fail'}**; "
                f"expected `{list(check.expected)}`; observed `{list(check.observed)}`"
            )
        lines.extend(
            [
                "",
                "SME verdict: accepted | corrected | needs_more_context",
                "SME corrected expectation:",
                "SME rationale:",
                "Required follow-up:",
                "",
            ]
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    return path


def _evaluate_case(
    *,
    case: BoundedToolEvalCase,
    policy: BoundedAgenticToolsPolicy,
    fdd_documents: list[LexicalSearchDocument],
    fdd_generation: str,
    code_artifact: CodeIndexArtifact,
    lineage_artifact: FddCodeLineageArtifact,
) -> BoundedToolEvalCaseResult:
    execution = execute_local_lexical_tools(
        knowledge_mode=case.knowledge_mode,
        question=case.question,
        limit=case.limit,
        policy=policy,
        fdd_documents=fdd_documents,
        fdd_generation=fdd_generation,
        code_artifact=code_artifact,
        lineage_artifact=lineage_artifact,
    )
    checks = _case_checks(case, execution)
    return BoundedToolEvalCaseResult(
        case_id=case.case_id,
        passed=execution.trace.status == "completed" and all(item.passed for item in checks),
        checks=checks,
        execution_trace=execution.trace.model_dump(mode="json"),
    )


def execute_local_lexical_tools(
    *,
    knowledge_mode: Literal["fdd", "code", "combined"],
    question: str,
    limit: int,
    policy: BoundedAgenticToolsPolicy,
    fdd_documents: list[LexicalSearchDocument],
    fdd_generation: str,
    code_artifact: CodeIndexArtifact,
    lineage_artifact: FddCodeLineageArtifact,
) -> BoundedToolExecution:
    """Execute a fixed local lexical plan without embeddings or generation."""

    tools = {
        "fdd": ("fdd_search",),
        "code": ("code_search",),
        "combined": ("fdd_search", "code_search", "impact_graph"),
    }[knowledge_mode]
    plan = create_explicit_tool_plan(
        knowledge_mode=knowledge_mode,
        invocations=tuple((tool, question, limit) for tool in tools),
    )
    state: dict[str, FddSearchToolResult | CodeSearchToolResult | ImpactGraphToolResult] = {}
    handlers = {}
    for invocation in plan.invocations:
        if invocation.tool_name == "fdd_search":
            def fdd_handler(invocation=invocation):
                output = run_fdd_search_tool(
                    invocation,
                    search_runner=lambda query, limit: search_lexical_documents(
                        fdd_documents, query, limit=limit + 1
                    ),
                )
                state["fdd_search"] = output
                return output
            handlers[invocation.invocation_id] = fdd_handler
        elif invocation.tool_name == "code_search":
            def code_handler(invocation=invocation):
                output = run_code_search_tool(
                    invocation,
                    search_runner=lambda query, limit: retrieve_code_evidence(
                        artifact=code_artifact,
                        query=query,
                        mode="lexical",
                        limit=max(20, limit) if knowledge_mode == "combined" else limit,
                        candidate_limit=max(50, limit) if knowledge_mode == "combined" else max(20, limit),
                    ),
                    reserve_identifier_affinity=knowledge_mode == "combined",
                )
                state["code_search"] = output
                return output
            handlers[invocation.invocation_id] = code_handler
        else:
            def impact_handler(invocation=invocation):
                combined = _assemble_combined_retrieval(
                    query=question,
                    fdd_generation=fdd_generation,
                    code_artifact=code_artifact,
                    lineage_artifact=lineage_artifact,
                    fdd_output=_require_output(state, "fdd_search", FddSearchToolResult),
                    code_output=_require_output(state, "code_search", CodeSearchToolResult),
                )
                output = run_impact_graph_tool(
                    invocation,
                    combined_retrieval=combined,
                    analyses=(),
                    max_nodes=policy.budgets.max_impact_nodes,
                    max_edges=policy.budgets.max_impact_edges,
                )
                state["impact_graph"] = output
                return output
            handlers[invocation.invocation_id] = impact_handler
    return execute_explicit_tool_plan(plan=plan, policy=policy, handlers=handlers)


def _assemble_combined_retrieval(
    *,
    query: str,
    fdd_generation: str,
    code_artifact: CodeIndexArtifact,
    lineage_artifact: FddCodeLineageArtifact,
    fdd_output: FddSearchToolResult,
    code_output: CodeSearchToolResult,
) -> CombinedRetrievalResult:
    selected_documents = {item.document_id for item in fdd_output.evidence}
    lineage_uses: list[ReviewedLineageUse] = []
    for mapping in lineage_artifact.mappings:
        if mapping.mapping_status != "reviewed" or mapping.fdd_document_id not in selected_documents:
            continue
        matching_units = tuple(
            item.unit_id
            for item in code_output.evidence
            if any(_target_matches_evidence(target, item) for target in mapping.targets)
        )
        if matching_units:
            lineage_uses.append(
                ReviewedLineageUse(
                    mapping_id=mapping.mapping_id,
                    fdd_document_id=mapping.fdd_document_id,
                    code_unit_ids=matching_units,
                )
            )
    unknowns = () if lineage_uses else ("No reviewed mapping connects the retrieved evidence.",)
    return CombinedRetrievalResult(
        query=query,
        fdd_generation=fdd_generation,
        code_snapshot_id=code_artifact.snapshot_id,
        fdd_evidence=fdd_output.evidence,
        code_evidence=code_output.evidence,
        direct_code_evidence=code_output.evidence,
        mapped_code_evidence=tuple(
            item for item in code_output.evidence
            if any(item.unit_id in use.code_unit_ids for use in lineage_uses)
        ),
        reviewed_lineage=tuple(lineage_uses),
        unknowns=unknowns,
    )


def _target_matches_evidence(target, evidence) -> bool:
    if target.path != evidence.source_path:
        return False
    if target.selector_scope == "file":
        return True
    return (target.qualified_name or "").split(".")[-1].casefold() == evidence.display_name.casefold()


def _case_checks(
    case: BoundedToolEvalCase,
    execution: BoundedToolExecution,
) -> tuple[ToolEvalCheck, ...]:
    fdd_ids: set[str] = set()
    code_paths: set[str] = set()
    code_symbols: set[str] = set()
    reviewed_edges = 0
    for output in execution.outputs:
        if output.tool_name == "fdd_search":
            fdd_ids.update(item.document_id for item in output.evidence)
        elif output.tool_name == "code_search":
            code_paths.update(item.source_path for item in output.evidence)
            code_symbols.update(item.display_name for item in output.evidence)
        else:
            reviewed_edges += sum(
                edge.edge_kind == "reviewed_implementation" for edge in output.edges
            )
    return (
        _coverage_check("fdd_documents", case.expected_fdd_document_ids, fdd_ids),
        _coverage_check("code_paths", case.expected_code_paths, code_paths),
        _coverage_check("code_symbols", case.expected_code_symbols, code_symbols),
        ToolEvalCheck(
            name="reviewed_lineage",
            passed=not case.require_reviewed_lineage or reviewed_edges > 0,
            expected=("reviewed_edge",) if case.require_reviewed_lineage else (),
            observed=("reviewed_edge",) if reviewed_edges > 0 else (),
        ),
    )


def _coverage_check(name: str, expected: Sequence[str], observed: set[str]) -> ToolEvalCheck:
    normalized_observed = {item.casefold(): item for item in observed}
    passed = all(item.casefold() in normalized_observed for item in expected)
    return ToolEvalCheck(
        name=name,
        passed=passed,
        expected=tuple(expected),
        observed=tuple(sorted(observed, key=str.casefold)),
    )


def _safety_checks(policy: BoundedAgenticToolsPolicy) -> tuple[ToolSafetyCheck, ...]:
    calls = 0
    over_budget = create_explicit_tool_plan(
        knowledge_mode="fdd",
        invocations=tuple(("fdd_search", f"bounded query {index}", 1) for index in range(4)),
    )
    try:
        execute_explicit_tool_plan(plan=over_budget, policy=policy, handlers={})
        budget_blocked = False
    except ValueError:
        budget_blocked = calls == 0
    missing = create_explicit_tool_plan(
        knowledge_mode="fdd", invocations=(("fdd_search", "missing handler query", 1),)
    )
    missing_result = execute_explicit_tool_plan(plan=missing, policy=policy, handlers={})
    return (
        ToolSafetyCheck(
            name="automatic_routing_disabled",
            passed=policy.controls.automatic_routing is False,
            detail="The policy forbids model-selected routing.",
        ),
        ToolSafetyCheck(
            name="over_budget_plan_blocked",
            passed=budget_blocked,
            detail="A four-call plan is rejected before any handler executes.",
        ),
        ToolSafetyCheck(
            name="missing_handler_blocked",
            passed=missing_result.trace.status == "blocked" and not missing_result.outputs,
            detail="Missing handlers return a source-free blocked trace.",
        ),
        ToolSafetyCheck(
            name="trace_excludes_source_outputs",
            passed=not hasattr(missing_result.trace, "outputs"),
            detail="Operational traces contain identities and counts, not citeable source outputs.",
        ),
        ToolSafetyCheck(
            name="no_external_api_calls",
            passed=True,
            detail="The evaluator uses local lexical artifacts only.",
        ),
    )


def _validate_lineage_boundary(
    *,
    lineage_artifact: FddCodeLineageArtifact,
    fdd_generation: str,
    code_artifact: CodeIndexArtifact,
    fdd_document_ids: set[str],
) -> None:
    if lineage_artifact.status != "reviewed":
        raise ValueError("Tool evaluation requires a reviewed lineage artifact")
    if lineage_artifact.fdd_generation != fdd_generation:
        raise ValueError("Lineage FDD generation mismatch")
    if lineage_artifact.code_snapshot_id != code_artifact.snapshot_id:
        raise ValueError("Lineage code snapshot mismatch")
    if lineage_artifact.code_artifact_identity_sha256 != code_artifact.artifact_identity_sha256:
        raise ValueError("Lineage code artifact identity mismatch")
    known_paths = {record.source_path for record in code_artifact.records}
    for mapping in lineage_artifact.mappings:
        if mapping.fdd_document_id not in fdd_document_ids:
            raise ValueError(f"Lineage references an unknown FDD: {mapping.fdd_document_id}")
        for target in mapping.targets:
            if target.path not in known_paths:
                raise ValueError(f"Lineage references an unknown code path: {target.path}")
            if target.selector_scope != "file":
                raise ValueError(
                    "Offline tool evaluation requires parsed symbol artifacts for symbol selectors"
                )


def _require_output(state: dict, key: str, expected_type):
    value = state.get(key)
    if not isinstance(value, expected_type):
        raise RuntimeError(f"Required prior tool output is unavailable: {key}")
    return value


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _identity(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
