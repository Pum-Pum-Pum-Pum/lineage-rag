from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from app.agentic_tools.models import create_explicit_tool_plan
from app.agentic_tools.orchestration import execute_explicit_tool_plan
from app.agentic_tools.policy import load_agentic_tools_policy
from app.agentic_tools.tools import (
    run_code_search_tool,
    run_fdd_search_tool,
    run_impact_graph_tool,
    select_identifier_affinity_evidence,
)
from app.code_ingestion.code_analysis_models import (
    CodeStaticAnalysisArtifact,
    DependencyEdge,
)
from app.code_ingestion.plsql_models import SourceMap
from app.code_retrieval.models import CodeEvidence, CodeRetrievalResult
from app.fdd_code_lineage.combined_retrieval import (
    CombinedRetrievalResult,
    FddEvidence,
    ReviewedLineageUse,
)


def _fdd_result(index: int, *, complete: bool = True) -> SimpleNamespace:
    payload = {
        "unit_id": f"fdd-unit-{index}",
        "document_id": f"FDD-{index}",
        "document_family": "FDD-FAMILY",
        "release_label": "R24",
        "source_kind": "paragraph",
        "text": f"Documented behavior {index}",
    }
    if not complete:
        payload.pop("document_id")
    return SimpleNamespace(payload=payload, score=1.0 - index / 100)


def _code_evidence(index: int = 1) -> CodeEvidence:
    return CodeEvidence(
        unit_id=f"code-unit-{index}",
        point_id=f"point-{index}",
        score=1.0 - index / 100,
        retrieval_method="hybrid",
        snapshot_id="code-snapshot-r1",
        module_id="fci-custom",
        source_path="pkg_aml_custom.sql",
        source_kind="procedure",
        display_name=f"SP_PROCESS_{index}",
        start_line=10 * index,
        end_line=10 * index + 9,
        parser_state="full_parse",
        conditional_state="unconditional",
        text=f"PROCEDURE sp_process_{index} IS BEGIN NULL; END;",
    )


def _code_result(query: str, count: int = 3) -> CodeRetrievalResult:
    return CodeRetrievalResult(
        query=query,
        mode="hybrid",
        snapshot_id="code-snapshot-r1",
        artifact_identity_sha256="a" * 64,
        collection_name="code_custom_r1_v2",
        evidence=tuple(_code_evidence(index) for index in range(1, count + 1)),
    )


def _combined(query: str, *, reviewed: bool = True) -> CombinedRetrievalResult:
    code = _code_evidence()
    return CombinedRetrievalResult(
        query=query,
        fdd_generation="functional_specs_v5",
        code_snapshot_id="code-snapshot-r1",
        fdd_evidence=(
            FddEvidence(
                unit_id="fdd-unit-1",
                document_id="FDD-1",
                document_family="FDD-FAMILY",
                release_label="R24",
                source_kind="paragraph",
                score=0.9,
                text="Documented AML behavior.",
            ),
        ),
        code_evidence=(code,),
        direct_code_evidence=(code,),
        mapped_code_evidence=(code,) if reviewed else (),
        reviewed_lineage=(
            ReviewedLineageUse(
                mapping_id="b" * 64,
                fdd_document_id="FDD-1",
                code_unit_ids=(code.unit_id,),
            ),
        )
        if reviewed
        else (),
    )


def _analysis() -> CodeStaticAnalysisArtifact:
    source_map = SourceMap(
        source_path="pkg_aml_custom.sql",
        start_line=12,
        end_line=12,
        start_offset=100,
        end_offset=120,
    )
    dependency = DependencyEdge(
        edge_id="c" * 64,
        dependency_kind="kernel_boundary",
        source_path="pkg_aml_custom.sql",
        source_map=source_map,
        target_display_name="PKG_AML_KERNEL.SEND",
        target_canonical_name="PKG_AML_KERNEL.SEND",
        resolution_state="kernel_unavailable",
        extraction_method="antlr_tree",
        confidence="high",
    )
    return CodeStaticAnalysisArtifact(
        module_id="fci-custom",
        snapshot_id="code-snapshot-r1",
        source_path="pkg_aml_custom.sql",
        source_sha256="d" * 64,
        analysis_policy_sha256="e" * 64,
        parser_state="full_parse",
        dependencies=(dependency,),
    )


def test_policy_is_hash_stable_and_forbids_automatic_routing(tmp_path: Path) -> None:
    policy = load_agentic_tools_policy()
    assert policy.controls.automatic_routing is False
    assert policy.sha256 == load_agentic_tools_policy().sha256

    unsafe = tmp_path / "unsafe.toml"
    unsafe.write_text(
        """schema_version = "bounded_agentic_tools_policy_v1"
[budgets]
max_calls = 3
max_results_per_call = 8
max_total_evidence_units = 16
max_impact_nodes = 24
max_impact_edges = 32
[controls]
automatic_routing = true
fdd_tools = ["fdd_search"]
code_tools = ["code_search"]
combined_tools = ["fdd_search", "code_search", "impact_graph"]
""",
        encoding="utf-8",
    )
    with pytest.raises(ValidationError):
        load_agentic_tools_policy(unsafe)


def test_plan_identity_and_mode_budget_fail_before_any_tool_call() -> None:
    policy = load_agentic_tools_policy()
    wrong_mode = create_explicit_tool_plan(
        knowledge_mode="fdd", invocations=(("code_search", "find code", 1),)
    )
    calls = 0

    def forbidden_handler():
        nonlocal calls
        calls += 1
        raise AssertionError("handler must not run")

    with pytest.raises(ValueError, match="not allowed"):
        execute_explicit_tool_plan(
            plan=wrong_mode,
            policy=policy,
            handlers={wrong_mode.invocations[0].invocation_id: forbidden_handler},
        )
    assert calls == 0

    with pytest.raises(ValidationError, match="identity"):
        wrong_mode.model_copy(
            update={"invocations": (wrong_mode.invocations[0].model_copy(update={"query": "tampered"}),)}
        ).model_validate(wrong_mode.model_copy(
            update={"invocations": (wrong_mode.invocations[0].model_copy(update={"query": "tampered"}),)}
        ).model_dump())


def test_fdd_search_is_bounded_and_requires_complete_citation_identity() -> None:
    plan = create_explicit_tool_plan(
        knowledge_mode="fdd", invocations=(("fdd_search", "AML behavior", 2),)
    )
    invocation = plan.invocations[0]
    result = run_fdd_search_tool(
        invocation, search_runner=lambda query, limit: [_fdd_result(i) for i in range(4)]
    )
    assert len(result.evidence) == 2
    assert result.truncated is True
    assert result.evidence[0].document_id == "FDD-0"

    produced = 0

    def excessive_results(query: str, limit: int):
        nonlocal produced
        index = 0
        while True:
            produced += 1
            yield _fdd_result(index)
            index += 1

    run_fdd_search_tool(invocation, search_runner=excessive_results)
    assert produced == invocation.limit + 1

    with pytest.raises(RuntimeError, match="document_id"):
        run_fdd_search_tool(
            invocation, search_runner=lambda query, limit: [_fdd_result(1, complete=False)]
        )


def test_code_search_is_bounded_and_rejects_query_mismatch() -> None:
    plan = create_explicit_tool_plan(
        knowledge_mode="code", invocations=(("code_search", "AML routine", 2),)
    )
    invocation = plan.invocations[0]
    result = run_code_search_tool(
        invocation, search_runner=lambda query, limit: _code_result(query, count=3)
    )
    assert len(result.evidence) == 2
    assert result.truncated is True
    assert result.snapshot_id == "code-snapshot-r1"

    with pytest.raises(RuntimeError, match="different query"):
        run_code_search_tool(
            invocation, search_runner=lambda query, limit: _code_result("other query")
        )


def test_combined_identifier_affinity_reserves_one_bounded_symbol_slot() -> None:
    evidence = tuple(
        _code_evidence(index).model_copy(update={"display_name": name})
        for index, name in enumerate(
            (
                "spRealtimeSubsTransaction",
                "spUHEndPoint",
                "spSMApprovalTxn",
                "spBatchTxnEndPoint",
                "spBatchTxnEventEndPoint",
                "spBatchUHEndPoint",
                "spBatchUHOffline",
                "spOfflineJobsWrapper",
                "spPopulateTxnList",
                "spSendBatchTxnEndData",
                "spSendBatchTxnEventData",
            ),
            start=1,
        )
    )
    selected = select_identifier_affinity_evidence(
        query="How is batch transaction data sent to FlagRight?",
        evidence=evidence,
        limit=8,
    )
    assert len(selected) == 8
    assert selected[-1].display_name == "spSendBatchTxnEndData"
    assert [item.display_name for item in selected[:7]] == [
        item.display_name for item in evidence[:7]
    ]

    unchanged = select_identifier_affinity_evidence(
        query="Explain visible implementation behavior",
        evidence=evidence,
        limit=8,
    )
    assert unchanged == evidence[:8]

    with pytest.raises(ValueError, match="bounds"):
        select_identifier_affinity_evidence(query="batch", evidence=evidence, limit=0)


def test_impact_graph_uses_reviewed_lineage_and_preserves_static_unknowns() -> None:
    query = "Explain AML impact"
    plan = create_explicit_tool_plan(
        knowledge_mode="combined", invocations=(("impact_graph", query, 8),)
    )
    result = run_impact_graph_tool(
        plan.invocations[0],
        combined_retrieval=_combined(query),
        analyses=(_analysis(),),
        max_nodes=10,
        max_edges=10,
    )
    assert {edge.edge_kind for edge in result.edges} == {
        "reviewed_implementation", "kernel_boundary"
    }
    kernel = next(edge for edge in result.edges if edge.edge_kind == "kernel_boundary")
    assert kernel.resolution_state == "kernel_unavailable"
    assert any(node.node_kind == "fdd_document" for node in result.nodes)


def test_impact_graph_without_reviewed_mapping_does_not_invent_lineage() -> None:
    query = "Explain AML impact"
    plan = create_explicit_tool_plan(
        knowledge_mode="combined", invocations=(("impact_graph", query, 8),)
    )
    result = run_impact_graph_tool(
        plan.invocations[0],
        combined_retrieval=_combined(query, reviewed=False),
        analyses=(_analysis(),),
        max_nodes=10,
        max_edges=10,
    )
    assert not any(edge.edge_kind == "reviewed_implementation" for edge in result.edges)
    assert "No reviewed lineage edges" in result.unknowns[-1]


def test_impact_graph_qualifies_requested_hidden_kernel_detail() -> None:
    query = "What exact hidden Java kernel method opens the connection and which defect line must be fixed?"
    plan = create_explicit_tool_plan(
        knowledge_mode="combined", invocations=(("impact_graph", query, 8),)
    )
    result = run_impact_graph_tool(
        plan.invocations[0],
        combined_retrieval=_combined(query),
        analyses=(),
        max_nodes=10,
        max_edges=10,
    )
    assert any("unavailable" in unknown.casefold() for unknown in result.unknowns)
    assert any("exact defect line" in unknown.casefold() for unknown in result.unknowns)

    ordinary = "Explain the visible custom connection implementation"
    ordinary_plan = create_explicit_tool_plan(
        knowledge_mode="combined", invocations=(("impact_graph", ordinary, 8),)
    )
    ordinary_result = run_impact_graph_tool(
        ordinary_plan.invocations[0],
        combined_retrieval=_combined(ordinary),
        analyses=(),
        max_nodes=10,
        max_edges=10,
    )
    assert not any("hidden java/kernel" in item.casefold() for item in ordinary_result.unknowns)


def test_impact_graph_enforces_node_and_edge_caps_during_construction() -> None:
    query = "Explain AML impact"
    plan = create_explicit_tool_plan(
        knowledge_mode="combined", invocations=(("impact_graph", query, 8),)
    )
    result = run_impact_graph_tool(
        plan.invocations[0],
        combined_retrieval=_combined(query),
        analyses=(_analysis(),),
        max_nodes=2,
        max_edges=1,
    )
    assert len(result.nodes) == 2
    assert len(result.edges) == 1
    assert result.truncated is True


def test_explicit_orchestration_stops_on_failure_and_trace_omits_source_text() -> None:
    policy = load_agentic_tools_policy()
    plan = create_explicit_tool_plan(
        knowledge_mode="combined",
        invocations=(
            ("fdd_search", "AML behavior", 2),
            ("code_search", "AML behavior", 2),
            ("impact_graph", "AML behavior", 2),
        ),
    )
    first, second, third = plan.invocations
    third_called = False

    def fail_code():
        raise RuntimeError("sensitive source must not enter trace")

    def forbidden_third():
        nonlocal third_called
        third_called = True
        raise AssertionError("later handler must not run")

    execution = execute_explicit_tool_plan(
        plan=plan,
        policy=policy,
        handlers={
            first.invocation_id: lambda: run_fdd_search_tool(
                first, search_runner=lambda query, limit: [_fdd_result(1)]
            ),
            second.invocation_id: fail_code,
            third.invocation_id: forbidden_third,
        },
    )
    assert execution.trace.status == "failed"
    assert [call.status for call in execution.trace.calls] == ["completed", "failed"]
    assert execution.trace.calls[-1].error_type == "RuntimeError"
    assert third_called is False
    assert "Documented behavior" not in execution.trace.model_dump_json()


def test_missing_tool_handler_returns_blocked_trace_without_partial_claims() -> None:
    policy = load_agentic_tools_policy()
    plan = create_explicit_tool_plan(
        knowledge_mode="fdd", invocations=(("fdd_search", "AML behavior", 1),)
    )
    execution = execute_explicit_tool_plan(plan=plan, policy=policy, handlers={})
    assert execution.trace.status == "blocked"
    assert execution.trace.calls[0].error_type == "MissingToolHandler"
    assert execution.outputs == ()
    assert execution.trace.total_evidence_units == 0


def test_complete_explicit_plan_records_identity_counts_without_automatic_routing() -> None:
    policy = load_agentic_tools_policy()
    query = "Explain AML impact"
    plan = create_explicit_tool_plan(
        knowledge_mode="combined",
        invocations=(
            ("fdd_search", query, 1),
            ("code_search", query, 1),
            ("impact_graph", query, 8),
        ),
    )
    fdd_call, code_call, graph_call = plan.invocations
    execution = execute_explicit_tool_plan(
        plan=plan,
        policy=policy,
        handlers={
            fdd_call.invocation_id: lambda: run_fdd_search_tool(
                fdd_call, search_runner=lambda requested_query, limit: [_fdd_result(1)]
            ),
            code_call.invocation_id: lambda: run_code_search_tool(
                code_call, search_runner=lambda requested_query, limit: _code_result(requested_query, 1)
            ),
            graph_call.invocation_id: lambda: run_impact_graph_tool(
                graph_call,
                combined_retrieval=_combined(query),
                analyses=(_analysis(),),
                max_nodes=policy.budgets.max_impact_nodes,
                max_edges=policy.budgets.max_impact_edges,
            ),
        },
    )
    assert execution.trace.status == "completed"
    assert execution.trace.total_evidence_units == 2
    assert execution.trace.automatic_routing_used is False
    assert len(execution.trace.calls) == 3
    assert all(call.evidence_identities for call in execution.trace.calls)
