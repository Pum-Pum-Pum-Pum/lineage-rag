from __future__ import annotations

from types import SimpleNamespace

from app.agentic_tools.uat import LocalToolUatReport, ManualToolUatCase
from app.code_indexing.models import CodeIndexArtifact
from app.code_retrieval.models import CodeRetrievalResult
from app.fdd_code_lineage.combined_answer import CombinedAnswerResponse
from app.fdd_code_lineage.combined_retrieval import (
    CombinedRetrievalResult,
    ReviewedLineageUse,
)
from app.fdd_code_lineage.paid_evaluation import evaluate_answer_structure


def build_paid_case(case: ManualToolUatCase) -> SimpleNamespace:
    return SimpleNamespace(
        mode="code" if case.knowledge_mode == "code" else "combined",
        question=case.question,
        analysis_kind=(
            "impact_analysis" if "impact" in case.case_id else "explanation"
        ),
        expected_unknown_kinds=(),
        should_abstain=case.expected_outcome == "qualified_unknown",
        expected_fdd_document_ids=case.expected_fdd_document_ids,
        expected_code_paths=case.expected_code_paths,
        expected_code_symbols=case.expected_code_symbols,
        expected_code_symbol_policy="all",
    )


def retrieval_from_local_uat(
    *, case: ManualToolUatCase, report: LocalToolUatReport, artifact: CodeIndexArtifact
) -> CodeRetrievalResult | CombinedRetrievalResult:
    outputs = {item.tool_name: item for item in report.execution.outputs}
    code_output = outputs.get("code_search")
    if case.knowledge_mode == "code":
        if code_output is None:
            raise RuntimeError("Code UAT report has no code-search output")
        return CodeRetrievalResult(
            query=case.question,
            mode="lexical",
            snapshot_id=artifact.snapshot_id,
            artifact_identity_sha256=artifact.artifact_identity_sha256,
            evidence=code_output.evidence,
        )

    fdd_output = outputs.get("fdd_search")
    if fdd_output is None:
        raise RuntimeError("FDD/combined UAT report has no FDD-search output")
    code_evidence = code_output.evidence if code_output is not None else ()
    graph = outputs.get("impact_graph")
    lineage: list[ReviewedLineageUse] = []
    unknowns: tuple[str, ...]
    if graph is None:
        unknowns = ("FDD-only mode did not request custom-code implementation evidence.",)
    else:
        grouped: dict[tuple[str, str], list[str]] = {}
        for edge in graph.edges:
            if edge.edge_kind != "reviewed_implementation":
                continue
            document_id = edge.source_node_id.removeprefix("fdd:")
            unit_id = edge.target_node_id.removeprefix("code:")
            grouped.setdefault((edge.evidence_identity, document_id), []).append(unit_id)
        lineage = [
            ReviewedLineageUse(
                mapping_id=mapping_id,
                fdd_document_id=document_id,
                code_unit_ids=tuple(sorted(set(unit_ids))),
            )
            for (mapping_id, document_id), unit_ids in sorted(grouped.items())
        ]
        unknowns = graph.unknowns
    mapped_ids = {unit for item in lineage for unit in item.code_unit_ids}
    return CombinedRetrievalResult(
        query=case.question,
        fdd_generation=report.fdd_generation,
        code_snapshot_id=artifact.snapshot_id,
        fdd_evidence=fdd_output.evidence,
        code_evidence=code_evidence,
        direct_code_evidence=code_evidence,
        mapped_code_evidence=tuple(
            item for item in code_evidence if item.unit_id in mapped_ids
        ),
        reviewed_lineage=tuple(lineage),
        unknowns=unknowns,
    )


def evaluate_paid_uat_answer(*, case: ManualToolUatCase, answer) -> dict:
    paid_case = build_paid_case(case)
    if case.knowledge_mode != "fdd":
        return evaluate_answer_structure(case=paid_case, answer=answer)
    failures: list[str] = []
    if not isinstance(answer, CombinedAnswerResponse):
        failures.append("FDD-only evaluation returned the wrong response contract")
    else:
        if not answer.requested_claim_supported:
            failures.append("Expected the requested FDD claim to be supported")
        if answer.documented_functionality.status != "answered":
            failures.append("Documented functionality did not answer")
        cited = {item.document_id for item in answer.fdd_citations}
        missing = sorted(set(case.expected_fdd_document_ids) - cited)
        if missing:
            failures.append(f"Missing expected cited FDD documents: {missing}")
    return {
        "passed": not failures,
        "failures": failures,
        "semantic_sme_review_required": True,
    }
