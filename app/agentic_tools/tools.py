from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from itertools import islice
import re
from typing import Any

from app.agentic_tools.models import (
    CodeSearchToolResult,
    FddSearchToolResult,
    ImpactGraphEdge,
    ImpactGraphNode,
    ImpactGraphToolResult,
    ToolInvocation,
    identity,
)
from app.code_ingestion.code_analysis_models import CodeStaticAnalysisArtifact
from app.code_retrieval.models import CodeRetrievalResult
from app.fdd_code_lineage.combined_retrieval import (
    CombinedRetrievalResult,
    FddEvidence,
)


FddSearchRunner = Callable[[str, int], Iterable[Any]]
CodeSearchRunner = Callable[[str, int], CodeRetrievalResult]


_IDENTIFIER_TOKEN_PATTERN = re.compile(
    r"[A-Z]+(?=[A-Z][a-z]|\d|$)|[A-Z]?[a-z]+|\d+"
)
_IDENTIFIER_ALIASES = {
    "txn": "transaction",
    "txns": "transaction",
    "sent": "send",
    "sending": "send",
}


def run_fdd_search_tool(
    invocation: ToolInvocation,
    *,
    search_runner: FddSearchRunner,
) -> FddSearchToolResult:
    """Execute one bounded FDD search through an injected read-only retriever."""

    if invocation.tool_name != "fdd_search":
        raise ValueError("FDD search tool received the wrong invocation type")
    raw_results = tuple(
        islice(search_runner(invocation.query, invocation.limit), invocation.limit + 1)
    )
    evidence = tuple(_fdd_evidence(item) for item in raw_results[:invocation.limit])
    return FddSearchToolResult(
        query=invocation.query,
        evidence=evidence,
        truncated=len(raw_results) > invocation.limit,
    )


def run_code_search_tool(
    invocation: ToolInvocation,
    *,
    search_runner: CodeSearchRunner,
    reserve_identifier_affinity: bool = False,
) -> CodeSearchToolResult:
    """Execute one bounded code search without embedding or generation side effects."""

    if invocation.tool_name != "code_search":
        raise ValueError("Code search tool received the wrong invocation type")
    retrieval = search_runner(invocation.query, invocation.limit + 1)
    if retrieval.query != invocation.query:
        raise RuntimeError("Code retriever returned evidence for a different query")
    evidence = retrieval.evidence
    if reserve_identifier_affinity:
        evidence = select_identifier_affinity_evidence(
            query=invocation.query,
            evidence=evidence,
            limit=invocation.limit,
        )
    else:
        evidence = evidence[: invocation.limit]
    return CodeSearchToolResult(
        query=invocation.query,
        snapshot_id=retrieval.snapshot_id,
        evidence=evidence,
        truncated=len(retrieval.evidence) > invocation.limit,
    )


def select_identifier_affinity_evidence(
    *, query: str, evidence: Sequence[Any], limit: int, minimum_matches: int = 3
) -> tuple[Any, ...]:
    """Reserve one bounded slot for a strong routine-name/query match.

    This operates only on already retrieved candidates. It does not increase the
    evidence budget, modify retrieval scores, or manufacture source evidence.
    """

    if limit <= 0 or minimum_matches <= 0:
        raise ValueError("Identifier-affinity selection bounds must be positive")
    if len(evidence) <= limit:
        return tuple(evidence)
    selected = list(evidence[:limit])
    query_tokens = _normalized_identifier_tokens(query)
    if not query_tokens:
        return tuple(selected)
    scored = [
        (len(query_tokens & _normalized_identifier_tokens(item.display_name)), rank, item)
        for rank, item in enumerate(evidence)
    ]
    best_matches, _, best = max(scored, key=lambda item: (item[0], -item[1]))
    if best_matches < minimum_matches or any(item.unit_id == best.unit_id for item in selected):
        return tuple(selected)
    weakest_matches = min(
        len(query_tokens & _normalized_identifier_tokens(item.display_name))
        for item in selected
    )
    if best_matches <= weakest_matches:
        return tuple(selected)
    selected[-1] = best
    return tuple(selected)


def _normalized_identifier_tokens(value: str) -> set[str]:
    expanded = value.replace("_", " ").replace("$", " ")
    raw = _IDENTIFIER_TOKEN_PATTERN.findall(expanded)
    return {_IDENTIFIER_ALIASES.get(token.casefold(), token.casefold()) for token in raw}


def run_impact_graph_tool(
    invocation: ToolInvocation,
    *,
    combined_retrieval: CombinedRetrievalResult,
    analyses: Sequence[CodeStaticAnalysisArtifact],
    max_nodes: int,
    max_edges: int,
) -> ImpactGraphToolResult:
    """Build a one-hop, evidence-bound graph from reviewed lineage and static edges."""

    if invocation.tool_name != "impact_graph":
        raise ValueError("Impact graph tool received the wrong invocation type")
    if combined_retrieval.query != invocation.query:
        raise RuntimeError("Impact graph input belongs to a different query")
    if max_nodes < 1 or max_edges < 1:
        raise ValueError("Impact graph bounds must be positive")
    edge_limit = min(max_edges, invocation.limit)

    nodes: dict[str, ImpactGraphNode] = {}
    edges: dict[str, ImpactGraphEdge] = {}
    truncated = False
    fdd_by_document = {
        item.document_id: item for item in combined_retrieval.fdd_evidence
    }
    code_by_unit = {item.unit_id: item for item in combined_retrieval.code_evidence}

    for lineage_use in sorted(
        combined_retrieval.reviewed_lineage, key=lambda item: item.mapping_id
    ):
        fdd = fdd_by_document.get(lineage_use.fdd_document_id)
        if fdd is None:
            continue
        fdd_node_id = f"fdd:{fdd.document_id}"
        if fdd_node_id not in nodes:
            if len(nodes) >= max_nodes:
                truncated = True
                continue
            nodes[fdd_node_id] = ImpactGraphNode(
                node_id=fdd_node_id,
                node_kind="fdd_document",
                label=fdd.document_id,
                source_identity=fdd.unit_id,
            )
        for unit_id in sorted(lineage_use.code_unit_ids):
            code = code_by_unit.get(unit_id)
            if code is None:
                continue
            code_node_id = f"code:{code.unit_id}"
            if code_node_id not in nodes:
                if len(nodes) >= max_nodes:
                    truncated = True
                    continue
                nodes[code_node_id] = _code_node(code)
            if len(edges) >= edge_limit:
                truncated = True
                continue
            edge_id = identity(
                {
                    "kind": "reviewed_implementation",
                    "mapping_id": lineage_use.mapping_id,
                    "fdd": fdd_node_id,
                    "code": code_node_id,
                }
            )
            edges[edge_id] = ImpactGraphEdge(
                edge_id=edge_id,
                edge_kind="reviewed_implementation",
                source_node_id=fdd_node_id,
                target_node_id=code_node_id,
                resolution_state="reviewed",
                evidence_identity=lineage_use.mapping_id,
            )

    analyses_by_path = {item.source_path: item for item in analyses}
    for code in sorted(code_by_unit.values(), key=lambda item: item.unit_id):
        code_node_id = f"code:{code.unit_id}"
        if code_node_id not in nodes:
            if len(nodes) >= max_nodes:
                truncated = True
                continue
            nodes[code_node_id] = _code_node(code)
        analysis = analyses_by_path.get(code.source_path)
        if analysis is None:
            continue
        for dependency in analysis.dependencies:
            if (
                dependency.source_map.start_line > code.end_line
                or dependency.source_map.end_line < code.start_line
            ):
                continue
            if len(edges) >= edge_limit:
                truncated = True
                break
            target_node_id = "dependency:" + identity(
                {
                    "name": dependency.target_canonical_name,
                    "state": dependency.resolution_state,
                }
            )
            if target_node_id not in nodes:
                if len(nodes) >= max_nodes:
                    truncated = True
                    continue
                nodes[target_node_id] = ImpactGraphNode(
                    node_id=target_node_id,
                    node_kind="static_dependency",
                    label=dependency.target_display_name,
                    source_identity=dependency.edge_id,
                )
            edges[dependency.edge_id] = ImpactGraphEdge(
                edge_id=dependency.edge_id,
                edge_kind=dependency.dependency_kind,
                source_node_id=code_node_id,
                target_node_id=target_node_id,
                resolution_state=dependency.resolution_state,
                evidence_identity=dependency.edge_id,
            )

    selected_nodes = tuple(sorted(
        nodes.values(),
        key=lambda item: (
            {"fdd_document": 0, "code_unit": 1, "static_dependency": 2}[item.node_kind],
            item.node_id,
        ),
    ))
    selected_edges = tuple(sorted(edges.values(), key=lambda item: item.edge_id))
    unknowns = list(combined_retrieval.unknowns)
    if not combined_retrieval.reviewed_lineage:
        unknowns.append("No reviewed lineage edges are available for this evidence set.")
    if _requests_unavailable_kernel_implementation(invocation.query):
        unknowns.append(
            "The requested hidden Java/kernel implementation and exact defect line "
            "are unavailable in the approved custom PL/SQL snapshot."
        )
    return ImpactGraphToolResult(
        query=invocation.query,
        nodes=selected_nodes,
        edges=selected_edges,
        truncated=truncated,
        unknowns=tuple(dict.fromkeys(unknowns)),
    )


def _fdd_evidence(result: Any) -> FddEvidence:
    payload = dict(result.payload)
    required = (
        "unit_id", "document_id", "document_family", "release_label", "source_kind", "text"
    )
    missing = [name for name in required if not str(payload.get(name, "")).strip()]
    if missing:
        raise RuntimeError(f"FDD tool evidence is missing required identity: {missing}")
    return FddEvidence(
        unit_id=str(payload["unit_id"]),
        document_id=str(payload["document_id"]),
        document_family=str(payload["document_family"]),
        release_label=str(payload["release_label"]),
        source_kind=str(payload["source_kind"]),
        score=float(result.score),
        text=str(payload["text"]),
    )


def _code_node(code) -> ImpactGraphNode:
    return ImpactGraphNode(
        node_id=f"code:{code.unit_id}",
        node_kind="code_unit",
        label=f"{code.source_path}:{code.display_name}",
        source_identity=code.unit_id,
    )


def _requests_unavailable_kernel_implementation(query: str) -> bool:
    tokens = _normalized_identifier_tokens(query)
    boundary = bool(tokens & {"kernel", "java", "j2ee"})
    unavailable_scope = bool(tokens & {"hidden", "unavailable", "internal"})
    implementation_detail = bool(tokens & {"method", "implementation", "line", "defect"})
    return boundary and unavailable_scope and implementation_detail
