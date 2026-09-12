"""Bounded, offline workflow context for explicit PL/SQL routine questions.

This module deliberately does not decide that an FDD is implemented.  It only
finds a uniquely named implementation routine already retrieved for the caller,
adds a small amount of caller context, and proposes lexical FDD evidence from
business text visible in that context.  Review is still required before a
candidate relationship becomes lineage.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import re
from pathlib import Path
from typing import Sequence

from app.code_indexing.models import CodeIndexArtifact, CodeIndexRecord
from app.code_ingestion.code_analysis_models import CodeStaticAnalysisArtifact, CodeSymbol
from app.code_ingestion.plsql_models import SourceMap
from app.code_retrieval.models import CodeEvidence
from app.retrieval.lexical_search import (
    LexicalSearchDocument,
    search_lexical_documents,
    tokenize,
)
from app.vectorstore.qdrant_search import QdrantSearchResult


MAX_CALLERS = 2
MAX_VALIDATION_CONTEXTS = 2
MAX_FDD_CANDIDATES = 3
MAX_WORKFLOW_TERMS = 40

# SQL/PLSQL mechanics do not help find a functional requirement.  This is a
# conservative deny-list, not an attempt to infer business aliases.
_NON_BUSINESS_TERMS = {
    "and", "begin", "code", "custom", "delete", "else", "end", "error",
    "exception", "from", "function", "into", "is", "not", "null", "or",
    "procedure", "select", "table", "then", "when", "where", "with",
    "pkg", "str", "tbl", "custom", "main", "info", "data", "record",
}
_COMMON_CONTEXT_TERMS = {
    "again", "before", "coming", "count", "creation", "deleted", "else",
    "field", "for", "handle", "handled", "here", "inside", "inisde",
    "mode", "new", "no_data_found", "others", "reason", "rows", "service",
    "the", "this", "that", "user", "web", "when", "with", "back",
}
_STRING_LITERAL = re.compile(r"'(?:''|[^'])*'")


@dataclass(frozen=True)
class PackageInventory:
    """Complete parsed implementation inventory for one package/source file."""

    source_path: str
    procedures: tuple[str, ...]
    functions: tuple[str, ...]
    parser_states: tuple[str, ...]


@dataclass(frozen=True)
class ResolvedCaller:
    """One statically resolved inbound call and the source range of that call."""

    symbol: CodeSymbol
    call_site: SourceMap


@dataclass(frozen=True)
class WorkflowDiscovery:
    """Unreviewed context derived for one exact routine identifier."""

    target: CodeSymbol
    caller_contexts: tuple[ResolvedCaller, ...]
    caller_evidence: tuple[CodeEvidence, ...]
    validation_evidence: tuple[CodeEvidence, ...]
    fdd_candidates: tuple[QdrantSearchResult, ...]
    status: str

    @property
    def caller_symbols(self) -> tuple[CodeSymbol, ...]:
        """Symbols retained for backward-readable workflow diagnostics."""

        return tuple(item.symbol for item in self.caller_contexts)


def enumerate_package_inventory(
    *, analysis_directory: Path, source_path: str
) -> PackageInventory:
    """Return every parsed implementation procedure/function in a source file.

    This is parser inventory, rather than ranked search.  It therefore makes no
    claim that a declaration is executable if the parse stage did not emit a
    corresponding implementation occurrence.
    """

    analyses = _load_analyses(analysis_directory)
    analysis = analyses.get(source_path)
    if analysis is None:
        raise LookupError("Requested source is unavailable.")
    implementations = [
        item
        for item in analysis.symbols
        if item.occurrence_role == "implementation"
    ]
    procedures = sorted(
        {item.name.display_name for item in implementations if item.symbol_kind == "procedure"},
        key=str.casefold,
    )
    functions = sorted(
        {item.name.display_name for item in implementations if item.symbol_kind == "function"},
        key=str.casefold,
    )
    return PackageInventory(
        source_path=source_path,
        procedures=tuple(procedures),
        functions=tuple(functions),
        parser_states=(analysis.parser_state,),
    )


def inventory_for_explicit_package_query(
    *,
    query: str,
    direct_code_evidence: Sequence[CodeEvidence],
    analysis_directory: Path,
) -> PackageInventory | None:
    """Return a complete parser inventory only for one explicit retrieved file.

    This preserves the normal top-k code search contract. A filename must be
    written in the question *and* already be represented in direct code
    evidence, so the helper never turns a generic request into a package scan.
    """

    normalized_query = _normalize_identifier(query)
    source_paths = {
        item.source_path
        for item in direct_code_evidence
        if _normalize_identifier(Path(item.source_path).name) in normalized_query
    }
    if len(source_paths) != 1:
        return None
    return enumerate_package_inventory(
        analysis_directory=analysis_directory,
        source_path=next(iter(source_paths)),
    )


def discover_explicit_routine_workflow(
    *,
    query: str,
    direct_code_evidence: Sequence[CodeEvidence],
    code_artifact: CodeIndexArtifact,
    analysis_directory: Path,
    fdd_documents: Sequence[LexicalSearchDocument],
    fdd_limit: int,
) -> WorkflowDiscovery | None:
    """Discover bounded context only when the caller supplied one exact routine.

    A generic query must continue through the existing retrieval and ranking
    paths unchanged.  The direct evidence requirement also stops a guessed name
    from causing an arbitrary package scan.
    """

    if fdd_limit <= 0:
        raise ValueError("FDD result limit must be greater than zero")
    analyses = _load_analyses(analysis_directory)
    target = _unique_explicit_target(
        query=query,
        direct_code_evidence=direct_code_evidence,
        analyses=analyses,
    )
    if target is None:
        return None

    callers = _resolved_callers(target, analyses)[:MAX_CALLERS]
    records_by_id = {record.unit_id: record for record in code_artifact.records}
    target_evidence = next(
        (
            item
            for item in direct_code_evidence
            if _same_symbol_name(item.display_name, target.name.display_name)
            and item.source_path == target.source_path
        ),
        None,
    )
    if target_evidence is None:
        return None
    caller_evidence = tuple(
        item
        for caller in callers
        if (item := _caller_evidence(caller, records_by_id, target_evidence)) is not None
    )
    validation_evidence = _validation_context_evidence(
        target=target,
        callers=callers,
        records_by_id=records_by_id,
        target_evidence=target_evidence,
        already_selected=caller_evidence,
    )
    terms = _workflow_terms(
        target_evidence, (*caller_evidence, *validation_evidence)
    )
    candidates = _workflow_fdd_candidates(
        query=query,
        terms=terms,
        documents=fdd_documents,
        limit=min(MAX_FDD_CANDIDATES, fdd_limit),
    )
    return WorkflowDiscovery(
        target=target,
        caller_contexts=callers,
        caller_evidence=caller_evidence,
        validation_evidence=validation_evidence,
        fdd_candidates=candidates,
        status="unreviewed_documentation_candidate",
    )


def merge_workflow_fdd_candidates(
    *,
    existing: Sequence[QdrantSearchResult],
    candidates: Sequence[QdrantSearchResult],
    limit: int,
) -> list[QdrantSearchResult]:
    """Reserve bounded FDD evidence slots for explicit-routine context.

    Candidate results are marked with their relationship and do not change
    global FDD weights.  Their order is deterministic: primary lexical match,
    then immediately adjacent units, then normal retrieval output.
    """

    if limit <= 0:
        raise ValueError("Result limit must be greater than zero")
    merged: list[QdrantSearchResult] = []
    seen: set[str] = set()
    for result in [*candidates, *existing]:
        unit_id = str(result.payload.get("unit_id", ""))
        if not unit_id or unit_id in seen:
            continue
        seen.add(unit_id)
        merged.append(result)
        if len(merged) == limit:
            break
    return merged


def promote_workflow_code_context(
    *,
    current: Sequence[CodeEvidence],
    discovery: WorkflowDiscovery,
    limit: int,
) -> tuple[CodeEvidence, ...]:
    """Retain the exact routine plus at most two caller contexts within bounds."""

    if limit <= 0:
        raise ValueError("Code result limit must be greater than zero")
    preferred: list[CodeEvidence] = []
    for item in current:
        if _same_symbol_name(item.display_name, discovery.target.name.display_name):
            metadata = dict(item.retrieval_metadata)
            metadata.update(
                retrieval_relation="workflow_target_routine",
                workflow_status=discovery.status,
            )
            preferred.append(item.model_copy(update={"retrieval_metadata": metadata}))
            break
    for item in discovery.caller_evidence:
        preferred.append(item)
    for item in discovery.validation_evidence:
        preferred.append(item)
    seen: set[str] = set()
    result: list[CodeEvidence] = []
    for item in [*preferred, *current]:
        if item.unit_id in seen:
            continue
        seen.add(item.unit_id)
        result.append(item)
        if len(result) == limit:
            break
    return tuple(result)


def workflow_unknown_boundary(discovery: WorkflowDiscovery) -> str:
    """A truthful boundary to include in combined-answer unknowns."""

    caller_count = len(discovery.caller_symbols)
    return (
        "Workflow FDD evidence was discovered from the exact routine and "
        f"{caller_count} resolved caller context(s). It is an unreviewed "
        "documentation candidate, not proof that the FDD is implemented or "
        "that a documentation/code difference is a defect."
    )


def _load_analyses(directory: Path) -> dict[str, CodeStaticAnalysisArtifact]:
    artifact_directory = directory / "analysis" if (directory / "analysis").is_dir() else directory
    result: dict[str, CodeStaticAnalysisArtifact] = {}
    for path in sorted(artifact_directory.glob("*.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        if payload.get("schema_version") != "code_static_analysis_v1":
            continue
        analysis = CodeStaticAnalysisArtifact.model_validate(payload)
        if analysis.source_path in result:
            raise ValueError("Duplicate static-analysis artifact for source path")
        result[analysis.source_path] = analysis
    return result


def _unique_explicit_target(
    *,
    query: str,
    direct_code_evidence: Sequence[CodeEvidence],
    analyses: dict[str, CodeStaticAnalysisArtifact],
) -> CodeSymbol | None:
    normalized_query = _normalize_identifier(query)
    direct_names = {
        _normalize_identifier(item.display_name)
        for item in direct_code_evidence
        if _normalize_identifier(item.display_name) in normalized_query
    }
    if len(direct_names) != 1:
        return None
    name = next(iter(direct_names))
    matches = [
        symbol
        for analysis in analyses.values()
        for symbol in analysis.symbols
        if symbol.occurrence_role == "implementation"
        and _normalize_identifier(symbol.name.display_name) == name
    ]
    # A name collision across packages is not an exact enough selector.
    return matches[0] if len(matches) == 1 else None


def _resolved_callers(
    target: CodeSymbol,
    analyses: dict[str, CodeStaticAnalysisArtifact],
) -> tuple[ResolvedCaller, ...]:
    symbols = {
        symbol.occurrence_id: symbol
        for analysis in analyses.values()
        for symbol in analysis.symbols
        if symbol.occurrence_role == "implementation"
    }
    callers: list[tuple[str, ResolvedCaller]] = []
    for analysis in analyses.values():
        for edge in analysis.dependencies:
            if (
                edge.dependency_kind != "routine_call"
                or edge.resolution_state != "resolved_in_snapshot"
                or target.occurrence_id not in edge.candidate_symbol_occurrence_ids
                or edge.source_symbol_occurrence_id not in symbols
            ):
                continue
            callers.append(
                (
                    edge.edge_id,
                    ResolvedCaller(
                        symbol=symbols[edge.source_symbol_occurrence_id],
                        call_site=edge.source_map,
                    ),
                )
            )
    unique: dict[str, ResolvedCaller] = {}
    for _, caller in sorted(callers, key=lambda item: item[0]):
        unique.setdefault(caller.symbol.occurrence_id, caller)
    return tuple(unique.values())


def _caller_evidence(
    caller: ResolvedCaller,
    records_by_id: dict[str, CodeIndexRecord],
    target_evidence: CodeEvidence,
) -> CodeEvidence | None:
    matches = [
        record
        for record in records_by_id.values()
        if record.source_path == caller.symbol.source_path
        and _overlaps(record.source_map, caller.call_site)
    ]
    # A parent-only index record remains usable, but an exact child unit around
    # the invocation is always preferred.  The routine declaration/header is
    # deliberately not used as validation context.
    if not matches:
        matches = [
            record
            for record in records_by_id.values()
            if record.source_path == caller.symbol.source_path
            and _overlaps(record.parent_source_map or record.source_map, caller.call_site)
        ]
    if not matches:
        return None
    record = min(
        matches,
        key=lambda item: (
            abs(item.source_map.start_line - caller.call_site.start_line),
            item.unit_id,
        ),
    )
    metadata = dict(target_evidence.retrieval_metadata)
    metadata.update(
        retrieval_relation="workflow_caller_context",
        workflow_status="visible_implementation_context",
    )
    return CodeEvidence(
        unit_id=record.unit_id,
        point_id=record.point_id,
        score=target_evidence.score,
        retrieval_method=target_evidence.retrieval_method,
        snapshot_id=record.snapshot_id,
        module_id=record.module_id,
        source_path=record.source_path,
        source_kind=record.source_kind,
        display_name=record.display_name,
        parent_unit_id=record.parent_unit_id,
        package_name=record.package_name,
        start_line=record.source_map.start_line,
        end_line=record.source_map.end_line,
        parser_state=record.parser_state,
        conditional_state=record.conditional_state,
        text=record.citation_text,
        retrieval_metadata=metadata,
    )


def _validation_context_evidence(
    *,
    target: CodeSymbol,
    callers: Sequence[ResolvedCaller],
    records_by_id: dict[str, CodeIndexRecord],
    target_evidence: CodeEvidence,
    already_selected: Sequence[CodeEvidence],
) -> tuple[CodeEvidence, ...]:
    """Find at most two invocation-related validation blocks.

    A caller's declaration/header explains neither why the call is allowed nor
    why it is rejected. We therefore inspect only other chunks in the target's
    selected source file that explicitly name the target routine. That admits a
    pre-validation routine (which often sets a state consumed by a later post
    routine) without scanning unrelated packages. This is a deterministic
    source-neighbourhood lookup, not graph expansion or semantic inference.
    """

    selected_ids = {target_evidence.unit_id, *(item.unit_id for item in already_selected)}
    target_name = _normalize_identifier(target.name.display_name)
    candidates: list[tuple[int, CodeIndexRecord]] = []
    call_offsets = [caller.call_site.start_offset for caller in callers]
    for record in records_by_id.values():
        if (
            record.unit_id in selected_ids
            or record.source_path != target.source_path
            or target_name not in _normalize_identifier(record.citation_text)
        ):
            continue
        distance = min(
            (abs(record.source_map.start_offset - offset) for offset in call_offsets),
            default=0,
        )
        candidates.append((distance, record))
    selected: list[CodeEvidence] = []
    for _, record in sorted(candidates, key=lambda item: (item[0], item[1].unit_id)):
        if record.unit_id in selected_ids:
            continue
        selected_ids.add(record.unit_id)
        metadata = dict(target_evidence.retrieval_metadata)
        metadata.update(
            retrieval_relation="workflow_validation_context",
            workflow_status="visible_implementation_context",
        )
        selected.append(
            CodeEvidence(
                unit_id=record.unit_id,
                point_id=record.point_id,
                score=target_evidence.score,
                retrieval_method=target_evidence.retrieval_method,
                snapshot_id=record.snapshot_id,
                module_id=record.module_id,
                source_path=record.source_path,
                source_kind=record.source_kind,
                display_name=record.display_name,
                parent_unit_id=record.parent_unit_id,
                package_name=record.package_name,
                start_line=record.source_map.start_line,
                end_line=record.source_map.end_line,
                parser_state=record.parser_state,
                conditional_state=record.conditional_state,
                text=record.citation_text,
                retrieval_metadata=metadata,
            )
        )
        if len(selected) == MAX_VALIDATION_CONTEXTS:
            break
    return tuple(selected)


def _workflow_terms(
    target: CodeEvidence, callers: Sequence[CodeEvidence]
) -> tuple[str, ...]:
    """Extract conservative business words from comments and string literals."""

    source_terms: list[list[str]] = []
    first_seen: dict[str, int] = {}
    for position, evidence in enumerate((target, *callers)):
        comments_and_literals = " ".join(_STRING_LITERAL.findall(evidence.text))
        # String literals carry the user-facing validation/error condition. We
        # intentionally exclude comments here: change markers often look like
        # FDD names but are historical implementation annotations, not proof of
        # a current functional relationship. Enhancement-comment matching stays
        # in the separate offline candidate-proposal workflow.
        terms: list[str] = []
        for term in tokenize(comments_and_literals):
            normalized = term.casefold()
            if (
                len(normalized) < 3
                or normalized in _NON_BUSINESS_TERMS
                or normalized in _COMMON_CONTEXT_TERMS
                or normalized.startswith(("pkg", "str", "p_", "t_"))
            ):
                continue
            if normalized not in terms:
                terms.append(normalized)
                first_seen.setdefault(normalized, position)
        source_terms.append(terms)

    coverage = {
        term: sum(term in terms for terms in source_terms)
        for terms in source_terms
        for term in terms
    }
    ranked = sorted(
        coverage,
        key=lambda term: (-coverage[term], first_seen[term], term),
    )
    return tuple(ranked[:MAX_WORKFLOW_TERMS])


def _workflow_fdd_candidates(
    *,
    query: str,
    terms: Sequence[str],
    documents: Sequence[LexicalSearchDocument],
    limit: int,
) -> tuple[QdrantSearchResult, ...]:
    if not terms:
        return ()
    # Do not let generic wording in the user's question outweigh the terms
    # actually visible at the routine and its resolved call site.  The exact
    # routine check happened before this point; it is not a corpus-wide query
    # expansion path.
    ranked = search_lexical_documents(list(documents), " ".join(terms), limit=limit)
    if not ranked:
        return ()
    by_id = {document.unit_id: document for document in documents}
    primary = ranked[0]
    primary_document = by_id[str(primary.payload["unit_id"])]
    adjacent = sorted(
        (
            item
            for item in documents
            if item.document_id == primary_document.document_id
            and item.unit_id != primary_document.unit_id
            and abs(item.unit_index - primary_document.unit_index) == 1
        ),
        key=lambda item: (item.unit_index, item.unit_id),
    )[: max(0, limit - 1)]
    result = [_workflow_result(primary, "workflow_fdd_candidate")]
    result.extend(_adjacent_result(item, primary.score) for item in adjacent)
    return tuple(result)


def _workflow_result(result, relation: str) -> QdrantSearchResult:
    payload = dict(result.payload)
    payload["retrieval_relation"] = relation
    payload["workflow_status"] = "unreviewed_documentation_candidate"
    return QdrantSearchResult(point_id=result.point_id, score=float(result.score), payload=payload)


def _adjacent_result(document: LexicalSearchDocument, score: float) -> QdrantSearchResult:
    return QdrantSearchResult(
        point_id=document.unit_id,
        score=score,
        payload={
            "unit_id": document.unit_id,
            "document_id": document.document_id,
            "document_name": document.document_name,
            "unit_index": document.unit_index,
            "source_kind": document.source_kind,
            "document_family": document.document_family,
            "release_label": document.release_label,
            "text": document.text,
            "retrieval_text": document.retrieval_text or document.text,
            "retrieval_method": "workflow_adjacent_context",
            "retrieval_relation": "same_document_adjacent_context",
            "workflow_status": "unreviewed_documentation_candidate",
        },
    )


def _normalize_identifier(value: str) -> str:
    return re.sub(r"[^a-z0-9]", "", value.casefold())


def _same_symbol_name(left: str, right: str) -> bool:
    return _normalize_identifier(left) == _normalize_identifier(right)


def _overlaps(left, right) -> bool:
    return left.start_offset < right.end_offset and right.start_offset < left.end_offset
