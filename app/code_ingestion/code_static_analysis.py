from __future__ import annotations

from dataclasses import dataclass

from app.code_ingestion.analysis_policy import CodeAnalysisPolicy
from app.code_ingestion.code_analysis_models import (
    AnalysisDiagnostic,
    CodeStaticAnalysisArtifact,
)
from app.code_ingestion.ddl_analysis import extract_ddl_structures, resolve_synonyms
from app.code_ingestion.plsql_dependency_analysis import (
    build_symbol_lookup,
    extract_dependencies,
)
from app.code_ingestion.plsql_models import PlSqlFileParseArtifact
from app.code_ingestion.plsql_symbol_analysis import diagnose_symbol_groups, extract_symbols


@dataclass(frozen=True)
class StaticAnalysisInput:
    parse_artifact: PlSqlFileParseArtifact
    source_text: str


def analyze_snapshot_sources(
    inputs: tuple[StaticAnalysisInput, ...],
    *,
    module_id: str,
    policy: CodeAnalysisPolicy,
) -> tuple[CodeStaticAnalysisArtifact, ...]:
    symbols_by_path = {
        item.parse_artifact.source_path: extract_symbols(
            item.parse_artifact,
            module_id=module_id,
        )
        for item in inputs
    }
    all_symbols = tuple(
        symbol
        for path in sorted(symbols_by_path, key=str.casefold)
        for symbol in symbols_by_path[path]
    )
    symbol_lookup = build_symbol_lookup(all_symbols)
    symbol_diagnostics = diagnose_symbol_groups(all_symbols)

    ddl_by_path = {}
    all_objects = []
    all_synonyms = []
    for item in inputs:
        objects, synonyms, diagnostics = extract_ddl_structures(
            item.source_text,
            item.parse_artifact,
        )
        ddl_by_path[item.parse_artifact.source_path] = (objects, synonyms, diagnostics)
        all_objects.extend(objects)
        all_synonyms.extend(synonyms)
    resolved_synonyms = resolve_synonyms(tuple(all_objects), tuple(all_synonyms))
    global_schema_diagnostics = _schema_identity_diagnostics(
        tuple(all_objects),
        resolved_synonyms,
    )
    resolved_synonyms_by_path = {
        path: tuple(item for item in resolved_synonyms if item.source_path == path)
        for path in ddl_by_path
    }

    artifacts = []
    for item in inputs:
        parse_artifact = item.parse_artifact
        file_symbols = symbols_by_path[parse_artifact.source_path]
        objects, _, ddl_diagnostics = ddl_by_path[parse_artifact.source_path]
        dependencies = extract_dependencies(
            item.source_text,
            parse_artifact,
            file_symbols=file_symbols,
            all_symbols=all_symbols,
            schema_objects=tuple(all_objects),
            policy=policy,
            symbol_lookup=symbol_lookup,
        )
        occurrence_ids = {symbol.occurrence_id for symbol in file_symbols}
        local_symbol_diagnostics = tuple(
            diagnostic
            for diagnostic in symbol_diagnostics
            if occurrence_ids.intersection(diagnostic.related_occurrence_ids)
        )
        boundary_diagnostics = _dependency_diagnostics(dependencies)
        synonym_diagnostics = _synonym_diagnostics(
            resolved_synonyms_by_path[parse_artifact.source_path]
        )
        local_schema_diagnostics = tuple(
            diagnostic
            for diagnostic in global_schema_diagnostics
            if diagnostic.source_path == parse_artifact.source_path
        )
        artifacts.append(
            CodeStaticAnalysisArtifact(
                module_id=module_id,
                snapshot_id=parse_artifact.snapshot_id,
                source_path=parse_artifact.source_path,
                source_sha256=parse_artifact.source_sha256,
                analysis_policy_sha256=policy.sha256,
                parser_state=parse_artifact.parser_state,
                symbols=file_symbols,
                dependencies=dependencies,
                schema_objects=objects,
                synonyms=resolved_synonyms_by_path[parse_artifact.source_path],
                diagnostics=(
                    *local_symbol_diagnostics,
                    *ddl_diagnostics,
                    *boundary_diagnostics,
                    *synonym_diagnostics,
                    *local_schema_diagnostics,
                ),
            )
        )
    return tuple(artifacts)


def _dependency_diagnostics(dependencies):
    diagnostics = []
    for edge in dependencies:
        if edge.resolution_state not in {
            "ambiguous",
            "unresolved",
            "dynamic_unknown",
            "custom_source_missing",
            "kernel_unavailable",
        }:
            continue
        severity = "warning"
        diagnostics.append(
            AnalysisDiagnostic(
                stage="dependency",
                severity=severity,
                code=f"dependency_{edge.resolution_state}",
                message=(
                    f"{edge.dependency_kind} target {edge.target_canonical_name!r} remains "
                    f"{edge.resolution_state}; downstream answers must preserve this boundary."
                ),
                source_path=edge.source_path,
                source_map=edge.source_map,
                related_occurrence_ids=(edge.source_symbol_occurrence_id,)
                if edge.source_symbol_occurrence_id
                else (),
            )
        )
    return tuple(diagnostics)


def _synonym_diagnostics(synonyms):
    return tuple(
        AnalysisDiagnostic(
            stage="snapshot_resolution",
            severity="warning",
            code=f"synonym_{synonym.resolution_state}",
            message=(
                f"Synonym {synonym.canonical_qualified_name!r} remains "
                f"{synonym.resolution_state}; live target behavior is not proven."
            ),
            source_path=synonym.source_path,
            source_map=synonym.source_map,
        )
        for synonym in synonyms
        if synonym.resolution_state != "resolved_in_snapshot"
    )


def _schema_identity_diagnostics(schema_objects, synonyms):
    diagnostics = []
    for kind, items in (("schema_object", schema_objects), ("synonym", synonyms)):
        grouped = {}
        for item in items:
            grouped.setdefault(item.canonical_qualified_name, []).append(item)
        for canonical_name, occurrences in grouped.items():
            if len(occurrences) < 2:
                continue
            for occurrence in occurrences:
                diagnostics.append(
                    AnalysisDiagnostic(
                        stage="snapshot_resolution",
                        severity="error",
                        code=f"duplicate_{kind}_identity",
                        message=(
                            f"Multiple {kind} definitions share canonical identity "
                            f"{canonical_name!r}; none may be selected implicitly."
                        ),
                        source_path=occurrence.source_path,
                        source_map=occurrence.source_map,
                    )
                )
    return tuple(diagnostics)
