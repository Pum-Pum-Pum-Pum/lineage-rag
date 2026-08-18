from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
from pathlib import Path

from app.code_ingestion.code_retrieval_artifact import build_code_retrieval_artifact
from app.code_ingestion.analysis_policy import CodeAnalysisPolicy, load_code_analysis_policy
from app.code_ingestion.code_static_analysis import StaticAnalysisInput, analyze_snapshot_sources
from app.code_ingestion.plsql_isolation import parse_file_isolated
from app.code_ingestion.plsql_models import CodeParseStageManifest
from app.code_ingestion.plsql_models import (
    CodeRetrievalArtifact,
    ParseReuseRecord,
    PlSqlFileParseArtifact,
)
from app.code_ingestion.snapshot_builder import load_snapshot_manifest
from app.code_ingestion.program_unit_validation import validate_custom_program_unit
from app.code_ingestion.plsql_segmentation import (
    inventory_routine_declarations,
    uncovered_routine_declarations,
)


PARSER_GENERATION_DIRECTORY = "plsql_antlr_4_13_2_analysis_v12"
PARSER_CONTRACT_VERSION = "plsql_parser_contract_v2"


def parse_code_snapshot(
    snapshot_directory: Path,
    staging_root: Path,
    *,
    timeout_seconds: float = 120.0,
    memory_limit_bytes: int = 1024 * 1024 * 1024,
    max_segment_characters: int = 500,
    max_retrieval_unit_characters: int = 6_000,
    retrieval_overlap_characters: int = 400,
    analysis_policy: CodeAnalysisPolicy | None = None,
    generation_directory: str = PARSER_GENERATION_DIRECTORY,
    reuse_generation_directory: Path | None = None,
) -> CodeParseStageManifest:
    """Parse one verified immutable snapshot and atomically publish local artifacts."""

    if timeout_seconds <= 0 or memory_limit_bytes <= 0 or max_segment_characters <= 0:
        raise ValueError("Parser resource boundaries must be greater than zero")
    if (
        max_retrieval_unit_characters <= 0
        or retrieval_overlap_characters < 0
        or retrieval_overlap_characters >= max_retrieval_unit_characters
    ):
        raise ValueError("Retrieval chunk boundaries are invalid")
    snapshot = load_snapshot_manifest(snapshot_directory, verify_sources=True)
    selected_analysis_policy = analysis_policy or load_code_analysis_policy()
    target = staging_root / snapshot.snapshot_id / generation_directory
    if target.exists():
        raise FileExistsError(f"Parse generation already exists and will not be overwritten: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    parse_paths: list[str] = []
    retrieval_paths: list[str] = []
    analysis_paths: list[str] = []
    analysis_inputs: list[StaticAnalysisInput] = []
    reuse_records: list[ParseReuseRecord] = []
    reuse_catalog, reused_from_generation = _load_reuse_catalog(
        reuse_generation_directory,
        file_contracts={
            entry.path: (entry.source_handler, entry.encoding)
            for entry in snapshot.files
        },
        compiler_context=snapshot.request.compiler_context.model_dump(mode="json"),
        snapshot_id=snapshot.snapshot_id,
        snapshot_content_sha256=snapshot.snapshot_content_sha256,
        timeout_seconds=timeout_seconds,
        memory_limit_bytes=memory_limit_bytes,
        max_segment_characters=max_segment_characters,
        max_retrieval_unit_characters=max_retrieval_unit_characters,
        retrieval_overlap_characters=retrieval_overlap_characters,
    )
    temporary = Path(tempfile.mkdtemp(prefix=".code-parse-", dir=target.parent))
    worker_root = temporary / ".workers"
    state_counts = {state: 0 for state in ("full_parse", "segmented_parse", "fallback_parse", "failed")}
    try:
        for entry in snapshot.files:
            if entry.source_handler not in {"plsql", "ddl"}:
                raise RuntimeError(f"No parser implementation for source handler: {entry.source_handler}")
            source_file = snapshot_directory / snapshot.source_directory_name / entry.path
            raw_bytes = source_file.read_bytes()
            observed_hash = hashlib.sha256(raw_bytes).hexdigest()
            if observed_hash != entry.sha256:
                raise RuntimeError(f"Immutable source changed before parsing: {entry.path}")
            source_text = raw_bytes.decode(entry.encoding)
            reuse_key = _parse_reuse_key(
                source_sha256=entry.sha256,
                source_handler=entry.source_handler,
                encoding=entry.encoding,
                compiler_context=snapshot.request.compiler_context.model_dump(mode="json"),
                timeout_seconds=timeout_seconds,
                memory_limit_bytes=memory_limit_bytes,
                max_segment_characters=max_segment_characters,
                max_retrieval_unit_characters=max_retrieval_unit_characters,
                retrieval_overlap_characters=retrieval_overlap_characters,
            )
            reusable = reuse_catalog.get(entry.path)
            if reusable is not None and reusable[2] == reuse_key:
                parsed, retrieval, _ = reusable
                reused = True
            else:
                parsed = parse_file_isolated(
                    source_file,
                    snapshot_id=snapshot.snapshot_id,
                    source_path=entry.path,
                    source_sha256=entry.sha256,
                    encoding=entry.encoding,
                    compiler_context=snapshot.request.compiler_context,
                    work_root=worker_root,
                    timeout_seconds=timeout_seconds,
                    memory_limit_bytes=memory_limit_bytes,
                    max_segment_characters=max_segment_characters,
                )
                retrieval = build_code_retrieval_artifact(
                    parsed,
                    source_text,
                    verified_source_sha256=observed_hash,
                    max_unit_characters=max_retrieval_unit_characters,
                    overlap_characters=retrieval_overlap_characters,
                )
                reused = False
            validate_custom_program_unit(
                parsed,
                source_handler=entry.source_handler,
                allowed_suffixes=selected_analysis_policy.boundaries.custom_program_unit_suffixes,
            )
            _validate_routine_parse_coverage(source_text, parsed, retrieval)
            stem = _artifact_stem(entry.path)
            parse_relative = f"parse/{stem}.json"
            retrieval_relative = f"retrieval/{stem}.json"
            _write_json(temporary / parse_relative, parsed.model_dump(mode="json"))
            _write_json(temporary / retrieval_relative, retrieval.model_dump(mode="json"))
            parse_paths.append(parse_relative)
            retrieval_paths.append(retrieval_relative)
            analysis_inputs.append(
                StaticAnalysisInput(parse_artifact=parsed, source_text=source_text)
            )
            reuse_records.append(
                ParseReuseRecord(
                    source_path=entry.path,
                    source_sha256=entry.sha256,
                    reuse_key_sha256=reuse_key,
                    reused=reused,
                    reused_from_generation=reused_from_generation if reused else None,
                )
            )
            state_counts[parsed.parser_state] += 1

        analysis_artifacts = analyze_snapshot_sources(
            tuple(analysis_inputs),
            module_id=snapshot.request.module_set,
            policy=selected_analysis_policy,
        )
        for analysis_input, analysis in zip(
            analysis_inputs,
            analysis_artifacts,
            strict=True,
        ):
            _validate_routine_symbol_coverage(analysis_input.parse_artifact, analysis)
        for analysis in analysis_artifacts:
            stem = _artifact_stem(analysis.source_path)
            analysis_relative = f"analysis/{stem}.json"
            _write_json(temporary / analysis_relative, analysis.model_dump(mode="json"))
            analysis_paths.append(analysis_relative)

        status = _stage_status(state_counts, analysis_artifacts)
        manifest = CodeParseStageManifest(
            status=status,
            snapshot_id=snapshot.snapshot_id,
            snapshot_content_sha256=snapshot.snapshot_content_sha256,
            parser_generation=generation_directory,
            parser_contract_version=PARSER_CONTRACT_VERSION,
            analysis_policy_sha256=selected_analysis_policy.sha256,
            file_count=len(snapshot.files),
            state_counts=state_counts,
            parse_artifacts=tuple(parse_paths),
            retrieval_artifacts=tuple(retrieval_paths),
            analysis_artifacts=tuple(analysis_paths),
            timeout_seconds=timeout_seconds,
            memory_limit_bytes=memory_limit_bytes,
            max_segment_characters=max_segment_characters,
            max_retrieval_unit_characters=max_retrieval_unit_characters,
            retrieval_overlap_characters=retrieval_overlap_characters,
            reused_from_generation=reused_from_generation,
            reused_parse_file_count=sum(record.reused for record in reuse_records),
            parse_reuse_records=tuple(reuse_records),
        )
        _write_json(temporary / "parse_stage_manifest.json", manifest.model_dump(mode="json"))
        shutil.rmtree(worker_root, ignore_errors=True)
        temporary.replace(target)
        return manifest
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def _stage_status(state_counts: dict[str, int], analysis_artifacts=()) -> str:
    if state_counts["failed"]:
        return "failed"
    if any(
        diagnostic.severity == "error"
        for artifact in analysis_artifacts
        for diagnostic in artifact.diagnostics
    ):
        return "failed"
    if state_counts["segmented_parse"] or state_counts["fallback_parse"]:
        return "complete_with_degradation"
    return "complete"


def _artifact_stem(source_path: str) -> str:
    readable = Path(source_path).name.replace(".", "_")[:80]
    suffix = hashlib.sha256(source_path.encode("utf-8")).hexdigest()[:12]
    return f"{readable}-{suffix}"


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )


def _load_reuse_catalog(
    directory: Path | None,
    *,
    file_contracts: dict[str, tuple[str, str]],
    compiler_context: dict[str, object],
    **expected: object,
) -> tuple[dict[str, tuple[PlSqlFileParseArtifact, CodeRetrievalArtifact, str]], str | None]:
    if directory is None:
        return {}, None
    manifest = CodeParseStageManifest.model_validate_json(
        (directory / "parse_stage_manifest.json").read_text(encoding="utf-8")
    )
    if manifest.parser_contract_version != PARSER_CONTRACT_VERSION:
        raise ValueError(
            "Reuse generation parser contract does not match the current segmentation contract"
        )
    for field, value in expected.items():
        if getattr(manifest, field) != value:
            raise ValueError(f"Reuse generation {field} does not match the requested parse contract")
    catalog = {}
    for parse_relative, retrieval_relative in zip(
        manifest.parse_artifacts,
        manifest.retrieval_artifacts,
        strict=True,
    ):
        parsed = PlSqlFileParseArtifact.model_validate_json(
            (directory / parse_relative).read_text(encoding="utf-8")
        )
        retrieval = CodeRetrievalArtifact.model_validate_json(
            (directory / retrieval_relative).read_text(encoding="utf-8")
        )
        if parsed.source_path not in file_contracts:
            continue
        source_handler, encoding = file_contracts[parsed.source_path]
        reuse_key = _parse_reuse_key(
            source_sha256=parsed.source_sha256,
            source_handler=source_handler,
            encoding=encoding,
            compiler_context=compiler_context,
            timeout_seconds=manifest.timeout_seconds,
            memory_limit_bytes=manifest.memory_limit_bytes,
            max_segment_characters=manifest.max_segment_characters,
            max_retrieval_unit_characters=manifest.max_retrieval_unit_characters,
            retrieval_overlap_characters=manifest.retrieval_overlap_characters,
        )
        catalog[parsed.source_path] = (parsed, retrieval, reuse_key)
    return catalog, manifest.parser_generation


def _parse_reuse_key(**values: object) -> str:
    payload = {
        "schema_version": "plsql_parse_reuse_key_v1",
        "antlr_tool_version": "4.13.2",
        "antlr_runtime_version": "4.13.2",
        "grammar_commit": "a7704d4c029c33a89818ac103f758f7c72d8d16c",
        "parser_contract_version": PARSER_CONTRACT_VERSION,
        **values,
    }
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _validate_routine_parse_coverage(source_text, parsed, retrieval) -> None:
    declarations = inventory_routine_declarations(
        source_text,
        source_path=parsed.source_path,
    )
    routine_segments = tuple(
        segment
        for segment in parsed.segments
        if segment.segment_kind in {"procedure", "procedure_spec", "function", "function_spec"}
    )
    node_offsets = {
        node.source_map.start_offset
        for node in parsed.extracted_nodes
        if node.node_kind in {"procedure", "procedure_spec", "function", "function_spec"}
    }
    routine_nodes = tuple(
        node
        for node in parsed.extracted_nodes
        if node.node_kind in {"procedure", "procedure_spec", "function", "function_spec"}
    )
    if parsed.parser_state == "segmented_parse":
        uncovered = uncovered_routine_declarations(declarations, parsed.segments)
    elif parsed.parser_state == "full_parse":
        uncovered = tuple(
            item
            for item in declarations
            if not any(
                node.display_name.casefold() == item.display_name.casefold()
                and node.source_map.start_offset
                <= item.source_map.start_offset
                < node.source_map.end_offset
                for node in routine_nodes
            )
        )
    else:
        uncovered = ()
    if uncovered:
        details = ", ".join(
            f"{item.display_name}@{item.source_map.start_line}" for item in uncovered[:10]
        )
        raise RuntimeError(
            f"Routine declaration coverage failed for {parsed.source_path}: {details}"
        )

    missing_nodes = [
        segment for segment in routine_segments if segment.source_map.start_offset not in node_offsets
    ]
    if missing_nodes:
        raise RuntimeError(f"Routine segments lack extracted nodes: {parsed.source_path}")

    retrieval_parent_maps = {
        unit.parent_source_map or unit.source_map for unit in retrieval.units
    }
    if any(segment.source_map not in retrieval_parent_maps for segment in routine_segments):
        raise RuntimeError(f"Routine segments lack citeable retrieval units: {parsed.source_path}")


def _validate_routine_symbol_coverage(parsed, analysis) -> None:
    expected_offsets = {
        node.source_map.start_offset
        for node in parsed.extracted_nodes
        if node.node_kind in {"procedure", "procedure_spec", "function", "function_spec"}
    }
    symbol_offsets = {symbol.source_map.start_offset for symbol in analysis.symbols}
    missing = sorted(expected_offsets - symbol_offsets)
    if missing:
        raise RuntimeError(
            f"Routine nodes lack symbol occurrences for {parsed.source_path}: {missing[:10]}"
        )
