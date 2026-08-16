from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.code_ingestion.code_parsing_pipeline import PARSER_GENERATION_DIRECTORY
from app.code_ingestion.code_retrieval_artifact import build_code_retrieval_artifact
from app.code_ingestion.code_analysis_models import CodeStaticAnalysisArtifact
from app.code_ingestion.plsql_models import (
    CodeParseStageManifest,
    CodeRetrievalArtifact,
    PlSqlFileParseArtifact,
)
from app.code_ingestion.snapshot_builder import load_snapshot_manifest


ROUTINE_KINDS = {"procedure", "procedure_spec", "function", "function_spec"}
KNOWN_FALSE_CALLS = {"AND", "EXISTS", "IN", "TBLBUNDLERPT", "TRUNC"}


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify real custom-code pre-index invariants.")
    parser.add_argument("snapshot_id")
    parser.add_argument("--snapshot-root", type=Path, required=True)
    parser.add_argument(
        "--staging-root",
        type=Path,
        default=ROOT_DIR / "data/staging/code",
    )
    parser.add_argument(
        "--generation",
        default=PARSER_GENERATION_DIRECTORY,
    )
    parser.add_argument("--output", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    snapshot_directory = args.snapshot_root / args.snapshot_id
    snapshot = load_snapshot_manifest(snapshot_directory, verify_sources=True)
    stage = args.staging_root / args.snapshot_id / args.generation
    manifest = CodeParseStageManifest.model_validate_json(
        (stage / "parse_stage_manifest.json").read_text(encoding="utf-8")
    )
    failures: list[str] = []
    file_results = []
    all_symbol_sets = []
    for entry, parse_relative, retrieval_relative, analysis_relative in zip(
        snapshot.files,
        manifest.parse_artifacts,
        manifest.retrieval_artifacts,
        manifest.analysis_artifacts,
        strict=True,
    ):
        source_path = snapshot_directory / snapshot.source_directory_name / entry.path
        source_text = source_path.read_bytes().decode(entry.encoding)
        parsed = PlSqlFileParseArtifact.model_validate_json(
            (stage / parse_relative).read_text(encoding="utf-8")
        )
        retrieval = CodeRetrievalArtifact.model_validate_json(
            (stage / retrieval_relative).read_text(encoding="utf-8")
        )
        analysis = CodeStaticAnalysisArtifact.model_validate_json(
            (stage / analysis_relative).read_text(encoding="utf-8")
        )
        rebuilt = build_code_retrieval_artifact(
            parsed,
            source_text,
            verified_source_sha256=entry.sha256,
            max_unit_characters=manifest.max_retrieval_unit_characters,
            overlap_characters=manifest.retrieval_overlap_characters,
        )
        deterministic = rebuilt == retrieval
        if not deterministic:
            failures.append(f"{entry.path}: retrieval artifact is not deterministic")
        exact_text = all(
            unit.text
            == source_text[unit.source_map.start_offset : unit.source_map.end_offset]
            for unit in retrieval.units
        )
        if not exact_text:
            failures.append(f"{entry.path}: retrieval text/source-map mismatch")
        routine_segments = [
            segment for segment in parsed.segments if segment.segment_kind in ROUTINE_KINDS
        ]
        retained = 0
        for segment in routine_segments:
            if any(
                (unit.parent_source_map or unit.source_map) == segment.source_map
                for unit in retrieval.units
            ):
                retained += 1
        if retained != len(routine_segments):
            failures.append(f"{entry.path}: one or more routine segments were not retained")
        calls = [
            edge for edge in analysis.dependencies if edge.dependency_kind == "routine_call"
        ]
        false_calls = sorted(
            {edge.target_canonical_name for edge in calls} & KNOWN_FALSE_CALLS
        )
        if false_calls:
            failures.append(f"{entry.path}: known false routine calls remain: {false_calls}")
        all_symbol_sets.append({symbol.symbol_key for symbol in analysis.symbols})
        file_results.append(
            {
                "source_path": entry.path,
                "parser_state": parsed.parser_state,
                "routine_segments": len(routine_segments),
                "retained_routine_segments": retained,
                "retrieval_units": retrieval.total_units,
                "child_units": sum(unit.parent_unit_id is not None for unit in retrieval.units),
                "child_parents": len(
                    {unit.parent_unit_id for unit in retrieval.units if unit.parent_unit_id}
                ),
                "max_text_characters": max((len(unit.text) for unit in retrieval.units), default=0),
                "max_retrieval_characters": max(
                    (len(unit.retrieval_text) for unit in retrieval.units), default=0
                ),
                "deterministic_rebuild": deterministic,
                "exact_source_mapping": exact_text,
                "routine_calls": len(calls),
                "unresolved_routine_calls": sum(
                    edge.resolution_state == "unresolved" for edge in calls
                ),
                "table_edges": sum(
                    edge.dependency_kind in {"table_read", "table_write"}
                    for edge in analysis.dependencies
                ),
                "known_false_calls": false_calls,
            }
        )
    spec_body_matches = len(all_symbol_sets[0] & all_symbol_sets[1]) if len(all_symbol_sets) == 2 else None
    if manifest.state_counts["fallback_parse"] or manifest.state_counts["failed"]:
        failures.append("Stage contains fallback or failed files")
    report = {
        "schema_version": "code_preindex_gate_v1",
        "status": "pass" if not failures else "fail",
        "snapshot_id": manifest.snapshot_id,
        "snapshot_content_sha256": manifest.snapshot_content_sha256,
        "parser_generation": manifest.parser_generation,
        "max_segment_characters": manifest.max_segment_characters,
        "max_retrieval_unit_characters": manifest.max_retrieval_unit_characters,
        "retrieval_overlap_characters": manifest.retrieval_overlap_characters,
        "state_counts": manifest.state_counts,
        "spec_body_matching_symbol_keys": spec_body_matches,
        "files": file_results,
        "failures": failures,
        "external_calls_performed": False,
    }
    rendered = json.dumps(report, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)
    raise SystemExit(0 if not failures else 1)


if __name__ == "__main__":
    main()
