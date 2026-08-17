from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.code_ingestion.analysis_policy import load_code_analysis_policy
from app.code_ingestion.code_parsing_pipeline import PARSER_GENERATION_DIRECTORY
from app.code_ingestion.dependency_review import (
    build_dependency_review_packet,
    render_dependency_review_markdown,
)
from app.code_ingestion.plsql_models import CodeParseStageManifest
from app.core.config import get_settings


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Export a local SME packet for ambiguous code dependencies.")
    parser.add_argument("snapshot_id")
    parser.add_argument("--snapshot-root", type=Path, default=settings.code_snapshots_dir)
    parser.add_argument("--staging-root", type=Path, default=settings.code_staging_dir)
    parser.add_argument("--generation", default=PARSER_GENERATION_DIRECTORY)
    parser.add_argument("--output-root", type=Path, default=settings.data_dir / "exports/code_analysis")
    parser.add_argument("--examples-per-case", type=int, default=3)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    stage = args.staging_root / args.snapshot_id / args.generation
    manifest = CodeParseStageManifest.model_validate_json(
        (stage / "parse_stage_manifest.json").read_text(encoding="utf-8")
    )
    policy = load_code_analysis_policy(get_settings().code_analysis_policy_path)
    if manifest.analysis_policy_sha256 != policy.sha256:
        raise RuntimeError("Parse stage policy hash does not match the current approved policy")
    packet = build_dependency_review_packet(
        args.snapshot_root / args.snapshot_id,
        stage,
        examples_per_case=args.examples_per_case,
    )
    args.output_root.mkdir(parents=True, exist_ok=True)
    base = f"{args.snapshot_id}-{args.generation}-dependency-review"
    json_path = args.output_root / f"{base}.json"
    markdown_path = args.output_root / f"{base}.md"
    if json_path.exists() or markdown_path.exists():
        raise FileExistsError("Dependency review output already exists and will not be overwritten")
    json_path.write_text(
        json.dumps(packet.model_dump(mode="json"), indent=2, ensure_ascii=False, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    markdown_path.write_text(render_dependency_review_markdown(packet), encoding="utf-8")
    print(json.dumps({
        "status": packet.review_status,
        "snapshot_id": packet.snapshot_id,
        "parser_generation": packet.parser_generation,
        "review_cases": packet.total_review_cases,
        "occurrences": packet.total_occurrences,
        "packet_identity_sha256": packet.packet_identity_sha256,
        "json_output": str(json_path.resolve()),
        "markdown_output": str(markdown_path.resolve()),
        "external_calls_performed": False,
    }, indent=2))


if __name__ == "__main__":
    main()
