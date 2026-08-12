from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.code_ingestion.code_parsing_pipeline import (
    PARSER_GENERATION_DIRECTORY,
    parse_code_snapshot,
)
from app.core.config import get_settings
from app.code_ingestion.analysis_policy import load_code_analysis_policy


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    settings = get_settings()
    parser = argparse.ArgumentParser(
        description=(
            "Parse one immutable custom-code snapshot into local PL/SQL structural and "
            "retrieval artifacts. No OpenAI or Qdrant calls are made."
        )
    )
    parser.add_argument("snapshot_id", help="Exact immutable snapshot directory name.")
    parser.add_argument("--snapshot-root", type=Path, default=settings.code_snapshots_dir)
    parser.add_argument("--staging-root", type=Path, default=settings.code_staging_dir)
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=settings.code_parse_timeout_seconds,
    )
    parser.add_argument(
        "--memory-limit-mib",
        type=int,
        default=settings.code_parse_memory_limit_mib,
    )
    parser.add_argument(
        "--max-segment-characters",
        type=int,
        default=settings.code_parse_max_segment_characters,
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    manifest = parse_code_snapshot(
        args.snapshot_root / args.snapshot_id,
        args.staging_root,
        timeout_seconds=args.timeout_seconds,
        memory_limit_bytes=args.memory_limit_mib * 1024 * 1024,
        max_segment_characters=args.max_segment_characters,
        analysis_policy=load_code_analysis_policy(get_settings().code_analysis_policy_path),
    )
    output_directory = (
        args.staging_root / manifest.snapshot_id / PARSER_GENERATION_DIRECTORY
    ).resolve()
    print(
        json.dumps(
            {
                "status": manifest.status,
                "snapshot_id": manifest.snapshot_id,
                "output_directory": str(output_directory),
                "file_count": manifest.file_count,
                "state_counts": manifest.state_counts,
                "analysis_policy_sha256": manifest.analysis_policy_sha256,
                "external_calls_performed": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
