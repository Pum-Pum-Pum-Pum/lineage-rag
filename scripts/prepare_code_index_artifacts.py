from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.code_indexing.contract import (
    CODE_INDEX_CONTRACT_DIRECTORY,
    build_code_index_artifact,
    write_code_index_artifact_no_overwrite,
)
from app.core.config import get_settings


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Prepare deterministic local code index contracts.")
    parser.add_argument("snapshot_id")
    parser.add_argument("--parse-generation", required=True)
    parser.add_argument("--parse-staging-root", type=Path, default=settings.code_staging_dir)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=settings.data_dir / "staging/code_indexes",
    )
    parser.add_argument("--embedding-model", default=settings.openai_embedding_model)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    parse_stage = args.parse_staging_root / args.snapshot_id / args.parse_generation
    artifact = build_code_index_artifact(parse_stage, embedding_model=args.embedding_model)
    target = args.output_root / args.snapshot_id / CODE_INDEX_CONTRACT_DIRECTORY
    output = write_code_index_artifact_no_overwrite(artifact, target)
    print(json.dumps({
        "status": artifact.status,
        "snapshot_id": artifact.snapshot_id,
        "parse_generation": artifact.parse_generation,
        "embedding_model": artifact.embedding_model,
        "records": artifact.total_records,
        "unique_embedding_inputs": len({record.cache_key for record in artifact.records}),
        "artifact_identity_sha256": artifact.artifact_identity_sha256,
        "output": str(output.resolve()),
        "external_calls_performed": False,
    }, indent=2))


if __name__ == "__main__":
    main()
