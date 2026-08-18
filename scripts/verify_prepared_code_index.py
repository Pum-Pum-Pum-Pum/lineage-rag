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
    load_code_index_artifact,
    verify_prepared_code_index_artifact,
)
from app.code_ingestion.analysis_policy import load_code_analysis_policy
from app.core.config import get_settings
from app.code_ingestion.dependency_review_ledger import load_dependency_review_ledger


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    settings = get_settings()
    parser = argparse.ArgumentParser(description="Verify a local prepared code-index contract by exact rebuild.")
    parser.add_argument("artifact", type=Path)
    parser.add_argument("--parse-staging-root", type=Path, default=settings.code_staging_dir)
    parser.add_argument("--dependency-review-ledger", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    artifact = load_code_index_artifact(args.artifact)
    parse_stage = args.parse_staging_root / artifact.snapshot_id / artifact.parse_generation
    policy = load_code_analysis_policy(get_settings().code_analysis_policy_path)
    ledger = (
        load_dependency_review_ledger(args.dependency_review_ledger)
        if args.dependency_review_ledger
        else None
    )
    result = verify_prepared_code_index_artifact(
        artifact,
        parse_stage,
        expected_policy_sha256=policy.sha256,
        dependency_review_ledger=ledger,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
