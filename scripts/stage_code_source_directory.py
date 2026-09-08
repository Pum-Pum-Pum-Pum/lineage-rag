from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.code_ingestion.snapshot_builder import SNAPSHOT_REQUEST_FILE, load_snapshot_request
from app.code_ingestion.source_import import stage_external_code_source
from app.core.config import get_settings
from app.core.ingestion_policy import load_ingestion_source_policy


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Read configured code extensions from an external directory into one local snapshot intake."
    )
    parser.add_argument("--source-directory", required=True, type=Path)
    parser.add_argument("--intake-directory", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    settings = get_settings()
    # Validate the request before copying a potentially large external tree.
    request = load_snapshot_request(args.intake_directory / SNAPSHOT_REQUEST_FILE)
    expected_directory_name = f"{request.module_set}-r{request.svn_revision}"
    if args.intake_directory.name != expected_directory_name:
        raise ValueError(
            "Snapshot request directory must match module_set and svn_revision: "
            f"expected {expected_directory_name!r}, got {args.intake_directory.name!r}."
        )
    receipt = stage_external_code_source(
        args.source_directory,
        args.intake_directory,
        source_policy=load_ingestion_source_policy(settings.ingestion_source_policy_path),
    )
    print(json.dumps(receipt, indent=2, ensure_ascii=False, sort_keys=True))


if __name__ == "__main__":
    main()
