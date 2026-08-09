from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.code_ingestion.intake_validation import validate_code_intake
from app.code_ingestion.snapshot_builder import (
    SNAPSHOT_REQUEST_FILE,
    build_code_snapshot,
    load_snapshot_request,
)
from app.core.config import get_settings


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    settings = get_settings()
    parser = argparse.ArgumentParser(
        description="Validate and publish one immutable custom-code snapshot without external API calls."
    )
    parser.add_argument(
        "intake_directory",
        type=Path,
        help="Directory containing snapshot_request.json and source/.",
    )
    parser.add_argument(
        "--snapshot-root",
        type=Path,
        default=settings.code_snapshots_dir,
        help="Immutable snapshot archive root.",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate and hash intake without writing a snapshot.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    request = load_snapshot_request(args.intake_directory / SNAPSHOT_REQUEST_FILE)
    if args.validate_only:
        report = validate_code_intake(args.intake_directory / "source")
        print(
            json.dumps(
                {
                    "status": "valid",
                    "module_set": request.module_set,
                    "svn_revision": request.svn_revision,
                    "file_count": len(report.files),
                    "warning_count": len(report.warnings),
                    "writes_performed": False,
                    "external_calls_performed": False,
                },
                indent=2,
            )
        )
        return

    manifest = build_code_snapshot(args.intake_directory, args.snapshot_root)
    diff = manifest.diff
    print(
        json.dumps(
            {
                "status": "published",
                "snapshot_id": manifest.snapshot_id,
                "snapshot_directory": str((args.snapshot_root / manifest.snapshot_id).resolve()),
                "file_count": len(manifest.files),
                "warning_count": sum(len(entry.warnings) for entry in manifest.files),
                "diff": {
                    "added": len(diff.added),
                    "modified": len(diff.modified),
                    "deleted": len(diff.deleted),
                    "unchanged": len(diff.unchanged),
                    "exact_renames": len(diff.exact_renames),
                    "missing_expected_changes": list(diff.missing_expected_changes),
                    "unexpected_changed_files": list(diff.unexpected_changed_files),
                },
                "external_calls_performed": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

