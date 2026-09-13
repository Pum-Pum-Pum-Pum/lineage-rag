from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.code_indexing.contract import load_code_index_artifact
from app.code_indexing.reuse_verification import verify_embedding_reuse
from app.code_ingestion.snapshot_builder import load_snapshot_manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Verify source-path embedding reuse without external calls."
    )
    parser.add_argument("--snapshot-manifest", type=Path, required=True)
    parser.add_argument("--base-artifact", type=Path, required=True)
    parser.add_argument("--embedded-artifact", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    report = verify_embedding_reuse(
        snapshot=load_snapshot_manifest(args.snapshot_manifest.parent),
        base_artifact=load_code_index_artifact(args.base_artifact),
        embedded_artifact=load_code_index_artifact(args.embedded_artifact),
    )
    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite reuse verification: {args.output}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8"
    )
    print(f"status={report['status']}")
    print(f"cached_records={report['cached_records']}")
    print(f"embedded_records={report['embedded_records']}")
    print(f"output={args.output}")
    print("external_calls_performed=false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
