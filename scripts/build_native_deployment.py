from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.deployment.native_package import build_native_package


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build the deterministic native deployment bundle."
    )
    parser.add_argument(
        "--output",
        default="data/exports/deployment/lineage-rag-native.zip",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = build_native_package(
        project_root=ROOT_DIR,
        output_path=ROOT_DIR / args.output,
    )
    payload = asdict(result)
    payload["archive_path"] = str(result.archive_path)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
