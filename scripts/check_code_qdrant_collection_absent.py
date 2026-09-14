from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.vectorstore.qdrant_schema import create_persistent_qdrant_client


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Fail closed unless a new local code Qdrant collection name is available."
    )
    parser.add_argument("--qdrant-path", type=Path, required=True)
    parser.add_argument("--collection-name", required=True)
    args = parser.parse_args(argv)
    if not args.collection_name.startswith("code_custom_"):
        raise ValueError("Code collection names must start with 'code_custom_'")
    client = create_persistent_qdrant_client(args.qdrant_path)
    try:
        if client.collection_exists(args.collection_name):
            raise FileExistsError(
                f"Code collection already exists and will not be modified: {args.collection_name}"
            )
    finally:
        client.close()
    print(
        json.dumps(
            {
                "status": "available",
                "collection_name": args.collection_name,
                "external_calls_performed": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
