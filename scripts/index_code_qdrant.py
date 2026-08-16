from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.code_indexing.contract import load_code_index_artifact
from app.code_indexing.qdrant import index_code_artifact_new_collection
from app.vectorstore.qdrant_schema import create_persistent_qdrant_client


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Index one embedded code artifact into a new isolated collection.")
    parser.add_argument("artifact", type=Path)
    parser.add_argument("--qdrant-path", type=Path, required=True)
    parser.add_argument("--collection-name", required=True)
    parser.add_argument("--batch-size", type=int, default=64)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    artifact = load_code_index_artifact(args.artifact)
    client = create_persistent_qdrant_client(args.qdrant_path)
    try:
        result = index_code_artifact_new_collection(
            client,
            collection_name=args.collection_name,
            artifact=artifact,
            batch_size=args.batch_size,
        )
    finally:
        client.close()
    print(json.dumps({**result.__dict__, "external_calls_performed": False}, indent=2))


if __name__ == "__main__":
    main()
