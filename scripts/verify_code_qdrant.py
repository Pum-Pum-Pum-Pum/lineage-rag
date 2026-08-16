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
from app.code_indexing.qdrant import verify_code_collection
from app.vectorstore.qdrant_schema import create_persistent_qdrant_client


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Verify exact code artifact-to-Qdrant identity.")
    parser.add_argument("artifact", type=Path)
    parser.add_argument("--qdrant-path", type=Path, required=True)
    parser.add_argument("--collection-name", required=True)
    args = parser.parse_args(argv)
    artifact = load_code_index_artifact(args.artifact)
    client = create_persistent_qdrant_client(args.qdrant_path)
    try:
        result = verify_code_collection(
            client,
            collection_name=args.collection_name,
            artifact=artifact,
        )
    finally:
        client.close()
    print(json.dumps({**result.__dict__, "external_calls_performed": False}, indent=2))


if __name__ == "__main__":
    main()
