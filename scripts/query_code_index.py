from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from qdrant_client import QdrantClient

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.code_indexing.contract import load_code_index_artifact
from app.code_retrieval.service import retrieve_code_evidence


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Query one explicit custom-code index generation without automatic routing."
    )
    parser.add_argument("artifact", type=Path)
    parser.add_argument("query")
    parser.add_argument("--mode", choices=("lexical", "dense", "hybrid"), default="lexical")
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--max-units-per-parent", type=int, default=2)
    parser.add_argument("--qdrant-path", type=Path)
    parser.add_argument("--collection-name")
    parser.add_argument(
        "--query-vector-json",
        type=Path,
        help="JSON array produced by an explicitly authorized query-embedding operation.",
    )
    args = parser.parse_args()

    artifact = load_code_index_artifact(args.artifact)
    vector = None
    if args.query_vector_json:
        vector = json.loads(args.query_vector_json.read_text(encoding="utf-8"))
        if not isinstance(vector, list):
            raise ValueError("Query vector JSON must contain one numeric array")
    needs_dense = args.mode in {"dense", "hybrid"}
    if needs_dense and (not args.qdrant_path or not args.collection_name):
        raise ValueError("Dense/hybrid mode requires --qdrant-path and --collection-name")
    client = QdrantClient(path=str(args.qdrant_path)) if needs_dense else None
    result = retrieve_code_evidence(
        artifact=artifact,
        query=args.query,
        mode=args.mode,
        limit=args.limit,
        client=client,
        collection_name=args.collection_name,
        query_vector=vector,
        max_units_per_parent=args.max_units_per_parent,
    )
    print(result.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
