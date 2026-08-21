from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from qdrant_client import QdrantClient

from app.code_indexing.contract import load_code_index_artifact
from app.fdd_code_lineage.combined_retrieval import retrieve_combined_evidence
from app.fdd_code_lineage.models import FddCodeLineageArtifact
from app.retrieval.lexical_search import (
    load_retrieval_ready_documents,
    search_lexical_artifacts,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run explicit local combined retrieval without automatic routing or LLM calls."
    )
    parser.add_argument("query")
    parser.add_argument("--fdd-generation", required=True)
    parser.add_argument("--fdd-directory", type=Path, required=True)
    parser.add_argument("--code-artifact", type=Path, required=True)
    parser.add_argument("--lineage-artifact", type=Path, required=True)
    parser.add_argument("--analysis-directory", type=Path, required=True)
    parser.add_argument("--code-mode", choices=("lexical", "dense", "hybrid"), default="lexical")
    parser.add_argument("--limit", type=int, default=5)
    parser.add_argument("--max-units-per-parent", type=int, default=2)
    parser.add_argument("--qdrant-path", type=Path)
    parser.add_argument("--collection-name")
    parser.add_argument("--query-vector-json", type=Path)
    args = parser.parse_args()

    code_artifact = load_code_index_artifact(args.code_artifact)
    lineage = FddCodeLineageArtifact.model_validate_json(
        args.lineage_artifact.read_text(encoding="utf-8")
    )
    fdd_results = search_lexical_artifacts(
        args.fdd_directory, args.query, limit=args.limit
    )
    known_fdd_ids = {
        item.document_id
        for item in load_retrieval_ready_documents(args.fdd_directory)
    }
    needs_dense = args.code_mode in {"dense", "hybrid"}
    if needs_dense and (
        not args.qdrant_path
        or not args.collection_name
        or not args.query_vector_json
    ):
        raise ValueError(
            "Dense/hybrid code retrieval requires Qdrant path, collection, and query vector"
        )
    vector = (
        json.loads(args.query_vector_json.read_text(encoding="utf-8"))
        if args.query_vector_json
        else None
    )
    client = QdrantClient(path=str(args.qdrant_path)) if needs_dense else None
    try:
        result = retrieve_combined_evidence(
            query=args.query,
            fdd_results=fdd_results,
            fdd_generation=args.fdd_generation,
            known_fdd_document_ids=known_fdd_ids,
            code_artifact=code_artifact,
            lineage_artifact=lineage,
            analysis_directory=args.analysis_directory,
            code_mode=args.code_mode,
            code_limit=args.limit,
            client=client,
            collection_name=args.collection_name,
            query_vector=vector,
            code_max_units_per_parent=args.max_units_per_parent,
        )
    finally:
        if client is not None:
            client.close()
    print(result.model_dump_json(indent=2))


if __name__ == "__main__":
    main()
