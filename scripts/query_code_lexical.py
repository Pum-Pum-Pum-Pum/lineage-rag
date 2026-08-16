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
from app.code_indexing.lexical import search_code_lexical_artifact


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Search one isolated local code lexical artifact.")
    parser.add_argument("artifact", type=Path)
    parser.add_argument("query")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--source-kind")
    args = parser.parse_args(argv)
    artifact = load_code_index_artifact(args.artifact)
    results = search_code_lexical_artifact(
        artifact,
        args.query,
        limit=args.limit,
        source_kind=args.source_kind,
    )
    print(json.dumps({
        "snapshot_id": artifact.snapshot_id,
        "artifact_identity_sha256": artifact.artifact_identity_sha256,
        "query": args.query,
        "result_count": len(results),
        "results": [
            {
                "unit_id": result.payload["unit_id"],
                "source_path": result.payload["document_name"],
                "source_kind": result.payload["source_kind"],
                "parent_unit_id": result.payload["parent_unit_id"],
                "score": result.score,
                "matched_query_terms": result.payload["matched_query_terms"],
            }
            for result in results
        ],
        "external_calls_performed": False,
    }, indent=2))


if __name__ == "__main__":
    main()
