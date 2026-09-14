from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.code_indexing.contract import (
    code_index_generation_name,
    load_code_index_artifact,
    write_code_index_artifact_no_overwrite,
)
from app.code_indexing.embedding import (
    embed_code_index_artifact,
    inspect_code_embedding_reuse,
)
from app.embeddings.client import get_embedding_client


AUTHORIZATION_TEXT = "I_AUTHORIZE_OPENAI_CODE_DISCLOSURE_AND_COST"


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Embed an approved code index contract with OpenAI.")
    parser.add_argument("prepared_artifact", type=Path)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--cache-artifact", type=Path, action="append", default=[])
    parser.add_argument("--request-batch-size", type=int, default=32)
    parser.add_argument(
        "--max-retries",
        type=int,
        default=0,
        help="OpenAI SDK retry count for paid code embeddings; defaults to zero.",
    )
    parser.add_argument("--authorization", default="")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    artifact = load_code_index_artifact(args.prepared_artifact)
    reuse_plan = inspect_code_embedding_reuse(
        artifact,
        cache_artifact_paths=args.cache_artifact,
    )
    if args.dry_run:
        print(json.dumps({
            "status": "dry_run",
            "snapshot_id": artifact.snapshot_id,
            "records": reuse_plan.total_records,
            "unique_embedding_inputs": reuse_plan.unique_embedding_inputs,
            "cached_records": reuse_plan.cached_records,
            "external_embedding_records": reuse_plan.external_embedding_records,
            "cached_unique_embedding_inputs": reuse_plan.cached_unique_embedding_inputs,
            "external_embedding_inputs": reuse_plan.external_embedding_inputs,
            "embedding_model": artifact.embedding_model,
            "cache_artifacts": [str(path) for path in args.cache_artifact],
            "external_code_would_be_sent": reuse_plan.external_embedding_inputs > 0,
            "external_calls_performed": False,
        }, indent=2))
        return
    if artifact.dependency_review_status != "reviewed":
        raise PermissionError(
            "Paid code embedding is blocked until the dependency evaluation labels "
            "are SME-reviewed and a reviewed index contract is prepared."
        )
    if args.authorization != AUTHORIZATION_TEXT:
        raise PermissionError(
            "Paid code embedding is blocked. Pass the exact authorization token only after "
            "approval to send internal code excerpts to OpenAI."
        )
    if args.max_retries < 0:
        raise ValueError("--max-retries must be zero or greater")
    embedded, summary = embed_code_index_artifact(
        artifact,
        client=get_embedding_client(max_retries=args.max_retries),
        cache_artifact_paths=args.cache_artifact,
        request_batch_size=args.request_batch_size,
    )
    target = args.output_root / artifact.snapshot_id / code_index_generation_name(
        artifact.embedding_model
    )
    output = write_code_index_artifact_no_overwrite(embedded, target)
    print(json.dumps({
        "status": embedded.status,
        **summary.__dict__,
        "artifact_identity_sha256": embedded.artifact_identity_sha256,
        "max_retries": args.max_retries,
        "output": str(output.resolve()),
        "external_calls_performed": summary.request_count > 0,
    }, indent=2))


if __name__ == "__main__":
    main()
