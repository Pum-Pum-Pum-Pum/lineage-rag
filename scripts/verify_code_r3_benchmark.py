from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.code_ingestion.r3_benchmark import (
    load_r3_benchmark_manifest,
    verify_r3_benchmark_against_snapshot,
    verify_r3_benchmark_fdd_coverage,
    write_r3_benchmark_template,
)
from app.code_ingestion.snapshot_builder import load_snapshot_manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Validate the reviewed R3 seven-package benchmark without external calls."
    )
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--snapshot-manifest", type=Path)
    parser.add_argument("--fdd-directory", type=Path)
    parser.add_argument("--require-reviewed", action="store_true")
    parser.add_argument("--print-template", action="store_true")
    args = parser.parse_args(argv)
    if args.print_template:
        if args.manifest or args.snapshot_manifest or args.fdd_directory or args.require_reviewed:
            parser.error("--print-template cannot be combined with validation arguments")
        print(write_r3_benchmark_template())
        return 0
    if args.manifest is None:
        parser.error("--manifest is required unless --print-template is used")
    benchmark = load_r3_benchmark_manifest(args.manifest)
    if args.require_reviewed and benchmark.review_status != "reviewed":
        raise ValueError("R3 benchmark is not SME-reviewed")
    if args.fdd_directory:
        from app.retrieval.lexical_search import load_retrieval_ready_documents

        documents = load_retrieval_ready_documents(args.fdd_directory)
        verify_r3_benchmark_fdd_coverage(
            benchmark=benchmark,
            fdd_document_ids={item.document_id for item in documents},
        )
    report: dict[str, object] = {
        "schema_version": "code_r3_benchmark_verification_v1",
        "status": "valid",
        "review_status": benchmark.review_status,
        "package_pairs": len(benchmark.package_pairs),
        "fdd_coverage_verified": args.fdd_directory is not None,
        "external_calls_performed": False,
    }
    if args.snapshot_manifest:
        report = verify_r3_benchmark_against_snapshot(
            benchmark=benchmark,
            snapshot=load_snapshot_manifest(args.snapshot_manifest.parent),
        )
        report["review_status"] = benchmark.review_status
    print(json.dumps(report, indent=2, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
