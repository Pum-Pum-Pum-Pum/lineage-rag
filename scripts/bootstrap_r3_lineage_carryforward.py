"""Create an R3 candidate that carries reviewed R2 lineage forward safely.

This does not copy review status.  It proves that every retrieval record for a
previously mapped source path is identical in the R2 and R3 embedded artifacts,
then emits a new *candidate* bound to R3.  An SME must still review and import
that candidate before it can be consolidated with newly reviewed R3 mappings.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.code_indexing.contract import load_code_index_artifact
from app.fdd_code_lineage.models import (
    build_lineage_artifact,
    create_mapping,
    validate_lineage_artifact,
    write_lineage_artifact_no_overwrite,
)
from app.fdd_code_lineage.reviewed_bundle import load_reviewed_lineage


def _fingerprints(artifact, source_path: str) -> Counter[tuple[object, ...]]:
    """Return the full retrieval identity for one logical source path.

    The cache key establishes the embedding input/model identity.  The remaining
    fields make line/range, displayed routine, and source content drift visible.
    A Counter, rather than a set, fails closed if duplicate units appear.
    """

    return Counter(
        (
            record.content_sha256,
            record.cache_key,
            record.source_kind,
            record.display_name,
            record.source_map.start_line,
            record.source_map.end_line,
            record.source_map.start_offset,
            record.source_map.end_offset,
            record.parent_source_map.start_line if record.parent_source_map else None,
            record.parent_source_map.end_line if record.parent_source_map else None,
        )
        for record in artifact.records
        if record.source_path == source_path
    )


def _fdd_document_ids(directory: Path) -> set[str]:
    values: set[str] = set()
    for path in directory.glob("*.retrieval_ready.json"):
        document_id = str(json.loads(path.read_text(encoding="utf-8")).get("document_id", "")).strip()
        if not document_id:
            raise ValueError(f"FDD artifact has no document ID: {path}")
        values.add(document_id)
    if not values:
        raise ValueError(f"No retrieval-ready FDD artifacts found in {directory}")
    return values


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Create an R3 candidate from unchanged, previously reviewed lineage."
    )
    parser.add_argument("--source-lineage", type=Path, required=True)
    parser.add_argument("--source-code-artifact", type=Path, required=True)
    parser.add_argument("--target-code-artifact", type=Path, required=True)
    parser.add_argument("--analysis-directory", type=Path, required=True)
    parser.add_argument("--fdd-processed-directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite R3 carry-forward candidate: {args.output}")
    source_lineage = load_reviewed_lineage(args.source_lineage)
    source_artifact = load_code_index_artifact(args.source_code_artifact)
    target_artifact = load_code_index_artifact(args.target_code_artifact)
    if source_lineage.code_artifact_identity_sha256 != source_artifact.artifact_identity_sha256:
        raise ValueError("Source lineage is not bound to the supplied source code artifact")
    if target_artifact.status != "embedded" or target_artifact.dependency_review_status != "reviewed":
        raise ValueError("Target R3 code artifact must be embedded and dependency-reviewed")
    if source_lineage.fdd_generation != "functional_specs_v9":
        raise ValueError("R3 carry-forward only supports the reviewed functional_specs_v9 FDD generation")

    paths = sorted({target.path for mapping in source_lineage.mappings for target in mapping.targets})
    path_counts: dict[str, int] = {}
    for path in paths:
        source_records = _fingerprints(source_artifact, path)
        target_records = _fingerprints(target_artifact, path)
        if not source_records or not target_records:
            raise ValueError(f"Carry-forward source path has no retrieval records: {path}")
        if source_records != target_records:
            raise ValueError(
                "Carry-forward requires unchanged retrieval records for mapped path: "
                f"{path}"
            )
        path_counts[path] = sum(source_records.values())

    mappings = [
        create_mapping(
            fdd_document_id=mapping.fdd_document_id,
            fdd_release_label=mapping.fdd_release_label,
            code_snapshot_id=target_artifact.snapshot_id,
            targets=mapping.targets,
            rationale=(
                "R3 carry-forward candidate from reviewed mapping "
                f"{mapping.mapping_id} in source lineage "
                f"{source_lineage.artifact_identity_sha256}. Every retrieval record "
                "for its mapped source path has an identical source-content, cache-key, "
                "routine, and source-range fingerprint in the R3 artifact. The earlier "
                "review is retained as provenance, but R3 requires fresh SME confirmation."
            ),
        )
        for mapping in source_lineage.mappings
    ]
    candidate = build_lineage_artifact(
        fdd_generation=source_lineage.fdd_generation,
        code_artifact=target_artifact,
        mappings=mappings,
    )
    validate_lineage_artifact(
        candidate,
        fdd_document_ids=_fdd_document_ids(args.fdd_processed_directory),
        code_artifact=target_artifact,
        analysis_directory=args.analysis_directory,
    )
    write_lineage_artifact_no_overwrite(candidate, args.output)
    print("status=candidate")
    print(f"source_lineage_artifact_identity_sha256={source_lineage.artifact_identity_sha256}")
    print(f"source_code_artifact_identity_sha256={source_artifact.artifact_identity_sha256}")
    print(f"target_code_artifact_identity_sha256={target_artifact.artifact_identity_sha256}")
    print(f"mappings={len(candidate.mappings)}")
    print(f"targets={sum(len(item.targets) for item in candidate.mappings)}")
    print(f"unchanged_paths={len(path_counts)}")
    print(f"unchanged_path_record_counts={json.dumps(path_counts, sort_keys=True)}")
    print(f"output={args.output}")
    print("external_calls_performed=false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
