from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.code_indexing.contract import load_code_index_artifact
from app.fdd_code_lineage.models import (
    FddCodeTarget,
    build_lineage_artifact,
    create_mapping,
    validate_lineage_artifact,
    write_lineage_artifact_no_overwrite,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build and validate a candidate FDD-to-code lineage artifact."
    )
    parser.add_argument("definition", type=Path)
    parser.add_argument("--code-artifact", type=Path, required=True)
    parser.add_argument("--analysis-directory", type=Path, required=True)
    parser.add_argument("--fdd-processed-directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    definition = json.loads(args.definition.read_text(encoding="utf-8"))
    code_artifact = load_code_index_artifact(args.code_artifact)
    document_ids = _load_fdd_document_ids(args.fdd_processed_directory)
    mappings = []
    for item in definition["mappings"]:
        if item.get("mapping_status", "candidate") != "candidate":
            raise ValueError("This preparation command creates candidate mappings only")
        mappings.append(
            create_mapping(
                fdd_document_id=item["fdd_document_id"],
                fdd_release_label=item["fdd_release_label"],
                code_snapshot_id=code_artifact.snapshot_id,
                targets=tuple(FddCodeTarget.model_validate(value) for value in item["targets"]),
                rationale=item["rationale"],
            )
        )
    artifact = build_lineage_artifact(
        fdd_generation=definition["fdd_generation"],
        code_artifact=code_artifact,
        mappings=mappings,
    )
    summary = validate_lineage_artifact(
        artifact,
        fdd_document_ids=document_ids,
        code_artifact=code_artifact,
        analysis_directory=args.analysis_directory,
    )
    write_lineage_artifact_no_overwrite(artifact, args.output)
    print(json.dumps({**summary, "output": str(args.output), "artifact_identity_sha256": artifact.artifact_identity_sha256}, indent=2))


def _load_fdd_document_ids(directory: Path) -> set[str]:
    document_ids: set[str] = set()
    for path in sorted(directory.glob("*.retrieval_ready.json")):
        payload = json.loads(path.read_text(encoding="utf-8"))
        document_id = str(payload.get("document_id", "")).strip()
        if not document_id:
            raise ValueError(f"FDD artifact has no document_id: {path}")
        document_ids.add(document_id)
    if not document_ids:
        raise ValueError(f"No retrieval-ready FDD artifacts found in {directory}")
    return document_ids


if __name__ == "__main__":
    main()
