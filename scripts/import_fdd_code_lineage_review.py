from __future__ import annotations

import argparse
import hashlib
import re
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.code_indexing.contract import load_code_index_artifact
from app.fdd_code_lineage.models import (
    FddCodeLineageArtifact,
    build_lineage_artifact,
    create_mapping,
    validate_lineage_artifact,
    write_lineage_artifact_no_overwrite,
)


SECTION_PATTERN = re.compile(
    r"^## \d+\. (?P<document>.+?)\r?\n(?P<body>.*?)(?=^## \d+\.|\Z)",
    re.MULTILINE | re.DOTALL,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Import a completed SME packet into a hash-bound reviewed lineage artifact."
    )
    parser.add_argument("candidate_artifact", type=Path)
    parser.add_argument("review_packet", type=Path)
    parser.add_argument("--reviewer", required=True)
    parser.add_argument("--code-artifact", type=Path, required=True)
    parser.add_argument("--analysis-directory", type=Path, required=True)
    parser.add_argument("--fdd-processed-directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    candidate = FddCodeLineageArtifact.model_validate_json(
        args.candidate_artifact.read_text(encoding="utf-8")
    )
    if candidate.status != "candidate":
        raise ValueError("Review import requires a candidate lineage artifact")
    packet_text = args.review_packet.read_text(encoding="utf-8")
    if f"Candidate artifact: `{candidate.artifact_identity_sha256}`" not in packet_text:
        raise ValueError("Review packet is not bound to the candidate artifact")
    decisions = _parse_decisions(packet_text)
    expected_ids = {item.mapping_id for item in candidate.mappings}
    if set(decisions) != expected_ids:
        raise ValueError("Review packet mapping IDs do not exactly match the candidate artifact")
    unresolved = {
        mapping_id: decision[0]
        for mapping_id, decision in decisions.items()
        if decision[0] != "reviewed"
    }
    if unresolved:
        raise ValueError(f"Lineage review is not fully approved: {unresolved}")

    reviewed_mappings = [
        create_mapping(
            fdd_document_id=item.fdd_document_id,
            fdd_release_label=item.fdd_release_label,
            code_snapshot_id=item.code_snapshot_id,
            targets=item.targets,
            rationale=decisions[item.mapping_id][1] or item.rationale,
            mapping_status="reviewed",
            reviewer=args.reviewer,
        )
        for item in candidate.mappings
    ]
    code_artifact = load_code_index_artifact(args.code_artifact)
    packet_hash = _sha256(args.review_packet.read_bytes())
    reviewed = build_lineage_artifact(
        fdd_generation=candidate.fdd_generation,
        code_artifact=code_artifact,
        mappings=reviewed_mappings,
        source_candidate_artifact_identity_sha256=candidate.artifact_identity_sha256,
        review_packet_sha256=packet_hash,
        reviewer=args.reviewer,
    )
    validate_lineage_artifact(
        reviewed,
        fdd_document_ids=_load_fdd_document_ids(args.fdd_processed_directory),
        code_artifact=code_artifact,
        analysis_directory=args.analysis_directory,
    )
    write_lineage_artifact_no_overwrite(reviewed, args.output)
    print(f"status={reviewed.status}")
    print(f"mappings={len(reviewed.mappings)}")
    print(f"review_packet_sha256={packet_hash}")
    print(f"artifact_identity_sha256={reviewed.artifact_identity_sha256}")
    print(f"output={args.output}")


def _parse_decisions(text: str) -> dict[str, tuple[str, str]]:
    decisions: dict[str, tuple[str, str]] = {}
    for match in SECTION_PATTERN.finditer(text):
        body = match.group("body")
        mapping_match = re.search(r"^- Mapping ID: `([0-9a-f]{64})`", body, re.MULTILINE)
        verdict_match = re.search(r"^SME verdict:\s*(\S+)", body, re.MULTILINE)
        rationale_match = re.search(r"^SME rationale:\s*(.*)$", body, re.MULTILINE)
        if not mapping_match or not verdict_match or not rationale_match:
            raise ValueError(f"Incomplete SME decision for {match.group('document')}")
        mapping_id = mapping_match.group(1)
        if mapping_id in decisions:
            raise ValueError(f"Duplicate SME mapping decision: {mapping_id}")
        rationale = rationale_match.group(1).strip()
        if not rationale:
            raise ValueError(f"SME rationale is blank for mapping {mapping_id}")
        decisions[mapping_id] = (verdict_match.group(1).strip(), rationale)
    if not decisions:
        raise ValueError("No SME mapping decisions found")
    return decisions


def _load_fdd_document_ids(directory: Path) -> set[str]:
    import json

    return {
        str(json.loads(path.read_text(encoding="utf-8"))["document_id"])
        for path in directory.glob("*.retrieval_ready.json")
    }


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


if __name__ == "__main__":
    main()
