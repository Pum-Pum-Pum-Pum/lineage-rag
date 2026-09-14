"""Create an unapproved R3 lineage definition from reviewed comment candidates."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.code_ingestion.r3_benchmark import load_r3_benchmark_manifest


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Create a candidate-only R3 lineage definition from comment-region proposals."
    )
    parser.add_argument("--benchmark-manifest", type=Path, required=True)
    parser.add_argument("--proposal-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite candidate lineage definition: {args.output}")
    benchmark = load_r3_benchmark_manifest(args.benchmark_manifest)
    if not benchmark.sme_reviewed or benchmark.review_status != "reviewed":
        raise ValueError("R3 lineage definition requires a reviewed R3 benchmark")
    proposal = json.loads(args.proposal_report.read_text(encoding="utf-8"))
    if proposal.get("status") != "candidate":
        raise ValueError("Proposal report must be an unapproved candidate report")
    if proposal.get("fdd_generation") != benchmark.fdd_generation:
        raise ValueError("Proposal FDD generation does not match reviewed R3 benchmark")

    documents = {item["document_id"]: item for item in proposal.get("documents", [])}
    mappings: list[dict[str, object]] = []
    selected_candidates = 0
    for pair in benchmark.package_pairs:
        if pair.category != "fdd_enhancement":
            continue
        fdd_document_id = pair.fdd_document_ids[0]
        document = documents.get(fdd_document_id)
        if document is None:
            raise ValueError(f"Proposal report omitted benchmark FDD: {fdd_document_id}")
        candidates = [
            candidate
            for candidate in document.get("candidates", [])
            if candidate.get("basis") == "comment_region_and_similarity"
            and candidate.get("target", {}).get("path") == pair.body_path
        ]
        if not candidates:
            raise ValueError(
                f"No comment-region implementation candidate for reviewed pair: {pair.pair_id}"
            )
        targets = []
        seen_targets: set[tuple[str, str, str]] = set()
        proposal_ids: list[str] = []
        for candidate in candidates:
            target = candidate["target"]
            identity = (
                target["path"],
                target["qualified_name"],
                target["overload_discriminator_hash"],
            )
            if identity in seen_targets:
                continue
            seen_targets.add(identity)
            proposal_ids.append(candidate["proposal_id"])
            targets.append(
                {
                    "module_id": target["module_id"],
                    "path": target["path"],
                    "qualified_name": target["qualified_name"],
                    "symbol_kind": target["symbol_kind"],
                    "selector_scope": "overload",
                    "overload_discriminator_hash": target["overload_discriminator_hash"],
                    "rationale": (
                        "Candidate implementation selected from reviewed comment-region "
                        f"proposal {candidate['proposal_id']}; similarity "
                        f"{candidate['cosine_similarity']}. This is not approval."
                    ),
                }
            )
        release = fdd_document_id.split("$ASNB_", 1)[1].split("_", 1)[0]
        mappings.append(
            {
                "fdd_document_id": fdd_document_id,
                "fdd_release_label": release,
                "mapping_status": "candidate",
                "rationale": (
                    f"Candidate R3 {pair.pair_id} relationship: exact reviewed enhancement "
                    "comment region and local similarity selected these implementation "
                    f"routines. Proposal IDs: {', '.join(proposal_ids)}. SME lineage review "
                    "is still required; no broader package behavior is claimed."
                ),
                "targets": targets,
            }
        )
        selected_candidates += len(targets)
    definition = {"fdd_generation": benchmark.fdd_generation, "mappings": mappings}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(definition, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("status=candidate_definition")
    print(f"mappings={len(mappings)}")
    print(f"implementation_targets={selected_candidates}")
    print(f"output={args.output}")
    print("external_calls_performed=false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
