"""Create a no-overwrite, source-scoped enhancement registry for reviewed R3."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.code_ingestion.r3_benchmark import load_r3_benchmark_manifest
from app.fdd_code_lineage.enhancement_comments import IDENTITY


def _parse_marker(marker: str) -> dict[str, str | None]:
    match = IDENTITY.search(marker)
    if match is None:
        raise ValueError(f"R3 enhancement marker is not a supported FCIS identity: {marker}")
    title = match["title"].strip().rstrip("*/").strip().split("::", 1)[0].strip()
    if not title:
        raise ValueError(f"R3 enhancement marker has no title: {marker}")
    return {
        "code_release": match["release"].upper(),
        "requirement": match["requirement"].upper() if match["requirement"] else None,
        "title": title,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Create a source-scoped enhancement registry from a reviewed R3 benchmark."
    )
    parser.add_argument("--base-registry", type=Path, required=True)
    parser.add_argument("--benchmark-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--basis", required=True)
    args = parser.parse_args(argv)

    if args.output.exists():
        raise FileExistsError(f"Refusing to overwrite enhancement registry: {args.output}")
    benchmark = load_r3_benchmark_manifest(args.benchmark_manifest)
    if not benchmark.sme_reviewed or benchmark.review_status != "reviewed":
        raise ValueError("R3 enhancement registry requires a reviewed benchmark")
    basis = args.basis.strip()
    if not basis:
        raise ValueError("Registry basis must be nonblank")
    registry = json.loads(args.base_registry.read_text(encoding="utf-8"))
    if registry.get("schema_version") != "enhancement_fdd_registry_v1":
        raise ValueError("Unknown base enhancement registry schema")
    mappings = list(registry.get("mappings", []))
    for pair in benchmark.package_pairs:
        if pair.category != "fdd_enhancement":
            continue
        if len(pair.enhancement_markers) != 1 or len(pair.fdd_document_ids) != 1:
            raise ValueError(
                f"R3 enhancement pair must have one marker and FDD ID: {pair.pair_id}"
            )
        entry = _parse_marker(pair.enhancement_markers[0])
        entry.update(
            {
                "fdd_document_id": pair.fdd_document_ids[0],
                "source_paths": [pair.body_path],
                "basis": basis,
            }
        )
        mappings.append(entry)
    output = {"schema_version": "enhancement_fdd_registry_v1", "mappings": mappings}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print("status=created")
    print(f"r3_scoped_mappings={len(mappings) - len(registry.get('mappings', []))}")
    print(f"output={args.output}")
    print("external_calls_performed=false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
