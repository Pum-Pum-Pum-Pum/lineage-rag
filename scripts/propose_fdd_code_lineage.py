"""Generate unapproved FDD/code matches from stored vectors and source comments."""
from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.fdd_code_lineage.semantic_proposals import build_proposals, write_report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--fdd-stage', type=Path, required=True)
    parser.add_argument('--snapshot-directory', type=Path, required=True)
    parser.add_argument('--analysis-directory', type=Path, required=True)
    parser.add_argument('--code-artifact', type=Path, required=True)
    parser.add_argument('--enhancement-registry', type=Path, required=True)
    parser.add_argument('--output-directory', type=Path, required=True)
    parser.add_argument('--top-k', type=int, default=5)
    parser.add_argument('--minimum-similarity', type=float, default=0.45)
    parser.add_argument('--ambiguity-margin', type=float, default=0.03)
    args = parser.parse_args()
    if args.output_directory.exists():
        raise FileExistsError('Proposal output already exists; choose a fresh run directory')
    report = build_proposals(fdd_stage=args.fdd_stage, snapshot_directory=args.snapshot_directory,
        analysis_directory=args.analysis_directory, code_artifact_path=args.code_artifact,
        registry_path=args.enhancement_registry, top_k=args.top_k,
        minimum_similarity=args.minimum_similarity, ambiguity_margin=args.ambiguity_margin,
        progress=lambda message: print(message, flush=True))
    write_report(report, args.output_directory)
    print(f'documents={len(report["documents"])}')
    print(f'candidate_pairs={sum(len(d["candidates"]) for d in report["documents"])}')
    print('status=candidate external_api_calls=0 automatic_approvals=0')
    print(f'review={args.output_directory / "review.html"}')


if __name__ == '__main__':
    main()
