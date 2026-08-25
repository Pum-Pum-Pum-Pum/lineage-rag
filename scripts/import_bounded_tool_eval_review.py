from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.agentic_tools.review import promote_reviewed_manifest


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Promote an accepted bounded-tool SME packet into reviewed artifacts."
    )
    parser.add_argument("--draft-manifest", type=Path, required=True)
    parser.add_argument("--review-packet", type=Path, required=True)
    parser.add_argument("--reviewed-manifest", type=Path, required=True)
    parser.add_argument("--ledger-file", type=Path, required=True)
    parser.add_argument("--reviewer", required=True)
    parser.add_argument("--approval-note", required=True)
    args = parser.parse_args()
    ledger = promote_reviewed_manifest(
        draft_manifest=args.draft_manifest,
        review_packet=args.review_packet,
        reviewed_manifest=args.reviewed_manifest,
        ledger_file=args.ledger_file,
        reviewer=args.reviewer,
        approval_note=args.approval_note,
    )
    print(f"reviewed_cases={ledger['summary']['accepted_cases']}")
    print(f"reviewed_manifest_sha256={ledger['reviewed_manifest_sha256']}")
    print(f"ledger_identity_sha256={ledger['ledger_identity_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
