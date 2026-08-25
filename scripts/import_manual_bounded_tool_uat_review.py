from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.agentic_tools.uat_review import promote_manual_uat_global_acceptance


def main() -> int:
    parser = argparse.ArgumentParser(description="Record global acceptance of a manual UAT batch.")
    parser.add_argument("--draft-manifest", type=Path, required=True)
    parser.add_argument("--batch-report", type=Path, required=True)
    parser.add_argument("--review-packet", type=Path, required=True)
    parser.add_argument("--reviewed-manifest", type=Path, required=True)
    parser.add_argument("--ledger-file", type=Path, required=True)
    parser.add_argument("--reviewer", required=True)
    parser.add_argument("--approval-note", required=True)
    parser.add_argument("--paid-use-authorized", action="store_true")
    parser.add_argument("--internal-evidence-disclosure-authorized", action="store_true")
    args = parser.parse_args()
    ledger = promote_manual_uat_global_acceptance(
        draft_manifest=args.draft_manifest,
        batch_report=args.batch_report,
        review_packet=args.review_packet,
        reviewed_manifest=args.reviewed_manifest,
        ledger_file=args.ledger_file,
        reviewer=args.reviewer,
        approval_note=args.approval_note,
        paid_use_authorized=args.paid_use_authorized,
        internal_evidence_disclosure_authorized=args.internal_evidence_disclosure_authorized,
    )
    print(f"reviewed_cases={ledger['case_count']}")
    print(f"maximum_answer_requests={ledger['maximum_answer_requests']}")
    print(f"maximum_query_embedding_requests={ledger['maximum_query_embedding_requests']}")
    print(f"ledger_identity_sha256={ledger['ledger_identity_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
