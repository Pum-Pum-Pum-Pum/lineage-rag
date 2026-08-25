from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.agentic_tools.replay_review import (
    build_replay_review_ledger,
    write_replay_review_ledger_no_overwrite,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Import one paid bounded-tool replay SME verdict."
    )
    parser.add_argument("--run-state", type=Path, required=True)
    parser.add_argument("--review-file", type=Path, required=True)
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--prior-review-ledger", type=Path, required=True)
    parser.add_argument("--reviewer", required=True)
    parser.add_argument("--global-acceptance-note", required=True)
    parser.add_argument("--output-file", type=Path, required=True)
    args = parser.parse_args()
    ledger = build_replay_review_ledger(
        run_state_path=args.run_state,
        review_file=args.review_file,
        authorization_path=args.authorization,
        prior_review_ledger_path=args.prior_review_ledger,
        reviewer=args.reviewer,
        global_acceptance_note=args.global_acceptance_note,
    )
    write_replay_review_ledger_no_overwrite(ledger, args.output_file)
    summary = ledger["summary"]
    print(f"case_id={ledger['case_id']}")
    print(f"case_remediation_closed={str(summary['case_remediation_closed']).lower()}")
    print(
        "effective_semantic_acceptances="
        f"{summary['effective_semantic_acceptances_after_replay']}/"
        f"{summary['original_suite_total_cases']}"
    )
    print(f"activation_authorized={str(summary['activation_authorized']).lower()}")
    print(f"ledger_identity_sha256={ledger['ledger_identity_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
