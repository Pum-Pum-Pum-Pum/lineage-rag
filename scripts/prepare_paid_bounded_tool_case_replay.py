from __future__ import annotations

import argparse
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.agentic_tools.replay import (
    build_case_replay_authorization,
    write_authorization_no_overwrite,
)
from app.core.config import get_settings


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare one hash-bound paid case replay.")
    parser.add_argument("--case-id", required=True)
    parser.add_argument("--reviewed-manifest", type=Path, required=True)
    parser.add_argument("--prior-review-ledger", type=Path, required=True)
    parser.add_argument("--prior-trace", type=Path, required=True)
    parser.add_argument("--local-uat-report", type=Path, required=True)
    parser.add_argument("--approval-note", required=True)
    parser.add_argument("--output-file", type=Path, required=True)
    args = parser.parse_args()
    value = build_case_replay_authorization(
        case_id=args.case_id,
        reviewed_manifest=args.reviewed_manifest,
        prior_review_ledger=args.prior_review_ledger,
        prior_trace=args.prior_trace,
        local_uat_report=args.local_uat_report,
        answer_model=get_settings().openai_chat_model,
        approval_note=args.approval_note,
    )
    write_authorization_no_overwrite(value, args.output_file)
    print(f"answer_requests_authorized={value['maximum_answer_requests']}")
    print(f"query_embedding_requests_authorized={value['maximum_query_embedding_requests']}")
    print(f"automatic_retries={value['automatic_retries']}")
    print(f"authorization_identity_sha256={value['authorization_identity_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
