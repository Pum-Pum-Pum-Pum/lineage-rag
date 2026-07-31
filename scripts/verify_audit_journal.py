from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.core.audit_journal import verify_audit_journal
from app.core.config import get_settings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Verify the local API audit journal without printing events.",
    )
    parser.add_argument("--expected-record-count", type=int)
    parser.add_argument("--expected-final-hmac")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = get_settings()
    result = verify_audit_journal(
        settings.audit_journal_path,
        settings.audit_hmac_key.get_secret_value(),
        expected_record_count=args.expected_record_count,
        expected_final_hmac=args.expected_final_hmac,
    )
    print(json.dumps(asdict(result), indent=2))
    if not result.valid:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
