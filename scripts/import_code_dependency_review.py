from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.code_ingestion.dependency_review_ledger import (
    import_dependency_review_markdown,
    write_dependency_review_ledger_no_overwrite,
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate an SME-edited code dependency packet and publish a local ledger."
    )
    parser.add_argument("packet_json", type=Path)
    parser.add_argument("reviewed_markdown", type=Path)
    parser.add_argument("--reviewer", required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    ledger = import_dependency_review_markdown(
        args.packet_json,
        args.reviewed_markdown,
        reviewer=args.reviewer,
    )
    output = write_dependency_review_ledger_no_overwrite(ledger, args.output)
    print(
        json.dumps(
            {
                "status": ledger.status,
                "snapshot_id": ledger.snapshot_id,
                "parser_generation": ledger.parser_generation,
                "packet_identity_sha256": ledger.packet_identity_sha256,
                "ledger_identity_sha256": ledger.ledger_identity_sha256,
                "decisions": len(ledger.decisions),
                "output": str(output.resolve()),
                "external_calls_performed": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
