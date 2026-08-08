from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.llm.fdd_sme_remediation import build_remediation_report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate unresolved SME findings against the active FDD artifacts."
    )
    parser.add_argument("--review-ledger", type=Path, required=True)
    parser.add_argument("--remediation-plan", type=Path, required=True)
    parser.add_argument("--artifact-directory", type=Path, required=True)
    parser.add_argument("--output-file", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = build_remediation_report(
        review_ledger_path=args.review_ledger,
        remediation_plan_path=args.remediation_plan,
        artifact_directory=args.artifact_directory,
    )
    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    args.output_file.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"phase_1_gate_status={report['phase_1_gate_status']}")
    print(f"blocking_case_ids={','.join(report['blocking_case_ids'])}")
    print(f"report={args.output_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
