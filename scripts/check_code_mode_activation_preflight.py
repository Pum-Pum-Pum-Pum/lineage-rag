from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.activation.code_modes import (
    ActivationApproval,
    ActivationRequest,
    evaluate_activation_preflight,
)
from app.core.config import Settings


def main() -> int:
    parser = argparse.ArgumentParser(description="Re-evaluate one activation request.")
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--readiness-report", type=Path, required=True)
    parser.add_argument("--env-file", type=Path, default=ROOT_DIR / ".env")
    parser.add_argument("--approval", type=Path)
    parser.add_argument("--output-file", type=Path, required=True)
    parser.add_argument("--action", choices=("activate", "rollback"), default="activate")
    args = parser.parse_args()
    if args.output_file.exists():
        raise FileExistsError(f"Refusing to overwrite preflight: {args.output_file}")
    request = ActivationRequest.model_validate_json(args.request.read_text(encoding="utf-8"))
    approval = (
        ActivationApproval.model_validate_json(args.approval.read_text(encoding="utf-8"))
        if args.approval
        else None
    )
    report = evaluate_activation_preflight(
        request=request,
        settings=Settings(_env_file=args.env_file),
        readiness_report_path=args.readiness_report,
        env_path=args.env_file,
        approval=approval,
        action=args.action,
    )
    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    args.output_file.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    failed = ",".join(item.name for item in report.checks if not item.passed)
    print(f"ready_to_apply={str(report.ready_to_apply).lower()}")
    print(f"failed_checks={failed or 'none'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
