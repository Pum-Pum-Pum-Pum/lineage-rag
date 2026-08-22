from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.activation.code_modes import (
    ActivationApproval,
    ActivationRequest,
    evaluate_activation_preflight,
    switch_code_modes,
)
from app.core.config import Settings


def main() -> int:
    parser = argparse.ArgumentParser(description="Approval-bound atomic CODE_MODES_ENABLED switch.")
    parser.add_argument("action", choices=("activate", "rollback"))
    parser.add_argument("--env-file", type=Path, default=ROOT_DIR / ".env")
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--approval", type=Path, required=True)
    parser.add_argument("--readiness-report", type=Path, required=True)
    parser.add_argument("--apply", action="store_true", help="Apply the atomic file change; otherwise dry-run.")
    args = parser.parse_args()
    request = ActivationRequest.model_validate_json(args.request.read_text(encoding="utf-8"))
    approval = ActivationApproval.model_validate_json(args.approval.read_text(encoding="utf-8"))
    settings = Settings(_env_file=args.env_file)
    preflight = evaluate_activation_preflight(
        request=request,
        settings=settings,
        readiness_report_path=args.readiness_report,
        env_path=args.env_file,
        approval=approval,
        action=args.action,
    )
    result = switch_code_modes(
        env_path=args.env_file,
        action=args.action,
        request=request,
        approval=approval,
        preflight=preflight,
        apply=args.apply,
    )
    print(json.dumps(result.model_dump(mode="json"), sort_keys=True))
    print("service_restart_required=true" if result.applied else "dry_run=true")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
