from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.activation.code_modes import build_activation_request, evaluate_activation_preflight
from app.core.config import Settings


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare a pending code-mode activation request.")
    parser.add_argument("--readiness-report", type=Path, required=True)
    parser.add_argument("--requested-by", required=True)
    parser.add_argument("--request-output", type=Path, required=True)
    parser.add_argument("--preflight-output", type=Path, required=True)
    args = parser.parse_args()
    if args.request_output.exists() or args.preflight_output.exists():
        raise FileExistsError("Refusing to overwrite activation artifacts")
    settings = Settings()
    request = build_activation_request(
        settings=settings,
        readiness_report_path=args.readiness_report,
        requested_by=args.requested_by,
    )
    preflight = evaluate_activation_preflight(
        request=request,
        settings=settings,
        readiness_report_path=args.readiness_report,
        env_path=ROOT_DIR / ".env",
    )
    for path, model in ((args.request_output, request), (args.preflight_output, preflight)):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(model.model_dump_json(indent=2), encoding="utf-8")
    print(f"request_identity_sha256={request.request_identity_sha256}")
    print(f"ready_to_apply={str(preflight.ready_to_apply).lower()}")
    print("approval_required=true")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
