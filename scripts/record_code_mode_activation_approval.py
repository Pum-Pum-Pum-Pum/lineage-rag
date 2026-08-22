from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.activation.code_modes import (
    ActivationRequest,
    build_activation_approval,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Record a human activation decision bound to one request."
    )
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--approved-by", required=True)
    parser.add_argument("--output-file", type=Path, required=True)
    parser.add_argument("--confirm-approved", action="store_true")
    parser.add_argument("--authorize-paid-smoke", action="store_true")
    parser.add_argument("--authorize-internal-evidence-disclosure", action="store_true")
    args = parser.parse_args()
    if not args.confirm_approved:
        raise PermissionError("Explicit --confirm-approved is required")
    if args.output_file.exists():
        raise FileExistsError(f"Refusing to overwrite approval: {args.output_file}")
    request = ActivationRequest.model_validate_json(
        args.request.read_text(encoding="utf-8")
    )
    approval = build_activation_approval(
        request=request,
        approved_by=args.approved_by,
        paid_smoke_authorized=args.authorize_paid_smoke,
        internal_evidence_disclosure_authorized=(
            args.authorize_internal_evidence_disclosure
        ),
    )
    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    args.output_file.write_text(
        approval.model_dump_json(indent=2), encoding="utf-8"
    )
    print(f"request_identity_sha256={request.request_identity_sha256}")
    print(f"approval_identity_sha256={approval.approval_identity_sha256}")
    print(f"paid_smoke_authorized={str(approval.paid_smoke_authorized).lower()}")
    print(
        "internal_evidence_disclosure_authorized="
        + str(approval.internal_evidence_disclosure_authorized).lower()
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
