"""Explicit generation promotion, separate from legacy flag-only activation."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.activation.code_generation import prepare, switch, verify_request, write_new
from app.activation.code_modes import ActivationApproval, build_activation_approval
from app.core.config import Settings


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    p = commands.add_parser('prepare')
    for name in ('code-artifact', 'analysis', 'lineage', 'dependency-ledger', 'code-report', 'combined-report', 'output'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--collection', required=True)
    p.add_argument('--requested-by', required=True)
    p = commands.add_parser('approve')
    p.add_argument('--request', type=Path, required=True)
    p.add_argument('--reviewer', required=True)
    p.add_argument('--output', type=Path, required=True)
    p = commands.add_parser('switch')
    p.add_argument('--request', type=Path, required=True)
    p.add_argument('--approval', type=Path, required=True)
    p.add_argument('--action', choices=('activate', 'rollback'), required=True)
    p.add_argument('--apply', action='store_true')
    p.add_argument('--services-stopped', action='store_true', help='Operator confirms FastAPI/MCP children are stopped; does not stop them.')
    args = parser.parse_args()
    os.chdir(ROOT)
    if args.command == 'prepare':
        if args.output.exists():
            raise FileExistsError('Refusing to overwrite the promotion request')
        request = prepare(root=ROOT, settings=Settings(), code_artifact=args.code_artifact,
                          analysis=args.analysis, lineage_path=args.lineage,
                          dependency_ledger=args.dependency_ledger, code_report=args.code_report,
                          combined_report=args.combined_report, collection=args.collection,
                          requested_by=args.requested_by)
        write_new(args.output, request)
        print(json.dumps({'status': 'pending_approval', 'request_identity_sha256': request['request_identity_sha256'],
                          'snapshot_id': request['snapshot_id'], 'output': str(args.output),
                          'external_api_calls': 0, 'activation_complete': False}, indent=2))
    else:
        request = json.loads(args.request.read_bytes())
        if args.command == 'approve':
            verify_request(request, ROOT)
            approval = build_activation_approval(
                request=SimpleNamespace(request_identity_sha256=request['request_identity_sha256']),
                approved_by=args.reviewer, paid_smoke_authorized=False,
                internal_evidence_disclosure_authorized=False)
            write_new(args.output, approval.model_dump(mode='json'))
            print('Approval recorded; no runtime state was changed. Paid requests authorized: 0.')
        else:
            if args.apply and not args.services_stopped:
                raise ValueError('Stop the intended serving processes before applying promotion or rollback')
            approval = ActivationApproval.model_validate_json(args.approval.read_bytes())
            result = switch(root=ROOT, request=request, approval=approval, action=args.action,
                            settings=Settings(), apply=args.apply)
            print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
