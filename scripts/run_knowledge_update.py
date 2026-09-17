"""One resumable command surface for coordinated FDD, code, and lineage updates."""
from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.code_updates.storage import now
from app.knowledge_updates.coordinator import Coordinator
from app.code_updates.storage import locked


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action", required=True, choices=(
        "init", "prepare", "build", "finalize", "activate", "verify", "status",
        "amend-review", "rollback",
    ))
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--mode", choices=("fdd", "code", "both", "review"))
    for name in (
        "fdd-generation", "fdd-source-directory", "code-source-directory", "svn-revision",
        "application-build", "reviewer", "price-per-million", "pricing-basis",
        "enhancement-registry", "max-usd", "embedding-approval", "runtime-receipt",
        "withdrawn-source", "replacement-manifest",
    ):
        parser.add_argument("--" + name)
    parser.add_argument("--services-stopped", action="store_true")
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    os.chdir(ROOT)
    coordinator = Coordinator(ROOT, args.run_id)
    if args.action == "status":
        coordinator.status(args)
        return 0
    started = time.monotonic()
    with locked(coordinator.run.directory):
        coordinator = Coordinator(ROOT, args.run_id)
        run = coordinator.run
        if run.state.get("last_finished_at"):
            run.state["waiting_seconds"] = run.state.get("waiting_seconds", 0) + max(
                0, (datetime.now(UTC) - datetime.fromisoformat(run.state["last_finished_at"])).total_seconds()
            )
        try:
            run.state.pop("last_error", None)
            coordinator.execute(args.action, args)
            return 0
        except Exception as exc:
            run.state.update(last_error=str(exc), next_action=args.action)
            run.event("action_failed", action=args.action, error=str(exc))
            print(str(exc), file=sys.stderr)
            return 1
        finally:
            run.state["processing_seconds"] = run.state.get("processing_seconds", 0.0) + (time.monotonic() - started)
            run.state["last_finished_at"] = now()
            run.save()
            coordinator.write_summary()
            print(f"Next: .\\scripts\\run_knowledge_update.ps1 -Action {run.state.get('next_action', 'init')} -RunId {args.run_id}")


if __name__ == "__main__":
    raise SystemExit(main())
