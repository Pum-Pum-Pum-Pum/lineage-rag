"""Entry point for persistent, resumable recurring code updates."""
from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime, UTC
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app.code_updates.storage import locked, now
from app.code_updates.coordinator import Coordinator


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--action", required=True, choices=["init", "prepare", "build", "finalize", "activate", "verify", "status"])
    parser.add_argument("--run-id", required=True)
    for name in ("source-directory", "svn-revision", "application-build", "reviewer", "price-per-million", "pricing-basis",
                 "enhancement-registry", "max-usd", "runtime-receipt"):
        parser.add_argument("--" + name)
    parser.add_argument("--services-stopped", action="store_true")
    args = parser.parse_args(argv)
    os.chdir(ROOT)
    coordinator = Coordinator(ROOT, args.run_id)
    run = coordinator.run
    if args.action == "status":
        # Status is strictly read-only and never constructs a provider client.
        import json
        print(json.dumps(run.state, indent=2))
        print(f"Next: .\\scripts\\run_code_update.ps1 -Action {run.state.get('next_action', 'init')} -RunId {args.run_id}")
        return 0
    started = time.monotonic()
    with locked(run.directory):
        # Reload after acquiring ownership; another invocation may have completed.
        coordinator = Coordinator(ROOT, args.run_id)
        run = coordinator.run
        prompt_wait_before = run.state.get("prompt_wait_seconds", 0)
        if run.state.get("last_finished_at"):
            run.state["waiting_seconds"] = run.state.get("waiting_seconds", 0) + max(0,
                (datetime.now(UTC) - datetime.fromisoformat(run.state["last_finished_at"])).total_seconds())
        try:
            run.state.pop("last_error", None)
            coordinator.execute(args.action, args)
            return 0
        except Exception as error:
            message = str(error)
            run.state.update(last_error=message, next_action=args.action)
            run.event("action_failed", action=args.action, error=message)
            print(message, file=sys.stderr)
            return 1
        finally:
            run.state["processing_seconds"] += max(0, time.monotonic() - started - (run.state.get("prompt_wait_seconds", 0) - prompt_wait_before))
            run.state["last_finished_at"] = now()
            run.save()
            print("Next:", run.summary())


if __name__ == "__main__":
    raise SystemExit(main())
