from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.core.audit_benchmark import (
    run_audit_journal_benchmark,
    write_audit_benchmark_report,
)


DEFAULT_OUTPUT = (
    ROOT_DIR
    / "data"
    / "exports"
    / "audit_benchmarks"
    / "audit-journal-benchmark.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark local HMAC audit append/fsync cost with synthetic events."
        ),
    )
    parser.add_argument("--events", type=int, default=200)
    parser.add_argument("--warmup-events", type=int, default=10)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = run_audit_journal_benchmark(
        measured_events=args.events,
        warmup_events=args.warmup_events,
        work_directory=args.output.parent,
    )
    output = write_audit_benchmark_report(result, args.output)
    print(
        json.dumps(
            {
                "output_path": str(output),
                "result": asdict(result),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
