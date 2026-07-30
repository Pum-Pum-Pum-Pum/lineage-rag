from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.core.config import get_settings
from app.deployment.preflight import run_deployment_preflight


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Check native deployment prerequisites without API calls."
    )
    parser.add_argument(
        "--allow-development",
        action="store_true",
        help="Allow dev/test ENVIRONMENT values for local package validation.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = run_deployment_preflight(
        get_settings(),
        project_root=ROOT_DIR,
        allow_development=args.allow_development,
    )
    print(json.dumps(asdict(report), indent=2))
    if not report.passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
