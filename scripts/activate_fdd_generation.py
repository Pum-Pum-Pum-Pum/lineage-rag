from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.activation.fdd_generation import (
    apply_fdd_generation_activation,
    build_fdd_generation_activation_plan,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Promote one verified FDD generation atomically.")
    parser.add_argument("--generation", required=True)
    parser.add_argument("--stage-directory", type=Path, required=True)
    parser.add_argument("--apply", action="store_true", help="Apply the validated promotion and .env switch.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    plan = build_fdd_generation_activation_plan(
        generation=args.generation,
        stage_directory=args.stage_directory,
        indexes_directory=ROOT_DIR / "data" / "indexes",
        env_path=ROOT_DIR / ".env",
    )
    if not args.apply:
        print(json.dumps({"status": "ready_to_apply", "plan": plan.__dict__}, indent=2))
        return
    evidence_path = apply_fdd_generation_activation(
        plan=plan,
        env_path=ROOT_DIR / ".env",
        evidence_directory=ROOT_DIR / "data" / "exports" / "activations" / "fdd",
    )
    print(json.dumps({"status": "activated", "evidence_path": str(evidence_path)}, indent=2))


if __name__ == "__main__":
    main()
