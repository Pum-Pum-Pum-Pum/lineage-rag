from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.activation.code_modes import initialize_disabled_baseline


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Safely initialize CODE_MODES_ENABLED=false when absent."
    )
    parser.add_argument("--env-file", type=Path, default=ROOT_DIR / ".env")
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    result = initialize_disabled_baseline(
        env_path=args.env_file,
        apply=args.apply,
    )
    print(json.dumps(result.model_dump(mode="json"), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
