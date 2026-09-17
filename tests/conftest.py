from __future__ import annotations

import sys
from pathlib import Path
from uuid import uuid4


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))


def pytest_configure(config):
    """Keep temporary fixtures local and give concurrent runs separate directories."""
    if config.option.basetemp is None:
        runs = ROOT_DIR / ".local" / "pytest" / "runs"
        runs.mkdir(parents=True, exist_ok=True)
        # pytest clears an explicit basetemp before using it. A fresh UUID prevents
        # an ordinary test run from erasing historical evidence or another run.
        config.option.basetemp = str(runs / uuid4().hex)
