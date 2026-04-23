from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.core.config import get_settings
from app.core.logging import configure_logging, get_logger


def main() -> None:
    settings = get_settings()
    configure_logging(settings.log_level)
    logger = get_logger("setup_check")

    directories = {
        "root_dir": settings.root_dir,
        "data_dir": settings.data_dir,
        "raw_specs_dir": settings.raw_specs_dir,
        "processed_dir": settings.processed_dir,
        "cache_dir": settings.cache_dir,
        "eval_dir": settings.eval_dir,
        "exports_dir": settings.exports_dir,
    }

    logger.info("Application: %s", settings.app_name)
    logger.info("Environment: %s", settings.environment)
    logger.info("Artifact version: %s", settings.artifact_version)
    logger.info("Index version: %s", settings.index_version)

    for name, path in directories.items():
        logger.info("%s=%s | exists=%s", name, path, path.exists())


if __name__ == "__main__":
    main()
