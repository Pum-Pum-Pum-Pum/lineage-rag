from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.core.config import get_settings
from app.core.logging import configure_logging, get_logger
from app.ingestion.docx_ingestion_artifact import ingest_docx_file
from app.ingestion.docx_loader import discover_docx_files
from app.ingestion.processed_artifact_writer import write_ingested_artifact_to_json


def main() -> None:
    settings = get_settings()
    configure_logging(settings.log_level)
    logger = get_logger("ingestion_pipeline")

    discovered_files = discover_docx_files(settings.raw_specs_dir)
    logger.info("Discovered %s DOCX files for ingestion", len(discovered_files))

    for discovered in discovered_files:
        artifact = ingest_docx_file(discovered.file_path)
        output_file = write_ingested_artifact_to_json(artifact, settings.processed_dir)
        logger.info(
            "Ingested %s -> %s | paragraphs=%s | tables=%s",
            discovered.file_name,
            output_file.name,
            artifact.extracted_text.non_empty_paragraph_count,
            artifact.extracted_tables.table_count,
        )


if __name__ == "__main__":
    main()
