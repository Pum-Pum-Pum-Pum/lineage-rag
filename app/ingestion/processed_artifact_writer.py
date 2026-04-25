from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from app.ingestion.docx_ingestion_artifact import IngestedDocxArtifact


def write_ingested_artifact_to_json(
    artifact: IngestedDocxArtifact,
    output_directory: str | Path,
) -> Path:
    """Write a unified ingestion artifact to a JSON file for debugging.

    The JSON artifact is stored locally so extracted document state can be
    inspected before normalization and chunking.
    """

    output_dir = Path(output_directory)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / f"{Path(artifact.document_name).stem}.json"
    output_file.write_text(
        json.dumps(asdict(artifact), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    return output_file
