from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from app.ingestion.retrieval_ready_artifact import RetrievalReadyArtifact


def write_retrieval_ready_artifact_to_json(
    artifact: RetrievalReadyArtifact,
    output_directory: str | Path,
) -> Path:
    """Write a combined retrieval-ready artifact to JSON for inspection."""

    output_dir = Path(output_directory)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / f"{Path(artifact.document_name).stem}.retrieval_ready.json"
    output_file.write_text(
        json.dumps(asdict(artifact), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    return output_file
