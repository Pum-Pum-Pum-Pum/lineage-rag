from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from app.ingestion.chunker import ChunkedDocument


def write_chunked_document_to_json(
    chunked_document: ChunkedDocument,
    output_directory: str | Path,
) -> Path:
    """Write chunked document output to a JSON artifact for inspection."""

    output_dir = Path(output_directory)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / f"{Path(chunked_document.document_name).stem}.chunks.json"
    output_file.write_text(
        json.dumps(asdict(chunked_document), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    return output_file
