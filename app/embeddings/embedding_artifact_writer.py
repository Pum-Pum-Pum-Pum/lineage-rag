from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

from app.embeddings.embedding_contract import EmbeddingBatch


def write_embedding_batch_to_json(
    batch: EmbeddingBatch,
    output_directory: str | Path,
    file_stem_suffix: str = "",
) -> Path:
    """Persist an embedding batch as a local JSON artifact.

    This creates a checkpoint between embedding generation and vector-store
    indexing so vectors and their metadata can be inspected and reused.
    """

    output_dir = Path(output_directory)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / f"{Path(batch.document_name).stem}{file_stem_suffix}.embeddings.json"
    output_file.write_text(
        json.dumps(asdict(batch), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    return output_file
