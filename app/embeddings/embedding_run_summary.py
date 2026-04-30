from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

from app.embeddings.embedding_contract import EmbeddingBatch
from app.embeddings.embedding_metrics import calculate_cache_hit_rate

@dataclass(frozen=True)
class EmbeddingRunSummary:
    document_name: str
    total_records: int
    cached_count: int
    embedded_count: int
    cache_miss_count: int
    cache_hit_rate: float
    embedding_models: list[str]
    artifact_versions: list[str]


def build_embedding_run_summary(batch: EmbeddingBatch) -> EmbeddingRunSummary:
    """Build a lightweight local observability summary for one embedding batch."""

    embedding_models = sorted({record.embedding_model for record in batch.records})
    artifact_versions = sorted({record.artifact_version for record in batch.records})

    return EmbeddingRunSummary(
        document_name=batch.document_name,
        total_records=batch.total_records,
        cached_count=batch.cached_count,
        embedded_count=batch.embedded_count,
        cache_miss_count=batch.cache_miss_count,
        cache_hit_rate=calculate_cache_hit_rate(batch),
        embedding_models=embedding_models,
        artifact_versions=artifact_versions,
    )


def write_embedding_run_summary_to_json(
    summary: EmbeddingRunSummary,
    output_directory: str | Path,
) -> Path:
    """Persist embedding run metrics as a JSON artifact."""

    output_dir = Path(output_directory)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / f"{Path(summary.document_name).stem}.embedding_summary.json"
    output_file.write_text(
        json.dumps(asdict(summary), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    return output_file
