from __future__ import annotations

from dataclasses import dataclass

from app.ingestion.docx_ingestion_artifact import IngestedDocxArtifact
from app.ingestion.text_normalizer import NormalizedTextResult, normalize_ingested_text


@dataclass(frozen=True)
class NormalizedDocxArtifact:
    raw_artifact: IngestedDocxArtifact
    normalized_text: NormalizedTextResult


def build_normalized_artifact(artifact: IngestedDocxArtifact) -> NormalizedDocxArtifact:
    """Combine raw ingestion output with normalized text output.

    This keeps raw extraction results and cleaned text together so downstream
    chunking can use cleaned text while debugging can still inspect the raw state.
    """

    normalized_text = normalize_ingested_text(artifact)
    return NormalizedDocxArtifact(
        raw_artifact=artifact,
        normalized_text=normalized_text,
    )
