from __future__ import annotations

from dataclasses import dataclass

from app.ingestion.docx_ingestion_artifact import IngestedDocxArtifact


NOISE_EXACT_MATCHES = {
    "You can delete any elements of this cover page that you do not need for your document.",
    "To add additional approval lines, press [Tab] from the last cell in the table above.",
}

NOISE_CONTAINS_PATTERNS = [
    "Title, Subject, Last Updated Date, Reference Number, and Version are marked by a Word Bookmark",
    "Show bookmarks option in the Show document content region",
]


@dataclass(frozen=True)
class NormalizedTextResult:
    original_non_empty_paragraph_count: int
    cleaned_paragraph_count: int
    removed_paragraph_count: int
    cleaned_paragraphs: list[str]
    cleaned_full_text: str


def is_noise_paragraph(paragraph: str) -> bool:
    stripped = paragraph.strip()
    if not stripped:
        return True
    if stripped in NOISE_EXACT_MATCHES:
        return True
    return any(pattern in stripped for pattern in NOISE_CONTAINS_PATTERNS)


def normalize_ingested_text(artifact: IngestedDocxArtifact) -> NormalizedTextResult:
    original_paragraphs = artifact.extracted_text.paragraphs
    cleaned_paragraphs = [
        paragraph.strip()
        for paragraph in original_paragraphs
        if not is_noise_paragraph(paragraph)
    ]

    cleaned_full_text = "\n".join(cleaned_paragraphs)
    original_non_empty = artifact.extracted_text.non_empty_paragraph_count
    cleaned_count = len(cleaned_paragraphs)

    return NormalizedTextResult(
        original_non_empty_paragraph_count=original_non_empty,
        cleaned_paragraph_count=cleaned_count,
        removed_paragraph_count=original_non_empty - cleaned_count,
        cleaned_paragraphs=cleaned_paragraphs,
        cleaned_full_text=cleaned_full_text,
    )
