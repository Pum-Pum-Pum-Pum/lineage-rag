from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from docx import Document


@dataclass(frozen=True)
class ExtractedDocxText:
    file_name: str
    paragraph_count: int
    non_empty_paragraph_count: int
    paragraphs: list[str]
    full_text: str


def extract_docx_text(file_path: str | Path) -> ExtractedDocxText:
    """Extract paragraph text from a DOCX file.

    This first extraction step intentionally handles paragraph text only.
    Table extraction comes later as a separate step.
    """

    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"DOCX file does not exist: {path}")
    if path.suffix.lower() != ".docx":
        raise ValueError(f"Expected a .docx file, got: {path.name}")

    document = Document(path)
    paragraphs = [paragraph.text.strip() for paragraph in document.paragraphs]
    non_empty_paragraphs = [paragraph for paragraph in paragraphs if paragraph]
    full_text = "\n".join(non_empty_paragraphs)

    return ExtractedDocxText(
        file_name=path.name,
        paragraph_count=len(paragraphs),
        non_empty_paragraph_count=len(non_empty_paragraphs),
        paragraphs=paragraphs,
        full_text=full_text,
    )
