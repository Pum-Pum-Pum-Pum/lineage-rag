from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from app.ingestion.docx_table_extractor import ExtractedDocxTables, extract_docx_tables
from app.ingestion.docx_text_extractor import ExtractedDocxText, extract_docx_text
from app.ingestion.filename_parser import ParsedDocumentName, parse_document_filename


@dataclass(frozen=True)
class IngestedDocxArtifact:
    document_name: str
    parsed_name: ParsedDocumentName
    extracted_text: ExtractedDocxText
    extracted_tables: ExtractedDocxTables


def ingest_docx_file(file_path: str | Path) -> IngestedDocxArtifact:
    """Build a single document-level ingestion artifact for one DOCX file.

    This combines filename parsing, paragraph extraction, and table extraction
    into one coherent output for debugging and later normalization.
    """

    path = Path(file_path)
    parsed_name = parse_document_filename(path)
    extracted_text = extract_docx_text(path)
    extracted_tables = extract_docx_tables(path)

    return IngestedDocxArtifact(
        document_name=path.name,
        parsed_name=parsed_name,
        extracted_text=extracted_text,
        extracted_tables=extracted_tables,
    )
