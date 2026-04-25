from pathlib import Path

from docx import Document

from app.ingestion.docx_ingestion_artifact import ingest_docx_file
from app.ingestion.text_normalizer import normalize_ingested_text


def test_normalize_ingested_text_removes_known_noise(tmp_path: Path) -> None:
    file_path = tmp_path / "FS_FCIS_14.7.0.0.0$ASNB_R24_Teller_Branch_Reports_Realignment_v1.0.docx"

    document = Document()
    document.add_paragraph("Functional Design Document")
    document.add_paragraph("")
    document.add_paragraph(
        "Title, Subject, Last Updated Date, Reference Number, and Version are marked by a Word Bookmark so that they can be easily reproduced in the header and footer of documents."
    )
    document.add_paragraph("Requirements Overview")
    document.add_paragraph("To add additional approval lines, press [Tab] from the last cell in the table above.")
    document.save(file_path)

    artifact = ingest_docx_file(file_path)
    normalized = normalize_ingested_text(artifact)

    assert normalized.original_non_empty_paragraph_count == 4
    assert normalized.cleaned_paragraph_count == 2
    assert normalized.removed_paragraph_count == 2
    assert normalized.cleaned_paragraphs == [
        "Functional Design Document",
        "Requirements Overview",
    ]
    assert normalized.cleaned_full_text == "Functional Design Document\nRequirements Overview"
