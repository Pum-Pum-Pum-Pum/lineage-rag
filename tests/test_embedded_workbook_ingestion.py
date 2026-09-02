from __future__ import annotations

import io
import zipfile
from pathlib import Path
from xml.etree import ElementTree

from docx import Document
import pytest

from app.ingestion.chunker import chunk_normalized_artifact
from app.ingestion.docx_ingestion_artifact import ingest_docx_file
from app.ingestion.normalized_artifact import build_normalized_artifact
from app.ingestion.retrieval_ready_artifact import build_retrieval_ready_artifact
from app.ingestion.table_chunker import chunk_tables_from_artifact


PACKAGE_NS = "http://schemas.openxmlformats.org/package/2006/relationships"
CONTENT_TYPES_NS = "http://schemas.openxmlformats.org/package/2006/content-types"
WORD_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
REL_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"


def _xlsx_bytes(*, external_link: bool = False) -> bytes:
    files = {
        "xl/workbook.xml": """<?xml version=\"1.0\"?>
        <workbook xmlns=\"http://schemas.openxmlformats.org/spreadsheetml/2006/main\"
                  xmlns:r=\"http://schemas.openxmlformats.org/officeDocument/2006/relationships\">
          <sheets><sheet name=\"Request\" sheetId=\"1\" r:id=\"rId1\"/>
                  <sheet name=\"Validation\" sheetId=\"2\" r:id=\"rId2\"/></sheets>
        </workbook>""",
        "xl/_rels/workbook.xml.rels": """<?xml version=\"1.0\"?>
        <Relationships xmlns=\"http://schemas.openxmlformats.org/package/2006/relationships\">
          <Relationship Id=\"rId1\" Target=\"worksheets/sheet1.xml\" Type=\"http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet\"/>
          <Relationship Id=\"rId2\" Target=\"worksheets/sheet2.xml\" Type=\"http://schemas.openxmlformats.org/officeDocument/2006/relationships/worksheet\"/>
        </Relationships>""",
        "xl/worksheets/sheet1.xml": """<?xml version=\"1.0\"?>
        <worksheet xmlns=\"http://schemas.openxmlformats.org/spreadsheetml/2006/main\"><sheetData>
          <row r=\"1\"><c r=\"A1\" t=\"inlineStr\"><is><t>API Name</t></is></c><c r=\"B1\" t=\"inlineStr\"><is><t>Request</t></is></c></row>
          <row r=\"2\"><c r=\"A2\" t=\"inlineStr\"><is><t>AMLStatus</t></is></c><c r=\"B2\" t=\"inlineStr\"><is><t>customerId</t></is></c></row>
        </sheetData></worksheet>""",
        "xl/worksheets/sheet2.xml": """<?xml version=\"1.0\"?>
        <worksheet xmlns=\"http://schemas.openxmlformats.org/spreadsheetml/2006/main\"><sheetData>
          <row r=\"1\"><c r=\"A1\" t=\"inlineStr\"><is><t>Rule</t></is></c></row>
          <row r=\"2\"><c r=\"A2\" t=\"inlineStr\"><is><t>customerId required</t></is></c></row>
        </sheetData></worksheet>""",
    }
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as archive:
        for path, content in files.items():
            archive.writestr(path, content)
        if external_link:
            archive.writestr("xl/externalLinks/externalLink1.xml", "<externalLink/>")
    return output.getvalue()


def _docx_with_xlsx(path: Path, *, external_link: bool = False) -> None:
    document = Document()
    document.add_paragraph("AML API request and validation contract")
    document.save(path)

    with zipfile.ZipFile(path) as source:
        parts = {entry.filename: source.read(entry) for entry in source.infolist()}
    rels = ElementTree.fromstring(parts["word/_rels/document.xml.rels"])
    ElementTree.SubElement(
        rels,
        f"{{{PACKAGE_NS}}}Relationship",
        {
            "Id": "rIdEmbeddedWorkbook",
            "Type": "http://schemas.openxmlformats.org/officeDocument/2006/relationships/package",
            "Target": "embeddings/api_contract.xlsx",
        },
    )
    parts["word/_rels/document.xml.rels"] = ElementTree.tostring(rels, encoding="utf-8", xml_declaration=True)
    document_xml = ElementTree.fromstring(parts["word/document.xml"])
    first_paragraph = document_xml.find(f".//{{{WORD_NS}}}p")
    assert first_paragraph is not None
    run = ElementTree.SubElement(first_paragraph, f"{{{WORD_NS}}}r")
    ElementTree.SubElement(run, f"{{{WORD_NS}}}object", {f"{{{REL_NS}}}id": "rIdEmbeddedWorkbook"})
    parts["word/document.xml"] = ElementTree.tostring(document_xml, encoding="utf-8", xml_declaration=True)
    content_types = ElementTree.fromstring(parts["[Content_Types].xml"])
    ElementTree.SubElement(
        content_types,
        f"{{{CONTENT_TYPES_NS}}}Default",
        {
            "Extension": "xlsx",
            "ContentType": "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        },
    )
    parts["[Content_Types].xml"] = ElementTree.tostring(content_types, encoding="utf-8", xml_declaration=True)
    parts["word/embeddings/api_contract.xlsx"] = _xlsx_bytes(external_link=external_link)
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as target:
        for part_path, content in parts.items():
            target.writestr(part_path, content)


def test_embedded_xlsx_becomes_linked_citeable_retrieval_units(tmp_path: Path) -> None:
    path = tmp_path / "FS_FCIS_14.4.0.0.0$ASNB_R4_REST API Services_v2.31.docx"
    _docx_with_xlsx(path)

    raw = ingest_docx_file(path)
    normalized = build_normalized_artifact(raw)
    artifact = build_retrieval_ready_artifact(
        normalized,
        chunk_normalized_artifact(normalized),
        chunk_tables_from_artifact(normalized),
    )
    workbook_units = [unit for unit in artifact.units if unit.source_kind == "embedded_workbook"]

    assert len(raw.embedded_workbooks.workbooks) == 1
    assert {unit.sheet_name for unit in workbook_units} == {"Request", "Validation"}
    assert {unit.sheet_role for unit in workbook_units} == {"request", "validation"}
    request = next(unit for unit in workbook_units if unit.sheet_name == "Request")
    assert request.attachment_path == "word/embeddings/api_contract.xlsx"
    assert request.source_range == "Request!1:2"
    assert "A2=AMLStatus" in request.text
    assert "DERIVED RETRIEVAL CONTEXT" not in request.text
    assert "DERIVED RETRIEVAL CONTEXT" in request.retrieval_text
    assert "AML API request and validation contract" in request.retrieval_text
    assert artifact.document_lineage_key == "FS_FCIS_14.4.0.0.0$ASNB_R4_REST API Services"
    assert artifact.document_revision == "2.31"


def test_embedded_xlsx_with_external_link_fails_closed(tmp_path: Path) -> None:
    path = tmp_path / "FS_FCIS_14.4.0.0.0$ASNB_R4_REST API Services_v2.31.docx"
    _docx_with_xlsx(path, external_link=True)

    with pytest.raises(ValueError, match="external links"):
        ingest_docx_file(path)


def test_plural_validation_sheet_name_uses_the_validation_role() -> None:
    from app.ingestion.embedded_workbook_chunker import _sheet_role

    assert _sheet_role("Validations") == "validation"
