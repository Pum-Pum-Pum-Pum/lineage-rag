from __future__ import annotations

import hashlib
import posixpath
import zipfile
from dataclasses import dataclass
from pathlib import Path
from xml.etree import ElementTree


WORD_NS = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
REL_NS = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
PACKAGE_REL_NS = "http://schemas.openxmlformats.org/package/2006/relationships"
SHEET_NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"

MAX_WORKBOOK_BYTES = 10 * 1024 * 1024
MAX_WORKBOOK_CELLS = 100_000


@dataclass(frozen=True)
class ExtractedWorkbookCell:
    reference: str
    value: str


@dataclass(frozen=True)
class ExtractedWorkbookRow:
    row_number: int
    cells: tuple[ExtractedWorkbookCell, ...]


@dataclass(frozen=True)
class ExtractedWorkbookSheet:
    sheet_name: str
    sheet_index: int
    visibility: str
    rows: tuple[ExtractedWorkbookRow, ...]
    row_count: int
    cell_count: int


@dataclass(frozen=True)
class ExtractedEmbeddedWorkbook:
    attachment_index: int
    relationship_id: str
    archive_path: str
    sha256: str
    size_bytes: int
    preceding_paragraph_index: int | None
    preceding_paragraph_text: str | None
    sheets: tuple[ExtractedWorkbookSheet, ...]


@dataclass(frozen=True)
class ExtractedEmbeddedWorkbooks:
    file_name: str
    workbook_count: int
    workbooks: tuple[ExtractedEmbeddedWorkbook, ...]
    unsupported_object_paths: tuple[str, ...]


def extract_embedded_workbooks(file_path: str | Path) -> ExtractedEmbeddedWorkbooks:
    """Extract real OOXML `.xlsx` attachments from a DOCX without executing them.

    Legacy OLE binary objects are reported as diagnostics. They are never parsed
    as spreadsheets or silently presented as workbook evidence.
    """

    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"DOCX file does not exist: {path}")
    if path.suffix.lower() != ".docx":
        raise ValueError(f"Expected DOCX input, got: {path.name}")
    with zipfile.ZipFile(path) as package:
        relationships = _load_relationships(package)
        anchors = _load_relationship_anchors(package)
        workbooks: list[ExtractedEmbeddedWorkbook] = []
        unsupported: list[str] = []
        for relationship_id, target, relationship_type in relationships:
            if relationship_type.endswith("/package") and target.lower().endswith(".xlsx"):
                archive_path = _word_target_path(target)
                entry = _require_entry(package, archive_path)
                if entry.file_size > MAX_WORKBOOK_BYTES:
                    raise ValueError(f"Embedded workbook exceeds {MAX_WORKBOOK_BYTES} bytes: {archive_path}")
                data = package.read(entry)
                paragraph_index, paragraph_text = anchors.get(relationship_id, (None, None))
                workbooks.append(
                    ExtractedEmbeddedWorkbook(
                        attachment_index=len(workbooks),
                        relationship_id=relationship_id,
                        archive_path=archive_path,
                        sha256=hashlib.sha256(data).hexdigest(),
                        size_bytes=len(data),
                        preceding_paragraph_index=paragraph_index,
                        preceding_paragraph_text=paragraph_text,
                        sheets=_parse_xlsx(data, archive_path),
                    )
                )
            elif relationship_type.endswith("/oleObject"):
                # Only interpret relationship targets for the object types this
                # extractor owns. A DOCX relationship part can legitimately
                # contain unrelated targets such as ../customXml/item2.xml;
                # treating those as workbook paths would turn harmless document
                # metadata into a false ingestion failure.
                unsupported.append(_word_target_path(target))
    return ExtractedEmbeddedWorkbooks(
        file_name=path.name,
        workbook_count=len(workbooks),
        workbooks=tuple(workbooks),
        unsupported_object_paths=tuple(sorted(set(unsupported))),
    )


def _load_relationships(package: zipfile.ZipFile) -> tuple[tuple[str, str, str], ...]:
    root = ElementTree.fromstring(package.read(_require_entry(package, "word/_rels/document.xml.rels")))
    items = []
    for relationship in root.findall(f"{{{PACKAGE_REL_NS}}}Relationship"):
        target = relationship.attrib.get("Target", "")
        relationship_id = relationship.attrib.get("Id", "")
        relationship_type = relationship.attrib.get("Type", "")
        if target and relationship_id and relationship_type:
            items.append((relationship_id, target, relationship_type))
    return tuple(items)


def _load_relationship_anchors(package: zipfile.ZipFile) -> dict[str, tuple[int | None, str | None]]:
    root = ElementTree.fromstring(package.read(_require_entry(package, "word/document.xml")))
    anchors: dict[str, tuple[int | None, str | None]] = {}
    preceding_index: int | None = None
    preceding_text: str | None = None
    for paragraph_index, paragraph in enumerate(root.iter(f"{{{WORD_NS}}}p")):
        text = "".join(node.text or "" for node in paragraph.iter(f"{{{WORD_NS}}}t")).strip()
        if text:
            preceding_index = paragraph_index
            preceding_text = text
        for node in paragraph.iter():
            for attribute, value in node.attrib.items():
                if attribute in {f"{{{REL_NS}}}id", f"{{{REL_NS}}}embed"}:
                    anchors[value] = (preceding_index, preceding_text)
    return anchors


def _parse_xlsx(data: bytes, archive_path: str) -> tuple[ExtractedWorkbookSheet, ...]:
    with zipfile.ZipFile(_BytesReader(data)) as workbook:
        if any(entry.filename.startswith("xl/externalLinks/") for entry in workbook.infolist()):
            raise ValueError(f"Embedded workbook has external links and is not accepted: {archive_path}")
        shared_strings = _load_shared_strings(workbook)
        relationships = _load_xlsx_relationships(workbook)
        workbook_root = ElementTree.fromstring(workbook.read(_require_entry(workbook, "xl/workbook.xml")))
        sheets: list[ExtractedWorkbookSheet] = []
        for sheet_index, sheet in enumerate(workbook_root.findall(f".//{{{SHEET_NS}}}sheet")):
            relationship_id = sheet.attrib.get(f"{{{REL_NS}}}id")
            if not relationship_id or relationship_id not in relationships:
                raise ValueError(f"Workbook sheet relationship is missing: {archive_path}")
            sheet_path = _xlsx_target_path(relationships[relationship_id])
            rows = _parse_sheet(workbook.read(_require_entry(workbook, sheet_path)), shared_strings)
            cells = sum(len(row.cells) for row in rows)
            if cells > MAX_WORKBOOK_CELLS:
                raise ValueError(f"Embedded workbook exceeds {MAX_WORKBOOK_CELLS} cells: {archive_path}")
            sheets.append(
                ExtractedWorkbookSheet(
                    sheet_name=sheet.attrib.get("name", f"Sheet{sheet_index + 1}"),
                    sheet_index=sheet_index,
                    visibility=sheet.attrib.get("state", "visible"),
                    rows=rows,
                    row_count=len(rows),
                    cell_count=cells,
                )
            )
    return tuple(sheets)


def _load_shared_strings(workbook: zipfile.ZipFile) -> tuple[str, ...]:
    try:
        root = ElementTree.fromstring(workbook.read("xl/sharedStrings.xml"))
    except KeyError:
        return ()
    return tuple("".join(node.text or "" for node in item.iter(f"{{{SHEET_NS}}}t")) for item in root.findall(f"{{{SHEET_NS}}}si"))


def _load_xlsx_relationships(workbook: zipfile.ZipFile) -> dict[str, str]:
    root = ElementTree.fromstring(workbook.read(_require_entry(workbook, "xl/_rels/workbook.xml.rels")))
    return {
        item.attrib["Id"]: item.attrib["Target"]
        for item in root.findall(f"{{{PACKAGE_REL_NS}}}Relationship")
        if "Id" in item.attrib and "Target" in item.attrib
    }


def _parse_sheet(data: bytes, shared_strings: tuple[str, ...]) -> tuple[ExtractedWorkbookRow, ...]:
    root = ElementTree.fromstring(data)
    rows: list[ExtractedWorkbookRow] = []
    for row in root.findall(f".//{{{SHEET_NS}}}row"):
        cells = tuple(_parse_cell(cell, shared_strings) for cell in row.findall(f"{{{SHEET_NS}}}c"))
        if cells:
            rows.append(ExtractedWorkbookRow(row_number=int(row.attrib.get("r", len(rows) + 1)), cells=cells))
    return tuple(rows)


def _parse_cell(cell: ElementTree.Element, shared_strings: tuple[str, ...]) -> ExtractedWorkbookCell:
    cell_type = cell.attrib.get("t")
    reference = cell.attrib.get("r", "")
    inline = "".join(node.text or "" for node in cell.findall(f".//{{{SHEET_NS}}}t"))
    value = cell.findtext(f"{{{SHEET_NS}}}v", default="")
    formula = cell.findtext(f"{{{SHEET_NS}}}f", default="")
    if cell_type == "s" and value:
        index = int(value)
        if index >= len(shared_strings):
            raise ValueError(f"Shared-string index out of range: {reference}")
        rendered = shared_strings[index]
    elif cell_type == "inlineStr":
        rendered = inline
    elif formula:
        rendered = f"={formula}" if not value else f"={formula} (cached: {value})"
    else:
        rendered = value
    return ExtractedWorkbookCell(reference=reference, value=rendered)


def _word_target_path(target: str) -> str:
    candidate = posixpath.normpath(posixpath.join("word", target)).lstrip("/")
    if candidate.startswith("../") or not candidate.startswith("word/"):
        raise ValueError(f"Unsafe DOCX relationship target: {target}")
    return candidate


def _xlsx_target_path(target: str) -> str:
    candidate = posixpath.normpath(posixpath.join("xl", target)).lstrip("/")
    if candidate.startswith("../") or not candidate.startswith("xl/"):
        raise ValueError(f"Unsafe XLSX relationship target: {target}")
    return candidate


def _require_entry(package: zipfile.ZipFile, name: str) -> zipfile.ZipInfo:
    try:
        return package.getinfo(name)
    except KeyError as exc:
        raise ValueError(f"Required OOXML part is missing: {name}") from exc


class _BytesReader:
    """Minimal seekable file object for ZipFile without persisting attachment bytes."""

    def __init__(self, data: bytes) -> None:
        import io

        self._stream = io.BytesIO(data)

    def __getattr__(self, name: str):
        return getattr(self._stream, name)
