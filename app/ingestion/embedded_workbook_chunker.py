from __future__ import annotations

from dataclasses import dataclass

from app.ingestion.embedding_input_limits import MAX_SOURCE_UNIT_BYTES, split_text_by_utf8_bytes
from app.ingestion.embedded_workbook_extractor import ExtractedEmbeddedWorkbook, ExtractedWorkbookRow


ROWS_PER_WORKBOOK_UNIT = 25
SHEET_ROLE_ALIASES = {
    "version": "version",
    "versions": "version",
    "request": "request",
    "requests": "request",
    "response": "response",
    "responses": "response",
    "validation": "validation",
    "validations": "validation",
}


@dataclass(frozen=True)
class EmbeddedWorkbookChunk:
    chunk_id: str
    workbook_index: int
    workbook_path: str
    workbook_sha256: str
    sheet_name: str
    sheet_role: str
    row_start: int
    row_end: int
    text: str
    preceding_paragraph_index: int | None
    preceding_paragraph_text: str | None


def chunk_embedded_workbook(
    *,
    document_name: str,
    workbook: ExtractedEmbeddedWorkbook,
    rows_per_unit: int = ROWS_PER_WORKBOOK_UNIT,
) -> tuple[EmbeddedWorkbookChunk, ...]:
    if rows_per_unit <= 0:
        raise ValueError("rows_per_unit must be greater than zero")
    chunks: list[EmbeddedWorkbookChunk] = []
    for sheet in workbook.sheets:
        row_groups = _bounded_row_groups(sheet.rows, rows_per_unit=rows_per_unit)
        for part_index, (row_start, row_end, text) in enumerate(row_groups):
            chunks.append(
                EmbeddedWorkbookChunk(
                    chunk_id=(
                        f"{document_name}::workbook_{workbook.attachment_index}"
                        f"::sheet_{sheet.sheet_index}::rows_{row_start}_{row_end}::part_{part_index}"
                    ),
                    workbook_index=workbook.attachment_index,
                    workbook_path=workbook.archive_path,
                    workbook_sha256=workbook.sha256,
                    sheet_name=sheet.sheet_name,
                    sheet_role=_sheet_role(sheet.sheet_name),
                    row_start=row_start,
                    row_end=row_end,
                    text=text,
                    preceding_paragraph_index=workbook.preceding_paragraph_index,
                    preceding_paragraph_text=workbook.preceding_paragraph_text,
                )
            )
    return tuple(chunks)


def _sheet_role(name: str) -> str:
    normalized = " ".join(name.casefold().split())
    return SHEET_ROLE_ALIASES.get(normalized, "other")


def _render_rows(rows: tuple[ExtractedWorkbookRow, ...]) -> str:
    return "\n".join(
        f"row {row.row_number}: " + " | ".join(f"{cell.reference}={cell.value}" for cell in row.cells)
        for row in rows
    )


def _bounded_row_groups(
    rows: tuple[ExtractedWorkbookRow, ...],
    *,
    rows_per_unit: int,
) -> tuple[tuple[int, int, str], ...]:
    """Preserve row provenance while bounding even a single enormous cell."""

    groups: list[tuple[int, int, str]] = []
    selected: list[ExtractedWorkbookRow] = []
    for row in rows:
        candidate = tuple(selected + [row])
        candidate_text = _render_rows(candidate)
        if selected and (
            len(candidate) > rows_per_unit
            or len(candidate_text.encode("utf-8")) > MAX_SOURCE_UNIT_BYTES
        ):
            groups.append((selected[0].row_number, selected[-1].row_number, _render_rows(tuple(selected))))
            selected = []

        row_text = _render_rows((row,))
        if len(row_text.encode("utf-8")) > MAX_SOURCE_UNIT_BYTES:
            for fragment in split_text_by_utf8_bytes(row_text, MAX_SOURCE_UNIT_BYTES):
                groups.append((row.row_number, row.row_number, fragment))
        else:
            selected.append(row)

    if selected:
        groups.append((selected[0].row_number, selected[-1].row_number, _render_rows(tuple(selected))))
    return tuple(groups)
