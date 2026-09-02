from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


FILENAME_PATTERN = re.compile(
    r"^(?P<document_family>FS_[^\s]+?)_R(?P<release_number>\d+)(?P<suffix>_.+)?$",
    re.IGNORECASE,
)
REVISION_SUFFIX_PATTERN = re.compile(r"^(?P<lineage>.+?)_v(?P<revision>\d+(?:\.\d+)+)$", re.IGNORECASE)


@dataclass(frozen=True)
class ParsedDocumentName:
    document_name: str
    document_id: str
    document_family: str
    release_label: str
    release_number: int
    variant_suffix: str | None = None
    document_lineage_key: str = ""
    document_revision: str | None = None
    document_revision_sort_key: tuple[int, ...] | None = None
    source_type: str = "docx"


def parse_document_filename(file_path: str | Path) -> ParsedDocumentName:
    """Parse a DOCX filename into release-aware metadata.

    Expected examples:
    FS_FCIS_14.4.0.0.0$ASNB_R1.docx
    FS_FCIS_14.4.0.0.0$ASNB_R2_PNB_Branch Online Reports(BOR)_v1.2.docx
    """

    path = Path(file_path)
    stem = path.stem
    match = FILENAME_PATTERN.match(stem)

    if not match:
        raise ValueError(
            "Filename does not match expected release-aware pattern: "
            f"{path.name}"
        )

    release_number = int(match.group("release_number"))
    document_family = match.group("document_family")
    release_label = f"R{release_number}"
    suffix = match.group("suffix")
    revision_match = REVISION_SUFFIX_PATTERN.match(stem)
    document_lineage_key = revision_match.group("lineage") if revision_match else stem
    document_revision = revision_match.group("revision") if revision_match else None
    revision_sort_key = (
        tuple(int(part) for part in document_revision.split("."))
        if document_revision is not None
        else None
    )

    return ParsedDocumentName(
        document_name=path.name,
        document_id=stem,
        document_family=document_family,
        release_label=release_label,
        release_number=release_number,
        variant_suffix=suffix[1:] if suffix else None,
        document_lineage_key=document_lineage_key,
        document_revision=document_revision,
        document_revision_sort_key=revision_sort_key,
    )
