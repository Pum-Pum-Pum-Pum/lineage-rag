from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path


FILENAME_PATTERN = re.compile(
    r"^(?P<document_family>FS_[^\s]+?)_R(?P<release_number>\d+)(?P<suffix>_.+)?$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class ParsedDocumentName:
    document_name: str
    document_family: str
    release_label: str
    release_number: int
    variant_suffix: str | None = None
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

    return ParsedDocumentName(
        document_name=path.name,
        document_family=document_family,
        release_label=release_label,
        release_number=release_number,
        variant_suffix=suffix[1:] if suffix else None,
    )
