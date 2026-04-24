from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


SUPPORTED_DOCX_SUFFIX = ".docx"
TEMP_DOCX_PREFIX = "~$"


@dataclass(frozen=True)
class DiscoveredDocxFile:
    file_path: Path
    file_name: str
    is_temporary: bool


def discover_docx_files(directory: str | Path) -> list[DiscoveredDocxFile]:
    """Discover valid DOCX files for ingestion.

    Filters out temporary Office lock files such as files starting with `~$`.
    """

    base_path = Path(directory)
    if not base_path.exists():
        raise FileNotFoundError(f"Input directory does not exist: {base_path}")

    discovered: list[DiscoveredDocxFile] = []

    for path in sorted(base_path.iterdir()):
        if not path.is_file():
            continue
        if path.suffix.lower() != SUPPORTED_DOCX_SUFFIX:
            continue

        is_temporary = path.name.startswith(TEMP_DOCX_PREFIX)
        if is_temporary:
            continue

        discovered.append(
            DiscoveredDocxFile(
                file_path=path,
                file_name=path.name,
                is_temporary=is_temporary,
            )
        )

    return discovered
