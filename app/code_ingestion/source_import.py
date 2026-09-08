from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
from datetime import UTC, datetime
from pathlib import Path

from app.core.ingestion_policy import IngestionSourcePolicy, load_ingestion_source_policy


SOURCE_IMPORT_RECEIPT_FILE = "source_import_receipt.json"
_STREAM_CHUNK_BYTES = 1024 * 1024


class SourceImportError(ValueError):
    """Raised when an external source tree cannot be staged safely."""


def stage_external_code_source(
    source_directory: Path,
    intake_directory: Path,
    *,
    source_policy: IngestionSourcePolicy | None = None,
) -> dict[str, object]:
    """Copy the configured code extensions from an external tree without modifying it.

    The destination is a normal local intake ``source/`` directory.  Hashes are
    taken before and after copying, so a changing SVN working copy cannot become
    an apparently immutable snapshot.
    """

    policy = source_policy or load_ingestion_source_policy()
    if not source_directory.is_dir():
        raise SourceImportError(f"Source directory is not a directory: {source_directory}")
    if not intake_directory.is_dir():
        raise SourceImportError(f"Intake directory is not a directory: {intake_directory}")
    source_root = source_directory.resolve(strict=True)
    intake_root = intake_directory.resolve(strict=True)

    destination = intake_root / "source"
    receipt_path = intake_root / SOURCE_IMPORT_RECEIPT_FILE
    if destination.exists() or receipt_path.exists():
        raise SourceImportError(
            "Source import destination already exists; create a new snapshot request rather than overwriting intake evidence."
        )

    allowed_extensions = frozenset(policy.extension_map("code"))
    selected, skipped = _select_source_files(source_root, allowed_extensions)
    if not selected:
        raise SourceImportError(
            "No configured code files were found in the external source directory."
        )

    before = {relative_path: _sha256(path) for relative_path, path in selected}
    temporary_root = Path(tempfile.mkdtemp(prefix=".code-source-import-", dir=intake_root))
    temporary_source = temporary_root / "source"
    try:
        copied: dict[str, str] = {}
        for relative_path, source_path in selected:
            destination_path = temporary_source / relative_path
            destination_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source_path, destination_path)
            copied[relative_path] = _sha256(destination_path)

        after = {relative_path: _sha256(path) for relative_path, path in selected if path.is_file()}
        if before != copied or before != after:
            raise SourceImportError(
                "External source changed during read-only import; no intake source was published. Update SVN, then rerun into a new empty request directory."
            )

        temporary_source.replace(destination)
        receipt = {
            "schema_version": "code_external_source_import_v1",
            "created_at_utc": datetime.now(UTC).isoformat(),
            "source_directory": str(source_root),
            "source_directory_sha256": hashlib.sha256(str(source_root).encode("utf-8")).hexdigest(),
            "selected_extensions": sorted(allowed_extensions),
            "selected_file_count": len(before),
            "selected_tree_sha256": _tree_sha256(before),
            "skipped_file_counts_by_extension": dict(sorted(skipped.items())),
            "writes_to_external_source": False,
        }
        receipt_path.write_text(
            json.dumps(receipt, indent=2, ensure_ascii=False, sort_keys=True),
            encoding="utf-8",
        )
        return receipt
    except Exception:
        if destination.exists():
            shutil.rmtree(destination, ignore_errors=True)
        receipt_path.unlink(missing_ok=True)
        raise
    finally:
        shutil.rmtree(temporary_root, ignore_errors=True)


def _select_source_files(
    source_root: Path,
    allowed_extensions: frozenset[str],
) -> tuple[list[tuple[str, Path]], dict[str, int]]:
    selected: list[tuple[str, Path]] = []
    skipped: dict[str, int] = {}
    seen_casefold_paths: set[str] = set()
    for path in sorted(source_root.rglob("*"), key=lambda item: item.as_posix().casefold()):
        if path.is_symlink():
            raise SourceImportError(
                f"Symlink is not accepted in external source import: {path.relative_to(source_root).as_posix()}"
            )
        if not path.is_file():
            continue
        relative_path = path.relative_to(source_root).as_posix()
        extension = path.suffix.lower()
        if extension not in allowed_extensions:
            skipped[extension or "<no_extension>"] = skipped.get(extension or "<no_extension>", 0) + 1
            continue
        casefold_path = relative_path.casefold()
        if casefold_path in seen_casefold_paths:
            raise SourceImportError(
                f"Case-insensitive source path collision: {relative_path}"
            )
        seen_casefold_paths.add(casefold_path)
        selected.append((relative_path, path))
    return selected, skipped


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(_STREAM_CHUNK_BYTES), b""):
            digest.update(block)
    return digest.hexdigest()


def _tree_sha256(entries: dict[str, str]) -> str:
    payload = [
        {"path": path, "sha256": digest}
        for path, digest in sorted(entries.items(), key=lambda item: item[0].casefold())
    ]
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
