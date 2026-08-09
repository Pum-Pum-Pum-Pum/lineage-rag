from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
from datetime import UTC, datetime
from pathlib import Path

from app.code_ingestion.intake_validation import (
    DEFAULT_LARGE_FILE_WARNING_BYTES,
    DEFAULT_STREAM_CHUNK_BYTES,
    validate_code_intake,
)
from app.code_ingestion.snapshot_models import (
    CodeFileManifestEntry,
    CodeSnapshotManifest,
    ExactRename,
    SnapshotDiff,
    SnapshotRequest,
)


SNAPSHOT_REQUEST_FILE = "snapshot_request.json"
SNAPSHOT_MANIFEST_FILE = "snapshot_manifest.json"


def load_snapshot_request(path: Path) -> SnapshotRequest:
    if not path.is_file():
        raise FileNotFoundError(f"Snapshot request not found: {path}")
    return SnapshotRequest.model_validate_json(path.read_text(encoding="utf-8"))


def load_snapshot_manifest(snapshot_directory: Path, *, verify_sources: bool = True) -> CodeSnapshotManifest:
    manifest_path = snapshot_directory / SNAPSHOT_MANIFEST_FILE
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Snapshot manifest not found: {manifest_path}")
    manifest = CodeSnapshotManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
    if snapshot_directory.name != manifest.snapshot_id:
        raise RuntimeError("Snapshot directory name does not match manifest snapshot_id")
    if _snapshot_content_hash(manifest.request, manifest.files) != manifest.snapshot_content_sha256:
        raise RuntimeError(f"Snapshot manifest identity check failed: {manifest.snapshot_id}")
    if verify_sources:
        current = _source_hashes(snapshot_directory / manifest.source_directory_name)
        expected = {entry.path: entry.sha256 for entry in manifest.files}
        if current != expected:
            raise RuntimeError(f"Immutable snapshot source verification failed: {manifest.snapshot_id}")
    return manifest


def build_code_snapshot(
    intake_directory: Path,
    snapshot_root: Path,
    *,
    large_file_warning_bytes: int = DEFAULT_LARGE_FILE_WARNING_BYTES,
    stream_chunk_bytes: int = DEFAULT_STREAM_CHUNK_BYTES,
) -> CodeSnapshotManifest:
    """Validate and atomically publish one no-overwrite custom-code snapshot."""

    request = load_snapshot_request(intake_directory / SNAPSHOT_REQUEST_FILE)
    source_directory = intake_directory / "source"
    validation = validate_code_intake(
        source_directory,
        large_file_warning_bytes=large_file_warning_bytes,
        stream_chunk_bytes=stream_chunk_bytes,
    )
    base_manifest = _load_requested_base(request, snapshot_root)
    diff = build_snapshot_diff(
        current_files=validation.files,
        base_manifest=base_manifest,
        expected_changed_packages=request.expected_changed_packages,
    )
    content_hash = _snapshot_content_hash(request, validation.files)
    snapshot_id = f"{request.module_set}-r{request.svn_revision}-{content_hash[:12]}"
    target_directory = snapshot_root / snapshot_id
    if target_directory.exists():
        raise FileExistsError(
            f"Snapshot already exists: {target_directory}. Immutable snapshots are never overwritten."
        )

    manifest = CodeSnapshotManifest(
        snapshot_id=snapshot_id,
        snapshot_content_sha256=content_hash,
        created_at_utc=datetime.now(UTC),
        request=request,
        files=validation.files,
        diff=diff,
    )

    snapshot_root.mkdir(parents=True, exist_ok=True)
    temporary_directory = Path(tempfile.mkdtemp(prefix=".code-snapshot-", dir=snapshot_root))
    try:
        temporary_source = temporary_directory / "source"
        shutil.copytree(source_directory, temporary_source)
        _write_json(temporary_directory / SNAPSHOT_MANIFEST_FILE, manifest.model_dump(mode="json"))
        _verify_staged_snapshot(temporary_directory, manifest)
        temporary_directory.replace(target_directory)
    except Exception:
        shutil.rmtree(temporary_directory, ignore_errors=True)
        raise
    return manifest


def build_snapshot_diff(
    *,
    current_files: tuple[CodeFileManifestEntry, ...],
    base_manifest: CodeSnapshotManifest | None,
    expected_changed_packages: tuple[str, ...] = (),
) -> SnapshotDiff:
    current = {entry.path: entry for entry in current_files}
    previous = {entry.path: entry for entry in base_manifest.files} if base_manifest else {}

    current_paths = set(current)
    previous_paths = set(previous)
    added = current_paths - previous_paths
    deleted = previous_paths - current_paths
    unchanged = {
        path for path in current_paths & previous_paths if current[path].sha256 == previous[path].sha256
    }
    modified = (current_paths & previous_paths) - unchanged
    formatting_only = {
        path
        for path in modified
        if current[path].normalized_text_sha256 == previous[path].normalized_text_sha256
    }

    renames: list[ExactRename] = []
    ambiguous_hashes: list[str] = []
    deleted_by_hash = _paths_by_hash(deleted, previous)
    added_by_hash = _paths_by_hash(added, current)
    for content_hash in sorted(set(deleted_by_hash) & set(added_by_hash)):
        old_paths = deleted_by_hash[content_hash]
        new_paths = added_by_hash[content_hash]
        if len(old_paths) == 1 and len(new_paths) == 1:
            old_path = old_paths[0]
            new_path = new_paths[0]
            renames.append(ExactRename(old_path=old_path, new_path=new_path, sha256=content_hash))
            deleted.remove(old_path)
            added.remove(new_path)
        else:
            ambiguous_hashes.append(content_hash)

    changed_paths = set(added) | set(modified) | set(deleted)
    changed_paths.update(rename.old_path for rename in renames)
    changed_paths.update(rename.new_path for rename in renames)
    missing_expected = tuple(
        expected
        for expected in expected_changed_packages
        if not _expected_path_matches(expected, changed_paths)
    )
    change_groups = [{path} for path in set(added) | set(modified) | set(deleted)]
    change_groups.extend({rename.old_path, rename.new_path} for rename in renames)
    covered_changed: set[str] = set()
    for group in change_groups:
        if any(_expected_path_matches(expected, group) for expected in expected_changed_packages):
            covered_changed.update(group)
    unexpected = changed_paths - covered_changed if expected_changed_packages else set()

    return SnapshotDiff(
        base_snapshot_id=base_manifest.snapshot_id if base_manifest else None,
        added=tuple(sorted(added, key=str.casefold)),
        modified=tuple(sorted(modified, key=str.casefold)),
        deleted=tuple(sorted(deleted, key=str.casefold)),
        unchanged=tuple(sorted(unchanged, key=str.casefold)),
        formatting_only_modified=tuple(sorted(formatting_only, key=str.casefold)),
        exact_renames=tuple(renames),
        ambiguous_rename_hashes=tuple(ambiguous_hashes),
        expected_changed_packages=expected_changed_packages,
        missing_expected_changes=missing_expected,
        unexpected_changed_files=tuple(sorted(unexpected, key=str.casefold)),
    )


def _load_requested_base(request: SnapshotRequest, snapshot_root: Path) -> CodeSnapshotManifest | None:
    if request.base_snapshot_id is None:
        return None
    base_directory = snapshot_root / request.base_snapshot_id
    manifest = load_snapshot_manifest(base_directory)
    if manifest.request.module_set != request.module_set:
        raise ValueError(
            "base_snapshot_id belongs to a different module_set: "
            f"{manifest.request.module_set!r} != {request.module_set!r}"
        )
    return manifest


def _snapshot_content_hash(
    request: SnapshotRequest,
    files: tuple[CodeFileManifestEntry, ...],
) -> str:
    payload = {
        "schema_version": "code_snapshot_identity_v1",
        "request": request.model_dump(mode="json"),
        "files": [
            {
                "path": entry.path,
                "sha256": entry.sha256,
                "size_bytes": entry.size_bytes,
            }
            for entry in sorted(files, key=lambda item: item.path.casefold())
        ],
    }
    return hashlib.sha256(_canonical_json(payload)).hexdigest()


def _canonical_json(payload: object) -> bytes:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def _write_json(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True), encoding="utf-8")


def _verify_staged_snapshot(directory: Path, expected_manifest: CodeSnapshotManifest) -> None:
    manifest = CodeSnapshotManifest.model_validate_json(
        (directory / SNAPSHOT_MANIFEST_FILE).read_text(encoding="utf-8")
    )
    if manifest != expected_manifest:
        raise RuntimeError("Staged snapshot manifest changed before publication")
    expected = {(entry.path, entry.sha256) for entry in expected_manifest.files}
    actual = set(_source_hashes(directory / "source").items())
    if actual != expected:
        raise RuntimeError("Copied snapshot sources do not match the validated intake manifest")


def _source_hashes(source_directory: Path) -> dict[str, str]:
    if not source_directory.is_dir():
        return {}
    result: dict[str, str] = {}
    for path in sorted(source_directory.rglob("*"), key=lambda item: item.as_posix().casefold()):
        if not path.is_file():
            continue
        relative_path = path.relative_to(source_directory).as_posix()
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(DEFAULT_STREAM_CHUNK_BYTES), b""):
                digest.update(block)
        result[relative_path] = digest.hexdigest()
    return result


def _paths_by_hash(
    paths: set[str],
    entries: dict[str, CodeFileManifestEntry],
) -> dict[str, list[str]]:
    result: dict[str, list[str]] = {}
    for path in sorted(paths, key=str.casefold):
        result.setdefault(entries[path].sha256, []).append(path)
    return result


def _expected_path_matches(expected: str, changed_paths: set[str]) -> bool:
    if "/" in expected:
        return expected.casefold() in {path.casefold() for path in changed_paths}
    return expected.casefold() in {Path(path).name.casefold() for path in changed_paths}
