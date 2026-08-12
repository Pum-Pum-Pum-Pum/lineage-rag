from __future__ import annotations

import hashlib
import json
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path


FIXED_ZIP_TIMESTAMP = (1980, 1, 1, 0, 0, 0)
ROOT_FILES = (
    ".env.example",
    "README.md",
    "config/ingestion_sources.toml",
    "config/code_analysis.toml",
    "pyproject.toml",
    "uv.lock",
)
SECRET_SUFFIXES = {
    ".crt",
    ".key",
    ".p12",
    ".pem",
    ".pfx",
    ".secret",
    ".secrets",
}


@dataclass(frozen=True)
class NativePackageFile:
    path: str
    sha256: str
    size_bytes: int


@dataclass(frozen=True)
class NativePackageManifest:
    schema_version: str
    python_version: str
    dependency_lock: str
    mutable_state_bundled: bool
    files: list[NativePackageFile]


@dataclass(frozen=True)
class NativePackageResult:
    archive_path: Path
    archive_sha256: str
    file_count: int


def build_native_package(
    *,
    project_root: str | Path,
    output_path: str | Path,
) -> NativePackageResult:
    """Build a deterministic source bundle for an approved native runtime."""

    root = Path(project_root).resolve()
    output = Path(output_path).resolve()
    files = _select_runtime_files(root)
    manifest = NativePackageManifest(
        schema_version="1.0",
        python_version="3.12",
        dependency_lock="uv.lock",
        mutable_state_bundled=False,
        files=[
            NativePackageFile(
                path=relative.as_posix(),
                sha256=_sha256(root / relative),
                size_bytes=(root / relative).stat().st_size,
            )
            for relative in files
        ],
    )
    manifest_bytes = json.dumps(
        asdict(manifest),
        indent=2,
        sort_keys=True,
    ).encode("utf-8")

    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(
        output,
        mode="w",
        compression=zipfile.ZIP_DEFLATED,
        compresslevel=9,
    ) as archive:
        for relative in files:
            _write_deterministic(
                archive,
                relative.as_posix(),
                (root / relative).read_bytes(),
            )
        _write_deterministic(
            archive,
            "deployment/package_manifest.json",
            manifest_bytes,
        )

    return NativePackageResult(
        archive_path=output,
        archive_sha256=_sha256(output),
        file_count=len(files) + 1,
    )


def _select_runtime_files(root: Path) -> list[Path]:
    candidates = [Path(item) for item in ROOT_FILES]
    candidates.extend(
        path.relative_to(root)
        for directory in ("app", "scripts", "deployment")
        for path in sorted((root / directory).rglob("*"))
        if path.is_file()
        and (
            path.suffix == ".py"
            or directory == "deployment"
        )
    )
    selected: list[Path] = []
    for relative in sorted(set(candidates), key=lambda item: item.as_posix()):
        absolute = (root / relative).resolve()
        if not absolute.is_relative_to(root):
            raise ValueError(f"Package path escapes project root: {relative}")
        if not absolute.is_file():
            raise FileNotFoundError(f"Required package file is missing: {relative}")
        if _is_secret(relative):
            raise ValueError(f"Secret-like file cannot be packaged: {relative}")
        selected.append(relative)
    return selected


def _is_secret(path: Path) -> bool:
    name = path.name.lower()
    return (
        (name.startswith(".env") and name != ".env.example")
        or path.suffix.lower() in SECRET_SUFFIXES
    )


def _write_deterministic(
    archive: zipfile.ZipFile,
    name: str,
    content: bytes,
) -> None:
    info = zipfile.ZipInfo(name, FIXED_ZIP_TIMESTAMP)
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = 0o644 << 16
    archive.writestr(info, content)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
