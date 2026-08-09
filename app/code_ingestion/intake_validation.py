from __future__ import annotations

import codecs
import hashlib
import re
from pathlib import Path

from app.code_ingestion.snapshot_models import (
    CodeFileManifestEntry,
    IntakeValidationReport,
    ValidationIssue,
)


ALLOWED_CODE_EXTENSIONS = frozenset({".sql", ".prc", ".fnc", ".ddl"})
DEFAULT_LARGE_FILE_WARNING_BYTES = 5 * 1024 * 1024
DEFAULT_STREAM_CHUNK_BYTES = 1024 * 1024

_SECRET_PATTERNS = {
    "private_key_material": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----", re.IGNORECASE),
    "openai_api_key": re.compile(r"\bsk-[A-Za-z0-9_-]{20,}\b"),
    "aws_access_key": re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b"),
    "assigned_password": re.compile(
        r"\b(?:password|passwd|pwd)\b\s*(?::=|=>|=)\s*['\"][^'\"\r\n]{8,}['\"]",
        re.IGNORECASE,
    ),
}


class CodeIntakeValidationError(ValueError):
    """Raised when code intake cannot safely become an immutable snapshot."""

    def __init__(self, issues: list[ValidationIssue]) -> None:
        self.issues = tuple(issues)
        summary = "; ".join(
            f"{issue.code}{f' ({issue.path})' if issue.path else ''}: {issue.message}"
            for issue in issues
        )
        super().__init__(summary)


def validate_code_intake(
    source_directory: Path,
    *,
    large_file_warning_bytes: int = DEFAULT_LARGE_FILE_WARNING_BYTES,
    stream_chunk_bytes: int = DEFAULT_STREAM_CHUNK_BYTES,
) -> IntakeValidationReport:
    """Validate and hash an allowlisted custom-code source tree without mutating it."""

    if large_file_warning_bytes <= 0:
        raise ValueError("large_file_warning_bytes must be greater than zero")
    if stream_chunk_bytes <= 0:
        raise ValueError("stream_chunk_bytes must be greater than zero")
    if not source_directory.is_dir():
        raise CodeIntakeValidationError(
            [
                ValidationIssue(
                    severity="error",
                    code="source_directory_missing",
                    path=str(source_directory),
                    message="The snapshot source directory does not exist.",
                )
            ]
        )

    entries: list[CodeFileManifestEntry] = []
    warnings: list[ValidationIssue] = []
    errors: list[ValidationIssue] = []
    seen_paths: set[str] = set()

    for path in sorted(source_directory.rglob("*"), key=lambda item: item.as_posix().casefold()):
        if path.is_symlink():
            errors.append(
                ValidationIssue(
                    severity="error",
                    code="symlink_not_allowed",
                    path=path.relative_to(source_directory).as_posix(),
                    message="Symlinks are not accepted in immutable code snapshots.",
                )
            )
            continue
        if not path.is_file():
            continue

        relative_path = path.relative_to(source_directory).as_posix()
        casefold_path = relative_path.casefold()
        if casefold_path in seen_paths:
            errors.append(
                ValidationIssue(
                    severity="error",
                    code="case_insensitive_path_collision",
                    path=relative_path,
                    message="Another source path differs only by character case.",
                )
            )
            continue
        seen_paths.add(casefold_path)

        extension = path.suffix.lower()
        if extension not in ALLOWED_CODE_EXTENSIONS:
            errors.append(
                ValidationIssue(
                    severity="error",
                    code="extension_not_allowed",
                    path=relative_path,
                    message=f"Allowed extensions are {sorted(ALLOWED_CODE_EXTENSIONS)}.",
                )
            )
            continue

        try:
            entry, file_warnings = _analyze_code_file(
                path,
                relative_path=relative_path,
                large_file_warning_bytes=large_file_warning_bytes,
                stream_chunk_bytes=stream_chunk_bytes,
            )
        except CodeIntakeValidationError as exc:
            errors.extend(exc.issues)
            continue
        entries.append(entry)
        warnings.extend(file_warnings)

    if not entries and not errors:
        errors.append(
            ValidationIssue(
                severity="error",
                code="no_code_files",
                path=None,
                message="No allowlisted PL/SQL or DDL files were found.",
            )
        )
    if errors:
        raise CodeIntakeValidationError(errors)

    return IntakeValidationReport(
        source_directory=str(source_directory.resolve()),
        files=tuple(sorted(entries, key=lambda entry: entry.path.casefold())),
        warnings=tuple(warnings),
    )


def _analyze_code_file(
    path: Path,
    *,
    relative_path: str,
    large_file_warning_bytes: int,
    stream_chunk_bytes: int,
) -> tuple[CodeFileManifestEntry, list[ValidationIssue]]:
    exact_digest = hashlib.sha256()
    size_bytes = 0
    control_bytes = 0
    first_bytes = b""

    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(stream_chunk_bytes), b""):
            if not first_bytes:
                first_bytes = block[:4]
            exact_digest.update(block)
            size_bytes += len(block)
            control_bytes += sum(byte < 32 and byte not in {9, 10, 13} for byte in block)

    encoding = _detect_encoding(path, first_bytes, stream_chunk_bytes)
    if not encoding.startswith("utf-16") and size_bytes and control_bytes / size_bytes > 0.01:
        raise CodeIntakeValidationError(
            [
                ValidationIssue(
                    severity="error",
                    code="binary_content_detected",
                    path=relative_path,
                    message="Control-byte density indicates binary rather than source text.",
                )
            ]
        )

    normalized_digest, line_count, secret_ids = _scan_decoded_text(
        path,
        encoding=encoding,
        stream_chunk_bytes=stream_chunk_bytes,
    )
    if secret_ids:
        raise CodeIntakeValidationError(
            [
                ValidationIssue(
                    severity="error",
                    code="potential_secret_detected",
                    path=relative_path,
                    message=(
                        "Potential secret categories were detected; values are intentionally omitted: "
                        + ", ".join(sorted(secret_ids))
                    ),
                )
            ]
        )

    warning_codes: list[str] = []
    warnings: list[ValidationIssue] = []
    if size_bytes > large_file_warning_bytes:
        warning_codes.append("large_file")
        warnings.append(
            ValidationIssue(
                severity="warning",
                code="large_file",
                path=relative_path,
                message=(
                    f"File is {size_bytes} bytes, above the {large_file_warning_bytes}-byte warning threshold; "
                    "it remains accepted for isolated downstream parsing."
                ),
            )
        )
    if encoding == "cp1252":
        warning_codes.append("legacy_encoding")
        warnings.append(
            ValidationIssue(
                severity="warning",
                code="legacy_encoding",
                path=relative_path,
                message="File is valid Windows-1252 rather than UTF-8; its decoded encoding is recorded.",
            )
        )
    if size_bytes == 0:
        warning_codes.append("empty_file")
        warnings.append(
            ValidationIssue(
                severity="warning",
                code="empty_file",
                path=relative_path,
                message="Empty source file is retained in the snapshot but will not produce retrieval units.",
            )
        )

    return (
        CodeFileManifestEntry(
            path=relative_path,
            extension=path.suffix.lower(),
            sha256=exact_digest.hexdigest(),
            normalized_text_sha256=normalized_digest,
            size_bytes=size_bytes,
            encoding=encoding,
            line_count=line_count,
            is_large_file=size_bytes > large_file_warning_bytes,
            warnings=tuple(warning_codes),
        ),
        warnings,
    )


def _detect_encoding(path: Path, first_bytes: bytes, stream_chunk_bytes: int) -> str:
    if first_bytes.startswith(codecs.BOM_UTF8):
        candidates = ("utf-8-sig",)
    elif first_bytes.startswith(codecs.BOM_UTF16_LE) or first_bytes.startswith(codecs.BOM_UTF16_BE):
        candidates = ("utf-16",)
    else:
        candidates = ("utf-8", "cp1252")

    for encoding in candidates:
        decoder = codecs.getincrementaldecoder(encoding)(errors="strict")
        try:
            with path.open("rb") as handle:
                for block in iter(lambda: handle.read(stream_chunk_bytes), b""):
                    decoder.decode(block)
                decoder.decode(b"", final=True)
            return encoding
        except UnicodeDecodeError:
            continue

    raise CodeIntakeValidationError(
        [
            ValidationIssue(
                severity="error",
                code="unsupported_encoding",
                path=path.name,
                message="Source is not valid UTF-8, UTF-16 with BOM, or Windows-1252 text.",
            )
        ]
    )


def _scan_decoded_text(
    path: Path,
    *,
    encoding: str,
    stream_chunk_bytes: int,
) -> tuple[str, int, set[str]]:
    decoder = codecs.getincrementaldecoder(encoding)(errors="strict")
    normalized_digest = hashlib.sha256()
    secret_ids: set[str] = set()
    overlap = ""
    pending_carriage_return = False
    newline_count = 0
    saw_text = False
    final_character = ""

    def consume(text: str) -> None:
        nonlocal overlap, newline_count, saw_text, final_character
        if not text:
            return
        normalized_digest.update(text.encode("utf-8"))
        newline_count += text.count("\n")
        saw_text = True
        final_character = text[-1]
        scan_text = overlap + text
        for pattern_id, pattern in _SECRET_PATTERNS.items():
            if pattern.search(scan_text):
                secret_ids.add(pattern_id)
        overlap = scan_text[-512:]

    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(stream_chunk_bytes), b""):
            decoded = decoder.decode(block)
            if pending_carriage_return:
                decoded = "\r" + decoded
                pending_carriage_return = False
            if decoded.endswith("\r"):
                decoded = decoded[:-1]
                pending_carriage_return = True
            consume(decoded.replace("\r\n", "\n").replace("\r", "\n"))

    final_text = decoder.decode(b"", final=True)
    if pending_carriage_return:
        final_text = "\r" + final_text
    consume(final_text.replace("\r\n", "\n").replace("\r", "\n"))
    line_count = 0 if not saw_text else newline_count + (0 if final_character == "\n" else 1)
    return normalized_digest.hexdigest(), line_count, secret_ids

