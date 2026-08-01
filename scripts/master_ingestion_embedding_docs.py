from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Sequence


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.core.config import get_settings
from app.core.logging import configure_logging, get_logger
from app.ingestion.docx_loader import DiscoveredDocxFile, discover_docx_files


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Run the existing FDD ingestion, full-document embedding, Qdrant indexing, "
            "verification, and archival stages in order."
        )
    )
    parser.add_argument(
        "--request-batch-size",
        type=_positive_int,
        default=64,
        help="Maximum uncached retrieval units per OpenAI embedding API request. Default: 64.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List the exact commands and files without calling OpenAI, Qdrant, or moving files.",
    )
    parser.add_argument(
        "--replace-existing-embedding-artifacts",
        action="store_true",
        help=(
            "Pass explicit conflict-repair artifact replacement to each selected "
            "document embedding stage."
        ),
    )
    parser.add_argument(
        "--rebuild-qdrant",
        action="store_true",
        help=(
            "Deprecated and intentionally unsupported for embedded local Qdrant. "
            "Use a new versioned QDRANT_COLLECTION_NAME instead."
        ),
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    settings = get_settings()
    configure_logging(settings.log_level)
    logger = get_logger("master_fdd_ingestion")

    if args.rebuild_qdrant:
        raise SystemExit(
            "--rebuild-qdrant is unsupported for embedded local Qdrant because delete-and-recreate "
            "can retain old points. Set QDRANT_COLLECTION_NAME to a new versioned collection instead."
        )

    documents = discover_docx_files(settings.raw_specs_dir)
    if not documents:
        raise RuntimeError(
            f"No FDD DOCX files found in {settings.raw_specs_dir}. "
            "Add reviewed documents before running the master ingestion command."
        )

    _ensure_archive_destinations_are_available(documents, settings.embedded_docs_dir)
    commands = build_pipeline_commands(
        documents=documents,
        cache_directory=settings.cache_dir / "embeddings",
        request_batch_size=args.request_batch_size,
        replace_existing_embedding_artifacts=args.replace_existing_embedding_artifacts,
    )

    logger.info(
        "Master FDD ingestion planned | documents=%s | input=%s | archive=%s",
        len(documents),
        settings.raw_specs_dir,
        settings.embedded_docs_dir,
    )
    for document in documents:
        logger.info("Selected FDD: %s", document.file_name)

    if args.dry_run:
        for command in commands:
            logger.info("DRY RUN command: %s", subprocess.list2cmdline(command))
        return

    for command in commands:
        logger.info("Running: %s", subprocess.list2cmdline(command))
        subprocess.run(command, cwd=ROOT_DIR, check=True)

    archive_documents(documents, settings.embedded_docs_dir)
    logger.info(
        "Master FDD ingestion complete | archived_documents=%s | archive=%s",
        len(documents),
        settings.embedded_docs_dir,
    )


def build_pipeline_commands(
    *,
    documents: Sequence[DiscoveredDocxFile],
    cache_directory: Path,
    request_batch_size: int,
    replace_existing_embedding_artifacts: bool = False,
) -> list[list[str]]:
    """Build the existing script calls for one controlled FDD batch."""

    python_executable = sys.executable
    scripts_dir = ROOT_DIR / "scripts"
    commands = [[python_executable, str(scripts_dir / "run_ingestion_pipeline.py")]]

    for document in documents:
        embedding_command = [
            python_executable,
            str(scripts_dir / "run_embedding_smoke_test.py"),
            "--document",
            document.file_name,
            "--all-units",
            "--request-batch-size",
            str(request_batch_size),
        ]
        if replace_existing_embedding_artifacts:
            embedding_command.append("--replace-existing-artifact")
        commands.append(embedding_command)

    commands.append([python_executable, str(scripts_dir / "run_qdrant_indexing.py")])
    verification_command = [python_executable, str(scripts_dir / "check_qdrant_index.py")]
    for document in documents:
        verification_command.extend(
            [
                "--embedding-artifact",
                str(cache_directory / f"{document.file_path.stem}.embeddings.json"),
            ]
        )
    commands.append(verification_command)
    return commands


def archive_documents(
    documents: Sequence[DiscoveredDocxFile],
    archive_directory: Path,
) -> None:
    """Move only fully verified source documents into the local FDD archive."""

    archive_directory.mkdir(parents=True, exist_ok=True)
    for document in documents:
        destination = archive_directory / document.file_name
        if destination.exists():
            raise FileExistsError(
                f"Refusing to overwrite already archived FDD: {destination}"
            )
        shutil.move(str(document.file_path), str(destination))


def _ensure_archive_destinations_are_available(
    documents: Sequence[DiscoveredDocxFile],
    archive_directory: Path,
) -> None:
    conflicts = [
        str(archive_directory / document.file_name)
        for document in documents
        if (archive_directory / document.file_name).exists()
    ]
    if conflicts:
        raise FileExistsError(
            "Refusing to ingest because archive destinations already exist: "
            + ", ".join(conflicts)
        )


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than 0")
    return parsed


if __name__ == "__main__":
    main()
