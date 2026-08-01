from __future__ import annotations

import argparse
import sys
from datetime import UTC, datetime
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.core.config import get_settings
from app.core.logging import configure_logging, get_logger
from app.embeddings.client import embed_batch
from app.embeddings.embedding_artifact_writer import write_embedding_batch_to_json
from app.embeddings.embedding_contract import build_embedding_batch_contract
from app.embeddings.embedding_run_summary import (
    build_embedding_run_summary,
    write_embedding_run_summary_to_json,
)
from app.embeddings.smoke_test import filter_embedding_batch_by_source_kind, limit_embedding_batch
from app.ingestion.chunker import chunk_normalized_artifact
from app.ingestion.docx_ingestion_artifact import ingest_docx_file
from app.ingestion.docx_loader import discover_docx_files
from app.ingestion.normalized_artifact import build_normalized_artifact
from app.ingestion.retrieval_ready_artifact import build_retrieval_ready_artifact
from app.ingestion.table_chunker import chunk_tables_from_artifact


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Embed a sample or all retrieval-ready units from one FDD."
    )
    selection_group = parser.add_mutually_exclusive_group()
    selection_group.add_argument(
        "--limit",
        type=int,
        default=2,
        help="Maximum number of retrieval-ready units to embed. Default: 2.",
    )
    selection_group.add_argument(
        "--all-units",
        action="store_true",
        help="Embed every retrieval-ready unit in the selected document.",
    )
    parser.add_argument(
        "--document",
        type=str,
        default="",
        help="Optional exact DOCX filename from data/raw_specs. Defaults to first valid DOCX.",
    )
    parser.add_argument(
        "--source-kind",
        choices=["paragraph", "table"],
        default=None,
        help="Optional retrieval unit source kind to smoke test: paragraph or table.",
    )
    parser.add_argument(
        "--request-batch-size",
        type=int,
        default=64,
        help="Maximum uncached retrieval units per OpenAI embedding API request. Default: 64.",
    )
    parser.add_argument(
        "--replace-existing-artifact",
        action="store_true",
        help=(
            "Quarantine the selected document's prior embedding artifact before "
            "rebuilding it. Use only for an explicitly diagnosed artifact conflict."
        ),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = get_settings()
    configure_logging(settings.log_level)
    logger = get_logger("embedding_smoke_test")

    if not settings.openai_api_key:
        raise RuntimeError("OPENAI_API_KEY is not configured. Set it in .env before running this script.")

    discovered_files = discover_docx_files(settings.raw_specs_dir)
    if args.document:
        discovered_files = [item for item in discovered_files if item.file_name == args.document]

    if not discovered_files:
        raise RuntimeError("No matching DOCX files found for embedding smoke test.")

    selected_file = discovered_files[0]
    embedding_cache_dir = settings.cache_dir / "embeddings"

    logger.info("Selected document: %s", selected_file.file_name)
    logger.info(
        "Embedding selection: %s",
        "all units" if args.all_units else f"first {args.limit} units",
    )
    logger.info("Source-kind filter: %s", args.source_kind or "none")
    logger.info("Embedding API request batch size: %s", args.request_batch_size)
    logger.info("Embedding cache directory: %s", embedding_cache_dir)

    raw_artifact = ingest_docx_file(selected_file.file_path)
    normalized_artifact = build_normalized_artifact(raw_artifact)
    paragraph_chunks = chunk_normalized_artifact(normalized_artifact)
    table_chunks = chunk_tables_from_artifact(normalized_artifact)
    retrieval_ready_artifact = build_retrieval_ready_artifact(
        normalized_artifact,
        paragraph_chunks,
        table_chunks,
    )
    embedding_batch = build_embedding_batch_contract(
        retrieval_ready_artifact,
        embedding_model=settings.openai_embedding_model,
        artifact_version=settings.artifact_version,
    )
    filtered_batch = filter_embedding_batch_by_source_kind(
        embedding_batch,
        source_kind=args.source_kind,
    )
    if not filtered_batch.records:
        raise RuntimeError(
            "No retrieval-ready units found for embedding smoke test "
            f"with source_kind={args.source_kind!r}."
        )

    selected_batch = filtered_batch if args.all_units else limit_embedding_batch(
        filtered_batch,
        limit=args.limit,
    )
    output_suffix = f".{args.source_kind}" if args.source_kind else ""
    existing_artifact = (
        embedding_cache_dir
        / f"{Path(selected_batch.document_name).stem}{output_suffix}.embeddings.json"
    )
    if args.replace_existing_artifact:
        quarantined = quarantine_embedding_artifact(existing_artifact)
        if quarantined is not None:
            logger.warning("Quarantined prior embedding artifact: %s", quarantined)

    embedded_batch = embed_batch(
        selected_batch,
        cache_directory=embedding_cache_dir,
        request_batch_size=args.request_batch_size,
    )
    embedding_output = write_embedding_batch_to_json(
        embedded_batch,
        embedding_cache_dir,
        file_stem_suffix=output_suffix,
    )
    summary = build_embedding_run_summary(embedded_batch)
    summary_output = write_embedding_run_summary_to_json(
        summary,
        embedding_cache_dir,
        file_stem_suffix=output_suffix,
    )

    first_vector_length = 0
    if embedded_batch.records and embedded_batch.records[0].vector is not None:
        first_vector_length = len(embedded_batch.records[0].vector)

    logger.info("Wrote embedding artifact: %s", embedding_output)
    logger.info("Wrote embedding summary: %s", summary_output)
    logger.info(
        "Smoke test complete | records=%s | cached=%s | embedded=%s | hit_rate=%.2f | vector_dim=%s",
        embedded_batch.total_records,
        embedded_batch.cached_count,
        embedded_batch.embedded_count,
        summary.cache_hit_rate,
        first_vector_length,
    )


def quarantine_embedding_artifact(artifact_path: Path) -> Path | None:
    """Preserve a diagnosed conflicting artifact outside the active cache glob."""

    if not artifact_path.exists():
        return None

    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    quarantined = artifact_path.with_name(f"{artifact_path.name}.conflict-{timestamp}")
    artifact_path.rename(quarantined)
    return quarantined


if __name__ == "__main__":
    main()
