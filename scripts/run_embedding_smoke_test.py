from __future__ import annotations

import argparse
import sys
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
from app.embeddings.smoke_test import limit_embedding_batch
from app.ingestion.chunker import chunk_normalized_artifact
from app.ingestion.docx_ingestion_artifact import ingest_docx_file
from app.ingestion.docx_loader import discover_docx_files
from app.ingestion.normalized_artifact import build_normalized_artifact
from app.ingestion.retrieval_ready_artifact import build_retrieval_ready_artifact
from app.ingestion.table_chunker import chunk_tables_from_artifact


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a tiny real embedding smoke test on retrieval-ready units."
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=2,
        help="Maximum number of retrieval-ready units to embed. Default: 2.",
    )
    parser.add_argument(
        "--document",
        type=str,
        default="",
        help="Optional exact DOCX filename from data/raw_specs. Defaults to first valid DOCX.",
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
    logger.info("Embedding smoke-test limit: %s", args.limit)
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
    smoke_batch = limit_embedding_batch(embedding_batch, limit=args.limit)
    embedded_batch = embed_batch(
        smoke_batch,
        cache_directory=embedding_cache_dir,
    )
    embedding_output = write_embedding_batch_to_json(embedded_batch, embedding_cache_dir)
    summary = build_embedding_run_summary(embedded_batch)
    summary_output = write_embedding_run_summary_to_json(summary, embedding_cache_dir)

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


if __name__ == "__main__":
    main()
