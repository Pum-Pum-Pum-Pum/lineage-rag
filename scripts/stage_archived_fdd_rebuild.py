from __future__ import annotations

import argparse
import hashlib
import json
import sys
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Sequence


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.core.config import get_settings
from app.core.logging import configure_logging, get_logger
from app.embeddings.client import embed_batch
from app.embeddings.embedding_artifact_writer import write_embedding_batch_to_json
from app.embeddings.embedding_contract import EmbeddingBatch, build_embedding_batch_contract
from app.ingestion.chunked_artifact_writer import write_chunked_document_to_json
from app.ingestion.chunker import chunk_normalized_artifact
from app.ingestion.docx_ingestion_artifact import ingest_docx_file
from app.ingestion.docx_loader import DiscoveredDocxFile, discover_docx_files
from app.ingestion.normalized_artifact import build_normalized_artifact
from app.ingestion.processed_artifact_writer import write_ingested_artifact_to_json
from app.ingestion.retrieval_ready_artifact import build_retrieval_ready_artifact
from app.ingestion.retrieval_ready_artifact_writer import write_retrieval_ready_artifact_to_json
from app.ingestion.table_chunker import chunk_tables_from_artifact
from app.vectorstore.qdrant_indexer import index_embedding_cache_directory
from app.vectorstore.qdrant_schema import QdrantCollectionConfig, create_persistent_qdrant_client
from scripts.check_qdrant_index import verify_embedding_artifacts


@dataclass(frozen=True)
class ArchivedSource:
    document_name: str
    sha256: str
    size_bytes: int


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    settings = get_settings()
    parser = argparse.ArgumentParser(
        description=(
            "Build a separate, validated FDD retrieval generation from archived DOCX files. "
            "This never changes the configured live collection."
        )
    )
    parser.add_argument(
        "--stage-directory",
        type=Path,
        default=settings.data_dir / "staging" / "table_context_v1",
        help="New directory for this staged generation.",
    )
    parser.add_argument(
        "--source-directory",
        type=Path,
        default=settings.embedded_docs_dir,
        help="Immutable archived DOCX source directory.",
    )
    parser.add_argument(
        "--collection-name",
        default="functional_specs_v3",
        help="New, empty Qdrant collection for this generation.",
    )
    parser.add_argument(
        "--embedding-input-version",
        default=settings.artifact_version,
        help=(
            "Embedding-input cache compatibility version. Keep this equal to the active "
            "version only when model, normalized retrieval text, and preprocessing contract "
            "are unchanged. This is not the index-generation name."
        ),
    )
    parser.add_argument(
        "--index-generation",
        default="table_context_v1",
        help=(
            "Human-readable retrieval/index generation identifier recorded in the staged "
            "manifest. It must change when retrieval representation or index semantics change."
        ),
    )
    parser.add_argument(
        "--request-batch-size",
        type=_positive_int,
        default=64,
        help="Maximum uncached units per OpenAI embedding request. Default: 64.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Read and hash archived inputs, then print the plan without writes or API calls.",
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    settings = get_settings()
    configure_logging(settings.log_level)
    logger = get_logger("stage_archived_fdd_rebuild")

    sources = discover_docx_files(args.source_directory)
    if not sources:
        raise RuntimeError(f"No archived FDD DOCX files found in {args.source_directory}.")

    source_manifest = build_source_manifest(sources)
    validate_stage_targets(
        stage_directory=args.stage_directory,
        target_collection_name=args.collection_name,
        active_collection_name=settings.qdrant_collection_name,
        qdrant_local_path=settings.qdrant_local_path,
    )
    logger.info(
        "Archived FDD staging plan | sources=%s | stage=%s | index_generation=%s | "
        "embedding_input_version=%s | target_collection=%s | cache_seed=%s",
        len(sources),
        args.stage_directory,
        args.index_generation,
        args.embedding_input_version,
        args.collection_name,
        settings.cache_dir / "embeddings",
    )
    for source in source_manifest:
        logger.info("Source | name=%s | sha256=%s | bytes=%s", source.document_name, source.sha256, source.size_bytes)

    if args.dry_run:
        logger.info("DRY RUN complete: no artifacts, OpenAI calls, Qdrant writes, or configuration changes.")
        return

    run_staged_rebuild(
        sources=sources,
        source_manifest=source_manifest,
        stage_directory=args.stage_directory,
        collection_name=args.collection_name,
        embedding_input_version=args.embedding_input_version,
        index_generation=args.index_generation,
        embedding_model=settings.openai_embedding_model,
        vector_size=settings.qdrant_vector_size,
        seed_cache_directory=settings.cache_dir / "embeddings",
        qdrant_local_path=settings.qdrant_local_path,
        request_batch_size=args.request_batch_size,
    )


def build_source_manifest(sources: Sequence[DiscoveredDocxFile]) -> list[ArchivedSource]:
    return [
        ArchivedSource(
            document_name=source.file_name,
            sha256=_sha256_file(source.file_path),
            size_bytes=source.file_path.stat().st_size,
        )
        for source in sources
    ]


def validate_stage_targets(
    *,
    stage_directory: Path,
    target_collection_name: str,
    active_collection_name: str,
    qdrant_local_path: Path,
) -> None:
    """Fail before paid work when a stage could overwrite or mix generations."""

    if not target_collection_name.strip():
        raise ValueError("Target Qdrant collection name must not be empty.")
    if target_collection_name == active_collection_name:
        raise ValueError(
            "Target collection must differ from the active collection; staged rebuilds cannot write "
            f"to {active_collection_name!r}."
        )
    if stage_directory.exists():
        raise FileExistsError(
            f"Stage directory already exists: {stage_directory}. Choose a new stage name; do not overwrite it."
        )

    client = create_persistent_qdrant_client(qdrant_local_path)
    try:
        if client.collection_exists(target_collection_name):
            raise FileExistsError(
                "Target Qdrant collection already exists: "
                f"{target_collection_name}. Choose a new versioned collection; do not delete or reuse it."
            )
    finally:
        client.close()


def run_staged_rebuild(
    *,
    sources: Sequence[DiscoveredDocxFile],
    source_manifest: Sequence[ArchivedSource],
    stage_directory: Path,
    collection_name: str,
    embedding_input_version: str,
    index_generation: str,
    embedding_model: str,
    vector_size: int,
    seed_cache_directory: Path,
    qdrant_local_path: Path,
    request_batch_size: int,
) -> None:
    """Create one isolated generation and prove every staged point exists exactly.

    The existing active embedding cache is read-only input. An unchanged retrieval
    text with the same model/version gets its vector reused; parent-enriched tables
    have a new content hash and are the only units sent to OpenAI.
    """

    processed_directory = stage_directory / "processed"
    embedding_directory = stage_directory / "cache" / "embeddings"
    manifest_path = stage_directory / "stage_manifest.json"
    stage_directory.mkdir(parents=True, exist_ok=False)
    _write_manifest(
        manifest_path,
        status="running",
        sources=source_manifest,
        collection_name=collection_name,
        embedding_input_version=embedding_input_version,
        index_generation=index_generation,
        embedding_model=embedding_model,
    )

    staged_embedding_paths: list[Path] = []
    totals = {"records": 0, "cached": 0, "embedded": 0}
    try:
        for source in sources:
            raw_artifact = ingest_docx_file(source.file_path)
            normalized_artifact = build_normalized_artifact(raw_artifact)
            paragraph_chunks = chunk_normalized_artifact(normalized_artifact)
            table_chunks = chunk_tables_from_artifact(normalized_artifact)
            retrieval_ready_artifact = build_retrieval_ready_artifact(
                normalized_artifact,
                paragraph_chunks,
                table_chunks,
            )
            write_ingested_artifact_to_json(raw_artifact, processed_directory)
            write_chunked_document_to_json(paragraph_chunks, processed_directory)
            write_retrieval_ready_artifact_to_json(retrieval_ready_artifact, processed_directory)

            batch = build_embedding_batch_contract(
                retrieval_ready_artifact,
                embedding_model=embedding_model,
                # EmbeddingRecord retains the legacy field name for on-disk
                # compatibility. Its value is specifically the input contract,
                # not the staged Qdrant/index generation.
                artifact_version=embedding_input_version,
            )
            embedded_batch = embed_batch(
                batch,
                cache_directory=seed_cache_directory,
                request_batch_size=request_batch_size,
            )
            _validate_vector_dimensions(embedded_batch, vector_size)
            staged_embedding_paths.append(
                write_embedding_batch_to_json(embedded_batch, embedding_directory)
            )
            totals["records"] += embedded_batch.total_records
            totals["cached"] += embedded_batch.cached_count
            totals["embedded"] += embedded_batch.embedded_count

        client = create_persistent_qdrant_client(qdrant_local_path)
        try:
            index_summary = index_embedding_cache_directory(
                client=client,
                collection_config=QdrantCollectionConfig(
                    collection_name=collection_name,
                    vector_size=vector_size,
                ),
                cache_directory=embedding_directory,
            )
            verified_records = verify_embedding_artifacts(
                client=client,
                collection_name=collection_name,
                artifact_paths=staged_embedding_paths,
            )
        finally:
            client.close()

        _write_manifest(
            manifest_path,
            status="verified",
            sources=source_manifest,
            collection_name=collection_name,
            embedding_input_version=embedding_input_version,
            index_generation=index_generation,
            embedding_model=embedding_model,
            totals=totals,
            qdrant={
                "attempted_records": index_summary.attempted_records,
                "upserted_points": index_summary.upserted_points,
                "skipped_records": index_summary.skipped_records,
                "verified_records": verified_records,
            },
        )
    except Exception as exc:
        _write_manifest(
            manifest_path,
            status="failed",
            sources=source_manifest,
            collection_name=collection_name,
            embedding_input_version=embedding_input_version,
            index_generation=index_generation,
            embedding_model=embedding_model,
            totals=totals,
            failure_type=type(exc).__name__,
            failure_message=str(exc),
        )
        raise


def _validate_vector_dimensions(batch: EmbeddingBatch, expected_vector_size: int) -> None:
    invalid = [
        record.unit_id
        for record in batch.records
        if record.vector is None or len(record.vector) != expected_vector_size
    ]
    if invalid:
        raise RuntimeError(
            "Embedding vector dimension validation failed before Qdrant indexing: "
            + ", ".join(invalid[:5])
        )


def _write_manifest(
    path: Path,
    *,
    status: str,
    sources: Sequence[ArchivedSource],
    collection_name: str,
    embedding_input_version: str,
    index_generation: str,
    embedding_model: str,
    totals: dict[str, int] | None = None,
    qdrant: dict[str, int] | None = None,
    failure_type: str | None = None,
    failure_message: str | None = None,
) -> None:
    payload = {
        "schema_version": "staged_fdd_rebuild_v1",
        "status": status,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "index_generation": index_generation,
        "representation": "table_parent_context_v1",
        "collection_name": collection_name,
        "embedding_input_version": embedding_input_version,
        "embedding_record_artifact_version": embedding_input_version,
        "embedding_model": embedding_model,
        "sources": [asdict(source) for source in sources],
        "totals": totals or {},
        "qdrant": qdrant or {},
        "failure_type": failure_type,
        "failure_message": failure_message,
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        for block in iter(lambda: file_handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be greater than 0")
    return parsed


if __name__ == "__main__":
    main()
