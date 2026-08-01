from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.core.config import get_settings
from app.core.logging import configure_logging, get_logger
from app.vectorstore.qdrant_indexer import index_embedding_cache_directory
from app.vectorstore.qdrant_schema import QdrantCollectionConfig, create_persistent_qdrant_client


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Index local embedding artifacts into the configured Qdrant collection."
    )
    parser.add_argument(
        "--rebuild",
        action="store_true",
        help=(
            "Deprecated and intentionally unsupported for embedded local Qdrant. "
            "Use a new versioned QDRANT_COLLECTION_NAME instead."
        ),
    )
    return parser.parse_args()


def main() -> None:
    settings = get_settings()
    configure_logging(settings.log_level)
    logger = get_logger("qdrant_indexing")

    embedding_cache_dir = settings.cache_dir / "embeddings"
    collection_config = QdrantCollectionConfig(
        collection_name=settings.qdrant_collection_name,
        vector_size=settings.qdrant_vector_size,
    )
    client = create_persistent_qdrant_client(settings.qdrant_local_path)

    try:
        logger.info("Embedding cache directory: %s", embedding_cache_dir)
        logger.info("Qdrant local path: %s", settings.qdrant_local_path)
        logger.info("Qdrant collection: %s", settings.qdrant_collection_name)
        logger.info("Qdrant vector size: %s", settings.qdrant_vector_size)

        summary = index_embedding_cache_directory(
            client=client,
            collection_config=collection_config,
            cache_directory=embedding_cache_dir,
        )

        logger.info(
            "Qdrant indexing complete | collection=%s | attempted=%s | upserted=%s | skipped=%s",
            summary.collection_name,
            summary.attempted_records,
            summary.upserted_points,
            summary.skipped_records,
        )
    finally:
        client.close()


if __name__ == "__main__":
    args = parse_args()
    if args.rebuild:
        raise SystemExit(
            "--rebuild is unsupported for embedded local Qdrant because delete-and-recreate "
            "can retain old points. Set QDRANT_COLLECTION_NAME to a new versioned collection instead."
        )
    main()
