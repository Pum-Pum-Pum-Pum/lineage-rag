from __future__ import annotations

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.core.config import get_settings
from app.core.logging import configure_logging, get_logger
from app.vectorstore.qdrant_schema import create_persistent_qdrant_client


def main() -> None:
    settings = get_settings()
    configure_logging(settings.log_level)
    logger = get_logger("qdrant_index_check")

    client = create_persistent_qdrant_client(settings.qdrant_local_path)
    collection_name = settings.qdrant_collection_name

    if not client.collection_exists(collection_name):
        logger.warning("Collection does not exist: %s", collection_name)
        client.close()
        return

    collection_info = client.get_collection(collection_name)
    count = client.count(collection_name).count

    logger.info("Collection: %s", collection_name)
    logger.info("Vector size: %s", collection_info.config.params.vectors.size)
    logger.info("Distance: %s", collection_info.config.params.vectors.distance)
    logger.info("Point count: %s", count)

    points, _ = client.scroll(
        collection_name=collection_name,
        limit=1,
        with_payload=True,
        with_vectors=False,
    )

    if points:
        payload = points[0].payload or {}
        logger.info("Sample point id: %s", points[0].id)
        logger.info("Sample unit_id: %s", payload.get("unit_id"))
        logger.info("Sample source_kind: %s", payload.get("source_kind"))
        logger.info("Sample document_family: %s", payload.get("document_family"))
        logger.info("Sample release_label: %s", payload.get("release_label"))
    else:
        logger.warning("Collection exists but contains no points.")

    client.close()


if __name__ == "__main__":
    main()
