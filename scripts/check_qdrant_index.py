from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.core.config import get_settings
from app.core.logging import configure_logging, get_logger
from app.embeddings.embedding_contract import EmbeddingRecord
from app.vectorstore.qdrant_schema import create_persistent_qdrant_client
from app.vectorstore.qdrant_upsert import build_qdrant_point_id


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect the configured Qdrant collection and optionally verify embedding artifacts."
    )
    parser.add_argument(
        "--embedding-artifact",
        action="append",
        default=[],
        help=(
            "Embedding JSON artifact whose exact Qdrant points must exist. "
            "Repeat for multiple documents."
        ),
    )
    return parser.parse_args()


def main(embedding_artifact_paths: list[str | Path] | None = None) -> None:
    settings = get_settings()
    configure_logging(settings.log_level)
    logger = get_logger("qdrant_index_check")
    artifact_paths = [Path(path) for path in (embedding_artifact_paths or [])]

    client = create_persistent_qdrant_client(settings.qdrant_local_path)
    collection_name = settings.qdrant_collection_name

    try:
        if not client.collection_exists(collection_name):
            if artifact_paths:
                raise RuntimeError(
                    "Qdrant collection does not exist; cannot verify the requested embedding artifacts."
                )
            logger.warning("Collection does not exist: %s", collection_name)
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

        if artifact_paths:
            verified_records = verify_embedding_artifacts(
                client=client,
                collection_name=collection_name,
                artifact_paths=artifact_paths,
            )
            logger.info(
                "Verified exact Qdrant points for embedding artifacts | artifacts=%s | records=%s",
                len(artifact_paths),
                verified_records,
            )
    finally:
        client.close()


def verify_embedding_artifacts(
    *,
    client,
    collection_name: str,
    artifact_paths: list[Path],
) -> int:
    """Confirm that every embedded record is present with its expected metadata.

    A collection-level point count is not proof that a newly ingested document
    was indexed. This check validates deterministic IDs and identifying payload
    fields for each supplied local embedding artifact.
    """

    records = _load_embedded_records(artifact_paths)
    expected_by_point_id = {build_qdrant_point_id(record): record for record in records}
    if len(expected_by_point_id) != len(records):
        raise RuntimeError(
            "Embedding artifacts contain duplicate Qdrant point IDs; cannot prove "
            "document-specific index integrity."
        )

    points = client.retrieve(
        collection_name=collection_name,
        ids=list(expected_by_point_id),
        with_payload=True,
        with_vectors=False,
    )
    points_by_id = {str(point.id): point for point in points}
    missing_ids = sorted(set(expected_by_point_id) - set(points_by_id))
    if missing_ids:
        raise RuntimeError(
            "Qdrant is missing expected embedding points: " + ", ".join(missing_ids[:5])
        )

    for point_id, expected in expected_by_point_id.items():
        payload = points_by_id[point_id].payload or {}
        for field_name, expected_value in {
            "cache_key": expected.cache_key,
            "unit_id": expected.unit_id,
            "document_family": expected.document_family,
            "release_label": expected.release_label,
            "content_hash": expected.content_hash,
            "document_id": expected.document_id,
        }.items():
            if payload.get(field_name) != expected_value:
                raise RuntimeError(
                    "Qdrant payload mismatch for point "
                    f"{point_id}: expected {field_name}={expected_value!r}."
                )

    return len(records)


def _load_embedded_records(artifact_paths: list[Path]) -> list[EmbeddingRecord]:
    records: list[EmbeddingRecord] = []
    for artifact_path in artifact_paths:
        if not artifact_path.is_file():
            raise FileNotFoundError(f"Embedding artifact does not exist: {artifact_path}")

        payload = json.loads(artifact_path.read_text(encoding="utf-8"))
        for record_payload in payload.get("records", []):
            record = EmbeddingRecord(**record_payload)
            if record.embedding_status not in {"embedded", "cached"} or record.vector is None:
                raise RuntimeError(
                    "Embedding artifact contains a record without a usable vector: "
                    f"{artifact_path} ({record.unit_id})"
                )
            records.append(record)

    if not records:
        raise RuntimeError("No usable embedded records were found in the requested artifacts.")

    return records


if __name__ == "__main__":
    args = parse_args()
    main(args.embedding_artifact)
