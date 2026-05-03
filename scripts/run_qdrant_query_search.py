from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.core.config import get_settings
from app.core.logging import configure_logging, get_logger
from app.retrieval.evidence_sufficiency import assess_evidence_sufficiency
from app.retrieval.query_search import search_query_text
from app.vectorstore.qdrant_schema import create_persistent_qdrant_client


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Embed a query and search persistent local Qdrant."
    )
    parser.add_argument("--query", required=True, help="Query text to search for.")
    parser.add_argument("--limit", type=int, default=5, help="Number of results to return.")
    parser.add_argument("--document-family", default=None, help="Optional document_family filter.")
    parser.add_argument("--release-label", default=None, help="Optional release_label filter, e.g. R24.")
    parser.add_argument("--source-kind", default=None, help="Optional source_kind filter: paragraph or table.")
    parser.add_argument(
        "--min-top-score",
        type=float,
        default=None,
        help="Minimum top score required for baseline evidence sufficiency. Defaults to config value.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = get_settings()
    configure_logging(settings.log_level)
    logger = get_logger("qdrant_query_search")

    client = create_persistent_qdrant_client(settings.qdrant_local_path)
    if not client.collection_exists(settings.qdrant_collection_name):
        client.close()
        raise RuntimeError(
            "Qdrant collection does not exist. Run scripts/run_qdrant_indexing.py first."
        )

    results = search_query_text(
        qdrant_client=client,
        collection_name=settings.qdrant_collection_name,
        query_text=args.query,
        embedding_model=settings.openai_embedding_model,
        limit=args.limit,
        document_family=args.document_family,
        release_label=args.release_label,
        source_kind=args.source_kind,
    )

    logger.info("Query: %s", args.query)
    logger.info("Results returned: %s", len(results))

    min_top_score = args.min_top_score if args.min_top_score is not None else settings.retrieval_min_top_score
    sufficiency = assess_evidence_sufficiency(
        results,
        min_results=1,
        min_top_score=min_top_score,
    )
    logger.info(
        "Evidence sufficiency | sufficient=%s | reason=%s | top_score=%s | min_top_score=%s",
        sufficiency.is_sufficient,
        sufficiency.reason,
        sufficiency.top_score,
        min_top_score,
    )

    for index, result in enumerate(results, start=1):
        payload = result.payload
        text = str(payload.get("text", ""))
        preview = text[:300].replace("\n", " ")
        logger.info(
            "Rank %s | score=%.4f | source=%s | family=%s | release=%s | unit=%s | text=%s",
            index,
            result.score,
            payload.get("source_kind"),
            payload.get("document_family"),
            payload.get("release_label"),
            payload.get("unit_id"),
            preview,
        )

    client.close()


if __name__ == "__main__":
    main()
