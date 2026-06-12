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
from app.retrieval.retrieval_config import build_retrieval_runtime_config
from app.services.query_retrieval import retrieve_query_evidence
from app.vectorstore.qdrant_schema import create_persistent_qdrant_client


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Search using the configured retrieval mode over local RAG artifacts."
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
    retrieval_config = build_retrieval_runtime_config(settings)

    client = None
    try:
        if _requires_qdrant_collection(retrieval_config.retrieval_mode):
            client = create_persistent_qdrant_client(settings.qdrant_local_path)
            if not client.collection_exists(settings.qdrant_collection_name):
                raise RuntimeError(
                    "Qdrant collection does not exist. Run scripts/run_qdrant_indexing.py first."
                )

        routed = retrieve_query_evidence(
            qdrant_client=client,
            collection_name=settings.qdrant_collection_name,
            query_text=args.query,
            embedding_model=settings.openai_embedding_model,
            retrieval_config=retrieval_config,
            lexical_artifact_directory=settings.processed_dir,
            limit=args.limit,
            document_family=args.document_family,
            release_label=args.release_label,
            source_kind=args.source_kind,
        )
        results = routed.results

        logger.info("Query: %s", args.query)
        logger.info(
            "Retrieval config | mode=%s | hybrid_dense_weight=%s | hybrid_lexical_weight=%s | "
            "hybrid_candidate_limit=%s | limit=%s | document_family=%s | release_label=%s | source_kind=%s",
            routed.retrieval_mode,
            retrieval_config.hybrid_dense_weight,
            retrieval_config.hybrid_lexical_weight,
            retrieval_config.hybrid_candidate_limit,
            args.limit,
            args.document_family,
            args.release_label,
            args.source_kind,
        )
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
                "Rank %s | score=%.4f | method=%s | source=%s | family=%s | release=%s | unit=%s | "
                "dense_score=%s | lexical_score=%s | text=%s",
                index,
                result.score,
                payload.get("retrieval_method", routed.retrieval_mode),
                payload.get("source_kind"),
                payload.get("document_family"),
                payload.get("release_label"),
                payload.get("unit_id"),
                payload.get("dense_score"),
                payload.get("lexical_score"),
                preview,
            )
    finally:
        if client is not None:
            client.close()


def _requires_qdrant_collection(retrieval_mode: str) -> bool:
    """Return whether the selected retrieval mode requires a Qdrant collection."""

    return retrieval_mode in {"dense", "hybrid"}


if __name__ == "__main__":
    main()
