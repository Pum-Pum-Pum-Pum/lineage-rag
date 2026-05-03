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
from app.services.answer_generation import generate_grounded_answer
from app.services.answer_trace import build_answer_trace, write_answer_trace
from app.vectorstore.qdrant_schema import create_persistent_qdrant_client


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a tiny grounded answer-generation smoke test."
    )
    parser.add_argument("--query", required=True, help="Question to answer.")
    parser.add_argument("--limit", type=int, default=5, help="Number of retrieval results to use.")
    parser.add_argument("--document-family", default=None, help="Optional document_family filter.")
    parser.add_argument("--release-label", default=None, help="Optional release_label filter.")
    parser.add_argument("--source-kind", default=None, help="Optional source_kind filter.")
    parser.add_argument(
        "--min-top-score",
        type=float,
        default=None,
        help="Optional sufficiency threshold override. Defaults to config value.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    settings = get_settings()
    configure_logging(settings.log_level)
    logger = get_logger("answer_smoke_test")

    min_top_score = args.min_top_score if args.min_top_score is not None else settings.retrieval_min_top_score

    client = create_persistent_qdrant_client(settings.qdrant_local_path)
    if not client.collection_exists(settings.qdrant_collection_name):
        client.close()
        raise RuntimeError("Qdrant collection does not exist. Run scripts/run_qdrant_indexing.py first.")

    retrieved_results = search_query_text(
        qdrant_client=client,
        collection_name=settings.qdrant_collection_name,
        query_text=args.query,
        embedding_model=settings.openai_embedding_model,
        limit=args.limit,
        document_family=args.document_family,
        release_label=args.release_label,
        source_kind=args.source_kind,
    )
    client.close()

    sufficiency = assess_evidence_sufficiency(
        retrieved_results,
        min_results=1,
        min_top_score=min_top_score,
    )

    response = generate_grounded_answer(
        query=args.query,
        retrieved_results=retrieved_results,
        sufficiency=sufficiency,
    )
    trace = build_answer_trace(
        query=args.query,
        filters={
            "document_family": args.document_family,
            "release_label": args.release_label,
            "source_kind": args.source_kind,
        },
        sufficiency=sufficiency,
        answer_response=response,
        retrieval_results=retrieved_results,
    )
    trace_output = write_answer_trace(trace, settings.exports_dir / "answer_runs")

    logger.info("Query: %s", response.query)
    logger.info("Request id: %s", trace.request_id)
    logger.info("Wrote answer trace: %s", trace_output)
    logger.info("Evidence sufficient: %s", sufficiency.is_sufficient)
    logger.info("Sufficiency reason: %s", sufficiency.reason)
    logger.info("Answered: %s", response.is_answered)
    if response.refusal_reason:
        logger.info("Refusal reason: %s", response.refusal_reason)

    logger.info("Answer:\n%s", response.answer)

    if response.usage is not None:
        logger.info(
            "LLM usage | model=%s | prompt_tokens=%s | completion_tokens=%s | total_tokens=%s",
            response.usage.model,
            response.usage.prompt_tokens,
            response.usage.completion_tokens,
            response.usage.total_tokens,
        )

    if response.cost is not None:
        logger.info(
            "LLM estimated cost | model=%s | input_cost=%s | output_cost=%s | total_cost=%s | currency=%s",
            response.cost.model,
            response.cost.input_cost,
            response.cost.output_cost,
            response.cost.total_cost,
            response.cost.currency,
        )

    for index, citation in enumerate(response.citations, start=1):
        logger.info(
            "Citation C%s | release=%s | source=%s | score=%.4f | unit=%s | text=%s",
            index,
            citation.release_label,
            citation.source_kind,
            citation.score,
            citation.unit_id,
            citation.text_preview.replace("\n", " "),
        )


if __name__ == "__main__":
    main()
