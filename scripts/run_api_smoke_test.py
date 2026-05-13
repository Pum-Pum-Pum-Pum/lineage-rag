from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import httpx

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.core.config import get_settings
from app.core.logging import configure_logging, get_logger


@dataclass(frozen=True)
class ApiSmokeResult:
    """Structured result for one API smoke-test run."""

    health_payload: dict[str, Any]
    query_payload: dict[str, Any] | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Smoke test the local FastAPI RAG backend over HTTP."
    )
    parser.add_argument(
        "--base-url",
        default="http://127.0.0.1:8000",
        help="Base URL for the running FastAPI backend.",
    )
    parser.add_argument(
        "--query",
        default=None,
        help="Optional question to send to POST /query. If omitted, only /health is checked.",
    )
    parser.add_argument("--limit", type=int, default=5, help="Number of retrieval results to request.")
    parser.add_argument("--document-family", default=None, help="Optional document_family filter.")
    parser.add_argument("--release-label", default=None, help="Optional release_label filter.")
    parser.add_argument("--source-kind", default=None, help="Optional source_kind filter: paragraph or table.")
    parser.add_argument(
        "--min-top-score",
        type=float,
        default=None,
        help="Optional sufficiency threshold override for POST /query.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=10.0,
        help="HTTP timeout in seconds for each API request.",
    )
    return parser.parse_args()


def run_api_smoke_test(
    client: httpx.Client,
    base_url: str,
    query: str | None = None,
    limit: int = 5,
    document_family: str | None = None,
    release_label: str | None = None,
    source_kind: str | None = None,
    min_top_score: float | None = None,
) -> ApiSmokeResult:
    """Call /health and optionally /query on a running FastAPI backend."""

    cleaned_base_url = _normalize_base_url(base_url)

    health_response = client.get(f"{cleaned_base_url}/health")
    health_payload = _extract_success_json(health_response, label="GET /health")

    query_payload = None
    if query is not None:
        query_response = client.post(
            f"{cleaned_base_url}/query",
            json=_build_query_payload(
                query=query,
                limit=limit,
                document_family=document_family,
                release_label=release_label,
                source_kind=source_kind,
                min_top_score=min_top_score,
            ),
        )
        query_payload = _extract_success_json(query_response, label="POST /query")

    return ApiSmokeResult(
        health_payload=health_payload,
        query_payload=query_payload,
    )


def main() -> None:
    args = parse_args()
    settings = get_settings()
    configure_logging(settings.log_level)
    logger = get_logger("api_smoke_test")

    with httpx.Client(timeout=args.timeout) as client:
        result = run_api_smoke_test(
            client=client,
            base_url=args.base_url,
            query=args.query,
            limit=args.limit,
            document_family=args.document_family,
            release_label=args.release_label,
            source_kind=args.source_kind,
            min_top_score=args.min_top_score,
        )

    health = result.health_payload
    logger.info(
        "Health | status=%s | app=%s | environment=%s | retrieval_mode=%s | "
        "qdrant_required=%s",
        health.get("status"),
        health.get("app_name"),
        health.get("environment"),
        health.get("retrieval_mode"),
        health.get("qdrant_required_for_current_mode"),
    )

    if result.query_payload is None:
        logger.info("No query supplied. Skipped POST /query.")
        return

    query_payload = result.query_payload
    sufficiency = query_payload.get("sufficiency") or {}
    logger.info("Query: %s", query_payload.get("query"))
    logger.info("Retrieval mode: %s", query_payload.get("retrieval_mode"))
    logger.info("Trace id: %s", query_payload.get("trace_id"))
    logger.info("Evidence sufficient: %s", sufficiency.get("is_sufficient"))
    logger.info("Sufficiency reason: %s", sufficiency.get("reason"))
    logger.info("Answered: %s", query_payload.get("is_answered"))
    if query_payload.get("refusal_reason"):
        logger.info("Refusal reason: %s", query_payload.get("refusal_reason"))
    logger.info("Answer:\n%s", query_payload.get("answer"))

    for index, citation in enumerate(query_payload.get("citations", []), start=1):
        logger.info(
            "Citation C%s | release=%s | source=%s | score=%s | unit=%s | text=%s",
            index,
            citation.get("release_label"),
            citation.get("source_kind"),
            citation.get("score"),
            citation.get("unit_id"),
            str(citation.get("text_preview", "")).replace("\n", " "),
        )


def _normalize_base_url(base_url: str) -> str:
    cleaned = base_url.strip().rstrip("/")
    if not cleaned:
        raise ValueError("base_url must not be blank")
    return cleaned


def _build_query_payload(
    query: str,
    limit: int,
    document_family: str | None = None,
    release_label: str | None = None,
    source_kind: str | None = None,
    min_top_score: float | None = None,
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "query": query,
        "limit": limit,
    }
    if document_family is not None:
        payload["document_family"] = document_family
    if release_label is not None:
        payload["release_label"] = release_label
    if source_kind is not None:
        payload["source_kind"] = source_kind
    if min_top_score is not None:
        payload["min_top_score"] = min_top_score
    return payload


def _extract_success_json(response: httpx.Response, label: str) -> dict[str, Any]:
    if response.status_code >= 400:
        raise RuntimeError(f"{label} failed with HTTP {response.status_code}")
    payload = response.json()
    if not isinstance(payload, dict):
        raise RuntimeError(f"{label} returned a non-object JSON response")
    return payload


if __name__ == "__main__":
    main()