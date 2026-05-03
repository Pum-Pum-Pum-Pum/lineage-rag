from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from app.llm.answer_contract import GroundedAnswerResponse
from app.retrieval.evidence_sufficiency import EvidenceSufficiencyDecision
from app.vectorstore.qdrant_search import QdrantSearchResult


@dataclass(frozen=True)
class AnswerTrace:
    request_id: str
    created_at_utc: str
    query: str
    filters: dict[str, str | None]
    sufficiency: EvidenceSufficiencyDecision
    answer_response: GroundedAnswerResponse
    retrieval_results: list[dict]


def build_answer_trace(
    query: str,
    filters: dict[str, str | None],
    sufficiency: EvidenceSufficiencyDecision,
    answer_response: GroundedAnswerResponse,
    retrieval_results: list[QdrantSearchResult],
    request_id: str | None = None,
) -> AnswerTrace:
    """Build a trace object for one answer-generation run."""

    return AnswerTrace(
        request_id=request_id or str(uuid4()),
        created_at_utc=datetime.now(UTC).isoformat(),
        query=query,
        filters=filters,
        sufficiency=sufficiency,
        answer_response=answer_response,
        retrieval_results=[
            {
                "point_id": result.point_id,
                "score": result.score,
                "payload": result.payload,
            }
            for result in retrieval_results
        ],
    )


def write_answer_trace(
    trace: AnswerTrace,
    output_directory: str | Path,
) -> Path:
    """Persist an answer trace as a local JSON artifact."""

    output_dir = Path(output_directory)
    output_dir.mkdir(parents=True, exist_ok=True)

    output_file = output_dir / f"{trace.request_id}.json"
    output_file.write_text(
        json.dumps(asdict(trace), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    return output_file
