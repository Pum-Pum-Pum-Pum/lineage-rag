from __future__ import annotations

import json
import uuid
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Literal

from qdrant_client import QdrantClient

from app.code_indexing.contract import load_code_index_artifact
from app.code_retrieval.answer_contract import CodeAnswerResponse
from app.code_retrieval.service import retrieve_code_evidence
from app.fdd_code_lineage.combined_answer import CombinedAnswerResponse
from app.fdd_code_lineage.combined_retrieval import retrieve_combined_evidence
from app.fdd_code_lineage.models import FddCodeLineageArtifact, validate_lineage_artifact
from app.fdd_code_lineage.paid_evaluation import (
    embed_one_query,
    generate_grounded_answer,
)
from app.llm.client import get_llm_client
from app.retrieval.lexical_search import load_retrieval_ready_documents
from app.retrieval.retrieval_config import RetrievalRuntimeConfig
from app.services.query_retrieval import retrieve_planned_query_evidence


@dataclass(frozen=True)
class KnowledgeModeOrchestrationResult:
    mode: Literal["code", "combined"]
    answer: CodeAnswerResponse | CombinedAnswerResponse
    retrieval: object
    embedding_call: dict
    answer_call: dict
    trace_id: str
    trace_output_path: Path


def run_code_or_combined_query(
    *,
    mode: Literal["code", "combined"],
    query: str,
    analysis_kind: Literal["explanation", "impact_analysis"],
    settings,
    retrieval_config: RetrievalRuntimeConfig,
    limit: int,
    correlation_id: str | None,
    conversation_context: str | None = None,
    openai_client=None,
) -> KnowledgeModeOrchestrationResult:
    """Run an explicit code/combined request behind the disabled-by-default gate."""

    if mode not in {"code", "combined"}:
        raise ValueError("Extended orchestration supports only code or combined mode")
    artifact = load_code_index_artifact(settings.code_index_artifact_path)
    client = openai_client or get_llm_client()
    retrieval_query = query
    if conversation_context and conversation_context.strip():
        retrieval_query = (
            f"{query}\n\nConversation context for reference resolution only:\n"
            f"{conversation_context.strip()}"
        )
    vector, embedding_call = embed_one_query(
        client=client,
        model=settings.openai_embedding_model,
        question=retrieval_query,
        expected_dimension=artifact.vector_dimension,
    )
    code_qdrant = QdrantClient(path=str(settings.code_qdrant_local_path))
    fdd_qdrant = None
    try:
        if not code_qdrant.collection_exists(settings.code_qdrant_collection_name):
            raise RuntimeError("Configured code collection is unavailable")
        if mode == "code":
            retrieval = retrieve_code_evidence(
                artifact=artifact,
                query=retrieval_query,
                mode="hybrid",
                limit=limit,
                candidate_limit=max(limit, retrieval_config.hybrid_candidate_limit),
                client=code_qdrant,
                collection_name=settings.code_qdrant_collection_name,
                query_vector=vector,
            )
        else:
            fdd_qdrant = QdrantClient(path=str(settings.qdrant_local_path))
            if not fdd_qdrant.collection_exists(settings.qdrant_collection_name):
                raise RuntimeError("Configured FDD collection is unavailable")
            documents = load_retrieval_ready_documents(settings.processed_dir)
            lineage = FddCodeLineageArtifact.model_validate_json(
                Path(settings.fdd_code_lineage_artifact_path).read_text(encoding="utf-8")
            )
            validate_lineage_artifact(
                lineage,
                fdd_document_ids={item.document_id for item in documents},
                code_artifact=artifact,
                analysis_directory=settings.code_analysis_directory,
            )
            if lineage.status != "reviewed" or lineage.fdd_generation != settings.fdd_generation:
                raise RuntimeError("Configured FDD/code lineage is not reviewed or generation-compatible")
            planned = retrieve_planned_query_evidence(
                qdrant_client=fdd_qdrant,
                collection_name=settings.qdrant_collection_name,
                query_text=retrieval_query,
                embedding_model=settings.openai_embedding_model,
                query_vector=vector,
                retrieval_config=retrieval_config,
                lexical_artifact_directory=settings.processed_dir,
                limit=limit,
            )
            retrieval = retrieve_combined_evidence(
                query=retrieval_query,
                fdd_results=planned.results,
                fdd_generation=settings.fdd_generation,
                known_fdd_document_ids={item.document_id for item in documents},
                code_artifact=artifact,
                lineage_artifact=lineage,
                analysis_directory=settings.code_analysis_directory,
                code_mode="hybrid",
                code_limit=limit,
                code_candidate_limit=max(limit, retrieval_config.hybrid_candidate_limit),
                client=code_qdrant,
                collection_name=settings.code_qdrant_collection_name,
                query_vector=vector,
            )
        case = SimpleNamespace(
            mode=mode,
            question=query,
            analysis_kind=analysis_kind,
            expected_unknown_kinds=(),
        )
        answer, answer_call = generate_grounded_answer(
            client=client,
            model=settings.openai_chat_model,
            case=case,
            retrieval=retrieval,
            conversation_context=conversation_context,
        )
    finally:
        code_qdrant.close()
        if fdd_qdrant is not None:
            fdd_qdrant.close()

    trace_id = correlation_id or str(uuid.uuid4())
    trace_path = Path(settings.exports_dir) / "answer_runs" / f"{trace_id}.json"
    trace = {
        "schema_version": "knowledge_mode_answer_trace_v1",
        "trace_id": trace_id,
        "knowledge_mode": mode,
        "query": query,
        "conversation_context_used": bool(
            conversation_context and conversation_context.strip()
        ),
        "embedding_call": embedding_call,
        "retrieval": retrieval.model_dump(mode="json"),
        "answer_call": answer_call,
        "answer": answer.model_dump(mode="json"),
    }
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    if trace_path.exists():
        raise FileExistsError(f"Refusing to overwrite answer trace: {trace_path}")
    trace_path.write_text(json.dumps(trace, indent=2, ensure_ascii=False), encoding="utf-8")
    return KnowledgeModeOrchestrationResult(
        mode=mode,
        answer=answer,
        retrieval=retrieval,
        embedding_call=embedding_call,
        answer_call=answer_call,
        trace_id=trace_id,
        trace_output_path=trace_path,
    )
