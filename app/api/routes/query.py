from __future__ import annotations

from fastapi import APIRouter, HTTPException, status

from app.core.config import get_settings
from app.core.logging import get_logger
from app.retrieval.retrieval_config import build_retrieval_runtime_config
from app.schemas.query_api import (
    CitationResponse,
    EvidenceSufficiencyResponse,
    LLMCostResponse,
    LLMUsageResponse,
    QueryRequest,
    QueryResponse,
)
from app.services.answer_orchestration import AnswerOrchestrationResult, run_grounded_answer_query
from app.vectorstore.qdrant_schema import create_persistent_qdrant_client


router = APIRouter(tags=["query"])
logger = get_logger("query_api")


@router.post("/query", response_model=QueryResponse)
def query_answer(request: QueryRequest) -> QueryResponse:
    """Run a grounded answer query through the shared orchestration service."""

    settings = get_settings()
    retrieval_config = build_retrieval_runtime_config(settings)
    min_top_score = (
        request.min_top_score
        if request.min_top_score is not None
        else settings.retrieval_min_top_score
    )

    client = None
    try:
        client = create_persistent_qdrant_client(settings.qdrant_local_path)

        if _requires_qdrant_collection(retrieval_config.retrieval_mode) and not client.collection_exists(
            settings.qdrant_collection_name
        ):
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Qdrant collection does not exist. Run indexing before querying.",
            )

        orchestration_result = run_grounded_answer_query(
            qdrant_client=client,
            collection_name=settings.qdrant_collection_name,
            query_text=request.query,
            embedding_model=settings.openai_embedding_model,
            retrieval_config=retrieval_config,
            lexical_artifact_directory=settings.processed_dir,
            trace_output_directory=settings.exports_dir / "answer_runs",
            limit=request.limit,
            min_results=1,
            min_top_score=min_top_score,
            document_family=request.document_family,
            release_label=request.release_label,
            source_kind=request.source_kind,
        )
        return _build_query_response(orchestration_result)
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        logger.exception("Unexpected query API failure")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal query processing error.",
        ) from exc
    finally:
        if client is not None:
            client.close()


def _build_query_response(result: AnswerOrchestrationResult) -> QueryResponse:
    answer = result.answer_response
    sufficiency = result.sufficiency

    usage = None
    if answer.usage is not None:
        usage = LLMUsageResponse(
            model=answer.usage.model,
            prompt_tokens=answer.usage.prompt_tokens,
            completion_tokens=answer.usage.completion_tokens,
            total_tokens=answer.usage.total_tokens,
        )

    cost = None
    if answer.cost is not None:
        cost = LLMCostResponse(
            model=answer.cost.model,
            input_cost=answer.cost.input_cost,
            output_cost=answer.cost.output_cost,
            total_cost=answer.cost.total_cost,
            currency=answer.cost.currency,
        )

    return QueryResponse(
        query=answer.query,
        answer=answer.answer,
        is_answered=answer.is_answered,
        refusal_reason=answer.refusal_reason,
        retrieval_mode=result.retrieval_mode,
        citations=[
            CitationResponse(
                unit_id=citation.unit_id,
                document_family=citation.document_family,
                release_label=citation.release_label,
                source_kind=citation.source_kind,
                score=citation.score,
                text_preview=citation.text_preview,
            )
            for citation in answer.citations
        ],
        sufficiency=EvidenceSufficiencyResponse(
            is_sufficient=sufficiency.is_sufficient,
            reason=sufficiency.reason,
            result_count=sufficiency.result_count,
            top_score=sufficiency.top_score,
        ),
        trace_id=result.trace.request_id,
        trace_output_path=str(result.trace_output_path),
        retrieval_metadata=result.trace.retrieval_metadata,
        usage=usage,
        cost=cost,
    )


def _requires_qdrant_collection(retrieval_mode: str) -> bool:
    """Return whether the selected retrieval mode requires a Qdrant collection."""

    return retrieval_mode in {"dense", "hybrid"}