from __future__ import annotations

from fastapi import APIRouter, HTTPException, status

from app.core.config import get_settings
from app.core.logging import get_logger
from app.core.request_observability import get_request_id
from app.retrieval.retrieval_config import build_retrieval_runtime_config
from app.schemas.query_api import (
    CitationResponse,
    CodeCitationResponse,
    CombinedSectionApiResponse,
    EvidenceSufficiencyResponse,
    LLMCostResponse,
    LLMUsageResponse,
    QueryRequest,
    QueryResponse,
)
from app.services.answer_orchestration import AnswerOrchestrationResult, run_grounded_answer_query
from app.services.knowledge_mode_orchestration import run_code_or_combined_query
from app.vectorstore.qdrant_schema import create_persistent_qdrant_client


router = APIRouter(tags=["query"])
logger = get_logger("query_api")


@router.post("/query", response_model=QueryResponse)
def query_answer(request: QueryRequest) -> QueryResponse:
    """Run a grounded answer query through the shared orchestration service."""

    return execute_query_request(request)


def execute_query_request(
    request: QueryRequest,
    *,
    conversation_context: str | None = None,
) -> QueryResponse:
    """Execute the query contract for single-turn or conversation callers."""

    settings = get_settings()
    retrieval_config = build_retrieval_runtime_config(settings)
    if request.knowledge_mode != "fdd":
        if not bool(getattr(settings, "code_modes_enabled", False)):
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Code and combined knowledge modes are not activated.",
            )
        try:
            result = run_code_or_combined_query(
                mode=request.knowledge_mode,
                query=request.query,
                analysis_kind=request.analysis_kind,
                settings=settings,
                retrieval_config=retrieval_config,
                limit=request.limit,
                correlation_id=get_request_id(),
                conversation_context=conversation_context,
            )
            return _build_extended_query_response(result)
        except HTTPException:
            raise
        except ValueError as exc:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc
        except Exception as exc:
            logger.exception("Code/combined query processing failed")
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="Internal code/combined query processing error.",
            ) from exc
    qdrant_required = _requires_qdrant_collection(retrieval_config.retrieval_mode)
    min_top_score = (
        request.min_top_score
        if request.min_top_score is not None
        else settings.retrieval_min_top_score
    )

    client = None
    try:
        if qdrant_required:
            try:
                client = create_persistent_qdrant_client(settings.qdrant_local_path)

                if not client.collection_exists(settings.qdrant_collection_name):
                    raise HTTPException(
                        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                        detail="Qdrant collection does not exist. Run indexing before querying.",
                    )
            except HTTPException:
                raise
            except Exception as exc:
                logger.exception("Qdrant dependency check failed during query")
                raise HTTPException(
                    status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                    detail="Qdrant dependency check failed. Verify vector-store availability before querying.",
                ) from exc

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
            conversation_context=conversation_context,
            correlation_id=get_request_id(),
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


def _build_extended_query_response(result) -> QueryResponse:
    answer = result.answer
    answer_call = result.answer_call
    if result.mode == "code":
        code_citations = answer.citations
        text = answer.answer
        is_answered = answer.is_answered
        refusal_reason = answer.refusal_reason
        requested_supported = answer.is_answered
        related = False
        fdd_citations = []
        sections = None
    else:
        code_citations = answer.code_citations
        ordered = {
            "documented_functionality": answer.documented_functionality,
            "visible_custom_implementation": answer.visible_custom_implementation,
            "impact_and_likely_change_locations": answer.impact_and_likely_change_locations,
            "unknown_or_unavailable_behavior": answer.unknown_or_unavailable_behavior,
        }
        text = "\n\n".join(
            f"{name.replace('_', ' ').title()}\n{section.text}"
            for name, section in ordered.items()
        )
        is_answered = answer.requested_claim_supported
        refusal_reason = None if is_answered else "requested_claim_unsupported"
        requested_supported = answer.requested_claim_supported
        related = answer.related_grounded_context_provided
        fdd_citations = [
            CitationResponse(
                unit_id=item.unit_id,
                document_family=item.document_family,
                release_label=item.release_label,
                source_kind=item.source_kind,
                score=item.score,
                text_preview=item.text_preview,
            )
            for item in answer.fdd_citations
        ]
        sections = {
            name: CombinedSectionApiResponse(**section.model_dump())
            for name, section in ordered.items()
        }
    scores = [item.score for item in code_citations] + [item.score for item in fdd_citations]
    return QueryResponse(
        query=answer.query,
        answer=text,
        is_answered=is_answered,
        refusal_reason=refusal_reason,
        retrieval_mode="hybrid",
        citations=fdd_citations,
        sufficiency=EvidenceSufficiencyResponse(
            is_sufficient=is_answered,
            reason=(
                "The requested claim was supported by the selected evidence."
                if is_answered
                else "The requested claim was not directly supported."
            ),
            result_count=len(code_citations) + len(fdd_citations),
            top_score=max(scores) if scores else None,
        ),
        trace_id=result.trace_id,
        trace_output_path=str(result.trace_output_path),
        retrieval_metadata={"knowledge_mode": result.mode},
        usage=LLMUsageResponse(
            model=answer_call["model"],
            prompt_tokens=answer_call["prompt_tokens"],
            completion_tokens=answer_call["completion_tokens"],
            total_tokens=answer_call["total_tokens"],
        ),
        knowledge_mode=result.mode,
        requested_claim_supported=requested_supported,
        related_grounded_context_provided=related,
        code_citations=[
            CodeCitationResponse(**item.model_dump()) for item in code_citations
        ],
        combined_sections=sections,
    )
