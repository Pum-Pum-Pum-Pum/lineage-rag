"""Isolated retrieval and answer contracts for the custom-code lane."""

from app.code_retrieval.answer_contract import (
    CodeAnswerResponse,
    CodeCitation,
    CodeUnknownBoundary,
    finalize_code_answer,
)
from app.code_retrieval.models import CodeEvidence, CodeRetrievalResult
from app.code_retrieval.service import retrieve_code_evidence

__all__ = [
    "CodeAnswerResponse",
    "CodeCitation",
    "CodeEvidence",
    "CodeRetrievalResult",
    "CodeUnknownBoundary",
    "finalize_code_answer",
    "retrieve_code_evidence",
]
