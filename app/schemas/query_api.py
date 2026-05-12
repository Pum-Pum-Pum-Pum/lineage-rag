from __future__ import annotations

from pydantic import BaseModel, Field, field_validator


class QueryRequest(BaseModel):
    """Request contract for the grounded answer query endpoint."""

    query: str = Field(..., min_length=1)
    limit: int = Field(default=5, gt=0)
    document_family: str | None = None
    release_label: str | None = None
    source_kind: str | None = None
    min_top_score: float | None = Field(default=None, ge=0)

    @field_validator("query")
    @classmethod
    def query_must_not_be_blank(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("query must not be blank")
        return cleaned

    @field_validator("source_kind")
    @classmethod
    def source_kind_must_be_supported(cls, value: str | None) -> str | None:
        if value is None:
            return None
        cleaned = value.strip().lower()
        if cleaned not in {"paragraph", "table"}:
            raise ValueError("source_kind must be either 'paragraph' or 'table'")
        return cleaned


class CitationResponse(BaseModel):
    unit_id: str
    document_family: str | None
    release_label: str | None
    source_kind: str | None
    score: float
    text_preview: str


class EvidenceSufficiencyResponse(BaseModel):
    is_sufficient: bool
    reason: str
    result_count: int
    top_score: float | None


class LLMUsageResponse(BaseModel):
    model: str
    prompt_tokens: int | None
    completion_tokens: int | None
    total_tokens: int | None


class LLMCostResponse(BaseModel):
    model: str
    input_cost: float | None
    output_cost: float | None
    total_cost: float | None
    currency: str


class QueryResponse(BaseModel):
    query: str
    answer: str
    is_answered: bool
    refusal_reason: str | None
    retrieval_mode: str
    citations: list[CitationResponse]
    sufficiency: EvidenceSufficiencyResponse
    trace_id: str
    trace_output_path: str
    retrieval_metadata: dict | None = None
    usage: LLMUsageResponse | None = None
    cost: LLMCostResponse | None = None