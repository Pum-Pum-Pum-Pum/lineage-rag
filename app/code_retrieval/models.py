from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class FrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class CodeCandidateSummary(FrozenModel):
    unit_id: str
    point_id: str
    score: float
    source_path: str
    display_name: str
    parent_unit_id: str | None = None
    start_line: int = Field(gt=0)
    end_line: int = Field(gt=0)


class CodeEvidence(FrozenModel):
    unit_id: str
    point_id: str
    score: float
    retrieval_method: Literal["dense", "lexical", "hybrid"]
    snapshot_id: str
    module_id: str
    source_path: str
    source_kind: str
    display_name: str
    parent_unit_id: str | None = None
    package_name: str | None = None
    start_line: int = Field(gt=0)
    end_line: int = Field(gt=0)
    parser_state: str
    conditional_state: str
    text: str
    retrieval_metadata: dict[str, Any] = Field(default_factory=dict)


class CodeRetrievalResult(FrozenModel):
    query: str
    mode: Literal["dense", "lexical", "hybrid"]
    snapshot_id: str
    artifact_identity_sha256: str
    collection_name: str | None = None
    evidence: tuple[CodeEvidence, ...]
    dense_candidates: tuple[CodeCandidateSummary, ...] = ()
    lexical_candidates: tuple[CodeCandidateSummary, ...] = ()
