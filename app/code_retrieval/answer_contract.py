from __future__ import annotations

import re
from typing import Literal, Sequence

from pydantic import BaseModel, ConfigDict, Field

from app.code_retrieval.models import CodeEvidence


CITATION_PATTERN = re.compile(r"\[C(\d+)\]")


class FrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class CodeCitation(FrozenModel):
    citation_id: str
    unit_id: str
    snapshot_id: str
    source_path: str
    display_name: str
    source_kind: str
    start_line: int = Field(gt=0)
    end_line: int = Field(gt=0)
    score: float
    text_preview: str


class CodeUnknownBoundary(FrozenModel):
    kind: Literal[
        "parser_degradation",
        "conditional_unknown",
        "kernel_unavailable",
        "dynamic_sql_unknown",
        "external_schema",
        "missing_snapshot",
    ]
    detail: str
    unit_id: str | None = None


class CodeAnswerResponse(FrozenModel):
    query: str
    analysis_kind: Literal["explanation", "impact_analysis"]
    answer: str
    is_answered: bool
    refusal_reason: str | None = None
    citations: tuple[CodeCitation, ...] = ()
    unknowns: tuple[CodeUnknownBoundary, ...] = ()
    patch_generation_allowed: Literal[False] = False
    impact_limitation: str | None = None


def build_code_citations(
    evidence: Sequence[CodeEvidence], *, preview_characters: int = 240
) -> tuple[CodeCitation, ...]:
    if preview_characters <= 0:
        raise ValueError("preview_characters must be greater than zero")
    citations: list[CodeCitation] = []
    for index, item in enumerate(evidence, start=1):
        if not item.source_path or not item.text.strip():
            raise ValueError("Code citations require source path and original source text")
        if item.end_line < item.start_line:
            raise ValueError("Code citation line range is invalid")
        preview = " ".join(item.text.split())[:preview_characters]
        citations.append(
            CodeCitation(
                citation_id=f"C{index}",
                unit_id=item.unit_id,
                snapshot_id=item.snapshot_id,
                source_path=item.source_path,
                display_name=item.display_name,
                source_kind=item.source_kind,
                start_line=item.start_line,
                end_line=item.end_line,
                score=item.score,
                text_preview=preview,
            )
        )
    return tuple(citations)


def derive_code_unknowns(evidence: Sequence[CodeEvidence]) -> tuple[CodeUnknownBoundary, ...]:
    unknowns: list[CodeUnknownBoundary] = []
    seen: set[tuple[str, str | None]] = set()
    for item in evidence:
        if item.parser_state not in {"full_parse", "complete"}:
            key = ("parser_degradation", item.unit_id)
            if key not in seen:
                unknowns.append(
                    CodeUnknownBoundary(
                        kind="parser_degradation",
                        detail=f"Evidence was produced with parser state {item.parser_state}.",
                        unit_id=item.unit_id,
                    )
                )
                seen.add(key)
        if item.conditional_state in {"conditional_unknown", "unresolved"}:
            key = ("conditional_unknown", item.unit_id)
            if key not in seen:
                unknowns.append(
                    CodeUnknownBoundary(
                        kind="conditional_unknown",
                        detail="The deployed conditional-compilation branch cannot be confirmed.",
                        unit_id=item.unit_id,
                    )
                )
                seen.add(key)
    return tuple(unknowns)


def finalize_code_answer(
    *,
    query: str,
    generated_content: str,
    evidence: Sequence[CodeEvidence],
    analysis_kind: Literal["explanation", "impact_analysis"] = "explanation",
    additional_unknowns: Sequence[CodeUnknownBoundary] = (),
    force_refusal_reason: str | None = None,
) -> CodeAnswerResponse:
    """Validate a future model response without making an LLM call.

    Generated content must begin with ``DECISION: ANSWER`` or
    ``DECISION: REFUSE``. Unsupported, malformed, or patch-producing output
    fails closed into a machine-readable refusal.
    """

    query = query.strip()
    if not query:
        raise ValueError("Code answer query must not be blank")
    citations = build_code_citations(evidence)
    unknowns = derive_code_unknowns(evidence) + tuple(additional_unknowns)
    limitation = (
        "Reported locations are candidates in visible custom code, not proven root causes."
        if analysis_kind == "impact_analysis"
        else None
    )

    if force_refusal_reason:
        return _refusal(query, analysis_kind, force_refusal_reason, citations, unknowns, limitation)
    if not evidence:
        missing = CodeUnknownBoundary(
            kind="missing_snapshot",
            detail="No approved code evidence was retrieved for the request.",
        )
        return _refusal(
            query,
            analysis_kind,
            "no_code_evidence",
            (),
            unknowns + (missing,),
            limitation,
        )

    content = generated_content.strip()
    first_line, _, body = content.partition("\n")
    if first_line not in {"DECISION: ANSWER", "DECISION: REFUSE"}:
        return _refusal(
            query, analysis_kind, "invalid_answer_contract", citations, unknowns, limitation
        )
    if _looks_like_patch(body):
        return _refusal(
            query, analysis_kind, "patch_generation_not_allowed", citations, unknowns, limitation
        )
    if first_line == "DECISION: REFUSE":
        return _refusal(
            query, analysis_kind, "model_refused", citations, unknowns, limitation, body
        )

    referenced = {f"C{number}" for number in CITATION_PATTERN.findall(body)}
    valid = {citation.citation_id for citation in citations}
    if not referenced or not referenced.issubset(valid):
        return _refusal(
            query, analysis_kind, "invalid_or_missing_citation", citations, unknowns, limitation
        )
    selected = tuple(item for item in citations if item.citation_id in referenced)
    return CodeAnswerResponse(
        query=query,
        analysis_kind=analysis_kind,
        answer=body.strip(),
        is_answered=True,
        citations=selected,
        unknowns=unknowns,
        impact_limitation=limitation,
    )


def _refusal(
    query: str,
    analysis_kind: Literal["explanation", "impact_analysis"],
    reason: str,
    citations: Sequence[CodeCitation],
    unknowns: Sequence[CodeUnknownBoundary],
    limitation: str | None,
    body: str = "",
) -> CodeAnswerResponse:
    answer = body.strip() or (
        "I cannot provide a grounded code answer from the currently approved "
        "evidence. Please provide a package, procedure, function, or source path."
    )
    return CodeAnswerResponse(
        query=query,
        analysis_kind=analysis_kind,
        answer=answer,
        is_answered=False,
        refusal_reason=reason,
        citations=tuple(citations),
        unknowns=tuple(unknowns),
        impact_limitation=limitation,
    )


def _looks_like_patch(text: str) -> bool:
    lines = text.splitlines()
    return any(
        line.startswith(("diff --git ", "@@ ", "+++ ", "--- ")) for line in lines
    )
