from __future__ import annotations

import re
from typing import Literal

from pydantic import BaseModel, ConfigDict

from app.code_retrieval.answer_contract import CodeCitation, build_code_citations
from app.fdd_code_lineage.combined_retrieval import CombinedRetrievalResult


FDD_CITATION_PATTERN = re.compile(r"\[F(\d+)\]")
CODE_CITATION_PATTERN = re.compile(r"\[C(\d+)\]")


class FrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class CombinedSectionDraft(FrozenModel):
    status: Literal["answered", "refused"]
    text: str


class CombinedAnswerDraft(FrozenModel):
    documented_functionality: CombinedSectionDraft
    visible_custom_implementation: CombinedSectionDraft
    impact_and_likely_change_locations: CombinedSectionDraft
    unknown_or_unavailable_behavior: CombinedSectionDraft


class FddCitation(FrozenModel):
    citation_id: str
    unit_id: str
    document_id: str
    document_family: str
    release_label: str
    source_kind: str
    score: float
    text_preview: str


class CombinedSectionResponse(FrozenModel):
    status: Literal["answered", "refused"]
    text: str
    refusal_reason: str | None = None


class CombinedAnswerResponse(FrozenModel):
    query: str
    documented_functionality: CombinedSectionResponse
    visible_custom_implementation: CombinedSectionResponse
    impact_and_likely_change_locations: CombinedSectionResponse
    unknown_or_unavailable_behavior: CombinedSectionResponse
    fdd_citations: tuple[FddCitation, ...] = ()
    code_citations: tuple[CodeCitation, ...] = ()
    reviewed_mapping_ids: tuple[str, ...] = ()
    patch_generation_allowed: Literal[False] = False


def finalize_combined_answer(
    *,
    retrieval: CombinedRetrievalResult,
    draft: CombinedAnswerDraft,
) -> CombinedAnswerResponse:
    """Validate four separately grounded combined-answer sections.

    FDD claims may cite only ``[F#]`` and implementation/impact claims may cite
    only ``[C#]``. Invalid sections fail independently instead of converting a
    valid lane into an unsupported combined claim.
    """

    fdd_citations = tuple(
        FddCitation(
            citation_id=f"F{index}",
            unit_id=item.unit_id,
            document_id=item.document_id,
            document_family=item.document_family,
            release_label=item.release_label,
            source_kind=item.source_kind,
            score=item.score,
            text_preview=" ".join(item.text.split())[:240],
        )
        for index, item in enumerate(retrieval.fdd_evidence, start=1)
    )
    code_citations = build_code_citations(retrieval.code_evidence)
    documented = _validate_section(
        draft.documented_functionality,
        required_lane="fdd",
        valid_fdd={item.citation_id for item in fdd_citations},
        valid_code={item.citation_id for item in code_citations},
    )
    implementation = _validate_section(
        draft.visible_custom_implementation,
        required_lane="code",
        valid_fdd={item.citation_id for item in fdd_citations},
        valid_code={item.citation_id for item in code_citations},
    )
    impact = _validate_section(
        draft.impact_and_likely_change_locations,
        required_lane="code",
        valid_fdd={item.citation_id for item in fdd_citations},
        valid_code={item.citation_id for item in code_citations},
        reject_patch=True,
    )
    unknown_text = draft.unknown_or_unavailable_behavior.text.strip()
    if retrieval.unknowns:
        suffix = "\n".join(f"- {item}" for item in retrieval.unknowns)
        unknown_text = f"{unknown_text}\n{suffix}".strip()
    unknown = CombinedSectionResponse(
        status=draft.unknown_or_unavailable_behavior.status,
        text=unknown_text or "No additional unknown boundary was recorded.",
    )
    referenced_fdd = _references(
        "\n".join(
            item.text
            for item in (documented, implementation, impact)
            if item.status == "answered"
        ),
        FDD_CITATION_PATTERN,
        "F",
    )
    referenced_code = _references(
        "\n".join(
            item.text
            for item in (documented, implementation, impact)
            if item.status == "answered"
        ),
        CODE_CITATION_PATTERN,
        "C",
    )
    return CombinedAnswerResponse(
        query=retrieval.query,
        documented_functionality=documented,
        visible_custom_implementation=implementation,
        impact_and_likely_change_locations=impact,
        unknown_or_unavailable_behavior=unknown,
        fdd_citations=tuple(
            item for item in fdd_citations if item.citation_id in referenced_fdd
        ),
        code_citations=tuple(
            item for item in code_citations if item.citation_id in referenced_code
        ),
        reviewed_mapping_ids=tuple(
            item.mapping_id for item in retrieval.reviewed_lineage
        ),
    )


def _validate_section(
    draft: CombinedSectionDraft,
    *,
    required_lane: Literal["fdd", "code"],
    valid_fdd: set[str],
    valid_code: set[str],
    reject_patch: bool = False,
) -> CombinedSectionResponse:
    text = draft.text.strip()
    if draft.status == "refused":
        return CombinedSectionResponse(status="refused", text=text or "Insufficient evidence.")
    fdd_refs = _references(text, FDD_CITATION_PATTERN, "F")
    code_refs = _references(text, CODE_CITATION_PATTERN, "C")
    invalid = not fdd_refs.issubset(valid_fdd) or not code_refs.issubset(valid_code)
    cross_lane = (required_lane == "fdd" and bool(code_refs)) or (
        required_lane == "code" and bool(fdd_refs)
    )
    missing = not (fdd_refs if required_lane == "fdd" else code_refs)
    if invalid or cross_lane or missing:
        return CombinedSectionResponse(
            status="refused",
            text="This section could not be grounded in its required evidence lane.",
            refusal_reason="invalid_or_cross_lane_citation",
        )
    if reject_patch and _looks_like_patch(text):
        return CombinedSectionResponse(
            status="refused",
            text="Patch generation is outside the approved Phase 2 scope.",
            refusal_reason="patch_generation_not_allowed",
        )
    return CombinedSectionResponse(status="answered", text=text)


def _references(text: str, pattern: re.Pattern[str], prefix: str) -> set[str]:
    return {f"{prefix}{number}" for number in pattern.findall(text)}


def _looks_like_patch(text: str) -> bool:
    return any(
        line.startswith(("diff --git ", "@@ ", "+++ ", "--- "))
        for line in text.splitlines()
    )
