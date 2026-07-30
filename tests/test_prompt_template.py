import pytest

from app.llm.prompt_template import (
    EvidenceBudgetExceededError,
    PromptEvidence,
    build_evidence_block,
    build_grounded_prompt,
    select_prompt_evidence,
)
from app.llm.answer_contract import GroundedAnswerRequest
from app.retrieval.evidence_sufficiency import EvidenceSufficiencyDecision
from app.vectorstore.qdrant_search import QdrantSearchResult


def _result(
    *,
    point_id: str = "point-1",
    text: str = (
        "The enhancements scoped in the document relate to multiple "
        "Teller reports."
    ),
) -> QdrantSearchResult:
    return QdrantSearchResult(
        point_id=point_id,
        score=0.75,
        payload={
            "unit_id": f"doc::{point_id}",
            "document_family": "FS_FCIS_14.7.0.0.0$ASNB",
            "release_label": "R24",
            "source_kind": "paragraph",
            "text": text,
        },
    )


def test_build_evidence_block_with_citations() -> None:
    block = build_evidence_block(
        [
            PromptEvidence(
                citation_id="C1",
                unit_id="doc::chunk_1",
                document_family="family",
                release_label="R24",
                source_kind="paragraph",
                score=0.75,
                text="Evidence text",
            )
        ]
    )

    assert "[C1]" in block
    assert "release_label: R24" in block
    assert "Evidence text" in block


def test_build_evidence_block_without_citations() -> None:
    assert build_evidence_block([]) == "No evidence was provided."


def test_build_grounded_prompt_contains_rules_query_and_evidence() -> None:
    request = GroundedAnswerRequest(
        query="What changed in branch reports?",
        retrieved_results=[_result()],
        sufficiency=EvidenceSufficiencyDecision(
            is_sufficient=True,
            reason="Retrieved evidence passed baseline sufficiency checks.",
            result_count=1,
            top_score=0.75,
        ),
    )

    prompt = build_grounded_prompt(request)

    assert "Answer only using the provided evidence" in prompt.system_prompt
    assert "What changed in branch reports?" in prompt.user_prompt
    assert "[C1]" in prompt.user_prompt
    assert "multiple Teller reports" in prompt.user_prompt
    assert len(prompt.citations) == 1


def test_grounded_prompt_uses_full_evidence_not_240_character_preview() -> None:
    acronym_text = (
        "Abbreviation table header\n"
        + ("metadata filler " * 20)
        + "\nBOR | Branch Online Reports"
    )
    realignment_text = (
        "Branch reports realignment\n"
        + ("removed report details " * 20)
        + "\nB-04 PNB Group Employee Transaction Report"
    )
    assert len(acronym_text) > 240
    assert len(realignment_text) > 240
    request = GroundedAnswerRequest(
        query="How many BOR reports remain after R24?",
        retrieved_results=[
            _result(point_id="acronyms", text=acronym_text),
            _result(point_id="realignment", text=realignment_text),
        ],
        sufficiency=EvidenceSufficiencyDecision(
            is_sufficient=True,
            reason="Evidence passed.",
            result_count=2,
            top_score=0.75,
        ),
    )

    prompt = build_grounded_prompt(request)

    assert "BOR | Branch Online Reports" in prompt.user_prompt
    assert "B-04 PNB Group Employee Transaction Report" in prompt.user_prompt
    assert len(prompt.citations[0].text_preview) == 240
    assert "BOR | Branch Online Reports" not in prompt.citations[0].text_preview


def test_prompt_evidence_budget_keeps_whole_ranked_units() -> None:
    results = [
        _result(point_id="first", text="First evidence"),
        _result(point_id="second", text="Second evidence"),
    ]

    selected = select_prompt_evidence(
        results,
        max_evidence_tokens=15,
        count_tokens=lambda text: text.count("unit_id:") * 10,
    )

    assert [item.unit_id for item in selected] == ["doc::first"]
    assert selected[0].text == "First evidence"


def test_prompt_evidence_budget_rejects_oversized_top_unit() -> None:
    with pytest.raises(
        EvidenceBudgetExceededError,
        match="highest-ranked evidence unit",
    ):
        select_prompt_evidence(
            [_result(text="Oversized evidence")],
            max_evidence_tokens=5,
            count_tokens=lambda text: 10,
        )


def test_build_grounded_prompt_marks_conversation_memory_as_non_evidence() -> None:
    request = GroundedAnswerRequest(
        query="What about that release?",
        retrieved_results=[_result()],
        sufficiency=EvidenceSufficiencyDecision(
            is_sufficient=True,
            reason="Retrieved evidence passed baseline sufficiency checks.",
            result_count=1,
            top_score=0.75,
        ),
        conversation_context="<conversation_memory>R24</conversation_memory>",
    )

    prompt = build_grounded_prompt(request)

    assert "<conversation_memory>R24</conversation_memory>" in prompt.user_prompt
    assert "context only; not documentary evidence" in prompt.user_prompt
    assert "using only the evidence above" in prompt.user_prompt


def test_current_state_prompt_treats_latest_release_baseline_as_historical() -> None:
    request = GroundedAnswerRequest(
        query="How many teller and branch reports are there currently?",
        retrieved_results=[_result()],
        sufficiency=EvidenceSufficiencyDecision(
            is_sufficient=True,
            reason="Evidence passed.",
            result_count=1,
            top_score=0.75,
        ),
        current_state_requested=True,
        effective_release_label="R24",
    )

    prompt = build_grounded_prompt(request)

    assert "current_state_requested: true" in prompt.user_prompt
    assert "effective_release_label: R24" in prompt.user_prompt
    assert "Existing Functionality" in prompt.system_prompt
    assert "pre-change baseline" in prompt.system_prompt
    assert "Do not count removed items" in prompt.system_prompt
