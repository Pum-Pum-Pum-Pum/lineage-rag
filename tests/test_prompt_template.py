from app.llm.prompt_template import build_evidence_block, build_grounded_prompt
from app.llm.answer_contract import GroundedAnswerRequest, Citation
from app.retrieval.evidence_sufficiency import EvidenceSufficiencyDecision
from app.vectorstore.qdrant_search import QdrantSearchResult


def _result() -> QdrantSearchResult:
    return QdrantSearchResult(
        point_id="point-1",
        score=0.75,
        payload={
            "unit_id": "doc::chunk_1",
            "document_family": "FS_FCIS_14.7.0.0.0$ASNB",
            "release_label": "R24",
            "source_kind": "paragraph",
            "text": "The enhancements scoped in the document relate to multiple Teller reports.",
        },
    )


def test_build_evidence_block_with_citations() -> None:
    block = build_evidence_block(
        [
            Citation(
                unit_id="doc::chunk_1",
                document_family="family",
                release_label="R24",
                source_kind="paragraph",
                score=0.75,
                text_preview="Evidence text",
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
