from __future__ import annotations

import pytest

from app.fdd_code_lineage.combined_retrieval import (
    FddEvidence,
    _reserve_fdd_topic_diversity_slot,
)


def evidence(document: str, text: str, *, unit: str = "", score: float = 1.0) -> FddEvidence:
    return FddEvidence(
        document_id=document, unit_id=unit or document,
        document_family="specs", release_label="R1", source_kind="paragraph",
        score=score, text=text,
    )


def select(query: str, *items: FddEvidence, limit: int = 2):
    return _reserve_fdd_topic_diversity_slot(query=query, candidates=items, limit=limit)


def test_mixed_case_topic_recovers_document_without_changing_scores_or_bounds():
    first = evidence("FS_APP_R1_Report", "system flow code", score=9)
    second = evidence("FS_APP_R2_Payments", "code behavior", score=8)
    topic = evidence("FS_APP_R3_ClearBridge_Integration", "ClearBridge transactions", score=2)
    unused = evidence("FS_APP_R4_ClearBridge_Additions", "ClearBridge new fields", score=1)
    result = select("How does ClearBridge integrate with APP?", first, second, topic, unused)
    assert result == (first, topic)
    assert result[1] is topic
    assert topic.score == 2


def test_acronym_preserves_additional_document_not_more_chunks_from_same_document():
    original = evidence("FS_APP_R1_KYC", "KYC screening", unit="a")
    generic = evidence("FS_APP_R1_Reports", "change behavior", unit="b")
    duplicate = evidence("FS_APP_R1_KYC", "KYC screening", unit="c")
    extension = evidence("FS_APP_R2_KYC_Extension", "KYC offline screening", unit="d")
    assert select("How is offline KYC handled?", original, generic, duplicate, extension) == (
        original, extension
    )


@pytest.mark.parametrize("query", ["how does screening work?", "How does screening work?", "APP behavior"])
def test_no_explicit_topic_or_shared_application_prefix_does_not_change_selection(query):
    items = (
        evidence("FS_APP_Report", "APP report"),
        evidence("FS_APP_Payment", "APP payment"),
        evidence("FS_APP_Screening", "APP screening"),
    )
    assert select(query, *items) == items[:2]


@pytest.mark.parametrize("document,text", [
    ("FS_KYC", "unrelated behavior"),  # title alone is not evidence
    ("FS_Reports", "KYC behavior"),   # source mention alone is not a title anchor
    ("FS_KYCA", "KYCA behavior"),     # no substring matches
    ("FS_KYC", "KYCA behavior"),
])
def test_requires_whole_topic_token_in_both_identity_and_source(document, text):
    items = (evidence("FS_Reports", "report"), evidence("FS_Payment", "payment"),
             evidence(document, text))
    assert select("How is KYC handled?", *items) == items[:2]


def test_never_replaces_equally_relevant_topic_evidence():
    items = tuple(evidence(f"FS_KYC_R{i}", "KYC behavior") for i in range(3))
    assert select("KYC behavior", *items) == items[:2]


def test_preserves_original_candidate_order_on_affinity_ties():
    # Input may be dense/hybrid ranked; do not reinterpret raw scores.
    items = (evidence("FS_Other", "other"), evidence("FS_Second", "second"),
             evidence("FS_KYC_First", "KYC", score=.01),
             evidence("FS_KYC_Second", "KYC", score=100))
    assert select("KYC behavior", *items) == (items[0], items[2])


@pytest.mark.parametrize("limit", [1, 5, 10])
def test_replaces_at_most_one_slot_within_existing_pool(limit):
    items = tuple(evidence(f"FS_Other_{i}", "generic") for i in range(limit)) + (
        evidence("FS_KYC_One", "KYC"), evidence("FS_KYC_Two", "KYC"))
    result = select("KYC", *items, limit=limit)
    assert len(result) == limit
    assert sum(a != b for a, b in zip(result, items)) == 1
    assert all(item in items for item in result)


def test_empty_and_underfilled_pools_stay_unchanged():
    assert select("KYC") == ()
    item = evidence("FS_KYC", "KYC")
    assert select("KYC", item) == (item,)
    with pytest.raises(ValueError, match="limit"):
        select("KYC", item, limit=0)
