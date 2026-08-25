from __future__ import annotations

from pathlib import Path

import pytest

from app.agentic_tools.evaluation import BoundedToolEvalCase, file_sha256, load_eval_cases
from app.agentic_tools.review import parse_accepted_review_packet, promote_reviewed_manifest


def _case() -> BoundedToolEvalCase:
    return BoundedToolEvalCase(
        case_id="tool-review-case-001",
        knowledge_mode="fdd",
        question="What documented behavior is visible for this function?",
        tools=("fdd_search",),
        limit=5,
        expected_fdd_document_ids=("doc-1",),
        rationale="Tests review promotion without changing the expectation.",
    )


def _packet(manifest_hash: str, *, verdict: str = "accepted", correction: str = "") -> str:
    return f"""# Bounded agentic tools SME review packet

- Manifest SHA-256: `{manifest_hash}`
- Report identity: `{'a' * 64}`

## 1. tool-review-case-001

SME verdict: {verdict}
SME corrected expectation: {correction}
SME rationale:
Required follow-up:
"""


def test_review_parser_rejects_nonaccepted_and_corrected_cases() -> None:
    with pytest.raises(ValueError, match="accepted unchanged"):
        parse_accepted_review_packet(_packet("1" * 64, verdict="needs_more_context"), approval_note="ok")
    with pytest.raises(ValueError, match="new manifest"):
        parse_accepted_review_packet(_packet("1" * 64, correction="remove symbol"), approval_note="ok")


def test_review_promotion_is_hash_bound_and_no_overwrite(tmp_path: Path) -> None:
    draft = tmp_path / "draft.jsonl"
    draft.write_text(_case().model_dump_json() + "\n", encoding="utf-8")
    packet = tmp_path / "review.md"
    packet.write_text(_packet(file_sha256(draft)), encoding="utf-8")
    reviewed = tmp_path / "reviewed.jsonl"
    ledger = tmp_path / "ledger.json"
    result = promote_reviewed_manifest(
        draft_manifest=draft,
        review_packet=packet,
        reviewed_manifest=reviewed,
        ledger_file=ledger,
        reviewer="Reviewer",
        approval_note="All reviewed expectations are accepted.",
    )
    promoted = load_eval_cases(reviewed)
    assert promoted[0].sme_reviewed is True
    assert promoted[0].review_status == "reviewed"
    assert result["evaluated_report_identity_sha256"] == "a" * 64
    with pytest.raises(FileExistsError, match="overwrite"):
        promote_reviewed_manifest(
            draft_manifest=draft,
            review_packet=packet,
            reviewed_manifest=reviewed,
            ledger_file=ledger,
            reviewer="Reviewer",
            approval_note="All reviewed expectations are accepted.",
        )


def test_review_promotion_rejects_manifest_mismatch(tmp_path: Path) -> None:
    draft = tmp_path / "draft.jsonl"
    draft.write_text(_case().model_dump_json() + "\n", encoding="utf-8")
    packet = tmp_path / "review.md"
    packet.write_text(_packet("0" * 64), encoding="utf-8")
    with pytest.raises(ValueError, match="not bound"):
        promote_reviewed_manifest(
            draft_manifest=draft,
            review_packet=packet,
            reviewed_manifest=tmp_path / "reviewed.jsonl",
            ledger_file=tmp_path / "ledger.json",
            reviewer="Reviewer",
            approval_note="All reviewed expectations are accepted.",
        )
