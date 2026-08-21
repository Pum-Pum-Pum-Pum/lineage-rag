from __future__ import annotations

from pathlib import Path

import pytest

from scripts.import_code_combined_eval_review import parse_review_packet


def test_review_parser_uses_explicit_global_note_for_blank_rationale() -> None:
    packet = _section("case-one", rationale="")
    decisions = parse_review_packet(
        packet, global_approval_note="SME confirmed the displayed expectations."
    )
    assert decisions[0].rationale == "SME confirmed the displayed expectations."
    assert decisions[0].rationale_source == "global_approval_note"


def test_review_parser_rejects_unapplied_correction() -> None:
    packet = _section("case-one", correction="Use a different symbol.")
    with pytest.raises(ValueError, match="corrected expectations"):
        parse_review_packet(packet, global_approval_note="Approved.")


def test_review_parser_rejects_nonaccepted_or_duplicate_case() -> None:
    with pytest.raises(ValueError, match="Only accepted"):
        parse_review_packet(
            _section("case-one", verdict="remove"),
            global_approval_note="Approved.",
        )
    duplicate = _section("case-one") + "\n" + _section("case-one")
    with pytest.raises(ValueError, match="Duplicate"):
        parse_review_packet(duplicate, global_approval_note="Approved.")


def test_review_parser_accepts_windows_crlf() -> None:
    packet = _section("case-one").replace("\n", "\r\n")
    decisions = parse_review_packet(packet, global_approval_note="Approved.")
    assert decisions[0].case_id == "case-one"


def _section(
    case_id: str,
    *,
    verdict: str = "accepted",
    correction: str = "",
    rationale: str = "Correct expectation.",
) -> str:
    return (
        f"## 1. {case_id}\n\n"
        f"SME verdict: {verdict}\n"
        f"SME corrected expectations: {correction}\n"
        f"SME rationale: {rationale}\n"
    )
