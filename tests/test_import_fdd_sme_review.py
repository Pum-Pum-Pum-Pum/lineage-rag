import json
from pathlib import Path

import pytest

from scripts.import_fdd_sme_review import (
    build_sme_review_ledger,
    parse_sme_review_markdown,
    validate_review_scope,
)


def test_review_import_normalizes_verdict_and_validates_scope(tmp_path: Path) -> None:
    review_path = tmp_path / "review.md"
    report_path = tmp_path / "report.json"
    review_path.write_text(
        """## case-1

What is supported?

SME verdict: conditionally accepted
SME rationale: The direct behavior is right, but lineage is incomplete.
Required follow-up: Add the baseline source.
""",
        encoding="utf-8",
    )
    report = {
        "cases": [
            {
                "case_id": "case-1",
                "question": "What is supported?",
                "claim_review_required": True,
            }
        ]
    }
    report_path.write_text(json.dumps(report), encoding="utf-8")

    decisions = parse_sme_review_markdown(review_path.read_text(encoding="utf-8"))
    validate_review_scope(decisions, report)
    ledger = build_sme_review_ledger(
        decisions=decisions,
        review_path=review_path,
        report_path=report_path,
        eval_path=review_path,
    )

    assert decisions[0].verdict == "conditionally_accepted"
    assert ledger["summary"]["unconditional_acceptance_rate"] == 0.0
    assert ledger["summary"]["requires_remediation_decision"] is True


def test_review_scope_rejects_missing_case() -> None:
    report = {
        "cases": [
            {"case_id": "case-1", "question": "Question?", "claim_review_required": True}
        ]
    }

    with pytest.raises(ValueError, match="scope mismatch"):
        validate_review_scope([], report)
