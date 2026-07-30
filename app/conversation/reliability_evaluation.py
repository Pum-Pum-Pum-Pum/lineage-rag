from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Sequence

from app.conversation.models import ConversationMessage


RELEASE_ANCHOR_PATTERN = re.compile(r"\bR\d+\b", flags=re.IGNORECASE)
REPORT_ANCHOR_PATTERN = re.compile(
    r"\b(?:B-\d{1,2}|T-\d{1,2})\b",
    flags=re.IGNORECASE,
)


@dataclass(frozen=True)
class SummaryDriftEvaluation:
    passed: bool
    invented_anchors: tuple[str, ...]
    missing_release_anchors: tuple[str, ...]


def evaluate_summary_drift(
    *,
    previous_summary: str | None,
    messages: Sequence[ConversationMessage],
    candidate_summary: str,
) -> SummaryDriftEvaluation:
    """Detect high-signal identifier drift in a generated rolling summary.

    This is an evaluation guard, not a complete semantic-faithfulness proof.
    It catches invented release/report identifiers and loss of explicit release
    scope because those errors can silently redirect later retrieval.
    """

    source_text = "\n".join(
        [
            previous_summary or "",
            *(message.content for message in messages),
        ]
    )
    source_releases = _extract(RELEASE_ANCHOR_PATTERN, source_text)
    source_reports = _extract(REPORT_ANCHOR_PATTERN, source_text)
    candidate_releases = _extract(RELEASE_ANCHOR_PATTERN, candidate_summary)
    candidate_reports = _extract(REPORT_ANCHOR_PATTERN, candidate_summary)

    invented = sorted(
        (candidate_releases | candidate_reports)
        - (source_releases | source_reports)
    )
    missing_releases = sorted(source_releases - candidate_releases)
    return SummaryDriftEvaluation(
        passed=not invented and not missing_releases,
        invented_anchors=tuple(invented),
        missing_release_anchors=tuple(missing_releases),
    )


def _extract(pattern: re.Pattern[str], text: str) -> set[str]:
    return {match.upper() for match in pattern.findall(text)}
