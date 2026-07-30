from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence


@dataclass(frozen=True)
class AnswerEvalExpectation:
    required_answer_regex_all: list[str]
    required_citation_release_label: str | None = None
    required_citation_unit_contains_all: list[str] | None = None
    forbidden_answer_regex_any: list[str] | None = None


@dataclass(frozen=True)
class AnswerEvalCase:
    case_id: str
    query: str
    expectation: AnswerEvalExpectation
    notes: str = ""


@dataclass(frozen=True)
class AnswerEvalResult:
    case_id: str
    passed: bool
    failures: list[str]


def load_answer_eval_cases(path: str | Path) -> list[AnswerEvalCase]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return [
        AnswerEvalCase(
            case_id=item["case_id"],
            query=item["query"],
            expectation=AnswerEvalExpectation(**item["expectation"]),
            notes=item.get("notes", ""),
        )
        for item in payload
    ]


def evaluate_serialized_answer(
    case: AnswerEvalCase,
    *,
    answer: str,
    citations: Sequence[dict[str, Any]],
) -> AnswerEvalResult:
    failures: list[str] = []
    expectation = case.expectation

    missing_patterns = [
        pattern
        for pattern in expectation.required_answer_regex_all
        if re.search(pattern, answer, flags=re.IGNORECASE | re.DOTALL) is None
    ]
    if missing_patterns:
        failures.append(
            f"Required answer patterns were missing: {missing_patterns}"
        )

    if expectation.forbidden_answer_regex_any:
        found_forbidden = [
            pattern
            for pattern in expectation.forbidden_answer_regex_any
            if re.search(pattern, answer, flags=re.IGNORECASE | re.DOTALL)
            is not None
        ]
        if found_forbidden:
            failures.append(
                f"Forbidden answer patterns were found: {found_forbidden}"
            )

    if expectation.required_citation_release_label:
        bad_releases = [
            citation.get("release_label")
            for citation in citations
            if citation.get("release_label")
            != expectation.required_citation_release_label
        ]
        if not citations:
            failures.append("No citations were returned.")
        elif bad_releases:
            failures.append(
                "Citations contained unexpected releases: "
                f"{bad_releases}"
            )

    if expectation.required_citation_unit_contains_all:
        citation_units = [
            str(citation.get("unit_id", ""))
            for citation in citations
        ]
        missing_units = [
            marker
            for marker in expectation.required_citation_unit_contains_all
            if not any(marker in unit for unit in citation_units)
        ]
        if missing_units:
            failures.append(
                f"Required citation units were missing: {missing_units}"
            )

    return AnswerEvalResult(
        case_id=case.case_id,
        passed=not failures,
        failures=failures,
    )
