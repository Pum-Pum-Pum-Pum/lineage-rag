from __future__ import annotations

import argparse
import hashlib
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path


SECTION_PATTERN = re.compile(r"^##\s+(.+?)\s*$", re.MULTILINE)
FIELD_PATTERN = re.compile(
    r"^SME verdict:\s*(?P<verdict>.+?)\s*$"
    r"(?P<after_verdict>.*?)"
    r"^SME rationale:\s*(?P<rationale>.*?)"
    r"^Required follow-up:\s*(?P<follow_up>.*)\Z",
    re.MULTILINE | re.DOTALL,
)
ALLOWED_VERDICTS = frozenset(
    {
        "accepted",
        "conditionally_accepted",
        "expected_case_incorrect",
        "retrieval_or_release_gap",
        "citation_contract_gap",
        "correct_safe_refusal",
        "other",
    }
)


@dataclass(frozen=True)
class SmeReviewDecision:
    case_id: str
    question: str
    verdict: str
    rationale: str
    required_follow_up: str


def parse_sme_review_markdown(markdown: str) -> list[SmeReviewDecision]:
    matches = list(SECTION_PATTERN.finditer(markdown))
    if not matches:
        raise ValueError("SME review did not contain any case headings")

    decisions: list[SmeReviewDecision] = []
    seen_case_ids: set[str] = set()
    for index, heading in enumerate(matches):
        case_id = heading.group(1).strip()
        if case_id in seen_case_ids:
            raise ValueError(f"Duplicate SME review case_id: {case_id}")
        seen_case_ids.add(case_id)
        section_end = matches[index + 1].start() if index + 1 < len(matches) else len(markdown)
        body = markdown[heading.end() : section_end].strip()
        field_match = FIELD_PATTERN.search(body)
        if field_match is None:
            raise ValueError(f"SME review fields are incomplete for case: {case_id}")
        question = body[: field_match.start()].strip()
        verdict = _normalize_verdict(field_match.group("verdict"))
        if verdict not in ALLOWED_VERDICTS:
            raise ValueError(f"Unsupported SME verdict for {case_id}: {verdict}")
        rationale = field_match.group("rationale").strip()
        required_follow_up = field_match.group("follow_up").strip()
        if not question or not rationale or not required_follow_up:
            raise ValueError(f"SME review contains blank fields for case: {case_id}")
        decisions.append(
            SmeReviewDecision(
                case_id=case_id,
                question=question,
                verdict=verdict,
                rationale=rationale,
                required_follow_up=required_follow_up,
            )
        )
    return decisions


def validate_review_scope(
    decisions: list[SmeReviewDecision],
    report: dict,
    expected_questions: dict[str, str] | None = None,
) -> None:
    report_cases = {
        str(case["case_id"]): str(case.get("question", ""))
        for case in report.get("cases", [])
        if case.get("claim_review_required")
    }
    expected = expected_questions or report_cases
    if set(expected) != set(report_cases):
        raise ValueError(
            "Evaluation manifest and grounded report claim-review scopes do not match"
        )
    actual = {decision.case_id: decision.question for decision in decisions}
    missing = sorted(set(expected).difference(actual))
    extra = sorted(set(actual).difference(expected))
    if missing or extra:
        raise ValueError(f"SME review scope mismatch; missing={missing}, extra={extra}")
    mismatched_questions = sorted(
        case_id
        for case_id, question in expected.items()
        if " ".join(actual[case_id].split()) != " ".join(question.split())
    )
    if mismatched_questions:
        raise ValueError(
            "SME review questions do not match the evaluated report: "
            f"{mismatched_questions}"
        )


def build_sme_review_ledger(
    *,
    decisions: list[SmeReviewDecision],
    review_path: Path,
    report_path: Path,
    eval_path: Path,
    minimum_acceptance_rate: float = 0.90,
) -> dict:
    if not 0.0 <= minimum_acceptance_rate <= 1.0:
        raise ValueError("minimum_acceptance_rate must be between 0 and 1")
    counts: dict[str, int] = {}
    for decision in decisions:
        counts[decision.verdict] = counts.get(decision.verdict, 0) + 1
    accepted_count = counts.get("accepted", 0)
    acceptance_rate = accepted_count / len(decisions) if decisions else 0.0
    unresolved_count = len(decisions) - accepted_count
    return {
        "schema_version": "fdd_grounded_sme_review_v1",
        "review_file": str(review_path),
        "review_sha256": hashlib.sha256(review_path.read_bytes()).hexdigest(),
        "report_file": str(report_path),
        "report_sha256": hashlib.sha256(report_path.read_bytes()).hexdigest(),
        "eval_file": str(eval_path),
        "eval_sha256": hashlib.sha256(eval_path.read_bytes()).hexdigest(),
        "summary": {
            "total_claim_reviews": len(decisions),
            "verdict_counts": counts,
            "unconditional_accepted_count": accepted_count,
            "unconditional_acceptance_rate": acceptance_rate,
            "minimum_acceptance_rate": minimum_acceptance_rate,
            "acceptance_threshold_passed": acceptance_rate >= minimum_acceptance_rate,
            "unresolved_review_count": unresolved_count,
            "requires_remediation_decision": unresolved_count > 0,
        },
        "decisions": [asdict(decision) for decision in decisions],
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Import an SME Markdown review into a validated JSON ledger."
    )
    parser.add_argument("--review-file", type=Path, required=True)
    parser.add_argument("--report-file", type=Path, required=True)
    parser.add_argument("--eval-file", type=Path, required=True)
    parser.add_argument("--output-file", type=Path, required=True)
    parser.add_argument("--minimum-acceptance-rate", type=float, default=0.90)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    review_markdown = args.review_file.read_text(encoding="utf-8")
    report = json.loads(args.report_file.read_text(encoding="utf-8"))
    expected_questions = _load_eval_questions(args.eval_file)
    decisions = parse_sme_review_markdown(review_markdown)
    validate_review_scope(decisions, report, expected_questions)
    ledger = build_sme_review_ledger(
        decisions=decisions,
        review_path=args.review_file,
        report_path=args.report_file,
        eval_path=args.eval_file,
        minimum_acceptance_rate=args.minimum_acceptance_rate,
    )
    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    args.output_file.write_text(
        json.dumps(ledger, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    summary = ledger["summary"]
    print(f"claim_reviews={summary['total_claim_reviews']}")
    print(f"accepted={summary['unconditional_accepted_count']}")
    print(f"acceptance_rate={summary['unconditional_acceptance_rate']:.4f}")
    print(f"threshold_passed={str(summary['acceptance_threshold_passed']).lower()}")
    print(f"unresolved={summary['unresolved_review_count']}")
    print(f"ledger={args.output_file}")
    return 0


def _normalize_verdict(verdict: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", verdict.strip().casefold()).strip("_")


def _load_eval_questions(path: Path) -> dict[str, str]:
    questions: dict[str, str] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        payload = json.loads(line)
        case_id = str(payload.get("case_id", "")).strip()
        question = str(payload.get("question", "")).strip()
        if not case_id or not question:
            raise ValueError(f"Evaluation case at line {line_number} lacks case_id or question")
        if not bool(payload.get("should_abstain")):
            questions[case_id] = question
    return questions


if __name__ == "__main__":
    raise SystemExit(main())
