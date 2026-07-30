from pathlib import Path

from app.llm.answer_evaluation import (
    evaluate_serialized_answer,
    load_answer_eval_cases,
)


EVAL_PATH = Path("data/eval/answer_eval.json")


def _citations() -> list[dict[str, str]]:
    return [
        {"unit_id": "r24::chunk_3", "release_label": "R24"},
        {"unit_id": "r24::table_chunk_6", "release_label": "R24"},
        {"unit_id": "r24::table_chunk_7", "release_label": "R24"},
    ]


def test_r24_current_state_answer_case_accepts_2_and_4_with_baseline() -> None:
    case = load_answer_eval_cases(EVAL_PATH)[0]
    answer = (
        "Baseline before R24: Teller 6 and Branch 17. "
        "Teller reports now: 2, consisting of T-1 and T-2. "
        "Branch reports now: 4, consisting of B-01, B-02, B-03, and B-04."
    )

    result = evaluate_serialized_answer(
        case,
        answer=answer,
        citations=_citations(),
    )

    assert result.passed is True
    assert result.failures == []


def test_r24_current_state_answer_case_rejects_baseline_as_current() -> None:
    case = load_answer_eval_cases(EVAL_PATH)[0]
    answer = (
        "There are currently Teller 6 and Branch 17 reports. "
        "R24 mentions T-1, T-2, B-01, B-02, B-03, and B-04."
    )

    result = evaluate_serialized_answer(
        case,
        answer=answer,
        citations=_citations(),
    )

    assert result.passed is False
    assert any("Required answer patterns" in item for item in result.failures)
    assert any("Forbidden answer patterns" in item for item in result.failures)


def test_r24_current_state_answer_case_rejects_r2_or_missing_table_citations() -> None:
    case = load_answer_eval_cases(EVAL_PATH)[0]
    answer = (
        "Baseline before R24: Teller 6 and Branch 17. "
        "Teller reports now: 2, T-1 and T-2. "
        "Branch reports now: 4, B-01, B-02, B-03, and B-04."
    )

    result = evaluate_serialized_answer(
        case,
        answer=answer,
        citations=[
            {"unit_id": "r24::chunk_3", "release_label": "R24"},
            {"unit_id": "r2::chunk_1", "release_label": "R2"},
        ],
    )

    assert result.passed is False
    assert any("unexpected releases" in item for item in result.failures)
    assert any("Required citation units" in item for item in result.failures)


def test_abstention_case_requires_refusal_state_and_reason() -> None:
    cases = {
        case.case_id: case
        for case in load_answer_eval_cases("data/eval/answer_eval.json")
    }
    case = cases["unsupported_mobile_login_abstention"]

    passing = evaluate_serialized_answer(
        case,
        answer=(
            "I could not find sufficient evidence in the indexed documents "
            "to answer this confidently."
        ),
        citations=[],
        is_answered=False,
        refusal_reason="Top score is below the required threshold.",
    )
    failing = evaluate_serialized_answer(
        case,
        answer="The mobile login uses an OTP.",
        citations=[],
        is_answered=True,
        refusal_reason=None,
    )

    assert passing.passed is True
    assert failing.passed is False
    assert any("Answer state" in item for item in failing.failures)
    assert any("Refusal reason" in item for item in failing.failures)
