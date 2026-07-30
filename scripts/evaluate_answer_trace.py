from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.llm.answer_evaluation import (
    evaluate_serialized_answer,
    load_answer_eval_cases,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a persisted answer trace against a named case."
    )
    parser.add_argument("--trace", required=True, help="Answer trace JSON path.")
    parser.add_argument(
        "--eval-file",
        default="data/eval/answer_eval.json",
        help="Answer evaluation JSON path.",
    )
    parser.add_argument("--case-id", required=True, help="Evaluation case ID.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cases = {
        case.case_id: case
        for case in load_answer_eval_cases(args.eval_file)
    }
    if args.case_id not in cases:
        raise ValueError(f"Unknown answer evaluation case: {args.case_id}")

    trace = json.loads(Path(args.trace).read_text(encoding="utf-8"))
    response = trace["answer_response"]
    result = evaluate_serialized_answer(
        cases[args.case_id],
        answer=str(response["answer"]),
        citations=list(response.get("citations", [])),
    )
    print(json.dumps(asdict(result), indent=2))
    if not result.passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
