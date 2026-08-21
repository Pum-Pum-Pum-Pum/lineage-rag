from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.fdd_code_lineage.evaluation import (
    load_code_combined_eval_cases,
    require_reviewed_code_combined_cases,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare the disclosure/cost boundary for a future paid code/combined "
            "answer evaluation. This command never calls OpenAI."
        )
    )
    parser.add_argument("--eval-file", type=Path, action="append", required=True)
    parser.add_argument("--retrieval-report", type=Path, required=True)
    parser.add_argument("--allow-unreviewed", action="store_true")
    parser.add_argument("--output-file", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cases = []
    for path in args.eval_file:
        cases.extend(load_code_combined_eval_cases(path))
    require_reviewed_code_combined_cases(
        cases, allow_unreviewed=args.allow_unreviewed
    )
    report = json.loads(args.retrieval_report.read_text(encoding="utf-8"))
    report_ids = {item["case_id"] for item in report.get("cases", [])}
    case_ids = {case.case_id for case in cases}
    if report_ids != case_ids:
        raise ValueError("Retrieval report case IDs do not exactly match the manifest")
    if not report.get("summary", {}).get("retrieval_threshold_passed"):
        raise ValueError("Paid answer evaluation is blocked by the retrieval gate")

    reviewed = all(case.sme_reviewed for case in cases)
    plan = {
        "schema_version": "code_combined_answer_eval_plan_v1",
        "status": "awaiting_explicit_paid_authorization",
        "release_gate_eligible": False,
        "reviewed_manifest": reviewed,
        "draft_baseline": not reviewed,
        "case_count": len(cases),
        "answer_generation_request_count": len(cases),
        "query_embedding_request_count": len(cases),
        "external_disclosure_scope": (
            "Evaluation questions plus retrieved internal FDD and PL/SQL evidence excerpts"
        ),
        "eval_files": {
            str(path): hashlib.sha256(path.read_bytes()).hexdigest()
            for path in args.eval_file
        },
        "retrieval_report": str(args.retrieval_report),
        "retrieval_report_sha256": hashlib.sha256(
            args.retrieval_report.read_bytes()
        ).hexdigest(),
        "required_next_action": (
            "Obtain explicit authorization for the stated OpenAI disclosure and cost; "
            "then run a separately recorded paid evaluation and SME-review its answers."
        ),
        "external_api_calls_performed": 0,
    }
    rendered = json.dumps(plan, indent=2, ensure_ascii=False, sort_keys=True)
    if args.output_file:
        if args.output_file.exists():
            raise FileExistsError(f"Answer-evaluation plan already exists: {args.output_file}")
        args.output_file.parent.mkdir(parents=True, exist_ok=True)
        args.output_file.write_text(rendered, encoding="utf-8")
        print(f"plan={args.output_file}")
    else:
        print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
