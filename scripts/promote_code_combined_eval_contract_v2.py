from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.fdd_code_lineage.evaluation import load_code_combined_eval_cases


ADVISORY_SYMBOL_CASE = "combined-aml-offline-impact-004"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Publish the reviewed v2 combined evaluation contract."
    )
    parser.add_argument("--input-file", type=Path, required=True)
    parser.add_argument("--answer-review-ledger", type=Path, required=True)
    parser.add_argument("--output-file", type=Path, required=True)
    parser.add_argument("--contract-ledger", type=Path, required=True)
    args = parser.parse_args()
    answer_review = json.loads(args.answer_review_ledger.read_text(encoding="utf-8"))
    if answer_review.get("summary", {}).get("semantic_acceptances") != 10:
        raise ValueError("Contract correction requires all ten semantic acceptances")
    cases = load_code_combined_eval_cases(args.input_file)
    if ADVISORY_SYMBOL_CASE not in {case.case_id for case in cases}:
        raise ValueError("Expected corrected impact case is absent")
    rendered = ""
    for case in cases:
        updates = {"schema_version": "code_combined_eval_case_v2"}
        if case.case_id == ADVISORY_SYMBOL_CASE:
            updates["expected_code_symbol_policy"] = "advisory"
        rendered += case.model_copy(update=updates).model_dump_json() + "\n"
    ledger = {
        "schema_version": "code_combined_eval_contract_promotion_v2",
        "source_manifest": str(args.input_file),
        "source_manifest_sha256": _sha256(args.input_file),
        "answer_review_ledger": str(args.answer_review_ledger),
        "answer_review_ledger_sha256": _sha256(args.answer_review_ledger),
        "output_manifest": str(args.output_file),
        "output_manifest_sha256": hashlib.sha256(rendered.encode()).hexdigest(),
        "corrections": [
            {
                "case_id": ADVISORY_SYMBOL_CASE,
                "change": "expected_code_symbol_policy=advisory",
                "reason": "SME accepted multiple grounded candidate change locations without requiring one exact symbol.",
            },
            {
                "case_id": "combined-kernel-http-negative-005",
                "change": "evaluate requested_claim_supported=false independently of related context",
                "reason": "SME accepted separately cited visible-code help while the exact hidden-kernel claim remained refused.",
            },
        ],
    }
    canonical = json.dumps(ledger, sort_keys=True, separators=(",", ":"))
    ledger["ledger_identity_sha256"] = hashlib.sha256(canonical.encode()).hexdigest()
    for path in (args.output_file, args.contract_ledger):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite v2 contract output: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
    args.output_file.write_text(rendered, encoding="utf-8")
    args.contract_ledger.write_text(
        json.dumps(ledger, indent=2, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
    load_code_combined_eval_cases(args.output_file)
    print(f"cases={len(cases)}")
    print(f"output_sha256={ledger['output_manifest_sha256']}")
    print(f"ledger_identity_sha256={ledger['ledger_identity_sha256']}")
    return 0


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
