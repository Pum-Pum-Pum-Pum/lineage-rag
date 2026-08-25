from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.agentic_tools.evaluation import (
    evaluate_bounded_tools,
    file_sha256,
    load_eval_cases,
    write_eval_report_no_overwrite,
    write_sme_review_packet_no_overwrite,
)
from app.agentic_tools.policy import load_agentic_tools_policy
from app.code_indexing.contract import load_code_index_artifact
from app.core.config import Settings
from app.fdd_code_lineage.models import FddCodeLineageArtifact
from app.retrieval.lexical_search import load_retrieval_ready_documents


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run local lexical-only evaluation of bounded Phase 2B tools."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-file", type=Path, required=True)
    parser.add_argument("--review-packet", type=Path)
    parser.add_argument("--allow-draft", action="store_true")
    args = parser.parse_args()
    cases = load_eval_cases(args.manifest)
    if any(not case.sme_reviewed for case in cases) and not args.allow_draft:
        raise PermissionError("Draft tool cases require explicit --allow-draft")
    settings = Settings()
    policy = load_agentic_tools_policy()
    code_artifact = load_code_index_artifact(settings.code_index_artifact_path)
    lineage = FddCodeLineageArtifact.model_validate_json(
        Path(settings.fdd_code_lineage_artifact_path).read_text(encoding="utf-8")
    )
    report = evaluate_bounded_tools(
        cases=cases,
        manifest_sha256=file_sha256(args.manifest),
        policy=policy,
        fdd_documents=load_retrieval_ready_documents(settings.processed_dir),
        fdd_generation=settings.fdd_generation,
        code_artifact=code_artifact,
        lineage_artifact=lineage,
    )
    write_eval_report_no_overwrite(report, args.output_file)
    if args.review_packet is not None:
        write_sme_review_packet_no_overwrite(
            cases=cases, report=report, path=args.review_packet
        )
    print(f"positive_passes={report.positive_passes}/{report.positive_total}")
    print(f"safety_passes={report.safety_passes}/{report.safety_total}")
    print(f"all_cases_reviewed={str(report.all_cases_reviewed).lower()}")
    print(f"release_gate_eligible={str(report.release_gate_eligible).lower()}")
    print(f"external_api_calls={report.external_api_calls}")
    print(f"report_identity_sha256={report.report_identity_sha256}")
    for case in report.cases:
        failures = [check.name for check in case.checks if not check.passed]
        print(
            f"case={case.case_id} passed={str(case.passed).lower()} "
            f"failed_checks={','.join(failures) or 'none'}"
        )
    return 0 if not report.all_cases_reviewed or report.release_gate_eligible else 1


if __name__ == "__main__":
    raise SystemExit(main())
