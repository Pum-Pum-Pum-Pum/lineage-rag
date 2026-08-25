from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.agentic_tools.evaluation import execute_local_lexical_tools
from app.agentic_tools.policy import load_agentic_tools_policy
from app.agentic_tools.uat import (
    ManualToolUatBatchReport,
    build_local_uat_report,
    build_manual_uat_batch_report,
    build_manual_uat_case_summary,
    load_manual_uat_cases,
    write_manual_uat_packet_no_overwrite,
)
from app.code_indexing.contract import load_code_index_artifact
from app.core.config import Settings
from app.fdd_code_lineage.models import FddCodeLineageArtifact
from app.retrieval.lexical_search import load_retrieval_ready_documents


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run a draft batch of explicit local bounded-tool UAT cases."
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--batch-report", type=Path, required=True)
    parser.add_argument("--review-packet", type=Path, required=True)
    parser.add_argument("--acknowledge-internal-evidence-output", action="store_true")
    args = parser.parse_args()
    if not args.acknowledge_internal_evidence_output:
        raise PermissionError(
            "Batch UAT reports contain internal source text; pass "
            "--acknowledge-internal-evidence-output explicitly"
        )
    cases = load_manual_uat_cases(args.manifest)
    report_paths = {
        case.case_id: args.output_directory / f"{case.case_id}.json" for case in cases
    }
    targets = [*report_paths.values(), args.batch_report, args.review_packet]
    existing = [str(path) for path in targets if path.exists()]
    if existing:
        raise FileExistsError(f"Refusing to overwrite UAT outputs: {existing}")

    settings = Settings()
    policy = load_agentic_tools_policy()
    code_artifact = load_code_index_artifact(settings.code_index_artifact_path)
    lineage = FddCodeLineageArtifact.model_validate_json(
        Path(settings.fdd_code_lineage_artifact_path).read_text(encoding="utf-8")
    )
    fdd_documents = load_retrieval_ready_documents(settings.processed_dir)
    reports = {}
    summaries = []
    for case in cases:
        execution = execute_local_lexical_tools(
            knowledge_mode=case.knowledge_mode,
            question=case.question,
            limit=case.limit,
            policy=policy,
            fdd_documents=fdd_documents,
            fdd_generation=settings.fdd_generation,
            code_artifact=code_artifact,
            lineage_artifact=lineage,
        )
        report = build_local_uat_report(
            knowledge_mode=case.knowledge_mode,
            question=case.question,
            fdd_generation=settings.fdd_generation,
            code_snapshot_id=code_artifact.snapshot_id,
            lineage_artifact_identity_sha256=lineage.artifact_identity_sha256,
            policy_sha256=policy.sha256,
            execution=execution,
        )
        reports[case.case_id] = report
        summaries.append(
            build_manual_uat_case_summary(
                case=case, report=report, report_file=report_paths[case.case_id]
            )
        )
    manifest_hash = hashlib.sha256(args.manifest.read_bytes()).hexdigest()
    batch = build_manual_uat_batch_report(
        manifest_sha256=manifest_hash, summaries=summaries
    )

    args.output_directory.mkdir(parents=True, exist_ok=True)
    for case_id, report in reports.items():
        with report_paths[case_id].open("w", encoding="utf-8", newline="") as handle:
            handle.write(report.model_dump_json(indent=2))
    args.batch_report.parent.mkdir(parents=True, exist_ok=True)
    with args.batch_report.open("w", encoding="utf-8", newline="") as handle:
        handle.write(batch.model_dump_json(indent=2))
    ManualToolUatBatchReport.model_validate_json(
        args.batch_report.read_text(encoding="utf-8")
    )
    write_manual_uat_packet_no_overwrite(
        cases=cases, batch=batch, path=args.review_packet
    )
    print(f"diagnostic_passes={batch.diagnostic_passes}/{batch.diagnostic_total}")
    print("all_cases_reviewed=false")
    print("external_api_calls=0")
    print(f"batch_identity_sha256={batch.batch_identity_sha256}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
