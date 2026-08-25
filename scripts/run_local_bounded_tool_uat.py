from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Literal, cast


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.agentic_tools.evaluation import execute_local_lexical_tools
from app.agentic_tools.policy import load_agentic_tools_policy
from app.agentic_tools.uat import build_local_uat_report, write_local_uat_report_no_overwrite
from app.code_indexing.contract import load_code_index_artifact
from app.core.config import Settings
from app.fdd_code_lineage.models import FddCodeLineageArtifact
from app.retrieval.lexical_search import load_retrieval_ready_documents


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run one explicit local lexical bounded-tool UAT case."
    )
    parser.add_argument("--mode", choices=("fdd", "code", "combined"), required=True)
    parser.add_argument("--question", required=True)
    parser.add_argument("--limit", type=int, default=8)
    parser.add_argument("--output-file", type=Path, required=True)
    parser.add_argument("--acknowledge-internal-evidence-output", action="store_true")
    args = parser.parse_args()
    if not args.acknowledge_internal_evidence_output:
        raise PermissionError(
            "Local UAT output contains internal source text; pass "
            "--acknowledge-internal-evidence-output explicitly"
        )
    settings = Settings()
    policy = load_agentic_tools_policy()
    if args.limit < 1 or args.limit > policy.budgets.max_results_per_call:
        raise ValueError("UAT limit exceeds the bounded-tool policy")
    code_artifact = load_code_index_artifact(settings.code_index_artifact_path)
    lineage = FddCodeLineageArtifact.model_validate_json(
        Path(settings.fdd_code_lineage_artifact_path).read_text(encoding="utf-8")
    )
    mode = cast(Literal["fdd", "code", "combined"], args.mode)
    execution = execute_local_lexical_tools(
        knowledge_mode=mode,
        question=args.question,
        limit=args.limit,
        policy=policy,
        fdd_documents=load_retrieval_ready_documents(settings.processed_dir),
        fdd_generation=settings.fdd_generation,
        code_artifact=code_artifact,
        lineage_artifact=lineage,
    )
    report = build_local_uat_report(
        knowledge_mode=mode,
        question=args.question,
        fdd_generation=settings.fdd_generation,
        code_snapshot_id=code_artifact.snapshot_id,
        lineage_artifact_identity_sha256=lineage.artifact_identity_sha256,
        policy_sha256=policy.sha256,
        execution=execution,
    )
    write_local_uat_report_no_overwrite(report, args.output_file)
    print(f"status={execution.trace.status}")
    print(f"calls={len(execution.trace.calls)}")
    print(f"evidence_units={execution.trace.total_evidence_units}")
    print("external_api_calls=0")
    print(f"report_identity_sha256={report.report_identity_sha256}")
    return 0 if execution.trace.status == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
