from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import UTC, datetime
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.agentic_tools.paid_uat import (
    build_paid_case,
    evaluate_paid_uat_answer,
    retrieval_from_local_uat,
)
from app.agentic_tools.uat import LocalToolUatReport, load_manual_uat_cases
from app.code_indexing.contract import load_code_index_artifact
from app.core.config import get_settings
from app.fdd_code_lineage.paid_evaluation import (
    create_no_retry_client,
    generate_grounded_answer,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run the authorized ten-case bounded-tool grounded-answer evaluation."
    )
    parser.add_argument("--reviewed-manifest", type=Path, required=True)
    parser.add_argument("--review-ledger", type=Path, required=True)
    parser.add_argument("--batch-report", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--confirm-authorized-disclosure", action="store_true")
    args = parser.parse_args()
    cases = load_manual_uat_cases(args.reviewed_manifest)
    ledger = json.loads(args.review_ledger.read_text(encoding="utf-8"))
    batch = json.loads(args.batch_report.read_text(encoding="utf-8"))
    _preflight(args, cases, ledger, batch)
    print(f"answer_requests_planned={len(cases)}")
    print("query_embedding_requests_planned=0")
    print("automatic_openai_retries=0")
    if not args.confirm_authorized_disclosure:
        raise PermissionError("Explicit disclosure confirmation flag is required")
    if args.output_directory.exists():
        raise FileExistsError(f"Output directory already exists: {args.output_directory}")
    args.output_directory.mkdir(parents=True)

    settings = get_settings()
    artifact = load_code_index_artifact(settings.code_index_artifact_path)
    client = create_no_retry_client(
        api_key=settings.openai_api_key, base_url=settings.openai_base_url
    )
    summaries_by_id = {item["case_id"]: item for item in batch["cases"]}
    run = {
        "schema_version": "paid_bounded_tool_uat_v1",
        "status": "running",
        "run_id": args.output_directory.name,
        "started_at": datetime.now(UTC).isoformat(),
        "reviewed_manifest_sha256": _sha256(args.reviewed_manifest),
        "review_ledger_sha256": _sha256(args.review_ledger),
        "review_ledger_identity_sha256": ledger["ledger_identity_sha256"],
        "batch_report_sha256": _sha256(args.batch_report),
        "batch_identity_sha256": batch["batch_identity_sha256"],
        "answer_model": settings.openai_chat_model,
        "query_embedding_requests_completed": 0,
        "answer_requests_completed": 0,
        "automatic_openai_retries": 0,
        "cases": [],
    }
    _write(args.output_directory / "run-state.json", run)
    for case in cases:
        record = {"case": case.model_dump(mode="json"), "status": "started"}
        try:
            summary = summaries_by_id[case.case_id]
            local_path = Path(summary["report_file"])
            if _sha256(local_path) != summary["report_sha256"]:
                raise RuntimeError("Local UAT evidence report hash mismatch")
            local_report = LocalToolUatReport.model_validate_json(
                local_path.read_text(encoding="utf-8")
            )
            retrieval = retrieval_from_local_uat(
                case=case, report=local_report, artifact=artifact
            )
            record["local_uat_report"] = str(local_path)
            record["local_uat_report_sha256"] = summary["report_sha256"]
            record["retrieval"] = retrieval.model_dump(mode="json")
            answer, call = generate_grounded_answer(
                client=client,
                model=settings.openai_chat_model,
                case=build_paid_case(case),
                retrieval=retrieval,
            )
            run["answer_requests_completed"] += 1
            record["answer_call"] = call
            record["answer"] = answer.model_dump(mode="json")
            record["structural_evaluation"] = evaluate_paid_uat_answer(
                case=case, answer=answer
            )
            record["status"] = "completed"
            trace_name = f"{case.case_id}.json"
            _write(args.output_directory / trace_name, record)
            run["cases"].append(
                {
                    "case_id": case.case_id,
                    "status": "completed",
                    "structural_passed": record["structural_evaluation"]["passed"],
                    "trace": trace_name,
                }
            )
            _replace(args.output_directory / "run-state.json", run)
            print(
                f"case={case.case_id} "
                f"structural_passed={str(record['structural_evaluation']['passed']).lower()}"
            )
        except Exception as error:
            record["status"] = "failed"
            record["error_type"] = type(error).__name__
            record["error"] = str(error)
            _write(args.output_directory / f"{case.case_id}-failed.json", record)
            run["status"] = "failed_closed"
            run["failed_case_id"] = case.case_id
            run["failure"] = {"type": type(error).__name__, "message": str(error)}
            run["completed_at"] = datetime.now(UTC).isoformat()
            _replace(args.output_directory / "run-state.json", run)
            raise
    passes = sum(item["structural_passed"] for item in run["cases"])
    run["status"] = "completed_pending_sme_review"
    run["completed_at"] = datetime.now(UTC).isoformat()
    run["summary"] = {
        "total_cases": len(cases),
        "structural_passes": passes,
        "semantic_sme_review_required": True,
        "activation_authorized": False,
    }
    _replace(args.output_directory / "run-state.json", run)
    _write_text(
        args.output_directory / "sme-review.md",
        _render_review(run, args.output_directory),
    )
    print(f"structural_passes={passes}/{len(cases)}")
    print(f"sme_review={args.output_directory / 'sme-review.md'}")
    print("activation_authorized=false")
    return 0


def _preflight(args, cases, ledger: dict, batch: dict) -> None:
    if len(cases) != 10 or not all(case.sme_reviewed for case in cases):
        raise ValueError("Paid UAT requires exactly ten reviewed cases")
    if not ledger.get("paid_use_authorized") or not ledger.get(
        "internal_evidence_disclosure_authorized"
    ):
        raise PermissionError("Paid use and internal-evidence disclosure must be authorized")
    if ledger.get("maximum_answer_requests") != 10 or ledger.get(
        "maximum_query_embedding_requests"
    ) != 0:
        raise ValueError("Authorization request bounds do not match the runner")
    if ledger.get("reviewed_manifest_sha256") != _sha256(args.reviewed_manifest):
        raise ValueError("Reviewed manifest differs from the authorization ledger")
    if ledger.get("batch_report_sha256") != _sha256(args.batch_report):
        raise ValueError("Batch report differs from the authorization ledger")
    if batch.get("diagnostic_passes") != batch.get("diagnostic_total"):
        raise ValueError("Local UAT diagnostic gate is not complete")


def _render_review(run: dict, directory: Path) -> str:
    lines = [
        "# Paid bounded-tool grounded-answer SME review",
        "",
        f"- Run ID: `{run['run_id']}`",
        f"- Answer requests: {run['answer_requests_completed']}",
        "- Query embedding requests: 0",
        f"- Structural passes: {run['summary']['structural_passes']}/{run['summary']['total_cases']}",
        "- Status: pending SME semantic review",
        "",
    ]
    for index, summary in enumerate(run["cases"], start=1):
        trace = json.loads((directory / summary["trace"]).read_text(encoding="utf-8"))
        lines.extend(
            [
                f"## {index}. {summary['case_id']}",
                "",
                f"**Question:** {trace['case']['question']}",
                "",
                "```json",
                json.dumps(trace["answer"], indent=2, ensure_ascii=False),
                "```",
                "",
                f"Structural result: **{'pass' if summary['structural_passed'] else 'fail'}**",
                "",
                "SME verdict: accepted | corrected | needs_more_context",
                "SME rationale:",
                "Required follow-up:",
                "",
            ]
        )
    return "\n".join(lines)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, value: dict) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite paid UAT artifact: {path}")
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def _replace(path: Path, value: dict) -> None:
    temp = path.with_suffix(".tmp")
    temp.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    temp.replace(path)


def _write_text(path: Path, value: str) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite paid UAT artifact: {path}")
    path.write_text(value, encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
