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
from app.agentic_tools.replay import validate_case_replay_authorization
from app.agentic_tools.uat import LocalToolUatReport, load_manual_uat_cases
from app.code_indexing.contract import load_code_index_artifact
from app.core.config import get_settings
from app.fdd_code_lineage.paid_evaluation import (
    CODE_SYSTEM_PROMPT,
    create_no_retry_client,
    generate_grounded_answer,
)


def main() -> int:
    parser = argparse.ArgumentParser(description="Run one authorized paid bounded-tool replay.")
    parser.add_argument("--authorization", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--confirm-authorized-disclosure", action="store_true")
    args = parser.parse_args()
    authorization = json.loads(args.authorization.read_text(encoding="utf-8"))
    validate_case_replay_authorization(authorization)
    _validate_bound_files(authorization)
    cases = load_manual_uat_cases(Path(authorization["reviewed_manifest"]))
    matching = [case for case in cases if case.case_id == authorization["case_id"]]
    if len(matching) != 1 or not matching[0].sme_reviewed:
        raise ValueError("Replay case is not one exact reviewed case")
    settings = get_settings()
    if settings.openai_chat_model != authorization["answer_model"]:
        raise ValueError("Configured answer model differs from replay authorization")
    prompt_hash = hashlib.sha256(CODE_SYSTEM_PROMPT.encode("utf-8")).hexdigest()
    if prompt_hash != authorization["code_system_prompt_sha256"]:
        raise ValueError("Code prompt differs from replay authorization")
    print("answer_requests_planned=1")
    print("query_embedding_requests_planned=0")
    print("automatic_openai_retries=0")
    if not args.confirm_authorized_disclosure:
        raise PermissionError("Explicit disclosure confirmation flag is required")
    if args.output_directory.exists():
        raise FileExistsError(f"Replay output already exists: {args.output_directory}")
    args.output_directory.mkdir(parents=True)
    state = {
        "schema_version": "paid_bounded_tool_case_replay_v1",
        "status": "running",
        "started_at": datetime.now(UTC).isoformat(),
        "authorization": str(args.authorization),
        "authorization_sha256": _sha256(args.authorization),
        "authorization_identity_sha256": authorization[
            "authorization_identity_sha256"
        ],
        "case_id": matching[0].case_id,
        "answer_requests_completed": 0,
        "query_embedding_requests_completed": 0,
        "automatic_openai_retries": 0,
    }
    _write(args.output_directory / "run-state.json", state)
    try:
        artifact = load_code_index_artifact(settings.code_index_artifact_path)
        local_report = LocalToolUatReport.model_validate_json(
            Path(authorization["local_uat_report"]).read_text(encoding="utf-8")
        )
        retrieval = retrieval_from_local_uat(
            case=matching[0], report=local_report, artifact=artifact
        )
        client = create_no_retry_client(
            api_key=settings.openai_api_key, base_url=settings.openai_base_url
        )
        answer, call = generate_grounded_answer(
            client=client,
            model=settings.openai_chat_model,
            case=build_paid_case(matching[0]),
            retrieval=retrieval,
        )
        result = {
            "case": matching[0].model_dump(mode="json"),
            "retrieval": retrieval.model_dump(mode="json"),
            "answer_call": call,
            "answer": answer.model_dump(mode="json"),
            "structural_evaluation": evaluate_paid_uat_answer(
                case=matching[0], answer=answer
            ),
        }
        _write(args.output_directory / f"{matching[0].case_id}.json", result)
        state.update(
            {
                "status": "completed_pending_sme_review",
                "completed_at": datetime.now(UTC).isoformat(),
                "answer_requests_completed": 1,
                "structural_passed": result["structural_evaluation"]["passed"],
                "trace": f"{matching[0].case_id}.json",
                "activation_authorized": False,
            }
        )
        _replace(args.output_directory / "run-state.json", state)
        _write_text(
            args.output_directory / "sme-review.md",
            _review_markdown(matching[0].case_id, matching[0].question, result),
        )
        print(f"structural_passed={str(state['structural_passed']).lower()}")
        print(f"sme_review={args.output_directory / 'sme-review.md'}")
        return 0
    except Exception as error:
        state.update(
            {
                "status": "failed_closed",
                "completed_at": datetime.now(UTC).isoformat(),
                "failure": {"type": type(error).__name__, "message": str(error)},
            }
        )
        _replace(args.output_directory / "run-state.json", state)
        raise


def _validate_bound_files(value: dict) -> None:
    bindings = {
        "reviewed_manifest": "reviewed_manifest_sha256",
        "prior_review_ledger": "prior_review_ledger_sha256",
        "prior_trace": "prior_trace_sha256",
        "local_uat_report": "local_uat_report_sha256",
    }
    for path_key, hash_key in bindings.items():
        if _sha256(Path(value[path_key])) != value[hash_key]:
            raise ValueError(f"Replay binding mismatch: {path_key}")


def _review_markdown(case_id: str, question: str, result: dict) -> str:
    return "\n".join(
        [
            "# Paid bounded-tool case replay SME review",
            "",
            f"## 1. {case_id}",
            "",
            f"**Question:** {question}",
            "",
            "```json",
            json.dumps(result["answer"], indent=2, ensure_ascii=False),
            "```",
            "",
            "Structural result: **"
            + ("pass" if result["structural_evaluation"]["passed"] else "fail")
            + "**",
            "",
            "SME verdict: accepted | corrected | needs_more_context",
            "SME rationale:",
            "Required follow-up:",
            "",
        ]
    )


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _write(path: Path, value: dict) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite replay artifact: {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        handle.write(json.dumps(value, indent=2, ensure_ascii=False))


def _replace(path: Path, value: dict) -> None:
    temp = path.with_suffix(".tmp")
    with temp.open("w", encoding="utf-8", newline="") as handle:
        handle.write(json.dumps(value, indent=2, ensure_ascii=False))
    temp.replace(path)


def _write_text(path: Path, value: str) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite replay artifact: {path}")
    with path.open("w", encoding="utf-8", newline="") as handle:
        handle.write(value)


if __name__ == "__main__":
    raise SystemExit(main())
