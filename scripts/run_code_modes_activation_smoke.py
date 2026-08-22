from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from app.activation.code_modes import (
    ActivationApproval,
    ActivationRequest,
    approval_identity,
)


SMOKE_CASES = (
    {
        "case_id": "activation-code-smoke-001",
        "knowledge_mode": "code",
        "analysis_kind": "explanation",
        "query": (
            "What does the custom spSendBatchTxnEndData routine do when sending "
            "batch transaction data to the AML integration?"
        ),
    },
    {
        "case_id": "activation-combined-smoke-001",
        "knowledge_mode": "combined",
        "analysis_kind": "explanation",
        "query": (
            "How does the system integrate FCIS transactions with FlagRight, and "
            "where is that flow visible in custom code?"
        ),
    },
)


def validate_authorization(request: ActivationRequest, approval: ActivationApproval) -> None:
    valid_identity = (
        approval_identity(approval.model_dump(mode="json"))
        == approval.approval_identity_sha256
    )
    if (
        approval.decision != "approved"
        or approval.request_identity_sha256 != request.request_identity_sha256
        or not valid_identity
        or not approval.paid_smoke_authorized
        or not approval.internal_evidence_disclosure_authorized
    ):
        raise PermissionError(
            "A valid approval authorizing paid smoke and internal-evidence disclosure is required"
        )


def run_smokes(
    *, client: httpx.Client, base_url: str,
    request: ActivationRequest, approval: ActivationApproval,
) -> dict[str, Any]:
    validate_authorization(request, approval)
    results: list[dict[str, Any]] = []
    for case in SMOKE_CASES:
        response = client.post(
            f"{base_url.rstrip('/')}/query",
            json={
                "query": case["query"],
                "knowledge_mode": case["knowledge_mode"],
                "analysis_kind": case["analysis_kind"],
                "limit": 8,
            },
        )
        try:
            payload = response.json()
        except ValueError:
            payload = {"unparseable_response_body_sha256": hashlib.sha256(response.content).hexdigest()}
        if response.status_code >= 400:
            failures = [f"http_status:{response.status_code}"]
        else:
            failures = validate_response(case, payload)
        results.append({
            "case": case,
            "passed": not failures,
            "failures": failures,
            "http_status": response.status_code,
            "response": payload,
        })
        if failures:
            break
    passed = len(results) == len(SMOKE_CASES) and all(item["passed"] for item in results)
    report: dict[str, Any] = {
        "schema_version": "code_modes_activation_smoke_v1",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "request_identity_sha256": request.request_identity_sha256,
        "approval_identity_sha256": approval.approval_identity_sha256,
        "authorized_request_limit": 2,
        "requests_attempted": len(results),
        "automatic_retries": 0,
        "passed": passed,
        "results": results,
    }
    report["report_identity_sha256"] = _canonical_sha256(report)
    return report


def validate_response(case: dict[str, str], payload: dict[str, Any]) -> list[str]:
    failures: list[str] = []
    mode = case["knowledge_mode"]
    if payload.get("knowledge_mode") != mode:
        failures.append("knowledge_mode_mismatch")
    if payload.get("requested_claim_supported") is not True:
        failures.append("requested_claim_not_supported")
    if not payload.get("trace_id"):
        failures.append("missing_trace_id")
    code_citations = payload.get("code_citations") or []
    if not code_citations:
        failures.append("missing_code_citations")
    for citation in code_citations:
        if not citation.get("source_path") or not citation.get("display_name"):
            failures.append("incomplete_code_citation_identity")
            break
        if not isinstance(citation.get("start_line"), int) or not isinstance(
            citation.get("end_line"), int
        ):
            failures.append("invalid_code_citation_lines")
            break
    fdd_citations = payload.get("citations") or []
    if mode == "code" and fdd_citations:
        failures.append("unexpected_fdd_citations")
    if mode == "combined":
        if not fdd_citations:
            failures.append("missing_fdd_citations")
        sections = payload.get("combined_sections") or {}
        for name in ("documented_functionality", "visible_custom_implementation"):
            if (sections.get(name) or {}).get("status") != "answered":
                failures.append(f"section_not_answered:{name}")
    return failures


def write_report(path: Path, report: dict[str, Any]) -> None:
    if path.exists():
        raise FileExistsError(f"Refusing to overwrite smoke evidence: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    data = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    handle, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(handle, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(data)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    finally:
        if os.path.exists(temporary_name):
            os.unlink(temporary_name)


def _canonical_sha256(payload: dict[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description="Run exactly two approval-bound code-mode smoke requests.")
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--approval", type=Path, required=True)
    parser.add_argument("--output-file", type=Path, required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--timeout", type=float, default=180.0)
    args = parser.parse_args()
    request = ActivationRequest.model_validate_json(args.request.read_text(encoding="utf-8"))
    approval = ActivationApproval.model_validate_json(args.approval.read_text(encoding="utf-8"))
    with httpx.Client(timeout=args.timeout) as client:
        report = run_smokes(
            client=client, base_url=args.base_url,
            request=request, approval=approval,
        )
    write_report(args.output_file, report)
    print(f"requests_attempted={report['requests_attempted']}")
    print(f"automatic_retries={report['automatic_retries']}")
    print(f"passed={str(report['passed']).lower()}")
    print(f"report_identity_sha256={report['report_identity_sha256']}")
    for result in report["results"]:
        response = result["response"]
        usage = response.get("usage") or {}
        print(
            f"case={result['case']['case_id']} passed={str(result['passed']).lower()} "
            f"trace_id={response.get('trace_id')} total_tokens={usage.get('total_tokens')}"
        )
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
