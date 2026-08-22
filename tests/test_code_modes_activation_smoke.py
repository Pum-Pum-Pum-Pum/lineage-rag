from __future__ import annotations

import httpx
import pytest
from datetime import UTC, datetime

from app.activation.code_modes import ActivationRequest, build_activation_approval
from scripts.run_code_modes_activation_smoke import run_smokes, validate_response


def _request() -> ActivationRequest:
    return ActivationRequest(
        schema_version="code_modes_activation_request_v1",
        created_at_utc=datetime.now(UTC),
        requested_by="operator",
        requested_modes=("code", "combined"),
        target_configuration={},
        rollback_configuration={},
        evidence_identities={},
        request_identity_sha256="a" * 64,
    )


def _payload(mode: str) -> dict:
    payload = {
        "knowledge_mode": mode,
        "requested_claim_supported": True,
        "trace_id": f"trace-{mode}",
        "citations": [],
        "code_citations": [{
            "source_path": "pkg_custom.sql", "display_name": "spExample",
            "start_line": 1, "end_line": 5,
        }],
    }
    if mode == "combined":
        payload["citations"] = [{"unit_id": "fdd-1"}]
        payload["combined_sections"] = {
            "documented_functionality": {"status": "answered"},
            "visible_custom_implementation": {"status": "answered"},
        }
    return payload


def test_smoke_requires_both_paid_and_disclosure_authorization() -> None:
    activation_request = _request()
    approval = build_activation_approval(
        request=activation_request, approved_by="SME",
        paid_smoke_authorized=True,
        internal_evidence_disclosure_authorized=False,
    )
    client = httpx.Client(transport=httpx.MockTransport(lambda request: pytest.fail("network called")))
    with client, pytest.raises(PermissionError):
        run_smokes(client=client, base_url="http://test", request=activation_request, approval=approval)


def test_smoke_attempts_exactly_two_requests_without_retries() -> None:
    activation_request = _request()
    approval = build_activation_approval(
        request=activation_request, approved_by="SME",
        paid_smoke_authorized=True,
        internal_evidence_disclosure_authorized=True,
    )
    seen: list[str] = []
    def handler(request: httpx.Request) -> httpx.Response:
        mode = __import__("json").loads(request.content)["knowledge_mode"]
        seen.append(mode)
        return httpx.Response(200, json=_payload(mode))
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        report = run_smokes(client=client, base_url="http://test", request=activation_request, approval=approval)
    assert report["passed"] is True
    assert report["requests_attempted"] == 2
    assert report["automatic_retries"] == 0
    assert seen == ["code", "combined"]


def test_response_validation_fails_closed_on_missing_lane_evidence() -> None:
    failures = validate_response(
        {"knowledge_mode": "combined"},
        {"knowledge_mode": "combined", "requested_claim_supported": True, "trace_id": "t"},
    )
    assert "missing_code_citations" in failures
    assert "missing_fdd_citations" in failures
    assert "section_not_answered:documented_functionality" in failures


def test_http_failure_is_recorded_without_retrying() -> None:
    request_model = _request()
    approval = build_activation_approval(
        request=request_model, approved_by="SME",
        paid_smoke_authorized=True,
        internal_evidence_disclosure_authorized=True,
    )
    seen = 0
    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal seen
        seen += 1
        if seen == 1:
            return httpx.Response(200, json=_payload("code"))
        return httpx.Response(400, json={"detail": "invalid combined answer contract"})
    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        report = run_smokes(
            client=client, base_url="http://test",
            request=request_model, approval=approval,
        )
    assert report["passed"] is False
    assert report["requests_attempted"] == 2
    assert report["results"][1]["failures"] == ["http_status:400"]
    assert report["results"][1]["response"]["detail"] == "invalid combined answer contract"
    assert seen == 2
