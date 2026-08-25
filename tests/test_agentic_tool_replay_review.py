from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from app.agentic_tools.replay import build_case_replay_authorization
from app.agentic_tools.replay_review import (
    build_replay_review_ledger,
    parse_replay_review_packet,
    write_replay_review_ledger_no_overwrite,
)


CASE_ID = "uat-code-aml-offline-impact-005"


def _write_json(path: Path, value: dict) -> Path:
    path.write_text(json.dumps(value), encoding="utf-8")
    return path


def _fixture(tmp_path: Path) -> dict[str, Path]:
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text("{}\n", encoding="utf-8")
    trace = _write_json(tmp_path / "prior-trace.json", {"case_id": CASE_ID})
    uat = _write_json(tmp_path / "uat.json", {"case_id": CASE_ID})
    prior_unsigned = {
        "summary": {"semantic_acceptances": 9, "total_cases": 10},
        "decisions": [
            {
                "case_id": CASE_ID,
                "sme_verdict": "corrected",
                "observed_structural_passed": False,
            }
        ],
    }
    canonical = json.dumps(prior_unsigned, sort_keys=True, separators=(",", ":"))
    prior_unsigned["ledger_identity_sha256"] = hashlib.sha256(
        canonical.encode()
    ).hexdigest()
    prior = _write_json(tmp_path / "prior-ledger.json", prior_unsigned)
    authorization = build_case_replay_authorization(
        case_id=CASE_ID,
        reviewed_manifest=manifest,
        prior_review_ledger=prior,
        prior_trace=trace,
        local_uat_report=uat,
        answer_model="test-model",
        approval_note="One authorized replay.",
    )
    authorization_path = _write_json(tmp_path / "authorization.json", authorization)
    replay_trace = _write_json(tmp_path / "replay-trace.json", {"passed": True})
    run = _write_json(
        tmp_path / "run-state.json",
        {
            "status": "completed_pending_sme_review",
            "case_id": CASE_ID,
            "trace": replay_trace.name,
            "answer_requests_completed": 1,
            "query_embedding_requests_completed": 0,
            "automatic_openai_retries": 0,
            "structural_passed": True,
            "authorization_sha256": hashlib.sha256(
                authorization_path.read_bytes()
            ).hexdigest(),
            "authorization_identity_sha256": authorization[
                "authorization_identity_sha256"
            ],
        },
    )
    review = tmp_path / "review.md"
    review.write_text(
        "\n".join(
            [
                f"## 1. {CASE_ID}",
                "Structural result: **pass**",
                "SME verdict: accepted",
                "SME rationale:",
                "Required follow-up:",
            ]
        ),
        encoding="utf-8",
    )
    return {
        "run": run,
        "review": review,
        "authorization": authorization_path,
        "prior": prior,
        "trace": replay_trace,
    }


def test_blank_rationale_does_not_consume_follow_up_label() -> None:
    packet = parse_replay_review_packet(
        f"## 1. {CASE_ID}\nStructural result: **pass**\n"
        "SME verdict: accepted\nSME rationale:\nRequired follow-up:\n"
    )
    assert packet["rationale"] == ""
    assert packet["required_follow_up"] == ""


def test_replay_acceptance_closes_only_the_corrected_case(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    ledger = build_replay_review_ledger(
        run_state_path=paths["run"],
        review_file=paths["review"],
        authorization_path=paths["authorization"],
        prior_review_ledger_path=paths["prior"],
        reviewer="AIAgentSmith",
        global_acceptance_note="User reviewed and accepted the replay packet.",
    )
    assert ledger["decision"]["rationale_source"] == "global_acceptance_note"
    assert ledger["summary"]["case_remediation_closed"] is True
    assert ledger["summary"]["effective_semantic_acceptances_after_replay"] == 10
    assert ledger["summary"]["activation_authorized"] is False


def test_replay_review_rejects_structural_mismatch(tmp_path: Path) -> None:
    paths = _fixture(tmp_path)
    paths["review"].write_text(
        f"## 1. {CASE_ID}\nStructural result: **fail**\n"
        "SME verdict: accepted\nSME rationale: accepted\nRequired follow-up:\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="structural results differ"):
        build_replay_review_ledger(
            run_state_path=paths["run"],
            review_file=paths["review"],
            authorization_path=paths["authorization"],
            prior_review_ledger_path=paths["prior"],
            reviewer="AIAgentSmith",
            global_acceptance_note="Accepted.",
        )


def test_replay_review_refuses_overwrite(tmp_path: Path) -> None:
    path = tmp_path / "ledger.json"
    write_replay_review_ledger_no_overwrite({"value": 1}, path)
    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        write_replay_review_ledger_no_overwrite({"value": 2}, path)
