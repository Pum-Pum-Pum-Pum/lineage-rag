from __future__ import annotations

import json
from pathlib import Path

import pytest

from app.agentic_tools.replay import (
    build_case_replay_authorization,
    validate_case_replay_authorization,
    write_authorization_no_overwrite,
)


def _source_files(tmp_path: Path) -> dict[str, Path]:
    paths = {}
    for name in ("manifest", "ledger", "trace", "uat"):
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps({"name": name}), encoding="utf-8")
        paths[name] = path
    return paths


def _authorization(tmp_path: Path) -> dict:
    paths = _source_files(tmp_path)
    return build_case_replay_authorization(
        case_id="uat-code-aml-offline-impact-005",
        reviewed_manifest=paths["manifest"],
        prior_review_ledger=paths["ledger"],
        prior_trace=paths["trace"],
        local_uat_report=paths["uat"],
        answer_model="test-model",
        approval_note="One answer request, no embeddings, and no retries.",
    )


def test_case_replay_authorization_is_exactly_bounded(tmp_path: Path) -> None:
    value = _authorization(tmp_path)

    validate_case_replay_authorization(value)

    assert value["case_id"] == "uat-code-aml-offline-impact-005"
    assert value["maximum_answer_requests"] == 1
    assert value["maximum_query_embedding_requests"] == 0
    assert value["automatic_retries"] == 0
    assert value["paid_use_authorized"] is True
    assert value["internal_evidence_disclosure_authorized"] is True


@pytest.mark.parametrize(
    ("field", "replacement"),
    [
        ("case_id", "another-case"),
        ("maximum_answer_requests", 2),
        ("maximum_query_embedding_requests", 1),
        ("automatic_retries", 1),
        ("paid_use_authorized", False),
        ("internal_evidence_disclosure_authorized", False),
    ],
)
def test_case_replay_authorization_rejects_tampering(
    tmp_path: Path, field: str, replacement: object
) -> None:
    value = _authorization(tmp_path)
    value[field] = replacement

    with pytest.raises((ValueError, PermissionError), match="identity mismatch"):
        validate_case_replay_authorization(value)


def test_case_replay_authorization_refuses_overwrite(tmp_path: Path) -> None:
    value = _authorization(tmp_path)
    output = tmp_path / "authorization.json"
    write_authorization_no_overwrite(value, output)

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        write_authorization_no_overwrite(value, output)


def test_case_replay_authorization_requires_approval_note(tmp_path: Path) -> None:
    paths = _source_files(tmp_path)

    with pytest.raises(ValueError, match="approval note"):
        build_case_replay_authorization(
            case_id="uat-code-aml-offline-impact-005",
            reviewed_manifest=paths["manifest"],
            prior_review_ledger=paths["ledger"],
            prior_trace=paths["trace"],
            local_uat_report=paths["uat"],
            answer_model="test-model",
            approval_note=" ",
        )
