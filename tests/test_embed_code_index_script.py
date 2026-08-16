from __future__ import annotations

import json

import pytest

from app.code_indexing.models import CodeIndexArtifact
from scripts import embed_code_index_artifacts


def _empty_artifact(tmp_path, *, review_status="draft"):
    artifact = CodeIndexArtifact(
        status="prepared",
        snapshot_id="fci-custom-r1-abc",
        snapshot_content_sha256="a" * 64,
        parse_generation="analysis-v1",
        analysis_policy_sha256="b" * 64,
        dependency_review_status=review_status,
        module_id="fci-custom",
        embedding_model="text-embedding-3-large",
        total_records=0,
        artifact_identity_sha256="c" * 64,
        records=(),
    )
    path = tmp_path / "prepared.json"
    path.write_text(json.dumps(artifact.model_dump(mode="json")), encoding="utf-8")
    return path


def test_dry_run_reports_external_disclosure_without_calling(tmp_path, capsys) -> None:
    path = _empty_artifact(tmp_path)
    embed_code_index_artifacts.main([
        str(path), "--output-root", str(tmp_path / "out"), "--dry-run"
    ])
    report = json.loads(capsys.readouterr().out)
    assert report["external_code_would_be_sent"] is True
    assert report["external_calls_performed"] is False


def test_paid_run_requires_sme_review_before_authorization(tmp_path) -> None:
    path = _empty_artifact(tmp_path)
    with pytest.raises(PermissionError, match="SME-reviewed"):
        embed_code_index_artifacts.main([
            str(path), "--output-root", str(tmp_path / "out")
        ])


def test_reviewed_artifact_still_requires_exact_disclosure_authorization(tmp_path) -> None:
    path = _empty_artifact(tmp_path, review_status="reviewed")
    with pytest.raises(PermissionError, match="exact authorization"):
        embed_code_index_artifacts.main([
            str(path), "--output-root", str(tmp_path / "out")
        ])
