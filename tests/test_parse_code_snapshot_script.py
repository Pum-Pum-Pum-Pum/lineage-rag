from __future__ import annotations

import json
from types import SimpleNamespace

from app.code_ingestion.plsql_models import CodeParseStageManifest
from scripts import parse_code_snapshot


def test_script_reports_local_stage_and_no_external_calls(monkeypatch, tmp_path, capsys) -> None:
    snapshot_root = tmp_path / "snapshots"
    staging_root = tmp_path / "staging"
    captured = {}

    def fake_parse(snapshot_directory, selected_staging_root, **limits):
        captured.update(
            snapshot_directory=snapshot_directory,
            staging_root=selected_staging_root,
            **limits,
        )
        return CodeParseStageManifest(
            status="complete",
            snapshot_id="fci-custom-r153-abc",
            snapshot_content_sha256="a" * 64,
            file_count=2,
            state_counts={
                "full_parse": 2,
                "segmented_parse": 0,
                "fallback_parse": 0,
                "failed": 0,
            },
            parse_artifacts=("parse/a.json", "parse/b.json"),
            retrieval_artifacts=("retrieval/a.json", "retrieval/b.json"),
            timeout_seconds=60,
            memory_limit_bytes=512 * 1024 * 1024,
        )

    monkeypatch.setattr(parse_code_snapshot, "parse_code_snapshot", fake_parse)
    monkeypatch.setattr(
        parse_code_snapshot,
        "get_settings",
        lambda: SimpleNamespace(
            code_snapshots_dir=snapshot_root,
            code_staging_dir=staging_root,
            code_parse_timeout_seconds=120,
            code_parse_memory_limit_mib=1024,
        ),
    )

    parse_code_snapshot.main(
        [
            "fci-custom-r153-abc",
            "--snapshot-root",
            str(snapshot_root),
            "--staging-root",
            str(staging_root),
            "--timeout-seconds",
            "60",
            "--memory-limit-mib",
            "512",
        ]
    )

    output = json.loads(capsys.readouterr().out)
    assert output["status"] == "complete"
    assert output["external_calls_performed"] is False
    assert captured["snapshot_directory"] == snapshot_root / "fci-custom-r153-abc"
    assert captured["timeout_seconds"] == 60
    assert captured["memory_limit_bytes"] == 512 * 1024 * 1024
