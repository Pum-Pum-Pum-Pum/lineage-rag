from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from scripts import build_code_snapshot


def test_validate_only_performs_no_snapshot_write(monkeypatch, tmp_path: Path, capsys) -> None:
    intake = tmp_path / "intake"
    source = intake / "source"
    source.mkdir(parents=True)
    (source / "pkg.sql").write_text("select 1 from dual;\n", encoding="utf-8")
    (intake / "snapshot_request.json").write_text(
        json.dumps(
            {
                "module_set": "fci-custom",
                "svn_revision": "123",
                "application_build": "14.7.1",
                "reviewer": "SME",
            }
        ),
        encoding="utf-8",
    )
    snapshot_root = tmp_path / "snapshots"
    monkeypatch.setattr(
        build_code_snapshot,
        "get_settings",
        lambda: SimpleNamespace(code_snapshots_dir=snapshot_root),
    )

    build_code_snapshot.main([str(intake), "--snapshot-root", str(snapshot_root), "--validate-only"])

    result = json.loads(capsys.readouterr().out)
    assert result["status"] == "valid"
    assert result["writes_performed"] is False
    assert result["external_calls_performed"] is False
    assert not snapshot_root.exists()

