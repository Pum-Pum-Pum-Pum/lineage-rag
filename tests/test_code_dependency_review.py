from __future__ import annotations

import json
from pathlib import Path

from app.code_ingestion.code_parsing_pipeline import parse_code_snapshot
from app.code_ingestion.dependency_review import (
    build_dependency_review_packet,
    render_dependency_review_markdown,
)
from app.code_ingestion.snapshot_builder import build_code_snapshot


def test_review_packet_groups_ambiguities_and_does_not_filter_tables(tmp_path: Path) -> None:
    intake = tmp_path / "intake"
    (intake / "source").mkdir(parents=True)
    (intake / "source/run_review_custom.sql").write_text(
        """CREATE OR REPLACE PROCEDURE run_review_custom(p_sql VARCHAR2) IS
  l_id NUMBER;
BEGIN
  pkg_visible_custom.do_work();
  pkg_kernel.do_work();
  unknown_local();
  SELECT id INTO l_id FROM app.business_table;
  EXECUTE IMMEDIATE p_sql;
END;
/
""",
        encoding="utf-8",
    )
    (intake / "snapshot_request.json").write_text(
        json.dumps({
            "module_set": "fci-custom",
            "svn_revision": "1",
            "application_build": "test",
            "reviewer": "test",
        }),
        encoding="utf-8",
    )
    snapshot_root = tmp_path / "snapshots"
    snapshot = build_code_snapshot(intake, snapshot_root)
    stage_root = tmp_path / "staging"
    manifest = parse_code_snapshot(snapshot_root / snapshot.snapshot_id, stage_root)
    stage = stage_root / snapshot.snapshot_id / manifest.parser_generation

    first = build_dependency_review_packet(snapshot_root / snapshot.snapshot_id, stage)
    second = build_dependency_review_packet(snapshot_root / snapshot.snapshot_id, stage)

    assert first == second
    by_target = {case.target_canonical_name: case for case in first.cases}
    assert "PKG_VISIBLE_CUSTOM.DO_WORK" in by_target
    assert by_target["PKG_KERNEL.DO_WORK"].proposed_resolution_state == "unresolved"
    assert "UNKNOWN_LOCAL" in by_target
    assert "P_SQL" in by_target
    assert "APP.BUSINESS_TABLE" not in by_target
    assert "SME verdict" in render_dependency_review_markdown(first)
