import json

import pytest

from app.code_ingestion import code_parsing_pipeline as pipeline
from app.code_ingestion.snapshot_builder import build_code_snapshot
from app.code_indexing.contract import build_code_index_artifact
from test_code_parsing_pipeline import _build_snapshot_files


def test_new_snapshot_reuses_unchanged_parse_with_new_ids_and_fresh_analysis(tmp_path, monkeypatch):
    source = "CREATE OR REPLACE PACKAGE BODY demo AS PROCEDURE work IS BEGIN NULL; END; END demo; /"
    before_change = "CREATE OR REPLACE PROCEDURE changed IS BEGIN NULL; END; /"
    base_dir, base = _build_snapshot_files(tmp_path, {"demo.sql": source, "changed.sql": before_change})
    stage_root = tmp_path / "stage"
    first = pipeline.parse_code_snapshot(base_dir, stage_root)
    base_stage = stage_root / base.snapshot_id / first.parser_generation
    intake = tmp_path / "next"
    (intake / "source").mkdir(parents=True)
    (intake / "source/demo.sql").write_text(source)
    (intake / "source/changed.sql").write_text(before_change.replace("NULL;", "demo.work;"))
    (intake / "source/added.sql").write_text("CREATE OR REPLACE PROCEDURE added IS BEGIN demo.work; END; /")
    request = base.request.model_dump(mode="json")
    request.update(svn_revision="154", base_snapshot_id=base.snapshot_id)
    (intake / "snapshot_request.json").write_text(json.dumps(request))
    target = build_code_snapshot(intake, base_dir.parent)
    parse = pipeline.parse_file_isolated
    calls = []

    def checked(*args, **kwargs):
        calls.append(kwargs["source_path"])
        assert kwargs["source_path"] != "demo.sql", "Unchanged source must not be reparsed"
        return parse(*args, **kwargs)

    monkeypatch.setattr(pipeline, "parse_file_isolated", checked)
    second = pipeline.parse_code_snapshot(base_dir.parent / target.snapshot_id, stage_root,
                                         base_generation_directory=base_stage)
    target_stage = stage_root / target.snapshot_id / second.parser_generation
    old = build_code_index_artifact(base_stage, embedding_model="text-embedding-3-large")
    new = build_code_index_artifact(target_stage, embedding_model="text-embedding-3-large")
    same = [r for r in new.records if r.source_path == "demo.sql"]
    assert sorted(calls) == ["added.sql", "changed.sql"]
    assert second.reused_parse_file_count == 1
    assert {r.cache_key for r in same} == {r.cache_key for r in old.records if r.source_path == "demo.sql"}
    assert {r.unit_id for r in same}.isdisjoint({r.unit_id for r in old.records})
    assert all(r.snapshot_id == target.snapshot_id for r in new.records)
    assert len(second.analysis_artifacts) == 3
    # Recovery can retain all newly parsed files while rebinding the baseline.
    monkeypatch.setattr(pipeline, "parse_file_isolated", lambda *a, **k: pytest.fail("Recovery must reuse all three files"))
    recovered = pipeline.parse_code_snapshot(base_dir.parent / target.snapshot_id, tmp_path / "recovery",
        reuse_generation_directory=target_stage, base_generation_directory=base_stage)
    assert recovered.reused_parse_file_count == 3
    recovered_index = build_code_index_artifact(tmp_path / "recovery" / target.snapshot_id / recovered.parser_generation,
                                              embedding_model="text-embedding-3-large")
    assert recovered_index == new
    # A resource-contract change cannot silently reuse old parsing.
    with pytest.raises(ValueError, match="timeout_seconds"):
        pipeline.parse_code_snapshot(base_dir.parent / target.snapshot_id, tmp_path / "mismatch",
            base_generation_directory=base_stage, timeout_seconds=121)
