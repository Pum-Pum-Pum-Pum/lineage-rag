from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

from app.code_ingestion.plsql_isolation import parse_file_isolated
from app.code_ingestion.plsql_parser_core import parse_plsql_segments_only
from app.code_ingestion.snapshot_models import CompilerContext


def _write_source(path: Path, text: str) -> str:
    path.write_text(text, encoding="utf-8")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _parse(path: Path, source_hash: str, work_root: Path, **limits):
    return parse_file_isolated(
        path,
        snapshot_id="snapshot-1",
        source_path=path.name,
        source_sha256=source_hash,
        encoding="utf-8",
        compiler_context=CompilerContext(),
        work_root=work_root,
        **limits,
    )


def test_isolated_worker_parses_standalone_procedure(tmp_path: Path) -> None:
    source = tmp_path / "customer.prc"
    source_hash = _write_source(
        source,
        "CREATE OR REPLACE PROCEDURE update_customer IS BEGIN NULL; END; /\n",
    )

    artifact = _parse(source, source_hash, tmp_path / "workers")

    assert artifact.parser_state == "full_parse"
    assert [(node.node_kind, node.display_name) for node in artifact.extracted_nodes] == [
        ("procedure", "update_customer")
    ]
    assert artifact.duration_ms > 0
    assert artifact.peak_memory_bytes > 0


def test_timeout_fails_over_to_bounded_original_source_chunks(tmp_path: Path) -> None:
    source = tmp_path / "customer.sql"
    source_hash = _write_source(source, "BEGIN NULL; END; /\n")

    artifact = _parse(
        source,
        source_hash,
        tmp_path / "workers",
        timeout_seconds=0.000001,
    )

    assert artifact.parser_state == "fallback_parse"
    assert artifact.segments[0].segment_kind == "fallback_chunk"
    assert [diagnostic.code for diagnostic in artifact.diagnostics] == [
        "segmented_parser_timeout_after_full_parser_timeout"
    ]


def test_memory_limit_fails_over_without_dropping_source(tmp_path: Path) -> None:
    source = tmp_path / "customer.sql"
    source_hash = _write_source(source, "BEGIN NULL; END; /\n")

    artifact = _parse(
        source,
        source_hash,
        tmp_path / "workers",
        memory_limit_bytes=1,
    )

    assert artifact.parser_state == "fallback_parse"
    with source.open("r", encoding="utf-8", newline="") as handle:
        exact_text = handle.read()
    assert artifact.segments[0].source_map.end_offset == len(exact_text)
    assert [diagnostic.code for diagnostic in artifact.diagnostics] == [
        "segmented_parser_memory_limit_exceeded_after_full_parser_memory_limit_exceeded"
    ]


def test_changed_source_hash_fails_closed(tmp_path: Path) -> None:
    source = tmp_path / "customer.sql"
    _write_source(source, "BEGIN NULL; END; /\n")

    artifact = _parse(source, "0" * 64, tmp_path / "workers")

    assert artifact.parser_state == "failed"
    assert not artifact.segments
    assert artifact.diagnostics[0].code == "parser_worker_failure"


def test_file_above_five_mib_is_bounded_and_preserved_on_timeout(tmp_path: Path) -> None:
    source = tmp_path / "large_package.sql"
    source_hash = _write_source(source, "-- " + ("x" * (5 * 1024 * 1024 + 1)) + "\n")

    artifact = _parse(
        source,
        source_hash,
        tmp_path / "workers",
        timeout_seconds=0.000001,
    )

    with source.open("r", encoding="utf-8", newline="") as handle:
        exact_text = handle.read()
    assert artifact.parser_state == "fallback_parse"
    assert artifact.segments[-1].source_map.end_offset == len(exact_text)
    assert artifact.diagnostics[0].code == "segmented_parser_timeout_after_full_parser_timeout"


def test_full_resource_boundary_runs_separate_segmented_worker(monkeypatch, tmp_path: Path) -> None:
    from app.code_ingestion import plsql_isolation

    source_text = "PROCEDURE p IS BEGIN NULL; END;\n"
    source = tmp_path / "customer.prc"
    source_hash = _write_source(source, source_text)
    segmented = parse_plsql_segments_only(
        source_text,
        snapshot_id="snapshot-1",
        source_path=source.name,
        source_sha256=source_hash,
    )
    modes = []

    def fake_run(request, **kwargs):
        modes.append(request.parse_mode)
        if request.parse_mode == "full":
            return plsql_isolation._WorkerResult(None, "parser_timeout", 1000, 100)
        return plsql_isolation._WorkerResult(segmented, None, 200, 200)

    monkeypatch.setattr(plsql_isolation, "_run_worker", fake_run)

    artifact = _parse(source, source_hash, tmp_path / "workers")

    assert modes == ["full", "segmented"]
    assert artifact.parser_state == "segmented_parse"
    assert artifact.duration_ms == 1200
    assert artifact.peak_memory_bytes == 200
    assert artifact.diagnostics[0].code == "full_parse_resource_boundary_segmented_retry"


@pytest.mark.parametrize(
    ("timeout_seconds", "memory_limit_bytes"),
    [(0, 1024), (1, 0)],
)
def test_resource_boundaries_must_be_positive(
    tmp_path: Path,
    timeout_seconds: float,
    memory_limit_bytes: int,
) -> None:
    source = tmp_path / "customer.sql"
    source_hash = _write_source(source, "BEGIN NULL; END; /\n")

    with pytest.raises(ValueError, match="greater than zero"):
        _parse(
            source,
            source_hash,
            tmp_path / "workers",
            timeout_seconds=timeout_seconds,
            memory_limit_bytes=memory_limit_bytes,
        )
