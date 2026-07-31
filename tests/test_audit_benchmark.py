import json
from pathlib import Path

import pytest

from app.core.audit_benchmark import (
    AuditBenchmarkResult,
    LatencySummary,
    run_audit_journal_benchmark,
    summarize_latencies,
    write_audit_benchmark_report,
)


def test_latency_summary_uses_stable_interpolated_percentiles() -> None:
    summary = summarize_latencies([4.0, 1.0, 3.0, 2.0])

    assert summary == LatencySummary(
        p50_ms=2.5,
        p95_ms=3.85,
        p99_ms=3.97,
        max_ms=4.0,
    )


@pytest.mark.parametrize(
    ("samples", "message"),
    [
        ([], "At least one"),
        ([1.0, -1.0], "must not be negative"),
    ],
)
def test_latency_summary_rejects_invalid_samples(
    samples: list[float],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        summarize_latencies(samples)


def test_benchmark_measures_real_fsync_writes_and_verifies_chain(
    tmp_path: Path,
) -> None:
    result = run_audit_journal_benchmark(
        measured_events=5,
        warmup_events=1,
        work_directory=tmp_path,
    )

    assert result.measured_events == 5
    assert result.warmup_events == 1
    assert result.append_latency.p50_ms >= 0
    assert result.measured_throughput_events_per_second > 0
    assert result.journal_size_bytes > 0
    assert result.average_bytes_per_record > 0
    assert result.verification_elapsed_ms >= 0
    assert result.verification_valid is True
    assert "not end-to-end API capacity" in result.benchmark_scope
    assert not list(tmp_path.glob("*/benchmark-audit.jsonl"))


@pytest.mark.parametrize(
    ("measured_events", "warmup_events", "message"),
    [
        (0, 0, "greater than zero"),
        (1, -1, "must not be negative"),
    ],
)
def test_benchmark_rejects_invalid_workloads(
    tmp_path: Path,
    measured_events: int,
    warmup_events: int,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        run_audit_journal_benchmark(
            measured_events=measured_events,
            warmup_events=warmup_events,
            work_directory=tmp_path,
        )


def test_report_contains_metrics_but_no_hmac_key_or_request_content(
    tmp_path: Path,
) -> None:
    result = AuditBenchmarkResult(
        schema_version="1.0",
        benchmark_scope="unit-test",
        measured_events=2,
        warmup_events=1,
        append_latency=LatencySummary(1.0, 2.0, 3.0, 4.0),
        measured_elapsed_ms=5.0,
        measured_throughput_events_per_second=400.0,
        journal_size_bytes=900,
        average_bytes_per_record=300.0,
        verification_elapsed_ms=0.5,
        verification_valid=True,
        python_version="3.12.0",
    )

    output = write_audit_benchmark_report(
        result,
        tmp_path / "reports" / "benchmark.json",
    )
    payload = json.loads(output.read_text(encoding="utf-8"))

    assert payload["append_latency"]["p99_ms"] == 3.0
    serialized = output.read_text(encoding="utf-8").lower()
    assert "hmac_key" not in serialized
    assert "query" not in serialized
