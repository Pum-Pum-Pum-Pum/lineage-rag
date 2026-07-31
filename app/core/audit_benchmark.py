from __future__ import annotations

import json
import platform
import secrets
from dataclasses import asdict, dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter_ns

from app.core.audit_journal import (
    ApiAuditEvent,
    AuditJournal,
    verify_audit_journal,
)


@dataclass(frozen=True)
class LatencySummary:
    p50_ms: float
    p95_ms: float
    p99_ms: float
    max_ms: float


@dataclass(frozen=True)
class AuditBenchmarkResult:
    schema_version: str
    benchmark_scope: str
    measured_events: int
    warmup_events: int
    append_latency: LatencySummary
    measured_elapsed_ms: float
    measured_throughput_events_per_second: float
    journal_size_bytes: int
    average_bytes_per_record: float
    verification_elapsed_ms: float
    verification_valid: bool
    python_version: str


def run_audit_journal_benchmark(
    *,
    measured_events: int = 200,
    warmup_events: int = 10,
    work_directory: str | Path | None = None,
) -> AuditBenchmarkResult:
    """Measure the local single-writer append/fsync and verification cost."""

    if measured_events <= 0:
        raise ValueError("measured_events must be greater than zero")
    if warmup_events < 0:
        raise ValueError("warmup_events must not be negative")

    parent = Path(work_directory) if work_directory is not None else None
    if parent is not None:
        parent.mkdir(parents=True, exist_ok=True)
    ephemeral_key = secrets.token_urlsafe(32)

    with TemporaryDirectory(dir=parent) as temporary_directory:
        journal_path = Path(temporary_directory) / "benchmark-audit.jsonl"
        journal = AuditJournal(journal_path, ephemeral_key)
        for index in range(warmup_events):
            journal.append(_benchmark_event(index, phase="warmup"))

        samples_ms: list[float] = []
        measured_started = perf_counter_ns()
        for index in range(measured_events):
            started = perf_counter_ns()
            journal.append(_benchmark_event(index, phase="measured"))
            samples_ms.append((perf_counter_ns() - started) / 1_000_000)
        measured_elapsed_ms = (
            perf_counter_ns() - measured_started
        ) / 1_000_000

        verification_started = perf_counter_ns()
        verification = verify_audit_journal(journal_path, ephemeral_key)
        verification_elapsed_ms = (
            perf_counter_ns() - verification_started
        ) / 1_000_000
        if not verification.valid:
            raise RuntimeError("Generated benchmark journal failed verification.")

        total_records = measured_events + warmup_events
        size_bytes = journal_path.stat().st_size
        return AuditBenchmarkResult(
            schema_version="1.0",
            benchmark_scope=(
                "Local single-writer AuditJournal.append with flush and fsync; "
                "not end-to-end API capacity"
            ),
            measured_events=measured_events,
            warmup_events=warmup_events,
            append_latency=summarize_latencies(samples_ms),
            measured_elapsed_ms=round(measured_elapsed_ms, 6),
            measured_throughput_events_per_second=round(
                measured_events / (measured_elapsed_ms / 1000),
                3,
            ),
            journal_size_bytes=size_bytes,
            average_bytes_per_record=round(size_bytes / total_records, 3),
            verification_elapsed_ms=round(verification_elapsed_ms, 6),
            verification_valid=True,
            python_version=platform.python_version(),
        )


def summarize_latencies(samples_ms: list[float]) -> LatencySummary:
    """Summarize measured latencies using linear-interpolated percentiles."""

    if not samples_ms:
        raise ValueError("At least one latency sample is required.")
    if any(sample < 0 for sample in samples_ms):
        raise ValueError("Latency samples must not be negative.")
    ordered = sorted(samples_ms)
    return LatencySummary(
        p50_ms=round(_percentile(ordered, 0.50), 6),
        p95_ms=round(_percentile(ordered, 0.95), 6),
        p99_ms=round(_percentile(ordered, 0.99), 6),
        max_ms=round(ordered[-1], 6),
    )


def write_audit_benchmark_report(
    result: AuditBenchmarkResult,
    output_path: str | Path,
) -> Path:
    """Write a content-safe local benchmark report."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(asdict(result), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return path


def _benchmark_event(index: int, *, phase: str) -> ApiAuditEvent:
    return ApiAuditEvent(
        event="api_request_completed",
        request_id=f"benchmark-{phase}-{index:06d}",
        method="GET",
        route="/benchmark",
        status_code=200,
        duration_ms=0.0,
    )


def _percentile(ordered: list[float], quantile: float) -> float:
    position = (len(ordered) - 1) * quantile
    lower_index = int(position)
    upper_index = min(lower_index + 1, len(ordered) - 1)
    fraction = position - lower_index
    return (
        ordered[lower_index]
        + (ordered[upper_index] - ordered[lower_index]) * fraction
    )
