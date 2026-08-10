from __future__ import annotations

import ctypes
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from app.code_ingestion.plsql_models import (
    ParseDiagnostic,
    ParserWorkerRequest,
    PlSqlFileParseArtifact,
)
from app.code_ingestion.plsql_segmentation import build_fallback_segments
from app.code_ingestion.snapshot_models import CompilerContext


DEFAULT_PARSE_TIMEOUT_SECONDS = 120.0
DEFAULT_PARSE_MEMORY_LIMIT_BYTES = 1024 * 1024 * 1024


def parse_file_isolated(
    input_file: Path,
    *,
    snapshot_id: str,
    source_path: str,
    source_sha256: str,
    encoding: str,
    compiler_context: CompilerContext,
    work_root: Path,
    timeout_seconds: float = DEFAULT_PARSE_TIMEOUT_SECONDS,
    memory_limit_bytes: int = DEFAULT_PARSE_MEMORY_LIMIT_BYTES,
) -> PlSqlFileParseArtifact:
    if timeout_seconds <= 0 or memory_limit_bytes <= 0:
        raise ValueError("Parser resource boundaries must be greater than zero")
    work_root.mkdir(parents=True, exist_ok=True)
    worker_directory = Path(tempfile.mkdtemp(prefix=".plsql-worker-", dir=work_root))
    request_path = worker_directory / "request.json"
    output_path = worker_directory / "result.json"
    request = ParserWorkerRequest(
        input_file=str(input_file.resolve()),
        snapshot_id=snapshot_id,
        source_path=source_path,
        source_sha256=source_sha256,
        encoding=encoding,
        compiler_context=compiler_context,
    )
    request_path.write_text(
        json.dumps(request.model_dump(mode="json"), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    command = [
        sys.executable,
        "-m",
        "app.code_ingestion.plsql_worker",
        "--request",
        str(request_path),
        "--output",
        str(output_path),
    ]
    started = time.perf_counter()
    process = subprocess.Popen(
        command,
        cwd=Path(__file__).resolve().parents[2],
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    observed_peak = 0
    termination_code: str | None = None
    try:
        while process.poll() is None:
            elapsed = time.perf_counter() - started
            observed_memory = _process_rss_bytes(process.pid)
            observed_peak = max(observed_peak, observed_memory)
            if observed_memory > memory_limit_bytes:
                termination_code = "parser_memory_limit_exceeded"
                process.kill()
                break
            if elapsed > timeout_seconds:
                termination_code = "parser_timeout"
                process.kill()
                break
            time.sleep(0.05)
        process.wait(timeout=10)
        elapsed_ms = (time.perf_counter() - started) * 1000
        if termination_code is not None:
            return _resource_fallback(
                input_file,
                snapshot_id=snapshot_id,
                source_path=source_path,
                source_sha256=source_sha256,
                encoding=encoding,
                duration_ms=elapsed_ms,
                peak_memory_bytes=observed_peak,
                code=termination_code,
            )
        if process.returncode != 0 or not output_path.is_file():
            return _resource_fallback(
                input_file,
                snapshot_id=snapshot_id,
                source_path=source_path,
                source_sha256=source_sha256,
                encoding=encoding,
                duration_ms=elapsed_ms,
                peak_memory_bytes=observed_peak,
                code="parser_worker_exit_failure",
            )
        artifact = PlSqlFileParseArtifact.model_validate_json(
            output_path.read_text(encoding="utf-8")
        )
        return artifact.model_copy(
            update={
                "duration_ms": elapsed_ms,
                "peak_memory_bytes": max(observed_peak, artifact.peak_memory_bytes),
            }
        )
    finally:
        if process.poll() is None:
            process.kill()
            process.wait(timeout=10)
        shutil.rmtree(worker_directory, ignore_errors=True)


def _resource_fallback(
    input_file: Path,
    *,
    snapshot_id: str,
    source_path: str,
    source_sha256: str,
    encoding: str,
    duration_ms: float,
    peak_memory_bytes: int,
    code: str,
) -> PlSqlFileParseArtifact:
    raw_bytes = input_file.read_bytes()
    if hashlib.sha256(raw_bytes).hexdigest() != source_sha256:
        return PlSqlFileParseArtifact(
            snapshot_id=snapshot_id,
            source_path=source_path,
            source_sha256=source_sha256,
            parser_state="failed",
            duration_ms=duration_ms,
            peak_memory_bytes=peak_memory_bytes,
            syntax_error_count=0,
            diagnostics=(
                ParseDiagnostic(
                    stage="worker",
                    severity="error",
                    code="source_changed_during_parse",
                    message="Source hash changed after snapshot verification; no fallback was created.",
                ),
            ),
        )
    source_text = raw_bytes.decode(encoding)
    return PlSqlFileParseArtifact(
        snapshot_id=snapshot_id,
        source_path=source_path,
        source_sha256=source_sha256,
        parser_state="fallback_parse",
        duration_ms=duration_ms,
        peak_memory_bytes=peak_memory_bytes,
        syntax_error_count=0,
        segments=build_fallback_segments(source_text, source_path=source_path),
        diagnostics=(
            ParseDiagnostic(
                stage="worker",
                severity="warning",
                code=code,
                message="The isolated parser exceeded its boundary; bounded original-source chunks were retained.",
            ),
        ),
    )


def _process_rss_bytes(process_id: int) -> int:
    if os.name == "nt":
        return _windows_process_rss_bytes(process_id)
    status_path = Path(f"/proc/{process_id}/status")
    if status_path.is_file():
        for line in status_path.read_text(encoding="utf-8").splitlines():
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) * 1024
    return 0


def _windows_process_rss_bytes(process_id: int) -> int:
    class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
        _fields_ = [
            ("cb", ctypes.c_ulong),
            ("PageFaultCount", ctypes.c_ulong),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
        ]

    process_query_information = 0x0400
    process_vm_read = 0x0010
    kernel32 = ctypes.windll.kernel32
    psapi = ctypes.windll.psapi
    handle = kernel32.OpenProcess(
        process_query_information | process_vm_read,
        False,
        process_id,
    )
    if not handle:
        return 0
    try:
        counters = PROCESS_MEMORY_COUNTERS()
        counters.cb = ctypes.sizeof(counters)
        if not psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb):
            return 0
        return int(counters.WorkingSetSize)
    finally:
        kernel32.CloseHandle(handle)

