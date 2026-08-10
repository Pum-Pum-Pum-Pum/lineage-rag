from __future__ import annotations

import argparse
import hashlib
import json
import time
import tracemalloc
from pathlib import Path
from typing import Sequence

from app.code_ingestion.plsql_models import (
    ParseDiagnostic,
    ParserWorkerRequest,
    PlSqlFileParseArtifact,
)
from app.code_ingestion.plsql_parser_core import parse_plsql_source


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Internal isolated PL/SQL parser worker.")
    parser.add_argument("--request", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    request = ParserWorkerRequest.model_validate_json(args.request.read_text(encoding="utf-8"))
    started = time.perf_counter()
    tracemalloc.start()
    try:
        raw_bytes = Path(request.input_file).read_bytes()
        if hashlib.sha256(raw_bytes).hexdigest() != request.source_sha256:
            raise RuntimeError("source_hash_mismatch")
        source_text = raw_bytes.decode(request.encoding)
        artifact = parse_plsql_source(
            source_text,
            snapshot_id=request.snapshot_id,
            source_path=request.source_path,
            source_sha256=request.source_sha256,
            compiler_context=request.compiler_context,
        )
    except Exception as exc:
        artifact = PlSqlFileParseArtifact(
            snapshot_id=request.snapshot_id,
            source_path=request.source_path,
            source_sha256=request.source_sha256,
            parser_state="failed",
            duration_ms=(time.perf_counter() - started) * 1000,
            peak_memory_bytes=0,
            syntax_error_count=0,
            diagnostics=(
                ParseDiagnostic(
                    stage="worker",
                    severity="error",
                    code="parser_worker_failure",
                    message=f"Parser worker failed safely with {type(exc).__name__}.",
                ),
            ),
        )
    _, peak_memory = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    artifact = artifact.model_copy(
        update={
            "duration_ms": (time.perf_counter() - started) * 1000,
            "peak_memory_bytes": peak_memory,
        }
    )
    args.output.write_text(
        json.dumps(artifact.model_dump(mode="json"), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()

