from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
from pathlib import Path

from app.code_ingestion.code_retrieval_artifact import build_code_retrieval_artifact
from app.code_ingestion.plsql_isolation import parse_file_isolated
from app.code_ingestion.plsql_models import CodeParseStageManifest
from app.code_ingestion.snapshot_builder import load_snapshot_manifest


PARSER_GENERATION_DIRECTORY = "plsql_antlr_4_13_2_v1"


def parse_code_snapshot(
    snapshot_directory: Path,
    staging_root: Path,
    *,
    timeout_seconds: float = 120.0,
    memory_limit_bytes: int = 1024 * 1024 * 1024,
) -> CodeParseStageManifest:
    """Parse one verified immutable snapshot and atomically publish local artifacts."""

    if timeout_seconds <= 0 or memory_limit_bytes <= 0:
        raise ValueError("Parser resource boundaries must be greater than zero")
    snapshot = load_snapshot_manifest(snapshot_directory, verify_sources=True)
    target = staging_root / snapshot.snapshot_id / PARSER_GENERATION_DIRECTORY
    if target.exists():
        raise FileExistsError(f"Parse generation already exists and will not be overwritten: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".code-parse-", dir=target.parent))
    worker_root = temporary / ".workers"
    parse_paths: list[str] = []
    retrieval_paths: list[str] = []
    state_counts = {state: 0 for state in ("full_parse", "segmented_parse", "fallback_parse", "failed")}
    try:
        for entry in snapshot.files:
            if entry.source_handler not in {"plsql", "ddl"}:
                raise RuntimeError(f"No parser implementation for source handler: {entry.source_handler}")
            source_file = snapshot_directory / snapshot.source_directory_name / entry.path
            raw_bytes = source_file.read_bytes()
            observed_hash = hashlib.sha256(raw_bytes).hexdigest()
            if observed_hash != entry.sha256:
                raise RuntimeError(f"Immutable source changed before parsing: {entry.path}")
            source_text = raw_bytes.decode(entry.encoding)
            parsed = parse_file_isolated(
                source_file,
                snapshot_id=snapshot.snapshot_id,
                source_path=entry.path,
                source_sha256=entry.sha256,
                encoding=entry.encoding,
                compiler_context=snapshot.request.compiler_context,
                work_root=worker_root,
                timeout_seconds=timeout_seconds,
                memory_limit_bytes=memory_limit_bytes,
            )
            retrieval = build_code_retrieval_artifact(
                parsed,
                source_text,
                verified_source_sha256=observed_hash,
            )
            stem = _artifact_stem(entry.path)
            parse_relative = f"parse/{stem}.json"
            retrieval_relative = f"retrieval/{stem}.json"
            _write_json(temporary / parse_relative, parsed.model_dump(mode="json"))
            _write_json(temporary / retrieval_relative, retrieval.model_dump(mode="json"))
            parse_paths.append(parse_relative)
            retrieval_paths.append(retrieval_relative)
            state_counts[parsed.parser_state] += 1

        status = _stage_status(state_counts)
        manifest = CodeParseStageManifest(
            status=status,
            snapshot_id=snapshot.snapshot_id,
            snapshot_content_sha256=snapshot.snapshot_content_sha256,
            file_count=len(snapshot.files),
            state_counts=state_counts,
            parse_artifacts=tuple(parse_paths),
            retrieval_artifacts=tuple(retrieval_paths),
            timeout_seconds=timeout_seconds,
            memory_limit_bytes=memory_limit_bytes,
        )
        _write_json(temporary / "parse_stage_manifest.json", manifest.model_dump(mode="json"))
        shutil.rmtree(worker_root, ignore_errors=True)
        temporary.replace(target)
        return manifest
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def _stage_status(state_counts: dict[str, int]) -> str:
    if state_counts["failed"]:
        return "failed"
    if state_counts["segmented_parse"] or state_counts["fallback_parse"]:
        return "complete_with_degradation"
    return "complete"


def _artifact_stem(source_path: str) -> str:
    readable = Path(source_path).name.replace(".", "_")[:80]
    suffix = hashlib.sha256(source_path.encode("utf-8")).hexdigest()[:12]
    return f"{readable}-{suffix}"


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True),
        encoding="utf-8",
    )
