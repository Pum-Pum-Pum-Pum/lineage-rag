"""Controlled materialisation and no-cost embedding planning for FDD updates."""
from __future__ import annotations

import hashlib
import json
import shutil
from collections import defaultdict
from pathlib import Path

import tiktoken

from app.embeddings.embedding_cache import load_embedding_cache
from app.embeddings.embedding_contract import build_embedding_batch_contract
from app.ingestion.chunker import chunk_normalized_artifact
from app.ingestion.docx_ingestion_artifact import ingest_docx_file
from app.ingestion.normalized_artifact import build_normalized_artifact
from app.ingestion.retrieval_ready_artifact import build_retrieval_ready_artifact
from app.ingestion.table_chunker import chunk_tables_from_artifact
from app.knowledge_updates.models import FddUpdatePlan


def materialize_sources(*, plan: FddUpdatePlan, baseline_directory: Path,
                        update_directory: Path, destination: Path) -> list[Path]:
    """Copy exactly the planned corpus to an immutable run-owned intake tree."""
    if destination.exists():
        raise FileExistsError(f"FDD intake already exists: {destination}")
    baseline = {p.relative_to(baseline_directory).as_posix(): p for p in baseline_directory.rglob("*.docx") if p.is_file()}
    updates = {p.relative_to(update_directory).as_posix(): p for p in update_directory.rglob("*.docx") if p.is_file()}
    destination.mkdir(parents=True)
    copied = []
    for source in plan.target_sources:
        candidate = updates.get(source.path) or baseline.get(source.path)
        if candidate is None:
            raise FileNotFoundError(f"Planned FDD source is no longer available: {source.path}")
        digest = hashlib.sha256(candidate.read_bytes()).hexdigest()
        if digest != source.sha256:
            raise RuntimeError(f"FDD source changed during materialization: {source.path}")
        target = destination / source.path
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(candidate, target)
        if hashlib.sha256(target.read_bytes()).hexdigest() != source.sha256:
            raise RuntimeError(f"FDD copy verification failed: {source.path}")
        copied.append(target)
    return copied


def plan_fdd_embeddings(*, source_directory: Path, cache_directories: list[Path],
                        embedding_model: str, artifact_version: str,
                        usd_per_million_tokens: str, pricing_basis: str) -> dict:
    """Prepare exact FDD embedding inputs without writing, indexing, or calling OpenAI."""
    cache = {}
    for directory in cache_directories:
        for key, record in load_embedding_cache(directory).items():
            previous = cache.get(key)
            if previous is not None and previous.vector != record.vector:
                raise RuntimeError(f"Conflicting FDD embedding cache key: {key}")
            cache[key] = record
    encoding = tiktoken.get_encoding("cl100k_base")
    records = {}
    paths = defaultdict(set)
    for source in sorted(source_directory.rglob("*.docx")):
        raw = ingest_docx_file(source)
        normalized = build_normalized_artifact(raw)
        ready = build_retrieval_ready_artifact(normalized, chunk_normalized_artifact(normalized), chunk_tables_from_artifact(normalized))
        batch = build_embedding_batch_contract(ready, embedding_model=embedding_model, artifact_version=artifact_version)
        for record in batch.records:
            prior = records.get(record.cache_key)
            if prior is not None and prior.text != record.text:
                raise RuntimeError(f"FDD cache-key collision: {record.cache_key}")
            records[record.cache_key] = record
            paths[record.cache_key].add(source.relative_to(source_directory).as_posix())
    missing = []
    for key, record in sorted(records.items()):
        if key in cache:
            continue
        tokens = len(encoding.encode(record.text, disallowed_special=()))
        if not 1 <= tokens <= 8191:
            raise ValueError(f"FDD embedding input outside token limit: {record.unit_id}")
        missing.append({"cache_key": key, "text_sha256": record.content_hash,
                        "source_paths": sorted(paths[key]), "token_upper_bound": tokens})
    from decimal import Decimal
    rate = Decimal(str(usd_per_million_tokens))
    if not rate.is_finite() or rate <= 0:
        raise ValueError("A positive FDD embedding price is required")
    value = {"schema_version": "knowledge_fdd_embedding_request_v1", "model": embedding_model,
             "input_version": artifact_version, "endpoint": "https://api.openai.com/v1",
             "caches": {str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest() for path in cache_directories if path.exists()},
             "unique_missing_inputs": len(missing), "missing": missing,
             "token_upper_bound": sum(item["token_upper_bound"] for item in missing),
             "usd_per_million_tokens": str(rate), "pricing_basis": pricing_basis,
             "max_retries": 0, "batch_size": 32}
    value["cost_upper_bound_usd"] = str(rate * value["token_upper_bound"] / 1_000_000)
    value["request_hash"] = hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    return value
