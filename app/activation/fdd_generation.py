from __future__ import annotations

import hashlib
import json
import os
import shutil
import tempfile
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path


ACTIVATION_KEYS = ("QDRANT_COLLECTION_NAME", "PROCESSED_DIR", "FDD_GENERATION")


@dataclass(frozen=True)
class FddGenerationActivationPlan:
    generation: str
    stage_manifest_path: str
    stage_manifest_sha256: str
    source_processed_directory: str
    source_artifact_sha256: str
    target_index_directory: str
    current_configuration: dict[str, str | None]
    target_configuration: dict[str, str]
    retrieval_index_path_present: bool


def build_fdd_generation_activation_plan(
    *,
    generation: str,
    stage_directory: Path,
    indexes_directory: Path,
    env_path: Path,
) -> FddGenerationActivationPlan:
    """Validate one verified FDD candidate without changing runtime state."""

    manifest_path = stage_directory / "stage_manifest.json"
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Verified stage manifest is missing: {manifest_path}")
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes.decode("utf-8"))
    if manifest.get("status") != "verified":
        raise ValueError(f"Stage is not verified: {stage_directory}")
    if manifest.get("collection_name") != generation:
        raise ValueError(
            "Stage collection does not match requested generation: "
            f"{manifest.get('collection_name')!r} != {generation!r}"
        )
    if not manifest.get("sources") or not manifest.get("qdrant", {}).get("verified_records"):
        raise ValueError("Verified stage manifest lacks source or exact-Qdrant verification evidence")

    source_processed = stage_directory / "processed"
    if not source_processed.is_dir() or not list(source_processed.glob("*.retrieval_ready.json")):
        raise FileNotFoundError(f"Verified lexical artifacts are missing: {source_processed}")

    env_values = _read_env_values(env_path)
    for key in ("QDRANT_COLLECTION_NAME", "PROCESSED_DIR"):
        if env_values.get(key) is None:
            raise ValueError(f".env must contain exactly one {key} entry before activation")
    if env_values["QDRANT_COLLECTION_NAME"] == generation:
        raise ValueError(f"Requested generation is already the active Qdrant collection: {generation}")

    target_index_directory = indexes_directory / generation
    if target_index_directory.exists():
        raise FileExistsError(
            f"Refusing to overwrite promoted index directory: {target_index_directory}"
        )

    target_processed_relative = f"data/indexes/{generation}/processed"
    target_configuration = {
        "QDRANT_COLLECTION_NAME": generation,
        "PROCESSED_DIR": target_processed_relative,
        "FDD_GENERATION": generation,
    }
    retrieval_index_path_present = env_values.get("RETRIEVAL_INDEX_PATH") is not None
    if retrieval_index_path_present:
        target_configuration["RETRIEVAL_INDEX_PATH"] = target_processed_relative

    return FddGenerationActivationPlan(
        generation=generation,
        stage_manifest_path=str(manifest_path),
        stage_manifest_sha256=_sha256_bytes(manifest_bytes),
        source_processed_directory=str(source_processed),
        source_artifact_sha256=_directory_sha256(source_processed),
        target_index_directory=str(target_index_directory),
        current_configuration={key: env_values.get(key) for key in (*ACTIVATION_KEYS, "RETRIEVAL_INDEX_PATH")},
        target_configuration=target_configuration,
        retrieval_index_path_present=retrieval_index_path_present,
    )


def apply_fdd_generation_activation(
    *,
    plan: FddGenerationActivationPlan,
    env_path: Path,
    evidence_directory: Path,
) -> Path:
    """Promote a verified index and atomically switch the paired FDD settings."""

    target_root = Path(plan.target_index_directory)
    if target_root.exists():
        raise FileExistsError(f"Refusing to overwrite promoted index directory: {target_root}")
    current_configuration = {
        key: _read_env_values(env_path).get(key)
        for key in (*ACTIVATION_KEYS, "RETRIEVAL_INDEX_PATH")
    }
    if current_configuration != plan.current_configuration:
        raise RuntimeError(".env changed after activation preflight; run a new preflight")
    source_processed = Path(plan.source_processed_directory)
    if _directory_sha256(source_processed) != plan.source_artifact_sha256:
        raise RuntimeError("Staged lexical artifacts changed after activation preflight")

    target_root.parent.mkdir(parents=True, exist_ok=True)
    evidence_directory.mkdir(parents=True, exist_ok=True)
    temporary_root = Path(tempfile.mkdtemp(prefix=f".{plan.generation}-promotion-", dir=target_root.parent))
    try:
        shutil.copytree(source_processed, temporary_root / "processed")
        if _directory_sha256(temporary_root / "processed") != plan.source_artifact_sha256:
            raise RuntimeError("Promoted lexical artifacts do not match the verified stage")
        os.replace(temporary_root, target_root)
    except Exception:
        if temporary_root.exists():
            shutil.rmtree(temporary_root)
        raise

    _atomic_update_env(env_path, plan.target_configuration)
    evidence_path = evidence_directory / (
        f"{plan.generation}-activation-{datetime.now(UTC).strftime('%Y%m%dT%H%M%S%fZ')}.json"
    )
    if evidence_path.exists():
        raise FileExistsError(f"Refusing to overwrite activation evidence: {evidence_path}")
    payload = {
        "schema_version": "fdd_generation_activation_v1",
        "activated_at_utc": datetime.now(UTC).isoformat(),
        "activation_complete": True,
        "restart_required": True,
        "rollback_configuration": plan.current_configuration,
        "plan": asdict(plan),
    }
    _atomic_write(evidence_path, json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    return evidence_path


def _read_env_values(path: Path) -> dict[str, str | None]:
    if not path.is_file():
        raise FileNotFoundError(f".env file does not exist: {path}")
    values: dict[str, str | None] = {}
    for key in (*ACTIVATION_KEYS, "RETRIEVAL_INDEX_PATH"):
        matches = [
            line.split("=", 1)[1].strip()
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip().startswith(f"{key}=")
        ]
        if len(matches) > 1:
            raise ValueError(f".env contains duplicate {key} entries")
        values[key] = matches[0] if matches else None
    return values


def _atomic_update_env(path: Path, replacements: dict[str, str]) -> None:
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    seen: set[str] = set()
    updated: list[str] = []
    for line in lines:
        stripped = line.strip()
        matched_key = next((key for key in replacements if stripped.startswith(f"{key}=")), None)
        if matched_key is None:
            updated.append(line)
            continue
        if matched_key in seen:
            raise ValueError(f".env contains duplicate {matched_key} entries")
        seen.add(matched_key)
        newline = "\r\n" if line.endswith("\r\n") else "\n"
        updated.append(f"{matched_key}={replacements[matched_key]}{newline}")
    for key, value in replacements.items():
        if key not in seen:
            if updated and not updated[-1].endswith(("\n", "\r")):
                updated[-1] += "\n"
            updated.append(f"{key}={value}\n")
    _atomic_write(path, "".join(updated))


def _atomic_write(path: Path, content: str) -> None:
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("w", encoding="utf-8", newline="") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _directory_sha256(directory: Path) -> str:
    digest = hashlib.sha256()
    for file_path in sorted(path for path in directory.rglob("*") if path.is_file()):
        relative = file_path.relative_to(directory).as_posix().encode("utf-8")
        digest.update(len(relative).to_bytes(8, "big"))
        digest.update(relative)
        with file_path.open("rb") as handle:
            for block in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(block)
    return digest.hexdigest()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()
