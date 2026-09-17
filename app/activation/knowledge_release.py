"""Hash-bound promotion of one compatible FDD, code, and lineage combination."""
from __future__ import annotations

import hashlib
import json
import os
from datetime import UTC, datetime
from pathlib import Path

from app.activation.code_generation import digest, local, runtime_files, sha
from app.activation.code_modes import ActivationApproval, _atomic_write, approval_identity
from app.activation.fdd_generation import build_fdd_generation_activation_plan
from app.knowledge_updates.models import KnowledgeReleaseManifest


KEYS = ("QDRANT_COLLECTION_NAME", "PROCESSED_DIR", "RETRIEVAL_INDEX_PATH", "FDD_GENERATION",
        "CODE_MODES_ENABLED", "CODE_INDEX_ARTIFACT_PATH", "CODE_ANALYSIS_DIRECTORY",
        "CODE_QDRANT_COLLECTION_NAME", "FDD_CODE_LINEAGE_ARTIFACT_PATH")


def _values(text: str) -> dict[str, str | None]:
    result = {}
    for key in KEYS:
        hits = [line.split("=", 1)[1].strip() for line in text.splitlines() if line.strip().startswith(key + "=")]
        if len(hits) > 1:
            raise ValueError(f"Duplicate release configuration key: {key}")
        result[key] = hits[0] if hits else None
    return result


def _render(text: str, values: dict[str, str]) -> str:
    if set(values) != set(KEYS):
        raise ValueError("A knowledge release must set every FDD and code configuration key")
    _values(text)
    lines, found = [], set()
    newline = "\r\n" if "\r\n" in text else "\n"
    for line in text.splitlines(keepends=True):
        key = line.split("=", 1)[0].strip() if "=" in line else ""
        if key not in values:
            lines.append(line)
        else:
            found.add(key)
            lines.append(f"{key}={values[key]}{newline}")
    for key in KEYS:
        if key not in found:
            lines.append(f"{key}={values[key]}{newline}")
    return "".join(lines)


def prepare(*, root: Path, manifest_path: Path, requested_by: str) -> dict:
    root = root.resolve()
    manifest = KnowledgeReleaseManifest.model_validate_json(manifest_path.read_bytes())
    if not requested_by.strip():
        raise ValueError("Requester is required")
    evidence = {str(local(root, manifest_path).relative_to(root)): sha(manifest_path)}
    for value in (manifest.code_artifact_path, manifest.lineage_path, manifest.fdd_processed_directory):
        path = local(root, value)
        if path.is_file():
            evidence[str(path.relative_to(root))] = sha(path)
        elif path.is_dir():
            for child in path.rglob("*.retrieval_ready.json"):
                evidence[str(child.relative_to(root))] = sha(child)
        else:
            raise FileNotFoundError(f"Release evidence missing: {path}")
    # Verify both staged collections before a human is asked to approve a
    # promotion.  A collection collision or incompatible embedding space is a
    # preparation error, never something activation is allowed to repair.
    from app.code_indexing.contract import load_code_index_artifact
    from app.code_indexing.qdrant import verify_code_collection
    from qdrant_client import QdrantClient
    artifact = load_code_index_artifact(local(root, manifest.code_artifact_path))
    code_client = QdrantClient(path=str(root / "data/qdrant_code_local"))
    try:
        verify_code_collection(code_client, collection_name=manifest.code_collection, artifact=artifact)
    finally:
        code_client.close()
    fdd_client = QdrantClient(path=str(root / "data/qdrant_local"))
    try:
        if not fdd_client.collection_exists(manifest.fdd_collection):
            raise ValueError(f"FDD collection is missing: {manifest.fdd_collection}")
        info = fdd_client.get_collection(manifest.fdd_collection)
        if not info.points_count or info.config.params.vectors.size != artifact.vector_dimension:
            raise ValueError("FDD collection is empty or uses an incompatible vector dimension")
    finally:
        fdd_client.close()
    env = root / ".env"
    before = env.read_text(encoding="utf-8")
    source_processed = local(root, manifest.fdd_processed_directory)
    if manifest.fdd_stage_directory:
        stage = local(root, manifest.fdd_stage_directory)
        plan = build_fdd_generation_activation_plan(
            generation=manifest.fdd_generation, stage_directory=stage,
            indexes_directory=root / "data/indexes", env_path=env,
        )
        # Its FDD-only preflight expects the stage collection to be new, which
        # is exactly the coordinated-release requirement.  The joint switch
        # performs the same verified copy before changing all settings.
        target_processed = f"data/indexes/{manifest.fdd_generation}/processed"
        fdd_promotion = {
            "stage_directory": str(stage.relative_to(root)),
            "source_processed_directory": str(source_processed.relative_to(root)),
            "source_processed_sha256": plan.source_artifact_sha256,
            "target_index_directory": str(Path(plan.target_index_directory).relative_to(root)),
        }
    else:
        target_processed = manifest.fdd_processed_directory
        fdd_promotion = None
    target = {"QDRANT_COLLECTION_NAME": manifest.fdd_collection,
              "PROCESSED_DIR": target_processed,
              "RETRIEVAL_INDEX_PATH": target_processed,
              "FDD_GENERATION": manifest.fdd_generation,
              "CODE_MODES_ENABLED": "true", "CODE_INDEX_ARTIFACT_PATH": manifest.code_artifact_path,
              "CODE_ANALYSIS_DIRECTORY": manifest.code_analysis_directory,
              "CODE_QDRANT_COLLECTION_NAME": manifest.code_collection,
              "FDD_CODE_LINEAGE_ARTIFACT_PATH": manifest.lineage_path}
    value = {"schema_version": "knowledge_release_promotion_request_v1", "requested_by": requested_by,
             "manifest_path": str(local(root, manifest_path).relative_to(root)),
             "manifest_identity_sha256": manifest.manifest_identity_sha256,
             "target_configuration": target, "rollback_configuration": {**_values(before), "CODE_MODES_ENABLED": "false"},
             "before_env_sha256": sha(env), "target_env_sha256": hashlib.sha256(_render(before, target).encode()).hexdigest(),
             "runtime_files": runtime_files(root), "evidence_files": evidence, "fdd_promotion": fdd_promotion,
             "restart_required": True, "activation_complete": False, "authorized_paid_requests": 0}
    value["request_identity_sha256"] = digest(value)
    return value


def switch(*, root: Path, request: dict, approval: ActivationApproval, action: str, apply: bool = False) -> dict:
    if action not in {"activate", "rollback"}:
        raise ValueError("Unknown release action")
    if digest({k: v for k, v in request.items() if k != "request_identity_sha256"}) != request["request_identity_sha256"]:
        raise ValueError("Release promotion request integrity failed")
    if approval.request_identity_sha256 != request["request_identity_sha256"] or approval.decision != "approved" or action not in approval.allowed_actions or approval_identity(approval.model_dump(mode="json")) != approval.approval_identity_sha256:
        raise PermissionError("Exact knowledge-release approval is required")
    root, env = root.resolve(), root.resolve() / ".env"
    expected = request["before_env_sha256"] if action == "activate" else request["target_env_sha256"]
    if sha(env) != expected:
        raise ValueError(".env differs from the recorded release state")
    if action == "activate" and runtime_files(root) != request["runtime_files"]:
        raise ValueError("Runtime changed; create a fresh promotion request")
    target = request["target_configuration"] if action == "activate" else request["rollback_configuration"]
    result = {"action": action, "request_identity_sha256": request["request_identity_sha256"], "applied": apply,
              "restart_required": True, "activation_complete": False, "external_api_calls": 0}
    if apply:
        lock = root / ".knowledge-release-promotion.lock"
        handle = lock.open("x")
        try:
            with handle:
                if sha(env) != expected:
                    raise ValueError(".env changed during release promotion")
                receipt = root / "data/exports/activation" / ("knowledge-release-" + datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ"))
                receipt.parent.mkdir(parents=True, exist_ok=True)
                _write_new(receipt.with_suffix(".intent.json"), {**result, "applied": False})
                if action == "activate" and request.get("fdd_promotion"):
                    _promote_fdd_lexical(root, request["fdd_promotion"])
                _atomic_write(env, _render(env.read_text(encoding="utf-8"), target))
                result["applied"] = True
                _write_new(receipt.with_suffix(".result.json"), result)
        finally:
            lock.unlink(missing_ok=True)
    return result


def _promote_fdd_lexical(root: Path, value: dict) -> None:
    """Copy the verified staged lexical generation without overwriting a target."""
    source = local(root, value["source_processed_directory"])
    target = local(root, value["target_index_directory"])
    if target.exists():
        raise FileExistsError(f"Refusing to overwrite promoted FDD index directory: {target}")
    from app.activation.fdd_generation import _directory_sha256
    if _directory_sha256(source) != value["source_processed_sha256"]:
        raise ValueError("Staged FDD lexical artifacts changed after promotion request")
    import shutil
    import tempfile
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=".knowledge-fdd-", dir=target.parent))
    try:
        shutil.copytree(source, temporary / "processed")
        if _directory_sha256(temporary / "processed") != value["source_processed_sha256"]:
            raise RuntimeError("Promoted FDD lexical artifacts failed exact verification")
        os.replace(temporary, target)
    except Exception:
        if temporary.exists(): shutil.rmtree(temporary)
        raise


def _write_new(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.flush(); os.fsync(stream.fileno())
