"""Restart attestations for a complete knowledge-release combination."""
from __future__ import annotations

import hashlib
import hmac
import secrets
from datetime import datetime
from pathlib import Path

from app.code_updates.runtime import process_identity, runtime_identity
from app.code_updates.storage import digest, immutable, now, read, sha


def create_restart_challenge(run, manifest: dict) -> None:
    path = run.directory / "restart_challenge.json"
    immutable(path, {"schema_version": "knowledge_update_restart_challenge_v1", "run_id": run.state["run_id"],
        "nonce": secrets.token_hex(32), "secret": secrets.token_hex(32), "created_at": now(),
        "expected": {"release_manifest_identity": manifest["manifest_identity_sha256"],
                     "fdd_generation": manifest["fdd_generation"], "fdd_collection": manifest["fdd_collection"],
                     "code_collection": manifest["code_collection"], "code_artifact_identity": manifest["code_artifact_identity_sha256"],
                     "lineage_identity": manifest["lineage_identity_sha256"]}})


def _identity(settings, manifest_identity: str) -> dict:
    code = runtime_identity(settings)
    return {"release_manifest_identity": manifest_identity, "fdd_generation": settings.fdd_generation,
            "fdd_collection": settings.qdrant_collection_name, "code_collection": settings.code_qdrant_collection_name,
            "code_artifact_identity": code["code_artifact_identity"], "lineage_identity": code["lineage_identity"]}


def publish_restart_receipts(settings, started_at: str, *, root: Path) -> None:
    """Publish only for already-promoted runs awaiting this exact process."""
    directory = root / "data/knowledge_updates"
    if not directory.is_dir(): return
    for challenge_path in directory.glob("*/restart_challenge.json"):
        run_dir = challenge_path.parent
        if not (run_dir / "activation.json").is_file() or not (run_dir / "state.json").is_file(): continue
        if read(run_dir / "state.json").get("status") != "AWAITING_RESTART": continue
        challenge = read(challenge_path)
        if datetime.fromisoformat(started_at) <= datetime.fromisoformat(challenge["created_at"]): continue
        observed = _identity(settings, challenge["expected"]["release_manifest_identity"])
        if observed != challenge["expected"]: continue
        alive, token = process_identity(__import__("os").getpid())
        if not alive or token is None: continue
        value = {"schema_version": "knowledge_update_runtime_receipt_v1", "run_id": challenge["run_id"], "nonce": challenge["nonce"],
                 "process_id": __import__("os").getpid(), "process_creation_token": token, "server_started_at": started_at,
                 "observed_at": now(), "release_manifest_identity": challenge["expected"]["release_manifest_identity"], "identity": observed}
        value["signature"] = hmac.new(bytes.fromhex(challenge["secret"]), digest(value).encode(), hashlib.sha256).hexdigest()
        output = run_dir / "runtime_receipts" / f"{value['process_id']}-{token}.json"
        if not output.exists(): immutable(output, value)


def verify_restart_receipt(run, path: Path) -> dict:
    challenge, value = read(run.directory / "restart_challenge.json"), read(path)
    signed = {k: v for k, v in value.items() if k != "signature"}
    expected = hmac.new(bytes.fromhex(challenge["secret"]), digest(signed).encode(), hashlib.sha256).hexdigest()
    if not hmac.compare_digest(expected, value.get("signature", "")) or value.get("nonce") != challenge["nonce"]:
        raise ValueError("Knowledge runtime receipt signature is invalid")
    if value.get("run_id") != run.state["run_id"] or value.get("identity") != challenge["expected"]:
        raise ValueError("MCP is serving a different knowledge release combination")
    alive, token = process_identity(value["process_id"])
    if not alive or token != value.get("process_creation_token"):
        raise ValueError("Attested MCP process is no longer running")
    return {"passed": True, "receipt_sha256": sha(path)}
