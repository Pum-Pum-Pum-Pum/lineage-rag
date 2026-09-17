"""Bounded stdio tests and challenge-bound live MCP restart attestations."""
from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import os
import secrets
import sys
from datetime import UTC, datetime
from pathlib import Path
from contextlib import contextmanager
from pydantic import BaseModel

from app.code_updates.storage import digest, immutable, read, run_directory, sha, now


@contextmanager
def stopped_mcp_guard(name=r"Local\CullingBladeLineageMcpStdio"):
    """Hold the launcher's mutex through promotion, closing the restart race."""
    if os.name != "nt":
        raise RuntimeError("Desktop promotion currently requires the Windows launcher ownership guard")
    import ctypes
    from ctypes import wintypes
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.CreateMutexW.argtypes = [ctypes.c_void_p, wintypes.BOOL, wintypes.LPCWSTR]
    kernel.CreateMutexW.restype = wintypes.HANDLE
    kernel.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    kernel.ReleaseMutex.argtypes = [wintypes.HANDLE]
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    handle = kernel.CreateMutexW(None, False, name)
    if not handle:
        raise RuntimeError("Cannot verify Desktop MCP ownership")
    acquired = False
    try:
        acquired = kernel.WaitForSingleObject(handle, 0) in (0, 128)
        if not acquired:
            raise RuntimeError("Desktop MCP is still running. Stop it before promotion.")
        yield
    finally:
        if acquired:
            kernel.ReleaseMutex(handle)
        kernel.CloseHandle(handle)


def desktop_mcp_running(name=r"Local\CullingBladeLineageMcpStdio") -> bool:
    """Read the stdio launch mutex without starting a competing MCP child."""
    if os.name != "nt":
        raise RuntimeError("Desktop MCP ownership probing currently requires Windows")
    import ctypes
    from ctypes import wintypes
    kernel = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel.CreateMutexW.argtypes = [ctypes.c_void_p, wintypes.BOOL, wintypes.LPCWSTR]
    kernel.CreateMutexW.restype = wintypes.HANDLE
    kernel.WaitForSingleObject.argtypes = [wintypes.HANDLE, wintypes.DWORD]
    kernel.ReleaseMutex.argtypes = [wintypes.HANDLE]
    kernel.CloseHandle.argtypes = [wintypes.HANDLE]
    handle = kernel.CreateMutexW(None, False, name)
    if not handle:
        raise RuntimeError("Cannot verify Desktop MCP ownership")
    acquired = False
    try:
        acquired = kernel.WaitForSingleObject(handle, 0) in (0, 128)
        return not acquired
    finally:
        if acquired:
            kernel.ReleaseMutex(handle)
        kernel.CloseHandle(handle)


class RuntimeReceipt(BaseModel):
    schema_version: str
    run_id: str
    nonce: str
    process_id: int
    process_creation_token: str | None = None
    server_started_at: str
    observed_at: str
    identity: dict
    signature: str


def runtime_identity(settings):
    from app.code_indexing.contract import load_code_index_artifact
    from app.fdd_code_lineage.reviewed_bundle import load_reviewed_lineage
    code = load_code_index_artifact(settings.code_index_artifact_path)
    lineage = load_reviewed_lineage(settings.fdd_code_lineage_artifact_path)
    return {"snapshot_id": code.snapshot_id, "code_artifact_identity": code.artifact_identity_sha256,
            "lineage_identity": lineage.artifact_identity_sha256,
            "collection": settings.code_qdrant_collection_name, "fdd_generation": settings.fdd_generation,
            "code_modes_enabled": settings.code_modes_enabled}


def create_restart_challenge(run, request):
    path = run.directory / "restart_challenge.json"
    if not path.exists():
        immutable(path, {"nonce": secrets.token_hex(32), "secret": secrets.token_hex(32), "created_at": now(),
            "expected": {"snapshot_id": request["snapshot_id"], "code_artifact_identity": request["code_artifact_identity_sha256"],
                         "lineage_identity": request["lineage_identity_sha256"], "fdd_generation": request["fdd_generation"],
                         "collection": request["target_configuration"]["CODE_QDRANT_COLLECTION_NAME"], "code_modes_enabled": True}})


def process_identity(pid):
    """Read-only process identity; never use os.kill on Windows."""
    if os.name == "nt":
        import ctypes
        from ctypes import wintypes
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.OpenProcess.restype = wintypes.HANDLE
        kernel.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
        kernel.CloseHandle.argtypes = [wintypes.HANDLE]
        kernel.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
        kernel.GetProcessTimes.argtypes = [wintypes.HANDLE, *([ctypes.POINTER(wintypes.FILETIME)] * 4)]
        handle = kernel.OpenProcess(0x1000, False, int(pid))
        if not handle:
            return False, None
        try:
            exit_code = wintypes.DWORD()
            created, exited, system, user = (wintypes.FILETIME() for _ in range(4))
            if not kernel.GetExitCodeProcess(handle, ctypes.byref(exit_code)) or exit_code.value != 259:
                return False, None
            if not kernel.GetProcessTimes(handle, ctypes.byref(created), ctypes.byref(exited), ctypes.byref(system), ctypes.byref(user)):
                return False, None
            return True, str((created.dwHighDateTime << 32) | created.dwLowDateTime)
        finally:
            kernel.CloseHandle(handle)
    try:
        return True, str(Path(f"/proc/{int(pid)}").stat().st_ctime_ns)
    except OSError:
        return False, None


def attest_runtime(settings, run_id, started_at, *, root=None):
    root = root or Path(__file__).resolve().parents[2]
    challenge = read(run_directory(root, run_id) / "restart_challenge.json")
    value = {"schema_version": "code_update_runtime_receipt_v1", "run_id": run_id,
             "nonce": challenge["nonce"], "process_id": os.getpid(), "server_started_at": started_at,
             "observed_at": now(), "identity": runtime_identity(settings)}
    alive, token = process_identity(os.getpid())
    if not alive or token is None:
        raise ValueError("Cannot attest this process identity")
    value["process_creation_token"] = token
    value["signature"] = hmac.new(bytes.fromhex(challenge["secret"]), digest(value).encode(), hashlib.sha256).hexdigest()
    return value


def publish_restart_receipts(settings, started_at, *, root=None):
    """Startup audit only for already-promoted runs waiting for this generation."""
    root = root or Path(__file__).resolve().parents[2]
    pending = [p for p in (root / "data/code_updates").glob("*/restart_challenge.json")
               if (p.parent / "activation.json").is_file() and (p.parent / "state.json").is_file()
               and read(p.parent / "activation.json").get("applied") is True
               and read(p.parent / "state.json").get("status") == "AWAITING_RESTART"]
    if not pending:
        return
    identity = runtime_identity(settings)
    for path in pending:
        challenge = read(path)
        if challenge["expected"] != identity or datetime.fromisoformat(started_at) <= datetime.fromisoformat(challenge["created_at"]):
            continue
        receipt = attest_runtime(settings, path.parent.name, started_at, root=root)
        output = path.parent / "runtime_receipts" / f"{receipt['process_id']}-{receipt['process_creation_token']}.json"
        if not output.exists():
            immutable(output, receipt)
    # Coordinated FDD/code releases use a distinct challenge schema but are
    # attested by this same already-started MCP process. Import lazily to avoid
    # a module cycle during normal code-update startup.
    from app.knowledge_updates.runtime import publish_restart_receipts as publish_knowledge_receipts
    publish_knowledge_receipts(settings, started_at, root=root)


def verify_restart_receipt(run, receipt_path):
    challenge = read(run.directory / "restart_challenge.json")
    value = read(receipt_path)
    signed = {k: v for k, v in value.items() if k != "signature"}
    signature = hmac.new(bytes.fromhex(challenge["secret"]), digest(signed).encode(), hashlib.sha256).hexdigest()
    if not hmac.compare_digest(signature, value.get("signature", "")) or value.get("nonce") != challenge["nonce"]:
        raise ValueError("Runtime receipt is not from the challenged local MCP runtime")
    if value.get("run_id") != run.state["run_id"] or value["identity"] != challenge["expected"]:
        raise ValueError("MCP is serving another generation or configuration")
    if datetime.fromisoformat(value["server_started_at"]) <= datetime.fromisoformat(challenge["created_at"]):
        raise ValueError("MCP has not restarted since promotion")
    alive, token = process_identity(value["process_id"])
    if not alive or token != value.get("process_creation_token"):
        raise ValueError("Attested MCP process is no longer running or its PID was reused")
    from app.core.config import Settings
    if runtime_identity(Settings()) != challenge["expected"]:
        raise ValueError("Current configured generation has changed")
    # This receipt proves execution in a restarted server, not merely Settings().
    # It is local-process evidence, not authentication against a hostile laptop user.
    return {"passed": True, "receipt": value, "receipt_sha256": sha(receipt_path), "verified_at": now()}


def unpack(result):
    if getattr(result, "is_error", getattr(result, "isError", False)):
        raise RuntimeError("MCP returned an error; inspect the bounded UAT log")
    data = getattr(result, "structured_content", getattr(result, "structuredContent", None))
    if data is None:
        blocks = [c.text for c in result.content if getattr(c, "type", None) == "text"]
        data = json.loads(blocks[0])
    # Some SDK versions wrap structured tool output in result.
    return data.get("result", data)


def _task_group_details(error: BaseException) -> str:
    """Return the leaf failures hidden by an anyio/asyncio exception group."""

    details: list[str] = []

    def collect(value: BaseException) -> None:
        children = getattr(value, "exceptions", None)
        if children:
            for child in children:
                collect(child)
            return
        message = str(value).strip() or value.__class__.__name__
        details.append(f"{value.__class__.__name__}: {message}")

    collect(error)
    return "; ".join(dict.fromkeys(details)) or error.__class__.__name__


def _reviewed_lineage_uat_case(final: dict, config: dict) -> dict:
    """Choose a reviewed combined expectation already eligible for this release.

    A UAT must prove the live MCP follows a reviewed relationship, but it must
    not select a mapping merely by artifact order. The final combined gate has
    already established the eligible questions and their exact code paths.
    """

    candidates = [final.get("new_combined_cases"), *config.get("combined_evals", [])]
    for value in candidates:
        if not value:
            continue
        path = Path(value)
        if not path.is_file():
            continue
        for line in path.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            item = json.loads(line)
            if (
                item.get("mode") == "combined"
                and item.get("review_status") == "reviewed"
                and item.get("sme_reviewed") is True
                and item.get("require_reviewed_lineage") is True
                and item.get("should_abstain") is False
                and item.get("expected_code_paths")
            ):
                return item
    raise RuntimeError(
        "No reviewed combined evaluation case is available for bounded MCP lineage UAT"
    )


def staged_uat(root, config, state, final, collection):
    from app.code_indexing.contract import load_code_index_artifact
    from app.fdd_code_lineage.reviewed_bundle import load_reviewed_lineage
    from qdrant_client import QdrantClient
    artifact = load_code_index_artifact(Path(final["code"]))
    lineage = load_reviewed_lineage(Path(final["lineage"]))
    # Do not launch a stdio child when Desktop already owns it. The launcher
    # reports this condition in PowerShell text, which is deliberately not a
    # JSON-RPC response and must never be fed to the MCP client parser.
    if desktop_mcp_running():
        raise RuntimeError(
            "Bounded MCP UAT cannot start while Desktop MCP is running. "
            "Stop the Desktop MCP, then resume finalize; no competing child was started."
        )
    # Open/close also detects local-store ownership before starting the child.
    from app.core.config import Settings
    settings = Settings()
    for directory in {config["code_store"], str(settings.qdrant_local_path)}:
        client = QdrantClient(path=directory)
        client.close()
    files = sorted({r.source_path for r in artifact.records})
    changed = state["steps"]["snapshot"]["result"]["diff"]
    from app.fdd_code_lineage.workflow_retrieval import _load_analyses, _resolved_callers, MAX_CALLERS
    analyses = _load_analyses(Path(state["analysis"]))
    implementations = [s for a in analyses.values() for s in a.symbols if s.occurrence_role == "implementation"]
    changed_paths = set(changed["added"] + changed["modified"])
    selected = sorted({s.source_path for s in implementations} & changed_paths) or files[:1]
    checks = [{"kind": "inventory", "mode": "code", "query": f"List procedures and functions in {selected[0]}", "path": selected[0]}]
    workflows = [(s, _resolved_callers(s, analyses)[:MAX_CALLERS]) for s in implementations if s.source_path in changed_paths]
    workflows = [(s, callers) for s, callers in workflows if callers]
    workflows.sort(key=lambda pair: (not any(c.symbol.source_path != pair[0].source_path for c in pair[1]), pair[0].canonical_qualified_name))
    if workflows:
        symbol, callers = workflows[0]
        checks.append({"kind": "caller_context", "mode": "combined", "path": symbol.source_path,
            "query": f"Explain {symbol.qualified_display_name} in {symbol.source_path}, its callers and validation context",
            "expected_callers": [c.symbol.source_path for c in callers]})
    symbols = {s.occurrence_id: s for s in implementations}
    tables = [(a.source_path, edge) for a in analyses.values() for edge in a.dependencies
              if a.source_path in changed_paths and edge.dependency_kind in {"table_read", "table_write"}
              and edge.source_symbol_occurrence_id in symbols]
    if tables:
        path, edge = sorted(tables, key=lambda pair: (pair[0], pair[1].edge_id))[0]
        symbol = symbols[edge.source_symbol_occurrence_id]
        checks.append({"kind": "table_behavior", "mode": "code", "path": path,
            "query": f"Explain {symbol.qualified_display_name} in {path} and its operation on {edge.target_canonical_name}",
            "expected_table": edge.target_canonical_name})
    lineage_case = _reviewed_lineage_uat_case(final, config)
    checks.append(
        {
            "kind": "lineage",
            "mode": "combined",
            "query": lineage_case["question"],
            "path": lineage_case["expected_code_paths"][0],
            "expected_fdd_documents": lineage_case.get("expected_fdd_document_ids", []),
        }
    )
    mapped = {t.path for m in lineage.mappings for t in m.targets}
    boundary = next((p for p in files if p not in mapped), None)
    if boundary:
        checks.append({"kind": "boundary", "mode": "combined", "query": f"Explain {boundary} and identify any reviewed FDD lineage", "path": boundary})
    env = dict(os.environ)
    env.pop("CONTROL_PLANE_API_KEY", None)
    env.update(INTERFACE_MODE="mcp", MCP_EVIDENCE_DISCLOSURE_ENABLED="true", RETRIEVAL_MODE="lexical",
               CODE_MODES_ENABLED="true", CODE_INDEX_ARTIFACT_PATH=final["code"],
               CODE_ANALYSIS_DIRECTORY=state["analysis"], CODE_QDRANT_COLLECTION_NAME=collection,
               FDD_CODE_LINEAGE_ARTIFACT_PATH=final["lineage"], FDD_GENERATION=config["fdd_generation"],
               PROCESSED_DIR=config["fdd_directory"], RETRIEVAL_INDEX_PATH=config["fdd_directory"])
    diagnostic_directory = run_directory(root, state["run_id"]) / "logs"
    diagnostic_directory.mkdir(parents=True, exist_ok=True)
    diagnostic_log = diagnostic_directory / f"mcp-uat-{secrets.token_hex(8)}.stderr.log"
    diagnostic_searches = diagnostic_log.with_suffix(".searches.json")
    attempted_searches = []

    async def run():
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client
        params = StdioServerParameters(command="powershell.exe", args=["-NoProfile", "-ExecutionPolicy", "Bypass",
            "-File", str(root / "scripts/run_mcp_stdio.ps1")], cwd=str(root), env=env)
        results = []
        with diagnostic_log.open("w", encoding="utf-8") as stderr:
            async with stdio_client(params, errlog=stderr) as (reader, writer):
                async with ClientSession(reader, writer) as session:
                    await session.initialize()
                    for check in checks[:5]:
                        mode, query, path = check["mode"], check["query"], check["path"]
                        response = unpack(await session.call_tool("search", {"query": query, "mode": mode}))
                        attempted_searches.append({"check": check, "search": response})
                        if response.get("retrieval_mode") != "lexical":
                            raise RuntimeError("UAT must run in lexical mode")
                        hits = [h for h in response.get("results", []) if h["source_type"] == "code"]
                        matches = [h for h in hits if path.casefold() in json.dumps(h).casefold()]
                        if not matches:
                            raise RuntimeError(f"MCP UAT did not retrieve {path}")
                        if check["kind"] == "inventory":
                            from app.fdd_code_lineage.workflow_retrieval import enumerate_package_inventory
                            expected = enumerate_package_inventory(analysis_directory=Path(state["analysis"]), source_path=path)
                            inventories = [h["metadata"].get("parser_inventory") for h in hits if h["metadata"].get("parser_inventory")]
                            if not any(i.get("source_path") == path and i.get("procedures") == list(expected.procedures)
                                       and i.get("functions") == list(expected.functions) for i in inventories):
                                raise RuntimeError(f"MCP package inventory is incomplete for {path}")
                        if check["kind"] == "caller_context":
                            for caller in check["expected_callers"]:
                                if not any(caller.casefold() in json.dumps(hit).casefold() for hit in hits):
                                    raise RuntimeError(
                                        f"MCP did not retrieve expected bounded caller context: {caller}; "
                                        f"query={query!r}; returned code IDs={[h.get('id') for h in hits]}"
                                    )
                        if check["kind"] == "lineage" and not any(h["metadata"].get("code_fdd_lineage_status") == "reviewed_mapping_available" for h in matches):
                            raise RuntimeError("MCP did not expose the expected reviewed relationship")
                        if boundary == path and mode == "combined":
                            if any(h["metadata"].get("code_fdd_lineage_status") != "no_reviewed_fdd_lineage" for h in matches):
                                raise RuntimeError("MCP falsely claimed reviewed lineage on boundary source")
                        fetched = unpack(await session.call_tool("fetch", {"id": matches[0]["id"]}))
                        if not fetched.get("text") or fetched.get("id") != matches[0]["id"]:
                            raise RuntimeError("MCP fetch failed for returned evidence ID")
                        if check["kind"] == "table_behavior" and check["expected_table"].casefold() not in (json.dumps(matches) + fetched["text"]).casefold():
                            raise RuntimeError("MCP table-behavior evidence did not include the expected operation target")
                        results.append({**check, "passed": True,
                                        "search": response, "fetch": fetched})
        return {"passed": True, "code_artifact_identity": artifact.artifact_identity_sha256,
                "lineage_identity": lineage.artifact_identity_sha256, "collection": collection,
                "external_api_calls": 0, "cases": results}
    try:
        return asyncio.run(asyncio.wait_for(run(), timeout=240))
    except BaseExceptionGroup as exc:
        immutable(diagnostic_searches, {"passed": False, "cases": attempted_searches,
                                      "failure": _task_group_details(exc), "external_api_calls": 0})
        raise RuntimeError(
            "Bounded MCP UAT failed: " + _task_group_details(exc)
            + f". Search diagnostics: {diagnostic_searches}. Server diagnostics: {diagnostic_log}"
        ) from exc
