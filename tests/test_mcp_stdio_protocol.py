from __future__ import annotations

import json
import os
import queue
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any


ROOT_DIR = Path(__file__).resolve().parents[1]


class MCPStdioProcess:
    """Minimal raw JSON-RPC harness for the actual MCP stdio child process."""

    def __init__(self) -> None:
        environment = os.environ.copy()
        environment.update(
            {
                "INTERFACE_MODE": "mcp",
                "MCP_EVIDENCE_DISCLOSURE_ENABLED": "false",
                "CODE_MODES_ENABLED": "false",
                "RETRIEVAL_MODE": "lexical",
                "PROCESSED_DIR": str(ROOT_DIR / "data" / "processed"),
                "RETRIEVAL_INDEX_PATH": str(ROOT_DIR / "data" / "processed"),
                "MCP_PROTOCOL_TEST_EMIT_DIAGNOSTICS": "1",
                "PYTHONWARNINGS": "default",
            }
        )
        environment.pop("CONTROL_PLANE_API_KEY", None)
        self.process = subprocess.Popen(
            [sys.executable, "-m", "app.mcp.server"],
            cwd=ROOT_DIR,
            env=environment,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding="utf-8",
            bufsize=1,
        )
        assert self.process.stdin is not None
        assert self.process.stdout is not None
        self._stdout_lines: queue.Queue[str] = queue.Queue()
        self._reader = threading.Thread(target=self._read_stdout, daemon=True)
        self._reader.start()
        self.frames: list[str] = []

    def _read_stdout(self) -> None:
        assert self.process.stdout is not None
        for line in self.process.stdout:
            self._stdout_lines.put(line)

    def request(self, request_id: int, method: str, params: dict[str, Any]) -> dict[str, Any]:
        self.process.stdin.write(
            json.dumps(
                {
                    "jsonrpc": "2.0",
                    "id": request_id,
                    "method": method,
                    "params": params,
                },
                separators=(",", ":"),
            )
            + "\n"
        )
        self.process.stdin.flush()
        while True:
            line = self._stdout_lines.get(timeout=10)
            self.frames.append(line)
            message = json.loads(line)
            if message.get("id") == request_id:
                return message

    def notify(self, method: str, params: dict[str, Any]) -> None:
        self.process.stdin.write(
            json.dumps(
                {"jsonrpc": "2.0", "method": method, "params": params},
                separators=(",", ":"),
            )
            + "\n"
        )
        self.process.stdin.flush()

    def close(self) -> str:
        if self.process.poll() is None:
            self.process.terminate()
        self.process.wait(timeout=10)
        self._reader.join(timeout=2)
        assert self.process.stderr is not None
        return self.process.stderr.read()


def test_actual_stdio_server_emits_only_jsonrpc_frames_and_keeps_diagnostics_on_stderr() -> None:
    server = MCPStdioProcess()
    try:
        initialized = server.request(
            1,
            "initialize",
            {
                "protocolVersion": "2026-07-28",
                "capabilities": {},
                "clientInfo": {"name": "protocol-test", "version": "1"},
            },
        )
        server.notify("notifications/initialized", {})
        tools = server.request(2, "tools/list", {})
        disabled = server.request(
            3,
            "tools/call",
            {"name": "search", "arguments": {"query": "internal query", "mode": "fdd"}},
        )
    finally:
        stderr = server.close()

    assert initialized["jsonrpc"] == "2.0"
    assert {tool["name"] for tool in tools["result"]["tools"]} == {"search", "fetch"}
    for tool in tools["result"]["tools"]:
        assert tool["annotations"]["readOnlyHint"] is True
        assert tool["annotations"]["destructiveHint"] is False
        assert tool["annotations"]["openWorldHint"] is False
        assert tool["outputSchema"]

    disabled_result = disabled["result"]
    assert disabled_result["isError"] is True
    assert "structuredContent" not in disabled_result
    assert disabled_result["content"] == [
        {"type": "text", "text": "Evidence disclosure is disabled."}
    ]
    assert all(json.loads(frame)["jsonrpc"] == "2.0" for frame in server.frames)
    assert "mcp-protocol-test-third-party-diagnostic" in stderr
    assert "mcp-protocol-test-warning" in stderr
