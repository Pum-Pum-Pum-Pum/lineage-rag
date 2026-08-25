from __future__ import annotations

import json
from pathlib import Path


def test_native_runtime_declares_tunnel_owned_mcp_stdio_child_without_secret_injection() -> None:
    payload = json.loads(Path("deployment/native_runtime.json").read_text(encoding="utf-8"))

    assert payload["processes"]["mcp_stdio"] == [
        "powershell",
        "-NoProfile",
        "-ExecutionPolicy",
        "Bypass",
        "-File",
        "scripts/run_mcp_stdio.ps1",
    ]
    mcp = payload["mcp_stdio"]
    assert mcp["process_owner"] == "tunnel-client"
    assert mcp["interface_mode"] == ["mcp", "both"]
    assert "CONTROL_PLANE_API_KEY" not in mcp["required_application_environment"]
    assert mcp["control_plane_key"]["must_not_be_inherited_by_mcp_child"] is True


def test_mcp_stdio_launcher_removes_parent_only_control_key_before_python_starts() -> None:
    source = Path("scripts/run_mcp_stdio.ps1").read_text(encoding="utf-8")

    assert 'Remove-Item -LiteralPath "Env:CONTROL_PLANE_API_KEY"' in source
    assert "-m app.mcp.server" in source
    assert "Write-Output" not in source
    assert "Write-Host" not in source
