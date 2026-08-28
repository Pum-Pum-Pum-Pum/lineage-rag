from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]


def test_mcp_stdio_launcher_strips_control_plane_key_and_serializes_children() -> None:
    script = (ROOT_DIR / "scripts" / "run_mcp_stdio.ps1").read_text(encoding="utf-8")

    assert 'Remove-Item -LiteralPath "Env:CONTROL_PLANE_API_KEY"' in script
    assert 'Local\\CullingBladeLineageMcpStdio' in script
    assert '$mutex.WaitOne(0, $false)' in script
    assert 'Culling Blade MCP server is already running.' in script
    assert '& $pythonPath -m app.mcp.server' in script
