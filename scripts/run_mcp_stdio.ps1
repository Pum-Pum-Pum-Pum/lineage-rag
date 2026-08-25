$ErrorActionPreference = "Stop"

# tunnel-client owns this child process. The control-plane credential is valid
# only for its parent process and must not be inherited by application Python.
Remove-Item -LiteralPath "Env:CONTROL_PLANE_API_KEY" -ErrorAction SilentlyContinue

$projectRoot = Split-Path -Parent $PSScriptRoot
$pythonPath = Join-Path $projectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $pythonPath -PathType Leaf)) {
    throw "Project virtual-environment Python is required for MCP stdio."
}

Set-Location -LiteralPath $projectRoot
& $pythonPath -m app.mcp.server
exit $LASTEXITCODE
