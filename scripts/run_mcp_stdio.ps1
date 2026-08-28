$ErrorActionPreference = "Stop"

# tunnel-client owns this child process. The control-plane credential is valid
# only for its parent process and must not be inherited by application Python.
Remove-Item -LiteralPath "Env:CONTROL_PLANE_API_KEY" -ErrorAction SilentlyContinue

$projectRoot = Split-Path -Parent $PSScriptRoot
$pythonPath = Join-Path $projectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path -LiteralPath $pythonPath -PathType Leaf)) {
    throw "Project virtual-environment Python is required for MCP stdio."
}

# Embedded local Qdrant is single-process storage. This mutex prevents a second
# MCP child from racing the first one and surfacing a storage-lock failure.
$mutex = New-Object System.Threading.Mutex($false, "Local\CullingBladeLineageMcpStdio")
if (-not $mutex.WaitOne(0, $false)) {
    $mutex.Dispose()
    throw "Culling Blade MCP server is already running. Stop the existing MCP child before starting another."
}

try {
    Set-Location -LiteralPath $projectRoot
    & $pythonPath -m app.mcp.server
    exit $LASTEXITCODE
}
finally {
    $mutex.ReleaseMutex()
    $mutex.Dispose()
}
