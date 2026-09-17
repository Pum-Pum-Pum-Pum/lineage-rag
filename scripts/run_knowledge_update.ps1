[CmdletBinding()]
param(
    [Parameter(Mandatory)][ValidateSet('init','prepare','build','finalize','activate','verify','status','amend-review','rollback')][string]$Action,
    [Parameter(Mandatory)][ValidatePattern('^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$')][string]$RunId,
    [ValidateSet('fdd','code','both','review')][string]$Mode,
    [string]$FddGeneration, [string]$FddSourceDirectory, [string]$CodeSourceDirectory,
    [string]$SvnRevision, [string]$ApplicationBuild, [string]$Reviewer,
    [string]$PricePerMillion, [string]$PricingBasis, [string]$EnhancementRegistry,
    [string]$MaxUsd, [string]$EmbeddingApproval, [string]$RuntimeReceipt,
    [string]$WithdrawnSource, [string]$ReplacementManifest, [switch]$ServicesStopped
)
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$root = Split-Path -Parent $PSScriptRoot
$python = Join-Path $root '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $python -PathType Leaf)) { throw 'Project Python is missing. Run uv sync --locked first.' }
$arguments = @((Join-Path $PSScriptRoot 'run_knowledge_update.py'), '--action', $Action, '--run-id', $RunId)
$optional = @{
    'mode'=$Mode; 'fdd-generation'=$FddGeneration; 'fdd-source-directory'=$FddSourceDirectory;
    'code-source-directory'=$CodeSourceDirectory; 'svn-revision'=$SvnRevision;
    'application-build'=$ApplicationBuild; 'reviewer'=$Reviewer;
    'price-per-million'=$PricePerMillion; 'pricing-basis'=$PricingBasis;
    'enhancement-registry'=$EnhancementRegistry; 'max-usd'=$MaxUsd;
    'embedding-approval'=$EmbeddingApproval; 'runtime-receipt'=$RuntimeReceipt;
    'withdrawn-source'=$WithdrawnSource; 'replacement-manifest'=$ReplacementManifest
}
foreach ($entry in $optional.GetEnumerator()) {
    if (-not [string]::IsNullOrWhiteSpace($entry.Value)) { $arguments += @('--' + $entry.Key, $entry.Value) }
}
if ($ServicesStopped) { $arguments += '--services-stopped' }
Push-Location -LiteralPath $root
try {
    & $python @arguments
    if ($LASTEXITCODE -ne 0) { throw "Knowledge update stopped. Read the reported error and resume action '$Action' for '$RunId'." }
}
finally { Pop-Location }
