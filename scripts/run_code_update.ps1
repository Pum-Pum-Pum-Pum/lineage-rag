[CmdletBinding()]
param(
    [Parameter(Mandatory)][ValidateSet('init','prepare','build','amend-review','finalize','activate','verify','status')][string]$Action,
    [Parameter(Mandatory)][ValidatePattern('^[A-Za-z0-9][A-Za-z0-9_-]{0,63}$')][string]$RunId,
    [string]$SourceDirectory, [string]$SvnRevision, [string]$ApplicationBuild, [string]$Reviewer,
    [string]$PricePerMillion, [string]$PricingBasis, [string]$EnhancementRegistry,
    [string]$MaxUsd, [string]$RuntimeReceipt, [string]$FddDocumentId, [string]$SourcePath,
    [string]$FddGeneration, [string]$FddDirectory, [string]$FddStage, [switch]$CoordinatedRelease,
    [string]$QualifiedName, [ValidateSet('procedure','function')][string]$SymbolKind,
    [string]$SourceMarker, [string]$EvidenceQuery, [switch]$ServicesStopped
)
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$python = Join-Path $projectRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $python -PathType Leaf)) { throw 'Project Python is missing. Run uv sync --locked first.' }
$arguments = @((Join-Path $PSScriptRoot 'run_code_update.py'), '--action', $Action, '--run-id', $RunId)
$optional = @{
    'source-directory'=$SourceDirectory; 'svn-revision'=$SvnRevision; 'application-build'=$ApplicationBuild;
    'reviewer'=$Reviewer; 'price-per-million'=$PricePerMillion; 'pricing-basis'=$PricingBasis;
    'enhancement-registry'=$EnhancementRegistry; 'max-usd'=$MaxUsd; 'runtime-receipt'=$RuntimeReceipt;
    'fdd-generation'=$FddGeneration; 'fdd-directory'=$FddDirectory; 'fdd-stage'=$FddStage;
    'fdd-document-id'=$FddDocumentId; 'source-path'=$SourcePath; 'qualified-name'=$QualifiedName;
    'symbol-kind'=$SymbolKind; 'source-marker'=$SourceMarker; 'evidence-query'=$EvidenceQuery
}
foreach ($entry in $optional.GetEnumerator()) {
    if (-not [string]::IsNullOrWhiteSpace($entry.Value)) { $arguments += @('--' + $entry.Key, $entry.Value) }
}
if ($ServicesStopped) { $arguments += '--services-stopped' }
if ($CoordinatedRelease) { $arguments += '--coordinated-release' }
Push-Location -LiteralPath $projectRoot
try {
    & $python @arguments
    if ($LASTEXITCODE -ne 0) { throw "Code update stopped. Read the reported error and resume action '$Action' for '$RunId'." }
}
finally { Pop-Location }
