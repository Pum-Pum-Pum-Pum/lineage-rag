[CmdletBinding()]
param(
    [Parameter(Mandatory)]
    [ValidatePattern('^[A-Za-z0-9][A-Za-z0-9_-]*$')]
    [string]$SnapshotRequest,

    [Parameter(Mandatory)]
    [ValidateSet('intake-parse', 'prepare-index', 'embed-index', 'evaluate', 'activate')]
    [string]$Stage,

    [string]$SourceDirectory,

    [string]$ParseGeneration = 'plsql_antlr_4_13_2_analysis_v15',
    # Required for post-intake stages only when a request name has more than
    # one immutable historical snapshot. The exact ID prevents any selection
    # by creation time or directory order.
    [string]$ImmutableSnapshotId,
    [string]$DependencyReviewLedger,
    [string]$CollectionName,
    [string]$EvaluationFile = 'data/evaluations/code_grounded_eval_v1_reviewed.jsonl',
    [string]$R3BenchmarkManifest,
    [ValidateSet('lexical', 'dense', 'hybrid')]
    [string]$CodeRetrievalMode = 'lexical',
    [string]$QueryVectorsJson,

    # Activation remains a separate, approval-bound runtime operation.  These
    # paths are required only for -Stage activate; they are deliberately not
    # inferred from directory ordering or a "latest" file.
    [string]$ActivationRequest,
    [string]$ActivationApproval,
    [string]$ActivationReadinessReport,
    [switch]$ApplyActivation,
    [switch]$ServicesStopped
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$RepositoryRoot = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $RepositoryRoot '.venv\Scripts\python.exe'
$IntakeDirectory = Join-Path $RepositoryRoot ("data\raw_code\$SnapshotRequest")
$SnapshotRoot = Join-Path $RepositoryRoot 'data\code_snapshots'
$CodeStageRoot = Join-Path $RepositoryRoot 'data\staging\code'
$IndexRoot = Join-Path $RepositoryRoot 'data\staging\code_indexes'
$EmbeddingRoot = Join-Path $RepositoryRoot 'data\staging\code_embeddings'
$CodeQdrantPath = Join-Path $RepositoryRoot 'data\qdrant_code_local'

if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
    throw "Project interpreter not found: $Python. Run 'uv sync --locked' first."
}

function Invoke-ProjectPython {
    param([Parameter(Mandatory)][string[]]$Arguments)
    & $Python @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Existing Python stage failed with exit code ${LASTEXITCODE}: $($Arguments -join ' ')"
    }
}

function Confirm-ExternalOperation {
    param([Parameter(Mandatory)][string]$Operation)
    $response = Read-Host "$Operation can send approved internal PL/SQL to OpenAI and may incur cost. Type APPROVE to continue"
    if ($response -cne 'APPROVE') {
        throw "$Operation was not approved. No external operation was started."
    }
}

function Resolve-SnapshotId {
    if (-not [string]::IsNullOrWhiteSpace($ImmutableSnapshotId)) {
        if ($ImmutableSnapshotId -notlike "$SnapshotRequest-*") {
            throw "-ImmutableSnapshotId must belong to request '$SnapshotRequest': $ImmutableSnapshotId"
        }
        $manifestPath = Join-Path $SnapshotRoot "$ImmutableSnapshotId\snapshot_manifest.json"
        if (-not (Test-Path -LiteralPath $manifestPath -PathType Leaf)) {
            throw "Immutable snapshot manifest does not exist: $manifestPath"
        }
        try {
            $manifest = Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json -ErrorAction Stop
        }
        catch {
            throw "Immutable snapshot manifest is not valid JSON: $manifestPath"
        }
        if ([string]$manifest.snapshot_id -ne $ImmutableSnapshotId -or
            [string]$manifest.request.module_set -ne ($SnapshotRequest -replace '-r[0-9]+$', '') -or
            [string]$manifest.request.svn_revision -ne ($SnapshotRequest -replace '^[A-Za-z0-9_-]+-r', '')) {
            throw "-ImmutableSnapshotId does not match the requested immutable snapshot: $ImmutableSnapshotId"
        }
        return $ImmutableSnapshotId
    }
    $matches = @(
        Get-ChildItem -LiteralPath $SnapshotRoot -Directory -ErrorAction SilentlyContinue |
        Where-Object { $_.Name -like "$SnapshotRequest-*" }
    )
    if ($matches.Count -ne 1) {
        throw "Expected exactly one immutable snapshot for '$SnapshotRequest'; found $($matches.Count). Use the exact request directory and do not choose a snapshot by directory order."
    }
    return $matches[0].Name
}

function Require-DependencyLedger {
    if ([string]::IsNullOrWhiteSpace($DependencyReviewLedger)) {
        throw 'prepare-index requires -DependencyReviewLedger pointing to the reviewed immutable ledger.'
    }
    if (-not (Test-Path -LiteralPath $DependencyReviewLedger -PathType Leaf)) {
        throw "Dependency review ledger does not exist: $DependencyReviewLedger"
    }
}

function Require-R3Benchmark {
    param([string]$SnapshotId)

    if ($SnapshotRequest -ne 'fci-custom-r3') {
        return
    }
    if ([string]::IsNullOrWhiteSpace($R3BenchmarkManifest)) {
        throw 'fci-custom-r3 requires -R3BenchmarkManifest pointing to its reviewed seven-package benchmark.'
    }
    if (-not (Test-Path -LiteralPath $R3BenchmarkManifest -PathType Leaf)) {
        throw "R3 benchmark manifest does not exist: $R3BenchmarkManifest"
    }
    $arguments = @(
        'scripts/verify_code_r3_benchmark.py',
        '--manifest', $R3BenchmarkManifest,
        '--require-reviewed',
        '--fdd-directory', 'data/staging/functional_specs_v9/processed'
    )
    if (-not [string]::IsNullOrWhiteSpace($SnapshotId)) {
        $arguments += @(
            '--snapshot-manifest',
            ("data/code_snapshots/$SnapshotId/snapshot_manifest.json")
        )
    }
    Invoke-ProjectPython -Arguments $arguments
}

function Resolve-BaseEmbeddingCacheArtifact {
    param([Parameter(Mandatory)][string]$SnapshotId)

    $manifestPath = Join-Path $SnapshotRoot "$SnapshotId\snapshot_manifest.json"
    if (-not (Test-Path -LiteralPath $manifestPath -PathType Leaf)) {
        throw "Immutable snapshot manifest is missing: $manifestPath"
    }
    try {
        $manifest = Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json -ErrorAction Stop
    }
    catch {
        throw "Immutable snapshot manifest is not valid JSON: $manifestPath"
    }
    if ([string]$manifest.snapshot_id -ne $SnapshotId) {
        throw "Immutable snapshot manifest identity does not match requested snapshot: $SnapshotId"
    }
    $baseSnapshotId = [string]$manifest.diff.base_snapshot_id
    if ([string]::IsNullOrWhiteSpace($baseSnapshotId)) {
        return $null
    }

    $cacheArtifact = Join-Path $EmbeddingRoot (
        "$baseSnapshotId\code_index_text_embedding_3_large_v1\code_index_artifact.json"
    )
    if (-not (Test-Path -LiteralPath $cacheArtifact -PathType Leaf)) {
        throw "Base snapshot '$baseSnapshotId' has no embedded code artifact at $cacheArtifact. Refusing a paid full re-embedding; embed and verify the approved base generation first."
    }
    try {
        $artifact = Get-Content -LiteralPath $cacheArtifact -Raw | ConvertFrom-Json -ErrorAction Stop
    }
    catch {
        throw "Base embedding artifact is not valid JSON: $cacheArtifact"
    }
    if ([string]$artifact.status -ne 'embedded' -or
        [string]$artifact.snapshot_id -ne $baseSnapshotId -or
        [string]$artifact.embedding_model -ne 'text-embedding-3-large') {
        throw "Base embedding artifact is not a compatible embedded text-embedding-3-large generation: $cacheArtifact"
    }
    return $cacheArtifact
}

function Require-ActivationArtifact {
    param(
        [Parameter(Mandatory)][string]$Name,
        [Parameter(Mandatory)][string]$Path
    )
    if ([string]::IsNullOrWhiteSpace($Path)) {
        throw "activate requires -$Name. Activation is hash-bound and will not infer this artifact."
    }
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        throw "Activation $Name does not exist: $Path"
    }
}

function Confirm-CodeActivation {
    param([Parameter(Mandatory)][string]$SnapshotId)
    $expected = "ACTIVATE $SnapshotId"
    $response = Read-Host "Activation will atomically set CODE_MODES_ENABLED=true in .env for $SnapshotId. Type '$expected' to continue"
    if ($response -cne $expected) {
        throw 'Activation was not confirmed. No configuration was changed.'
    }
}

Push-Location $RepositoryRoot
try {
    if ($Stage -ne 'intake-parse' -and -not [string]::IsNullOrWhiteSpace($SourceDirectory)) {
        throw '-SourceDirectory is valid only with -Stage intake-parse.'
    }
    if ($Stage -eq 'intake-parse' -and -not [string]::IsNullOrWhiteSpace($ImmutableSnapshotId)) {
        throw '-ImmutableSnapshotId is valid only after a snapshot has been published.'
    }
    switch ($Stage) {
        'intake-parse' {
            Require-R3Benchmark
            if (-not (Test-Path -LiteralPath $IntakeDirectory -PathType Container)) {
                throw "Snapshot intake does not exist: $IntakeDirectory"
            }
            if (-not [string]::IsNullOrWhiteSpace($SourceDirectory)) {
                Invoke-ProjectPython -Arguments @(
                    'scripts/stage_code_source_directory.py',
                    '--source-directory', $SourceDirectory,
                    '--intake-directory', ("data/raw_code/$SnapshotRequest")
                )
            }
            Invoke-ProjectPython -Arguments @('scripts/build_code_snapshot.py', ("data/raw_code/$SnapshotRequest"), '--validate-only')
            $published = & $Python 'scripts/build_code_snapshot.py' ("data/raw_code/$SnapshotRequest")
            if ($LASTEXITCODE -ne 0) { throw "Existing Python stage failed with exit code ${LASTEXITCODE}: build_code_snapshot.py" }
            $publication = ($published | Out-String | ConvertFrom-Json)
            $snapshotId = [string]$publication.snapshot_id
            if ([string]::IsNullOrWhiteSpace($snapshotId)) { throw 'Snapshot publication returned no snapshot_id.' }
            Require-R3Benchmark -SnapshotId $snapshotId
            Invoke-ProjectPython -Arguments @(
                'scripts/parse_code_snapshot.py', $snapshotId,
                '--generation', $ParseGeneration
            )
            $gateOutput = "data/exports/code_analysis/$snapshotId-$ParseGeneration-preindex-gate.json"
            if (Test-Path -LiteralPath $gateOutput -PathType Leaf) {
                throw "Pre-index gate output already exists: $gateOutput. Preserve it and publish a new snapshot/generation; do not overwrite review evidence."
            }
            Invoke-ProjectPython -Arguments @(
                'scripts/check_code_preindex_gate.py', $snapshotId,
                '--snapshot-root', 'data/code_snapshots',
                '--generation', $ParseGeneration,
                '--output', $gateOutput
            )
            Write-Output "INTAKE/PARSE COMPLETE: immutable snapshot=$snapshotId. No OpenAI call or Qdrant write occurred."
        }
        'prepare-index' {
            Require-DependencyLedger
            $snapshotId = Resolve-SnapshotId
            Require-R3Benchmark -SnapshotId $snapshotId
            Invoke-ProjectPython -Arguments @(
                'scripts/prepare_code_index_artifacts.py', $snapshotId,
                '--parse-generation', $ParseGeneration,
                '--dependency-review-ledger', $DependencyReviewLedger
            )
            $artifact = "data/staging/code_indexes/$snapshotId/code_index_contract_v5/code_index_artifact.json"
            Invoke-ProjectPython -Arguments @(
                'scripts/verify_prepared_code_index.py', $artifact,
                '--dependency-review-ledger', $DependencyReviewLedger
            )
            Write-Output "PREPARED ONLY: $artifact. No OpenAI call, Qdrant write, or activation occurred."
        }
        'embed-index' {
            if ([string]::IsNullOrWhiteSpace($CollectionName) -or $CollectionName -notmatch '^code_custom_[A-Za-z0-9_]+$') {
                throw 'embed-index requires a new -CollectionName beginning code_custom_ (for example, code_custom_r2_v1).'
            }
            $snapshotId = Resolve-SnapshotId
            Require-R3Benchmark -SnapshotId $snapshotId
            $prepared = "data/staging/code_indexes/$snapshotId/code_index_contract_v5/code_index_artifact.json"
            if (-not (Test-Path -LiteralPath $prepared -PathType Leaf)) {
                throw "Prepared reviewed code artifact is missing: $prepared. Run prepare-index first."
            }
            $embeddedDirectory = Join-Path $EmbeddingRoot (
                "$snapshotId\code_index_text_embedding_3_large_v1"
            )
            if (Test-Path -LiteralPath $embeddedDirectory) {
                throw "Embedded code generation already exists: $embeddedDirectory. Refusing a duplicate paid embedding run."
            }
            Invoke-ProjectPython -Arguments @(
                'scripts/check_code_qdrant_collection_absent.py',
                '--qdrant-path', 'data/qdrant_code_local',
                '--collection-name', $CollectionName
            )
            $baseCacheArtifact = Resolve-BaseEmbeddingCacheArtifact -SnapshotId $snapshotId
            if ($null -ne $baseCacheArtifact) {
                Write-Output "Embedding reuse enabled: base snapshot cache=$baseCacheArtifact"
            }
            else {
                Write-Output 'Embedding reuse unavailable: this is a first-generation snapshot with no base snapshot.'
            }
            $preflightArguments = @(
                'scripts/embed_code_index_artifacts.py', $prepared,
                '--output-root', 'data/staging/code_embeddings',
                '--dry-run'
            )
            if ($null -ne $baseCacheArtifact) {
                $preflightArguments += @('--cache-artifact', $baseCacheArtifact)
            }
            Invoke-ProjectPython -Arguments $preflightArguments
            Confirm-ExternalOperation -Operation 'Code embed-index'
            $embeddingArguments = @(
                'scripts/embed_code_index_artifacts.py', $prepared,
                '--output-root', 'data/staging/code_embeddings',
                '--authorization', 'I_AUTHORIZE_OPENAI_CODE_DISCLOSURE_AND_COST'
            )
            if ($null -ne $baseCacheArtifact) {
                $embeddingArguments += @('--cache-artifact', $baseCacheArtifact)
            }
            Invoke-ProjectPython -Arguments $embeddingArguments
            $embedded = "data/staging/code_embeddings/$snapshotId/code_index_text_embedding_3_large_v1/code_index_artifact.json"
            if ($SnapshotRequest -eq 'fci-custom-r3') {
                $reuseReport = "data/exports/code_analysis/$snapshotId-code-embedding-reuse-verification.json"
                Invoke-ProjectPython -Arguments @(
                    'scripts/verify_code_embedding_reuse.py',
                    '--snapshot-manifest', "data/code_snapshots/$snapshotId/snapshot_manifest.json",
                    '--base-artifact', $baseCacheArtifact,
                    '--embedded-artifact', $embedded,
                    '--output', $reuseReport
                )
                Write-Output "R3 embedding reuse verification passed: $reuseReport"
            }
            Invoke-ProjectPython -Arguments @(
                'scripts/index_code_qdrant.py', $embedded,
                '--qdrant-path', 'data/qdrant_code_local',
                '--collection-name', $CollectionName
            )
            Invoke-ProjectPython -Arguments @(
                'scripts/verify_code_qdrant.py', $embedded,
                '--qdrant-path', 'data/qdrant_code_local',
                '--collection-name', $CollectionName
            )
            Write-Output "STAGED ONLY: collection=$CollectionName is not active. Retain the prior code collection for rollback."
        }
        'evaluate' {
            $snapshotId = Resolve-SnapshotId
            Require-R3Benchmark -SnapshotId $snapshotId
            $embedded = "data/staging/code_embeddings/$snapshotId/code_index_text_embedding_3_large_v1/code_index_artifact.json"
            if (-not (Test-Path -LiteralPath $embedded -PathType Leaf)) {
                throw "Embedded code artifact is missing: $embedded. Run embed-index first."
            }
            if ($CodeRetrievalMode -in @('dense', 'hybrid') -and [string]::IsNullOrWhiteSpace($QueryVectorsJson)) {
                throw 'Dense/hybrid code evaluation requires a reviewed, precomputed -QueryVectorsJson file. This launcher never creates query embeddings.'
            }
            $arguments = @(
                'scripts/run_code_combined_retrieval_eval.py',
                '--eval-file', $EvaluationFile,
                '--code-artifact', $embedded,
                '--analysis-directory', ("data/staging/code/$snapshotId/$ParseGeneration"),
                '--code-mode', $CodeRetrievalMode
            )
            if ($CodeRetrievalMode -in @('dense', 'hybrid')) {
                if ([string]::IsNullOrWhiteSpace($CollectionName)) {
                    throw 'Dense/hybrid code evaluation requires -CollectionName for the isolated code Qdrant generation.'
                }
                $arguments += @('--qdrant-path', 'data/qdrant_code_local', '--collection-name', $CollectionName, '--query-vectors-json', $QueryVectorsJson)
            }
            Invoke-ProjectPython -Arguments $arguments
            Write-Output 'Code retrieval evaluation completed. Combined evaluation still requires explicit reviewed FDD generation and lineage inputs; paid answer evaluation remains separate.'
        }
        'activate' {
            $snapshotId = Resolve-SnapshotId
            Require-R3Benchmark -SnapshotId $snapshotId
            Require-ActivationArtifact -Name 'ActivationRequest' -Path $ActivationRequest
            Require-ActivationArtifact -Name 'ActivationApproval' -Path $ActivationApproval
            $requestPayload = Get-Content -LiteralPath $ActivationRequest -Raw | ConvertFrom-Json
            if ($requestPayload.schema_version -eq 'code_generation_promotion_request_v1') {
                if ($requestPayload.snapshot_id -ne $snapshotId) {
                    throw 'Promotion request belongs to a different immutable snapshot.'
                }
                $promotionArguments = @('scripts/promote_code_generation.py', 'switch',
                    '--action', 'activate', '--request', $ActivationRequest, '--approval', $ActivationApproval)
                Invoke-ProjectPython -Arguments $promotionArguments
                if ($ApplyActivation) {
                    if (-not $ServicesStopped) { throw 'Stop serving processes and confirm -ServicesStopped before applying.' }
                    Confirm-CodeActivation -SnapshotId $snapshotId
                    Invoke-ProjectPython -Arguments ($promotionArguments + @('--apply', '--services-stopped'))
                    Write-Output 'GENERATION CONFIGURATION APPLIED: restart the intended client and verify readiness. Activation is not yet complete.'
                } else {
                    Write-Output 'GENERATION PREFLIGHT ONLY: no .env change or service restart occurred.'
                }
                break
            }
            Require-ActivationArtifact -Name 'ActivationReadinessReport' -Path $ActivationReadinessReport

            $embedded = "data/staging/code_embeddings/$snapshotId/code_index_text_embedding_3_large_v1/code_index_artifact.json"
            if (-not (Test-Path -LiteralPath $embedded -PathType Leaf)) {
                throw "Embedded code artifact is missing: $embedded. Run embed-index and the required evaluation gates first."
            }

            # First perform the exact same no-write preflight that will govern
            # the change.  It verifies the request, approval, readiness bytes,
            # current configuration, and the disabled starting state.
            Invoke-ProjectPython -Arguments @(
                'scripts/switch_code_modes.py', 'activate',
                '--request', $ActivationRequest,
                '--approval', $ActivationApproval,
                '--readiness-report', $ActivationReadinessReport
            )
            if (-not $ApplyActivation) {
                Write-Output 'ACTIVATION PREFLIGHT ONLY: .env is unchanged. Re-run with -ApplyActivation after the deliberate confirmation gate.'
                break
            }
            Confirm-CodeActivation -SnapshotId $snapshotId
            Invoke-ProjectPython -Arguments @(
                'scripts/switch_code_modes.py', 'activate',
                '--request', $ActivationRequest,
                '--approval', $ActivationApproval,
                '--readiness-report', $ActivationReadinessReport,
                '--apply'
            )
            Write-Output "ACTIVATED CONFIGURATION: CODE_MODES_ENABLED=true is now atomically persisted for $snapshotId."
            Write-Output 'Restart FastAPI and Streamlit if running. In Codex Desktop, toggle the local MCP server off and on so its child process reloads .env. Run the approved runtime readiness and smoke gates; roll back with scripts/switch_code_modes.py if a gate fails.'
        }
    }
}
finally {
    Pop-Location
}
