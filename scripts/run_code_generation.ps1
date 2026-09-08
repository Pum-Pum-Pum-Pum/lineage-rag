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
    [string]$DependencyReviewLedger,
    [string]$CollectionName,
    [string]$EvaluationFile = 'data/evaluations/code_grounded_eval_v1_reviewed.jsonl',
    [ValidateSet('lexical', 'dense', 'hybrid')]
    [string]$CodeRetrievalMode = 'lexical',
    [string]$QueryVectorsJson,

    # Activation remains a separate, approval-bound runtime operation.  These
    # paths are required only for -Stage activate; they are deliberately not
    # inferred from directory ordering or a "latest" file.
    [string]$ActivationRequest,
    [string]$ActivationApproval,
    [string]$ActivationReadinessReport,
    [switch]$ApplyActivation
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
    switch ($Stage) {
        'intake-parse' {
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
            Confirm-ExternalOperation -Operation 'Code embed-index'
            if ([string]::IsNullOrWhiteSpace($CollectionName) -or $CollectionName -notmatch '^code_custom_[A-Za-z0-9_]+$') {
                throw 'embed-index requires a new -CollectionName beginning code_custom_ (for example, code_custom_r2_v1).'
            }
            $snapshotId = Resolve-SnapshotId
            $prepared = "data/staging/code_indexes/$snapshotId/code_index_contract_v5/code_index_artifact.json"
            if (-not (Test-Path -LiteralPath $prepared -PathType Leaf)) {
                throw "Prepared reviewed code artifact is missing: $prepared. Run prepare-index first."
            }
            Invoke-ProjectPython -Arguments @(
                'scripts/embed_code_index_artifacts.py', $prepared,
                '--output-root', 'data/staging/code_embeddings',
                '--authorization', 'I_AUTHORIZE_OPENAI_CODE_DISCLOSURE_AND_COST'
            )
            $embedded = "data/staging/code_embeddings/$snapshotId/code_index_text_embedding_3_large_v1/code_index_artifact.json"
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
            Require-ActivationArtifact -Name 'ActivationRequest' -Path $ActivationRequest
            Require-ActivationArtifact -Name 'ActivationApproval' -Path $ActivationApproval
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
