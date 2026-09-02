[CmdletBinding()]
param(
    [Parameter(Mandatory)]
    [ValidatePattern('^functional_specs_v[0-9]+$')]
    [string]$Generation,

    [Parameter(Mandatory)]
    [ValidateSet('prepare', 'embed-index', 'evaluate', 'activate')]
    [string]$Stage,

    [string]$EvaluationFile = 'data/evaluations/fdd_grounded_eval_v2_reviewed.jsonl'
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$RepositoryRoot = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $RepositoryRoot '.venv\Scripts\python.exe'
$StageDirectory = Join-Path $RepositoryRoot ("data\staging\$Generation")
$IntakeCollection = "${Generation}_intake"
$IntakeOutput = Join-Path $RepositoryRoot ("data\staging\${Generation}_intake\processed")
$TargetCollection = $Generation

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
    $response = Read-Host "$Operation can send approved internal evidence to OpenAI and may incur cost. Type APPROVE to continue"
    if ($response -cne 'APPROVE') {
        throw "$Operation was not approved. No external operation was started."
    }
}

function Confirm-Activation {
    param([Parameter(Mandatory)][string]$TargetGeneration)
    $expected = "ACTIVATE $TargetGeneration"
    $response = Read-Host "Activation will promote verified lexical artifacts and atomically update .env to $TargetGeneration. Type '$expected' to continue"
    if ($response -cne $expected) {
        throw "Activation was not confirmed. No artifacts or configuration were changed."
    }
}

function Invoke-StagedRebuild {
    param([switch]$DryRun)
    $arguments = @(
        'scripts/stage_archived_fdd_rebuild.py',
        '--source-directory', 'data/docs_embedded',
        '--stage-directory', ("data/staging/$Generation"),
        '--collection-name', $TargetCollection,
        '--index-generation', $Generation
    )
    if ($DryRun) { $arguments += '--dry-run' }
    Invoke-ProjectPython -Arguments $arguments
}

Push-Location $RepositoryRoot
try {
    switch ($Stage) {
        'prepare' {
            # Intake preview checks raw documents and duplicate archive names.  It does
            # not create artifacts, contact OpenAI, open Qdrant for indexing, or move files.
            $env:QDRANT_COLLECTION_NAME = $IntakeCollection
            $env:INGESTION_OUTPUT_DIR = "data/staging/${Generation}_intake/processed"
            Invoke-ProjectPython -Arguments @('scripts/master_ingestion_embedding_docs.py', '--dry-run')
            Invoke-StagedRebuild -DryRun
            Write-Output "PREPARED PLAN ONLY: no OpenAI call, Qdrant write, archive move, or activation occurred."
        }
        'embed-index' {
            Confirm-ExternalOperation -Operation 'FDD embed-index'
            # Keep the per-document intake collection separate from the current serving
            # collection.  The complete staged rebuild below creates the release candidate.
            $rawDocuments = @(Get-ChildItem -LiteralPath 'data/raw_specs' -Filter '*.docx' -File -ErrorAction SilentlyContinue)
            if ($rawDocuments.Count -gt 0) {
                $env:QDRANT_COLLECTION_NAME = $IntakeCollection
                $env:INGESTION_OUTPUT_DIR = "data/staging/${Generation}_intake/processed"
                Invoke-ProjectPython -Arguments @('scripts/master_ingestion_embedding_docs.py')
            }
            else {
                Write-Output 'No DOCX files in data/raw_specs; skipping intake/archive and rebuilding from the verified archive only.'
            }
            Invoke-StagedRebuild
            Write-Output "STAGED ONLY: $Generation is not active. Review data/staging/$Generation/stage_manifest.json."
        }
        'evaluate' {
            Confirm-ExternalOperation -Operation 'FDD retrieval evaluation'
            if (-not (Test-Path -LiteralPath $StageDirectory -PathType Container)) {
                throw "Staged generation does not exist: $StageDirectory"
            }
            Invoke-ProjectPython -Arguments @(
                'scripts/run_fdd_retrieval_gate.py',
                '--eval-file', $EvaluationFile,
                '--collection-name', $TargetCollection,
                '--lexical-artifact-directory', ("data/staging/$Generation/processed")
            )
            Write-Output 'Retrieval-only evaluation completed. Paid grounded-answer evaluation remains a separate explicit operation.'
        }
        'activate' {
            if (-not (Test-Path -LiteralPath (Join-Path $StageDirectory 'stage_manifest.json') -PathType Leaf)) {
                throw "Verified stage manifest is missing: $StageDirectory\stage_manifest.json"
            }
            # This dry preflight validates the verified stage, current .env pair, and
            # target namespace before the operator confirms the state change.
            Invoke-ProjectPython -Arguments @(
                'scripts/activate_fdd_generation.py',
                '--generation', $Generation,
                '--stage-directory', ("data/staging/$Generation")
            )
            Confirm-Activation -TargetGeneration $Generation
            Invoke-ProjectPython -Arguments @(
                'scripts/activate_fdd_generation.py',
                '--generation', $Generation,
                '--stage-directory', ("data/staging/$Generation"),
                '--apply'
            )
            Write-Output "ACTIVATED CONFIGURATION: $Generation is now the configured FDD vector and lexical generation."
            Write-Output 'Restart FastAPI and Streamlit if they are running. Toggle the Desktop-owned MCP server off and on so its child process reloads .env.'
        }
    }
}
finally {
    Pop-Location
}
