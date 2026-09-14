# Custom-code generation launcher

For recurring updates above an active reviewed generation, use the
[resumable code update runbook](Code_Update_Runbook.md). It consolidates commands,
review and recovery into one persistent run. This document remains the low-level
reference and bootstrap route; historical R3 instructions are retained separately.

Use this runbook to create a complete, immutable custom-code generation from a
read-only SVN working copy. Run each numbered step separately from the repository
root, in the same PowerShell terminal. Stop whenever a command reports an error.
Do not paste the whole runbook as one script: two steps require an SME to edit
and save a review before continuing.

For the controlled `fci-custom-r3` seven-package expansion, first follow the
[R3 seven-package expansion runbook](R3_Seven_Package_Expansion_Runbook.md).
R3 additionally requires its reviewed benchmark manifest on every launcher
command, validates the complete 14-new-file/one-modified-base-file delta, and
records cache reuse before it permits a paid embedding prompt.

Only `.sql`, `.spc`, `.prc`, and `.fnc` files are included, case-insensitively.
`.ddl`, `.cmt`, and every other extension are skipped. The source directory must
contain the complete intended code set, including unchanged files. Files absent
from that directory are treated as deleted when comparing with the base snapshot.
Included files still undergo program-unit and source-identity validation.

## Where to start

- **New source generation:** start at Step 0.
- **Current R2:** intake, dependency review, preparation, embedding, and code-only
  lexical evaluation already completed for `fci-custom-r2-ffd9732906d4`.
  The three cases in the focused Neo Day2 Part2 packet have now been imported
  under reviewer `Pum` into the
  [reviewed Part2 artifact](../data/staging/fdd_code_lineage/fci-custom-r2-v3-part2/reviewed_lineage_artifact.json).
  Its saved Markdown and candidate are approval-bound history; do not edit them
  or repeat Steps 0-9 for this packet.
  Consolidation and the full offline combined evaluation are now complete.
  The [reviewed bundle](../data/staging/fdd_code_lineage/fci-custom-r2-consolidated-v1/reviewed_lineage_bundle.json)
  retains six unchanged mappings from the two original review chains.
  After the bounded FDD selection fix, the
  [combined report](../data/exports/evaluations/code-combined-retrieval-20260912T063429Z.json)
  passes all four positive cases with FDD/code recall of 1.0; the negative case
  reports no failures. The earlier failed report is retained as history.
  Generation-aware promotion and verified serving support for the bundle are
  implemented. The operator approved request `f60b4ccf...`, and its configuration
  switch has been applied. Code readiness passed 4/4, combined readiness passed
  7/7, and rollback dry-run passed against the saved settings.
  **Current checkpoint:** the authorized disabled rollback was applied after
  unsuccessful restart verification. Desktop's `CODE_MODES_ENABLED` override
  has now been removed and its saved configuration checked. `.env` code modes
  remain disabled. The old request cannot be reapplied because the rollback
  starting state differs from its bound before-state. A fresh request is ready:
  `data/exports/activation/fci-custom-r2-promotion-v3-request.json`, identity
  `928c46bec3ed2dfb40baa413b25221b2074a0fe33562c3464e1c1d85ca6140be`.
  The operator approved this request and promotion has now been applied:
  `.env` has `CODE_MODES_ENABLED=true` and the approved R2 artifact selections.
  Fresh local readiness passed code 4/4 and combined 7/7; rollback dry-run passed.
  **Desktop restart verified:** fresh wrapper 24384, launcher 11612 and server
  28732 were observed under Desktop host 24796 after apply. Three process samples
  and a follow-up confirmed the server remained present. Saved configuration
  and runtime/evidence hashes match; search/fetch metadata is exposed.
  Configuration promotion and restart verification are complete. Do not repeat
  approval/apply or add Desktop generation overrides. Live retrieval/answer
  testing was not performed and needs its own bounded cost/disclosure authority.
  No paid/evidence query was run;
  manual answer testing remains separately authorized work.
  Do not re-ingest, re-embed, or repeat the completed SME reviews.
  The Part2-only artifact must not replace the broader AML lineage:
  the reviewed benchmark also expects the original R22 Neo FDD. Existing R2 v1
  provides broad file-level relationships; these do not imply newly reviewed
  exact routine selectors. R2 configuration is selected and Desktop process
  restart is verified; live answer quality is a separate check. The earlier `-v1` directory with a null FDD generation
  is an incomplete draft, not an input to resume from.
- **New terminal:** restore your variables using the resume block, substituting
  your actual request, immutable snapshot, reviewer, collection, and FDD generation.
  Skip every creation step whose output already exists.

```powershell
$ErrorActionPreference = 'Stop'
$request = 'fci-custom-r2'
$snapshotId = 'fci-custom-r2-ffd9732906d4'
$parseGeneration = 'plsql_antlr_4_13_2_analysis_v15'
$reviewer = 'Pum'
$collection = 'code_custom_r2_v1'
$fddGeneration = 'functional_specs_v9'

# Helper stops a block after a failed Python command; it does not run anything yet.
function Invoke-CodePython {
  & .\.venv\Scripts\python.exe @args
  if ($LASTEXITCODE -ne 0) { throw "Python stage failed: exit $LASTEXITCODE" }
}
```

The current R2 base was `fci-custom-r1-b1c79c6dc2c5`. For a future snapshot,
choose the exact previous complete snapshot you intend to compare against.
R2 can serve as that comparison/cache base once its retained artifacts are verified;
being a comparison base does not make it the active runtime generation.

## 0. Create the request directory and JSON file

Edit the values at the top of this block. Leave required values blank until
you know them; the block refuses to create a request while they are blank.
`$svnRevision` is the actual source revision, not a retry counter.
The request directory must use `<module_set>-r<svn_revision>`.
Do not append `-retry1`: the existing importer and snapshot resolver do not
support that naming convention. Earlier chat suggestions to do so were incorrect.

```powershell
$ErrorActionPreference = 'Stop'
$moduleSet = 'fci-custom'
$svnRevision = ''
$applicationBuild = ''
$reviewer = ''
$baseSnapshotId = '' # Exact previous immutable ID; use $null for a first snapshot.
$sourceDirectory = 'C:\SVN\BACKEND'
$parseGeneration = 'plsql_antlr_4_13_2_analysis_v15'
$fddGeneration = 'functional_specs_v9'

function Invoke-CodePython {
  & .\.venv\Scripts\python.exe @args
  if ($LASTEXITCODE -ne 0) { throw "Python stage failed: exit $LASTEXITCODE" }
}

if ($svnRevision -notmatch '^[1-9][0-9]*$') { throw 'Enter the actual SVN revision.' }
if ([string]::IsNullOrWhiteSpace($applicationBuild) -or
    [string]::IsNullOrWhiteSpace($reviewer)) { throw 'Enter build and reviewer.' }
if ($null -ne $baseSnapshotId) {
  if ($baseSnapshotId -notmatch '^[A-Za-z0-9][A-Za-z0-9._-]{0,255}$') {
    throw 'Enter a valid base snapshot ID, or $null for a first snapshot.'
  }
  $baseManifest = Join-Path "data\code_snapshots\$baseSnapshotId" 'snapshot_manifest.json'
  if (-not (Test-Path -LiteralPath $baseManifest -PathType Leaf)) {
    throw "Base snapshot manifest not found: $baseManifest"
  }
}
if (-not (Test-Path -LiteralPath $sourceDirectory -PathType Container)) {
  throw "Source directory not found: $sourceDirectory"
}
$request = "$moduleSet-r$svnRevision"
$collection = "code_custom_r${svnRevision}_v1" # Must be unused in the code store.
$intake = Join-Path 'data\raw_code' $request
$requestFile = Join-Path $intake 'snapshot_request.json'
if (Test-Path -LiteralPath $intake) { throw "Snapshot request already exists: $intake" }
$existingSnapshots = @(Get-ChildItem 'data\code_snapshots' -Directory |
  Where-Object { $_.Name -like "$request-*" })
if ($existingSnapshots.Count -gt 0) { throw 'This request already has a published snapshot.' }

$requestJson = [ordered]@{
  schema_version = 'code_snapshot_request_v1'
  module_set = $moduleSet
  svn_revision = $svnRevision
  application_build = $applicationBuild
  reviewer = $reviewer
  base_snapshot_id = $baseSnapshotId
  expected_changed_packages = @()
  compiler_context = @{ oracle_version = $null; plsql_ccflags = $null }
} | ConvertTo-Json -Depth 5
New-Item -ItemType Directory -Path $intake | Out-Null
# UTF-8 without BOM also works with Windows PowerShell 5.1.
[System.IO.File]::WriteAllText(
  (Join-Path (Get-Location).Path $requestFile),
  $requestJson, [System.Text.UTF8Encoding]::new($false)
)
Get-Content -LiteralPath $requestFile
```

This creates both the directory and the JSON **file**. Never run `mkdir` with
`snapshot_request.json` at the end. The JSON schema stays
`"schema_version": "code_snapshot_request_v1"` for every source revision.
`expected_changed_packages: []` is allowed: actual source hashes determine changes.
Unknown compiler context stays JSON `null`. Confirm the displayed values before intake.

## 1. Intake, snapshot, parse, and local gate

```powershell
.\scripts\run_code_generation.ps1 -SnapshotRequest $request -Stage intake-parse `
  -SourceDirectory $sourceDirectory -ParseGeneration $parseGeneration
```

For example, `$sourceDirectory` supplies the same argument as
`-SourceDirectory 'C:\SVN\BACKEND'`. The script copies source into controlled
intake without modifying the external folder. No OpenAI call or Qdrant write occurs.

Wait for `INTAKE/PARSE COMPLETE`. Copy only the value after `immutable snapshot=`
and before the sentence-ending period. The completed R2 run emitted:

```text
fci-custom-r2-ffd9732906d4
```

Set the ID for **your** run, then inspect its diff:

```powershell
$snapshotId = Read-Host 'Paste the exact emitted immutable snapshot ID'
$manifestPath = "data\code_snapshots\$snapshotId\snapshot_manifest.json"
$manifest = Get-Content -LiteralPath $manifestPath -Raw | ConvertFrom-Json
if ($manifest.snapshot_id -ne $snapshotId -or
    "$($manifest.request.module_set)-r$($manifest.request.svn_revision)" -ne $request) {
  throw 'Snapshot does not belong to this request.'
}
$manifest.files | Select-Object path, sha256
$manifest.diff | ConvertTo-Json -Depth 6
```

Check file count, base ID, added, modified, and deleted files before continuing.
R2 contained six files: five unchanged and `utpks_utduh_custom.sql` added.
Unexpected deletions mean the input set needs investigation; do not embed it.

## 2. Export and review the dependency packet

```powershell
Invoke-CodePython scripts\export_code_dependency_review.py $snapshotId `
  --generation $parseGeneration
$dependencyPrefix = "data\exports\code_analysis\$snapshotId-$parseGeneration-dependency-review"
Write-Output "Review: $dependencyPrefix.md"
```

The exporter creates its output directory and JSON/Markdown pair. Edit the
Markdown only. Each decision requires a nonblank `SME rationale` on the same
line as its label.

| SME verdict | SME corrected kind/state | Result |
| --- | --- | --- |
| `accepted` | Leave blank | Keeps the proposed classification |
| `corrected` | Exact `kind / state`, e.g. `dynamic_sql / dynamic_known` | Uses the SME correction |
| `needs_more_context` | Leave blank | Produces a pending ledger; preparation must wait |

The example correction is syntax guidance; apply it only when supported by
the reviewed source. Save all decisions before Step 3.

## 3. Import the reviewed dependency ledger

```powershell
$dependencyPrefix = "data\exports\code_analysis\$snapshotId-$parseGeneration-dependency-review"
$dependencyLedger = "data\exports\code_analysis\reviews\$snapshotId-dependency-review-ledger.json"
Invoke-CodePython scripts\import_code_dependency_review.py `
  "$dependencyPrefix.json" "$dependencyPrefix.md" `
  --reviewer $reviewer --output $dependencyLedger
```

The importer creates the reviews directory and the ledger
(`<snapshot-id>-dependency-review-ledger.json`). Continue only when
`status=reviewed`. If import fails, correct the draft review fields and retry
the import; once a ledger exists, retain it and its exact reviewed inputs.

## 4. Prepare and verify reviewed index artifacts

```powershell
$dependencyLedger = "data\exports\code_analysis\reviews\$snapshotId-dependency-review-ledger.json"
.\scripts\run_code_generation.ps1 -SnapshotRequest $request -Stage prepare-index `
  -ParseGeneration $parseGeneration -DependencyReviewLedger $dependencyLedger
```

The launcher creates and verifies the complete prepared artifact. No paid call,
Qdrant write, or activation occurs. Every command pins the same parser generation;
v15 is also the current launcher default.

### 4.1 R3 mandatory reuse preflight

For the controlled `fci-custom-r3` expansion, run this local preflight **after
Step 4 and before any embedding approval**. It reports the exact count of
unique code excerpts that would be sent to OpenAI after compatible R2 cache
reuse. It does not create an OpenAI client, disclose code, write Qdrant, or
activate a collection.

```powershell
$prepared = "data\staging\code_indexes\$r3SnapshotId\code_index_contract_v5\code_index_artifact.json"
$baseArtifact = 'data\staging\code_embeddings\fci-custom-r2-ffd9732906d4\code_index_text_embedding_3_large_v1\code_index_artifact.json'

.\.venv\Scripts\python.exe scripts\embed_code_index_artifacts.py `
  $prepared --output-root data\staging\code_embeddings `
  --cache-artifact $baseArtifact --dry-run
```

Record the prepared-artifact identity, base-artifact identity,
`cached_embedding_inputs`, and `external_embedding_inputs`. Obtain separate
bounded cost/disclosure approval for that reported cache-miss count before
continuing to Step 5. For R3, pass both `-ImmutableSnapshotId $r3SnapshotId`
and `-R3BenchmarkManifest $r3Reviewed` to every later launcher command.

## 5. Embed, index, and verify the isolated collection

Before using the embedded local Qdrant store, stop the MCP/FastAPI process that
owns it. Keep the single-client laptop model; do not delete a Qdrant lock file.

```powershell
.\scripts\run_code_generation.ps1 -SnapshotRequest $request -Stage embed-index `
  -CollectionName $collection
```

Type `APPROVE` only after reviewing the intended internal-code disclosure and
embedding cost. The launcher creates the embedding output directories and
complete new collection, then verifies its points. Keep the previous collection.

When a snapshot names an embedded base snapshot, the launcher automatically
uses that exact base artifact as its cache source. Identical embedding text and
model/cache identities reuse vectors, including unchanged units inside a changed
package. Changed units are embedded with their complete context, not only the
edited lines. Parser/chunking/context changes can require new embeddings even
when a source file is unchanged. The complete collection still contains all
current units.

A missing compatible base artifact or an existing candidate embedding directory
blocks the launcher before the paid prompt. If embedding already completed but
indexing failed, do not repeat this paid stage: use the local indexing recovery
commands at the end of this runbook. Keep smoke collections as test evidence.

## 6. Evaluate code retrieval

```powershell
.\scripts\run_code_generation.ps1 -SnapshotRequest $request -Stage evaluate `
  -ParseGeneration $parseGeneration
```

Continue after `release_gate_eligible=true`. This default evaluation is lexical
and makes no external calls. It covers the reviewed code cases; it does not
establish combined-mode or live-answer correctness. Add SME-reviewed cases for
new package behavior before relying on it beyond the existing benchmark.

Use exact logical code paths in new reviewed cases. Legacy filename-only
expectations match only when a retrieved basename resolves uniquely. Keep
historical failed reports when a later evaluation passes.

## 7. Create the lineage directory and candidate definition

For a large corpus, first use the optional
[automatic lineage proposal generator](Automatic_Lineage_Proposal_Runbook.md).
It reuses stored embeddings and enhancement comments to produce a local ranked
review report. Select supported candidates from that report; it does not replace
the SME review or automatically update the approved lineage artifact.

Use this section when creating a new lineage draft, not when reviewing the
already-created focused Part2 packet linked at the top.
The existing AML definition is a starting draft. This block copies its mappings
into a new definition for the selected FDD generation; it does not approve them.

```powershell
$lineageVersion = 'v1' # Use a fresh version for a revised lineage proposal.
$lineageDirectory = "data\staging\fdd_code_lineage\$request-$lineageVersion"
$definitionPath = Join-Path $lineageDirectory 'candidate_definition.json'
$seedDefinition = 'data\staging\fdd_code_lineage\neo_aml_v1\candidate_definition.json'
if (Test-Path -LiteralPath $definitionPath) { throw 'Lineage definition already exists; review it and resume at Step 8.' }
$definition = Get-Content -LiteralPath $seedDefinition -Raw | ConvertFrom-Json
$definition.fdd_generation = $fddGeneration
foreach ($mapping in $definition.mappings) { $mapping.mapping_status = 'candidate' }
New-Item -ItemType Directory -Force -Path $lineageDirectory | Out-Null
[System.IO.File]::WriteAllText(
  (Join-Path (Get-Location).Path $definitionPath),
  ($definition | ConvertTo-Json -Depth 30), [System.Text.UTF8Encoding]::new($false)
)
Write-Output "Edit and check: $definitionPath"
```

Review the new definition before Step 8. Check exact FDD document IDs, release
labels, logical code paths, and rationale. Prefer exact symbol/overload selectors
when known. A file-scoped link remains useful provenance, but it cannot steer
combined FDD ranking because a package can implement several FDDs. Only a
reviewed, unambiguous symbol-level link may reserve one existing FDD evidence
slot; it never increases the caller-visible FDD limit. Existing AML links must
still be valid for this FDD/code pair.
Add a link for `utpks_utduh_custom.sql` only when its documented relationship is
supported; an unmapped package remains available for code retrieval without a
claim that it implements a particular FDD.

## 8. Build and render the candidate lineage review

```powershell
$lineageDirectory = "data\staging\fdd_code_lineage\$request-$lineageVersion"
$definitionPath = Join-Path $lineageDirectory 'candidate_definition.json'
$candidateLineage = Join-Path $lineageDirectory 'candidate_lineage_artifact.json'
$lineageReview = Join-Path $lineageDirectory 'lineage_review.md'
$codeArtifact = "data\staging\code_embeddings\$snapshotId\code_index_text_embedding_3_large_v1\code_index_artifact.json"
$analysisDirectory = "data\staging\code\$snapshotId\$parseGeneration"
$fddDirectory = "data\staging\$fddGeneration\processed"

.\.venv\Scripts\python.exe scripts\prepare_fdd_code_lineage.py $definitionPath `
  --code-artifact $codeArtifact --analysis-directory $analysisDirectory `
  --fdd-processed-directory $fddDirectory --output $candidateLineage

.\.venv\Scripts\python.exe scripts\render_fdd_code_lineage_review.py $candidateLineage `
  --output $lineageReview
```

These are local operations. Open `lineage_review.md` and complete each mapping
with `SME verdict: reviewed` and a nonblank `SME rationale` only if you accept it.
This packet uses `reviewed`, unlike the dependency packet's `accepted` verdict.

If a target needs correction, change the draft definition and create a fresh
lineage version, then repeat Steps 7-8 and review that version. The current
importer does **not** apply changes typed into `SME corrected targets/symbols`.
Leave that field blank for accepted targets. `rejected` or `needs_symbol_scope`
blocks review import.

## 9. Import the reviewed lineage artifact

After saving the completed Markdown:

```powershell
$reviewedLineage = Join-Path $lineageDirectory 'reviewed_lineage_artifact.json'
.\.venv\Scripts\python.exe scripts\import_fdd_code_lineage_review.py `
  $candidateLineage $lineageReview --reviewer $reviewer `
  --code-artifact $codeArtifact --analysis-directory $analysisDirectory `
  --fdd-processed-directory $fddDirectory --output $reviewedLineage
```

Expect `status=reviewed`. The artifact binds the candidate, review, FDD generation,
and code artifact. This records the mapping review without changing runtime.

## 10. Evaluate combined FDD/code retrieval

```powershell
.\.venv\Scripts\python.exe scripts\run_code_combined_retrieval_eval.py `
  --eval-file data\evaluations\combined_grounded_eval_v2_reviewed.jsonl `
  --code-artifact $codeArtifact --analysis-directory $analysisDirectory `
  --fdd-generation $fddGeneration --fdd-directory $fddDirectory `
  --lineage-artifact $reviewedLineage --code-mode lexical `
  --fdd-candidate-limit 30
```

Expect `release_gate_eligible=true`; retain the emitted report path. This is a
local lexical retrieval gate. Dense/hybrid evaluation requires separately
reviewed precomputed query vectors and the candidate collection; these commands
do not generate query embeddings. The FDD candidate limit is internal only: the
returned FDD lane remains bound by `--limit` (ten by default). Live answer/citation evaluation and its SME
acceptance remain separate, with explicit authorization for any new paid calls.

### Multiple existing reviews: provenance-preserving consolidation

For the current R2, this operation has already completed. Do not rerun creation
against its existing output. For a future consolidation, use an unused output
directory and supply each original reviewed artifact, its candidate, and its saved
review Markdown. The command verifies exact byte hashes, candidate/decision
bindings, common generations, and source targets. It neither creates new SME
decisions nor broadens file-level relationships into routine-level relationships.
Hashes establish integrity, not reviewer authentication; access controls remain
necessary.

The following records the exact R2 consolidation inputs. Run from the repository
root; no helper function or earlier terminal variables are needed:

```powershell
.\.venv\Scripts\python.exe scripts\consolidate_reviewed_lineage.py `
  --source data\staging\fdd_code_lineage\fci-custom-r2-v1\reviewed_lineage_artifact.json data\staging\fdd_code_lineage\fci-custom-r2-v1\candidate_lineage_artifact.json data\staging\fdd_code_lineage\fci-custom-r2-v1\candidate_review.md `
  --source data\staging\fdd_code_lineage\fci-custom-r2-v3-part2\reviewed_lineage_artifact.json data\staging\fdd_code_lineage\fci-custom-r2-v3-part2\candidate_lineage_artifact.json data\staging\fdd_code_lineage\fci-custom-r2-v3-part2\lineage_review.md `
  --code-artifact data\staging\code_embeddings\fci-custom-r2-ffd9732906d4\code_index_text_embedding_3_large_v1\code_index_artifact.json `
  --analysis-directory data\staging\code\fci-custom-r2-ffd9732906d4\plsql_antlr_4_13_2_analysis_v15 `
  --fdd-directory data\staging\functional_specs_v9\processed `
  --output data\staging\fdd_code_lineage\fci-custom-r2-consolidated-v1\reviewed_lineage_bundle.json
if ($LASTEXITCODE -ne 0) { throw 'Lineage consolidation failed; stop here.' }
```

For Step 10, after restoring the existing R2 evaluation variables, select:

```powershell
$reviewedLineage = 'data\staging\fdd_code_lineage\fci-custom-r2-consolidated-v1\reviewed_lineage_bundle.json'
```

The evaluator accepts this bundle as the unchanged union of its verified parents.
The shared retrieval and readiness paths now use an explicit verified serving
loader for the bundle; raw legacy v1 model parsing still rejects this format.
Select it only through the approved promotion below, not a manual `.env` edit. The earlier
combined failure was a release-gate result, not a consolidation failure; retain
the failed report and do not weaken the manifest or silently expand retrieval limits.

The bounded combined-only FDD fallback retains one additional document matching
an explicit acronym/mixed-case topic in both its document identity and source
text. It uses only the already retrieved candidates and an existing output slot;
it does not change scores, increase limits, or infer new lineage. A newly selected
exact reviewed-symbol anchor takes precedence over this fallback. Generic or
all-lowercase wording does not trigger the conservative topic heuristic. Passing
this lexical benchmark does not establish corpus-wide relevance or live answer
quality; dense/hybrid and manual answer checks remain separate evidence.

### 10a. Code-to-FDD workflow retrieval checks

For an exact named routine, combined retrieval now adds a small, deterministic
workflow context without changing general ranking: the directly retrieved
routine, up to two resolved callers, and up to two same-source validation
blocks that explicitly mention that routine. It then proposes at most three FDD
units from the local lexical corpus and adds immediate adjacent units from the
same document. This helps retain the condition and outcome together.

The returned metadata marks these FDD units as
`workflow_status=unreviewed_documentation_candidate`. They are useful source
evidence, but are **not** reviewed lineage and must not be described as proof
that code implements the FDD or that a difference is a defect. Add an accepted
symbol-level mapping through Steps 7-9 only after SME review.

For an exact logical filename, for example `utpks_utduh_custom.sql`, a code or
combined search also returns `metadata.parser_inventory` on one returned code
result. It contains the complete parser-extracted procedure/function inventory
for that retrieved source file. Generic questions never trigger package
inventory or package-wide workflow scanning. The existing MCP surface remains
only `search` and `fetch`.

Before a release, retain focused regression evidence for all four outcomes:

- the expected FDD candidate and its adjacent context are selected;
- a generic but plausible FDD is not promoted over the specific workflow;
- absent documentation remains explicitly unreviewed; and
- a later correction is added as a new reviewed lineage artifact, not by
  overwriting an earlier candidate or review record.

## 11. Runtime promotion checkpoint

Steps 0-10 prepare and evaluate the candidate. They do not select it for serving.
`-Stage activate` now distinguishes a generation-promotion request from the legacy
flag-only request. The new request atomically switches all five code-mode settings:
flag, embedded artifact, analysis directory, collection and reviewed lineage.
It leaves FDD selection, interface mode, disclosure and retrieval strategy unchanged.
The old assessor remains a historical flag-activation tool; do not reuse its fixed
v4/v5/r1 collection checks as R2 generation readiness.

### 11a. Prepare a pending request (no runtime changes)

For the current R2 this is already prepared at
`data/exports/activation/fci-custom-r2-promotion-v2-request.json`.
The v1 request is retained but stale after the additional rollback safeguard.
Do not rerun against an existing output. For future requests, substitute the
actual artifact/review/report paths and use an unused output filename:

```powershell
.\.venv\Scripts\python.exe scripts\promote_code_generation.py prepare `
  --code-artifact data\staging\code_embeddings\fci-custom-r2-ffd9732906d4\code_index_text_embedding_3_large_v1\code_index_artifact.json `
  --analysis data\staging\code\fci-custom-r2-ffd9732906d4\plsql_antlr_4_13_2_analysis_v15 `
  --lineage data\staging\fdd_code_lineage\fci-custom-r2-consolidated-v1\reviewed_lineage_bundle.json `
  --dependency-ledger data\exports\code_analysis\reviews\fci-custom-r2-ffd9732906d4-dependency-review-ledger.json `
  --code-report data\exports\evaluations\code-combined-retrieval-20260912T063626Z.json `
  --combined-report data\exports\evaluations\code-combined-retrieval-20260912T063429Z.json `
  --collection code_custom_r2_v1 --requested-by Pum `
  --output data\exports\activation\fci-custom-r2-promotion-v2-request.json
if ($LASTEXITCODE -ne 0) { throw 'Promotion preparation failed; stop here.' }
```

Preparation verifies the reviewed parse-to-embedding contract, both passed retrieval
reports and their manifests, FDD serving/evaluated unit equivalence, lineage, exact
code collection identities/count and FDD collection availability/dimension. It
binds input bytes, directory membership, effective non-secret settings and runtime
Python/PowerShell/config/lock files. It does not perform a paid query, prove live
hybrid/answer quality, change `.env`, or restart any process. A local Qdrant lock
conflict blocks this check; do not delete lock files or spawn competing clients.

### 11b. Explicit approval, then dry-run

Stop here until the operator approves the exact emitted request hash. Approval
permits this configuration operation and disabled rollback only: **zero paid
requests and no additional evidence disclosure**. Prior paid authorizations are
not reused. The approval file is an audit record, not identity authentication.

After that explicit approval:

```powershell
$promotionRequest = 'data\exports\activation\fci-custom-r2-promotion-v2-request.json'
$promotionApproval = 'data\exports\activation\fci-custom-r2-promotion-v2-approval.json'
.\.venv\Scripts\python.exe scripts\promote_code_generation.py approve `
  --request $promotionRequest --reviewer Pum --output $promotionApproval
if ($LASTEXITCODE -ne 0) { throw 'Approval import failed; stop here.' }
```

Run the generation dry-run separately:

```powershell
.\scripts\run_code_generation.ps1 -SnapshotRequest fci-custom-r2 -Stage activate `
  -ActivationRequest $promotionRequest -ActivationApproval $promotionApproval
```

It rechecks current `.env`, settings, runtime/evidence hashes and local collections.
New generation requests do not use `-ActivationReadinessReport`; their checks are
generation-aware and rerun at apply. A changed input requires a new request and
approval, not editing an approved record.

### 11c. Apply only after approval and stopped-process confirmation

Stop FastAPI and the Desktop-owned MCP child before applying. Verify they have
stopped; `-ServicesStopped` is an operator assertion, not a process-killing command.
In the Desktop MCP settings remove overrides for the five promoted keys, especially
`CODE_MODES_ENABLED=true`: it would defeat an `.env` rollback. Let those keys inherit
from `.env`. Interface/disclosure overrides are separate controls and must be
reviewed before restarting the client; do not enable disclosure as part of promotion.

```powershell
.\scripts\run_code_generation.ps1 -SnapshotRequest fci-custom-r2 -Stage activate `
  -ActivationRequest $promotionRequest -ActivationApproval $promotionApproval `
  -ApplyActivation -ServicesStopped
```

Type the prompted `ACTIVATE <immutable-snapshot-id>` only for the approved request.
The switch preserves unrelated `.env` entries, emits immutable intent/result
receipts, and reports `activation_complete=false`: configuration is not proof of a
successfully restarted serving process. Restart only the intended client, confirm
its effective R2 artifact/collection/lineage and FDD v9, then check mode readiness.
Plan/authorize any evidence-returning or paid embedding smoke separately before
running it. Never infer that the ChatGPT plan covers application embedding cost.

If a runtime gate fails, stop the serving child and run the approved rollback:

```powershell
.\.venv\Scripts\python.exe scripts\promote_code_generation.py switch `
  --request $promotionRequest --approval $promotionApproval --action rollback
if ($LASTEXITCODE -ne 0) { throw 'Rollback preflight failed; reconcile configuration.' }
.\.venv\Scripts\python.exe scripts\promote_code_generation.py switch `
  --request $promotionRequest --approval $promotionApproval --action rollback --apply --services-stopped
```

Rollback restores the earlier selection (removing keys previously absent) with
`CODE_MODES_ENABLED=false`. It does not re-enable an old potentially incompatible
code/FDD pair. It remains available if new runtime/evidence files become invalid,
but refuses an altered request/approval, changed `.env`, or conflicting process
overrides. Restart and verify the disabled state. Retain every request and receipt;
never edit or overwrite approval history.

## Reference: feature-flag activation preflight and atomic `.env` update

This reference applies only to an already selected, verified runtime configuration
starting with `CODE_MODES_ENABLED=false`. It is not the next command for current R2.
Use the exact existing request, approval, and readiness paths for that configuration:

```powershell
$activationRequest = Read-Host 'Exact activation request JSON path'
$activationApproval = Read-Host 'Exact approval JSON path'
$activationReadiness = Read-Host 'Exact readiness report JSON path'
.\scripts\run_code_generation.ps1 -SnapshotRequest $request -Stage activate `
  -ActivationRequest $activationRequest -ActivationApproval $activationApproval `
  -ActivationReadinessReport $activationReadiness
```

Only after the no-write preflight and human approval pass, repeat that command
with `-ApplyActivation` and the prompted `ACTIVATE <snapshot-id>` confirmation.
Restart only the intended serving processes and verify the loaded configuration.
For the laptop MCP model, the Desktop client owns the MCP child; restart that
connection and avoid a competing process opening the embedded Qdrant store.
Existing feature-flag rollback restores the flag, not a different code generation:

```powershell
Invoke-CodePython scripts\switch_code_modes.py rollback `
  --request $activationRequest --approval $activationApproval `
  --readiness-report $activationReadiness --apply
```

## Recovery and resuming

- Keep immutable snapshots, review packets, ledgers, embeddings, and reports.
  Do not delete them to reuse a request name or change SVN revision to hide a retry.
- If source import already exists, inspect the receipt and source before retrying.
  Do not rerun `intake-parse -SourceDirectory` blindly; it refuses overwrite.
- For a parser repair, retain the source snapshot and use a new parser generation.
  Run `parse_code_snapshot.py` and `check_code_preindex_gate.py` for that exact
  generation, then export a new dependency packet. Do not reuse a v15 ledger for
  different parser output.
- If a review import failed before writing its output, correct the draft fields
  and retry only the importer. A saved ledger/artifact requires a new review version.
- Restore session variables before resuming in a new terminal. For Steps 9-10,
  restore the path assignments from Step 8 and `$reviewedLineage` from Step 9
  without rerunning creation commands.

If embeddings succeeded but indexing failed, verify the existing embedded
artifact and choose an unused collection (or investigate an incomplete collection
before trying again). These existing commands only index/verify local vectors:

```powershell
$codeArtifact = "data\staging\code_embeddings\$snapshotId\code_index_text_embedding_3_large_v1\code_index_artifact.json"
Invoke-CodePython scripts\index_code_qdrant.py $codeArtifact `
  --qdrant-path data\qdrant_code_local --collection-name $collection
Invoke-CodePython scripts\verify_code_qdrant.py $codeArtifact `
  --qdrant-path data\qdrant_code_local --collection-name $collection
```
