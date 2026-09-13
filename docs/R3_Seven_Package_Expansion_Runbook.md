# R3 seven-package expansion runbook

This runbook creates a controlled R3 candidate above the active R2 baseline.
It does **not** activate R3, change `.env`, launch MCP, or make paid calls by
itself. R2 remains the serving/rollback generation until the separate promotion
controls pass.

R3 is deliberately fixed to the following scope:

- Base snapshot: `fci-custom-r2-ffd9732906d4`.
- FDD generation: `functional_specs_v9`.
- Seven new complete package pairs: exactly 14 new source files.
- One real modified R2 source file, outside those 14 paths.
- No source deletion or formatting-only modification.

## 1. Record and review the package-selection benchmark

Create a new draft from the local template. It is intentionally invalid until
all seven real package pairs are entered.

```powershell
$r3Draft = 'data\evaluations\code_r3_benchmark_draft.json'
if (Test-Path -LiteralPath $r3Draft) { throw "Draft already exists: $r3Draft" }
$r3Template = & .\.venv\Scripts\python.exe scripts\verify_code_r3_benchmark.py --print-template
if ($LASTEXITCODE -ne 0) { throw 'Could not create the R3 benchmark template.' }
[System.IO.File]::WriteAllText(
  (Join-Path (Get-Location).Path $r3Draft),
  ($r3Template + [Environment]::NewLine), [System.Text.UTF8Encoding]::new($false)
)
```

Edit the JSON. Use exact logical paths relative to the selected source root;
they must match later snapshot paths exactly. Enter the seven pairs in this
required mix: two `fdd_enhancement`, two `multi_routine_validation`, one
`table_heavy`, one `no_approved_fdd_mapping`, and one
`cross_package_dependency`.

For each pair enter its spec/body, key routine names, expected callers/callees,
FDD document IDs where applicable, exact enhancement marker text, expected
outcome, and a human rationale. The cross-package pair requires a caller and a
callee. The no-approved-FDD pair must contain no FDD ID or marker. Set
`modified_base_source_path` to the one real changed R2 source path; it cannot
be one of the fourteen new paths.

Validate FDD coverage before SME review:

```powershell
& .\.venv\Scripts\python.exe scripts\verify_code_r3_benchmark.py `
  --manifest $r3Draft `
  --fdd-directory data\staging\functional_specs_v9\processed
```

After the SME accepts the completed draft, create immutable reviewed evidence:

```powershell
$r3Reviewed = 'data\evaluations\code_r3_benchmark_reviewed.json'
$r3Ledger = 'data\exports\evaluations\code_r3_benchmark_review_ledger.json'
& .\.venv\Scripts\python.exe scripts\promote_code_r3_benchmark_review.py `
  --draft-manifest $r3Draft --reviewer '<reviewer>' `
  --approval-note '<identified SME acceptance rationale>' `
  --output $r3Reviewed --ledger $r3Ledger
```

The reviewed file and ledger refuse overwrite. Keep both with R3 evidence.

## 2. Build the complete source snapshot

The source folder must contain every retained R2 source file, all fourteen new
files, and the changed R2 file. Do not pass only the additions: a missing base
source is a deletion. In `data\raw_code\fci-custom-r3\snapshot_request.json`,
set `base_snapshot_id` to `fci-custom-r2-ffd9732906d4` and list the fourteen
new files plus the changed R2 path in `expected_changed_packages`.

Run the normal read-only source import and parse, now bound to the reviewed
benchmark:

```powershell
.\scripts\run_code_generation.ps1 -SnapshotRequest fci-custom-r3 `
  -Stage intake-parse -SourceDirectory '<complete-R3-source-root>' `
  -R3BenchmarkManifest $r3Reviewed
```

The launcher validates the reviewed selection before copying and again against
the immutable snapshot. It fails if added files are not exactly the fourteen
benchmark paths, the changed R2 path differs, a deletion exists, a change is
formatting-only, or a referenced v9 FDD does not exist.

Continue with dependency review and preparation using the normal steps in
[Code_Generation_Launcher_Runbook.md](Code_Generation_Launcher_Runbook.md),
passing `-R3BenchmarkManifest $r3Reviewed` to every R3 launcher command.

## 3. Inspect reuse before authorizing embeddings

After `prepare-index`, run the local preflight. It reads only prepared and R2
embedded artifacts; it does not create a client, call OpenAI, or disclose code.

```powershell
$snapshotId = '<exact immutable R3 snapshot ID>'
$prepared = "data\staging\code_indexes\$snapshotId\code_index_contract_v5\code_index_artifact.json"
$base = 'data\staging\code_embeddings\fci-custom-r2-ffd9732906d4\code_index_text_embedding_3_large_v1\code_index_artifact.json'
& .\.venv\Scripts\python.exe scripts\embed_code_index_artifacts.py $prepared `
  --output-root data\staging\code_embeddings --cache-artifact $base --dry-run
```

Use `external_embedding_inputs` as the bounded number of unique code excerpts
that could be sent to OpenAI. Obtain a new approval that binds that count, the
prepared artifact, the R2 cache artifact, cost, and code disclosure. The R3
launcher repeats this preflight before its own `APPROVE` prompt.

```powershell
.\scripts\run_code_generation.ps1 -SnapshotRequest fci-custom-r3 `
  -Stage embed-index -CollectionName code_custom_r3_v1 `
  -R3BenchmarkManifest $r3Reviewed
```

After an authorized embedding run, the launcher writes a no-overwrite reuse
report. It fails if an unchanged R2 source path contains a new embedding. New
and modified paths may contain cache misses; unchanged units inside a modified
source can still be cached. R3 remains staged only.

## 4. Evaluate code, combined lineage, and no-lineage boundaries

Create a separate R3 code/combined draft manifest using the existing
`code_combined_eval_case_v2` schema. Include at least one reviewed case for
each positive category: enhancement link, multi-routine flow, table-heavy
behavior, and cross-package caller/callee. Do not edit R2's reviewed manifests.
Promote its accepted review using `import_code_combined_eval_review.py`, then
run code-only and combined retrieval evaluation against R3's staged artifact.

For the no-approved-FDD pair, create a separate JSONL draft with this shape:

```json
{"schema_version":"code_documentation_boundary_case_v1","case_id":"r3-no-fdd-001","question":"<exact routine question>","expected_code_paths":["<logical body path>"],"expected_code_symbols":["<routine>"],"expected_code_symbol_policy":"all","expected_documentation_state":"no_reviewed_fdd_lineage","sme_reviewed":false,"review_status":"draft","rationale":"<why no approved v9 FDD/code lineage is expected>"}
```

Promote it without overwriting its draft, then run the local diagnostic after
R3 lineage review:

```powershell
& .\.venv\Scripts\python.exe scripts\promote_code_documentation_boundary_review.py `
  --draft-manifest data\evaluations\code_r3_documentation_boundary_draft.jsonl `
  --reviewer '<reviewer>' --approval-note '<identified SME acceptance rationale>' `
  --output data\evaluations\code_r3_documentation_boundary_reviewed.jsonl `
  --ledger data\exports\evaluations\code_r3_documentation_boundary_review_ledger.json

& .\.venv\Scripts\python.exe scripts\run_code_documentation_boundary_eval.py `
  --eval-file data\evaluations\code_r3_documentation_boundary_reviewed.jsonl `
  --code-artifact '<R3 embedded artifact>' --analysis-directory '<R3 analysis directory>' `
  --fdd-generation functional_specs_v9 --fdd-directory data\staging\functional_specs_v9\processed `
  --lineage-artifact '<R3 reviewed lineage or reviewed bundle>'
```

The boundary diagnostic passes only when the expected code evidence is present
and no reviewed mapping applies. If FDD chunks are retrieved, its report labels
them `unreviewed_candidate`; it never treats them as a confirmed relationship.

Only after code retrieval, combined retrieval, lineage, documentation-boundary,
and bounded MCP UAT pass should a new hash-bound R3 promotion request be made.
