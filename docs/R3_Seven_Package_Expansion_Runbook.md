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
$r3Ledger = 'data\exports\code_analysis\reviews\code_r3_benchmark_review_ledger.json'
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

Copy the emitted immutable snapshot ID. If an earlier historical snapshot uses
the same request name, preserve it and pass the reviewed ID explicitly to every
post-intake launcher stage; never select by folder order or delete history:

```powershell
$r3SnapshotId = '<exact reviewed immutable snapshot ID>'

.\scripts\run_code_generation.ps1 -SnapshotRequest fci-custom-r3 `
  -ImmutableSnapshotId $r3SnapshotId `
  -Stage prepare-index -R3BenchmarkManifest $r3Reviewed `
  -ParseGeneration plsql_antlr_4_13_2_analysis_v15 `
  -DependencyReviewLedger '<reviewed dependency ledger path>'
```

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

Start with the no-cost bootstrap below. It creates one precise **code-only**
draft case per selected routine (sixteen cases for the current R3 benchmark)
and one separate draft for the
intentionally undocumented Data Center pair. It does not create a combined
case: combined cases must wait for the later reviewed R3 lineage artifact.
Neither command changes R2, writes Qdrant, calls OpenAI, or activates R3.

```powershell
$r3CodeDraft = 'data\evaluations\code_r3_grounded_eval_v2_draft.jsonl'
$r3BoundaryDraft = 'data\evaluations\code_r3_documentation_boundary_v2_draft.jsonl'
$r3CodeReview = 'data\exports\evaluations\code_r3_grounded_eval_v2_review.md'

& .\.venv\Scripts\python.exe scripts\bootstrap_code_r3_evaluation_drafts.py `
  --benchmark-manifest $r3Reviewed `
  --code-output $r3CodeDraft --boundary-output $r3BoundaryDraft

& .\.venv\Scripts\python.exe scripts\render_code_combined_eval_review.py `
  --eval-file $r3CodeDraft --output-file $r3CodeReview
```

Review `code_r3_grounded_eval_review.md`. For every case, set
`SME verdict: accepted`, leave `SME corrected expectations:` blank, and provide
a nonblank rationale. If a question, path, or expected routine is wrong, correct
the **draft JSONL**, discard the unreviewed review packet, and use a new review
packet filename. Do not manually set `sme_reviewed` to true.

After that review is saved, promote only the code draft:

```powershell
$r3CodeReviewedDirectory = 'data\evaluations'
$r3CodeEvalLedger = 'data\exports\evaluations\code_r3_grounded_eval_v2_review_ledger.json'

& .\.venv\Scripts\python.exe scripts\import_code_combined_eval_review.py `
  --eval-file $r3CodeDraft --review-file $r3CodeReview --reviewer $reviewer `
  --global-approval-note 'Reviewed R3 code-only retrieval expectations.' `
  --output-directory $r3CodeReviewedDirectory --ledger-file $r3CodeEvalLedger
```

This writes `data\evaluations\code_r3_grounded_eval_v2_reviewed.jsonl`. Run the
new code-only gate and the unchanged R2 regression gate separately:

```powershell
.\scripts\run_code_generation.ps1 -SnapshotRequest fci-custom-r3 `
  -ImmutableSnapshotId $r3SnapshotId -Stage evaluate `
  -R3BenchmarkManifest $r3Reviewed `
  -EvaluationFile data\evaluations\code_r3_grounded_eval_v2_reviewed.jsonl

.\scripts\run_code_generation.ps1 -SnapshotRequest fci-custom-r3 `
  -ImmutableSnapshotId $r3SnapshotId -Stage evaluate `
  -R3BenchmarkManifest $r3Reviewed `
  -EvaluationFile data\evaluations\code_grounded_eval_v1_reviewed.jsonl
```

Both are lexical by default and make no external calls. Do **not** run the
launcher without `-EvaluationFile` at this stage: its default is the R2 suite
and would omit the new R3 test cases.

## 5. Create the R3 scoped enhancement registry and candidate proposals

R3's two Zakat pairs reuse the same R25/REQ03 comment identity but point to
different FDDs. Create a new source-scoped registry so that identity is a
precise discovery hint for its own package body only. The existing R2 registry
is retained unchanged.

```powershell
$r3Registry = 'data\evaluations\enhancement_fdd_registry_r3_v1.json'

& .\.venv\Scripts\python.exe scripts\bootstrap_r3_enhancement_registry.py `
  --base-registry data\evaluations\enhancement_fdd_registry_v1.json `
  --benchmark-manifest $r3Reviewed --output $r3Registry `
  --basis 'R3 benchmark review accepted the exact package-body R25 enhancement comments and their stated FDD identities; discovery only, not lineage approval.'
```

Then create a new proposal directory. This reads local vectors and source
comments only; it does not open Qdrant, call OpenAI, write `.env`, or activate
R3.

```powershell
$r3ProposalDirectory = 'data\exports\lineage_proposals\fci-custom-r3-fdd-v9-v1'

& .\.venv\Scripts\python.exe scripts\propose_fdd_code_lineage.py `
  --fdd-stage data\staging\functional_specs_v9 `
  --snapshot-directory "data\code_snapshots\$r3SnapshotId" `
  --analysis-directory "data\staging\code\$r3SnapshotId\plsql_antlr_4_13_2_analysis_v15" `
  --code-artifact "data\staging\code_embeddings\$r3SnapshotId\code_index_text_embedding_3_large_v1\code_index_artifact.json" `
  --enhancement-registry $r3Registry --output-directory $r3ProposalDirectory
```

Read its `review.html` before selecting any proposed mapping. A proposal is
candidate evidence, never an approved FDD-to-code relationship.

For the reviewed current R3 Zakat candidates, create a new definition containing
only the `comment_region_and_similarity` implementation candidates for the two
benchmark FDDs. Similarity-only suggestions and the intentionally undocumented
pair are excluded by construction.

```powershell
$r3LineageDirectory = 'data\staging\fdd_code_lineage\fci-custom-r3-v1'
$r3LineageDefinition = Join-Path $r3LineageDirectory 'candidate_definition.json'

& .\.venv\Scripts\python.exe scripts\bootstrap_r3_lineage_definition.py `
  --benchmark-manifest $r3Reviewed `
  --proposal-report "$r3ProposalDirectory\proposals.json" `
  --output $r3LineageDefinition
```

Read the resulting JSON before building the candidate lineage artifact. It is a
candidate-only input; it cannot make a reviewed mapping or alter runtime.

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

## 6. Evaluate reviewed R3 FDD/code lineage

After the separate lineage-review import produces a reviewed R3 lineage
artifact, create a new combined-evaluation draft. This is deliberately a
separate manifest from the R2 AML combined suite: it derives one exact case per
reviewed R3 implementation target and binds that case to its reviewed FDD
mapping. It makes no external calls and does not change runtime configuration.

```powershell
$r3CombinedDraft = 'data\evaluations\code_r3_combined_eval_v1_draft.jsonl'
$r3CombinedReview = 'data\exports\evaluations\code_r3_combined_eval_v1_review.md'

& .\.venv\Scripts\python.exe scripts\bootstrap_r3_combined_evaluation_draft.py `
  --benchmark-manifest $r3Reviewed `
  --lineage-artifact $r3ReviewedLineage `
  --output $r3CombinedDraft

& .\.venv\Scripts\python.exe scripts\render_code_combined_eval_review.py `
  --eval-file $r3CombinedDraft --output-file $r3CombinedReview
```

Review every case in `code_r3_combined_eval_v1_review.md`. Confirm the FDD and
routine identity; set `SME verdict: accepted`, leave corrected expectations
blank, and give a nonblank rationale. Then promote the accepted draft:

```powershell
$r3CombinedLedger = 'data\exports\evaluations\code_r3_combined_eval_v1_review_ledger.json'

& .\.venv\Scripts\python.exe scripts\import_code_combined_eval_review.py `
  --eval-file $r3CombinedDraft --review-file $r3CombinedReview `
  --reviewer $reviewer `
  --global-approval-note 'Reviewed R3 combined FDD/code retrieval expectations.' `
  --output-directory data\evaluations --ledger-file $r3CombinedLedger
```

Run the resulting `data\evaluations\code_r3_combined_eval_v1_reviewed.jsonl`
with `scripts\run_code_combined_retrieval_eval.py`, the R3 embedded artifact,
the reviewed R3 lineage artifact, and the active `functional_specs_v9` FDD
generation. This is lexical/local by default; dense or hybrid evaluation needs
separate precomputed query vectors and explicit approval.

## 7. Re-review unchanged R2 lineage for the R3 artifact

R3 contains the R2 AML source files, but the old reviewed lineage is bound to
the R2 code-artifact identity and cannot be selected for R3 serving directly.
Never omit it silently and never combine cross-generation artifacts. Instead,
derive an R3 *candidate* only when every retrieval record for every previously
mapped path has the same content, cache key, routine identity, and source range
in R3. The candidate remains unreviewed; it needs a new R3 SME review/import.

```powershell
$r3CarryDirectory = 'data\staging\fdd_code_lineage\fci-custom-r3-carryforward-v1'
$r3CarryCandidate = Join-Path $r3CarryDirectory 'candidate_lineage_artifact.json'
$r3CarryReview = Join-Path $r3CarryDirectory 'candidate_review.md'

& .\.venv\Scripts\python.exe scripts\bootstrap_r3_lineage_carryforward.py `
  --source-lineage data\staging\fdd_code_lineage\fci-custom-r2-consolidated-v1\reviewed_lineage_bundle.json `
  --source-code-artifact data\staging\code_embeddings\fci-custom-r2-ffd9732906d4\code_index_text_embedding_3_large_v1\code_index_artifact.json `
  --target-code-artifact $r3CodeArtifact `
  --analysis-directory $r3AnalysisDirectory `
  --fdd-processed-directory $r3FddDirectory `
  --output $r3CarryCandidate

& .\.venv\Scripts\python.exe scripts\render_fdd_code_lineage_review.py `
  $r3CarryCandidate --output $r3CarryReview
```

If the carry-forward check reports a changed mapped source path, stop and make
a new specific candidate from the changed R3 source instead. If it succeeds,
review every carried mapping and import it to
`reviewed_lineage_artifact.json` using the standard lineage-review importer.
Only then may a same-generation R3 bundle consolidate the carried R2 mappings
with the new R3 Zakat mappings.

Only after code retrieval, combined retrieval, lineage, documentation-boundary,
and bounded MCP UAT pass should a new hash-bound R3 promotion request be made.
