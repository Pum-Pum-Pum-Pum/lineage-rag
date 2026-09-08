# Custom-code generation launcher

Use this runbook to create a complete, immutable custom-code generation from a
read-only SVN working copy. Run one stage at a time from the repository root.
Do not reuse a request directory, immutable snapshot, review packet, ledger,
index collection, or activation artifact.

Terminology matters throughout this runbook:

- A **request directory** is the one-use intake label, for example
  `fci-custom-r3`.
- An **immutable snapshot ID** is emitted only after intake succeeds, for
  example `fci-custom-r3-dca298b717b1`.
- An embedded Qdrant collection is **staged**, not active, until a separately
  approved runtime configuration selects it.
- A collection whose name contains `smoke` is test evidence only. Never use it
  as a candidate for a live code or combined capability.

The launcher reads `C:\SVN\BACKEND`, copies approved source files into local
controlled intake, and never modifies the SVN working copy.

## What is included

Only these extensions are copied, case-insensitively:

```text
.sql  .spc  .prc  .fnc
```

`.ddl`, `.cmt`, and all other extensions are skipped. This does not remove
future Text-to-SQL capability: approved live Oracle metadata, not incomplete
deployment DDL files, will be its schema authority.

All files with an included extension are eligible. A parsed file must still
declare one top-level package, procedure, or function whose owner name matches
the filename stem. This is source-identity validation, not a `_CUSTOM` or
`_MAIN` filename filter.

## 0. Create the request directory and JSON file

Choose the actual numeric SVN revision first. The request directory must be exactly
`<module_set>-r<svn_revision>`. For example, an SVN revision of `3`
uses `fci-custom-r3`.

Do **not** run `mkdir data/raw_code/fci-custom-r3/snapshot_request.json`:
that creates a directory called `snapshot_request.json`, not a file.

```powershell
$svnRevision = '3' # Replace with the actual numeric SVN revision.
$request = "fci-custom-r$svnRevision"
$intake = Join-Path 'data\raw_code' $request
$requestFile = Join-Path $intake 'snapshot_request.json'

if (Test-Path -LiteralPath $requestFile) {
  throw "Snapshot request already exists: $requestFile"
}

New-Item -ItemType Directory -Force -Path $intake | Out-Null

@'
{
  "schema_version": "code_snapshot_request_v1",
  "module_set": "fci-custom",
  "svn_revision": "3",
  "application_build": "14.7",
  "reviewer": "Pavan",
  "base_snapshot_id": null,
  "expected_changed_packages": [],
  "compiler_context": {
    "oracle_version": null,
    "plsql_ccflags": null
  }
}
'@ | Set-Content -LiteralPath $requestFile -Encoding utf8
```

Edit the created file before intake:

- `application_build`: application/FCIS build.
- `reviewer`: accountable reviewer or approved role.
- `base_snapshot_id`: exact previous immutable snapshot ID, such as
  `fci-custom-r1-a47f5d4d54e1`. Use the JSON value `null` (not an empty
  string) only for the first ever snapshot.
- `expected_changed_packages`: leave `[]` unless it is a reviewed expectation;
  the exact hash comparison remains the source of truth.
- `oracle_version` and `plsql_ccflags`: use `null` when unknown. The system
  keeps conditional compilation behaviour explicitly unknown.

## 1. Intake, snapshot, parse, and local pre-index gate

```powershell
.\scripts\run_code_generation.ps1 -SnapshotRequest fci-custom-r3 -Stage intake-parse `
  -SourceDirectory 'C:\SVN\BACKEND'
```

This is local only: it performs no OpenAI call and no Qdrant write. Its final
line has the immutable snapshot identity, for example:

```text
INTAKE/PARSE COMPLETE: immutable snapshot=fci-custom-r3-dca298b717b1. No OpenAI call or Qdrant write occurred.
```

Copy only the value after `immutable snapshot=`. In the example it is:

```text
fci-custom-r3-dca298b717b1
```

Use that exact value in every later export, review, and ledger filename. Do
not substitute the request name (`fci-custom-r3`) for the immutable ID, and do
not guess a snapshot from directory order.

### If parsing needs a repaired parser generation

`intake-parse` is deliberately no-overwrite. Once it has copied the source
and published an immutable snapshot, do **not** run it again against the same
request directory and do not delete its intake evidence. If a parser defect is
repaired after publication, analyse the emitted snapshot under a new parser
generation, then run the pre-index gate for that exact generation:

```powershell
.\.venv\Scripts\python.exe scripts\parse_code_snapshot.py <snapshot-id> `
  --generation plsql_antlr_4_13_2_analysis_v15

.\.venv\Scripts\python.exe scripts\check_code_preindex_gate.py <snapshot-id> `
  --snapshot-root data\code_snapshots `
  --generation plsql_antlr_4_13_2_analysis_v15 `
  --output data\exports\code_analysis\<snapshot-id>-plsql_antlr_4_13_2_analysis_v15-preindex-gate.json
```

Use a new request only when the copied source itself must change. A new parser
generation preserves the historical failed parser evidence without rewriting
the immutable source snapshot.

## 2. Export and review the dependency packet

Replace `<snapshot-id>` below with the emitted immutable ID.

```powershell
.\.venv\Scripts\python.exe scripts\export_code_dependency_review.py `
  <snapshot-id>
```

This creates two no-overwrite files under `data\exports\code_analysis\`:

```text
<snapshot-id>-plsql_antlr_4_13_2_analysis_v15-dependency-review.json
<snapshot-id>-plsql_antlr_4_13_2_analysis_v15-dependency-review.md
```

Review the Markdown with the SME. Keep the canonical JSON unchanged; record
the SME verdict, corrected classification where needed, and rationale in the
Markdown. This step is local only.

Use the Markdown review fields exactly as follows. The importer is intentionally
strict so a narrative note cannot be mistaken for a machine-readable correction.

| SME verdict | `SME corrected kind/state` | Where to put the explanation |
| --- | --- | --- |
| `accepted` | Leave blank | `SME rationale` |
| `corrected` | Exact `kind / state`, for example `dynamic_sql / dynamic_known` | `SME rationale` |
| `needs_more_context` | Leave blank | `SME rationale` |

Do not place prose such as a SQL statement in `SME corrected kind/state`.
Correct the field and rerun only the **import** command; the original packet
and Markdown remain the reviewed source record.

## 3. Import the immutable reviewed dependency ledger

```powershell
.\.venv\Scripts\python.exe scripts\import_code_dependency_review.py `
  data\exports\code_analysis\<snapshot-id>-plsql_antlr_4_13_2_analysis_v15-dependency-review.json `
  data\exports\code_analysis\<snapshot-id>-plsql_antlr_4_13_2_analysis_v15-dependency-review.md `
  --reviewer Pavan `
  --output data\exports\code_analysis\reviews\<snapshot-id>-dependency-review-ledger.json
```

The ledger is hash-bound to both the canonical packet and reviewed Markdown.
It is no-overwrite and does not call OpenAI.

## 4. Prepare and verify reviewed index artifacts

```powershell
.\scripts\run_code_generation.ps1 -SnapshotRequest fci-custom-r3 -Stage prepare-index `
  -DependencyReviewLedger data\exports\code_analysis\reviews\<snapshot-id>-dependency-review-ledger.json
```

This produces and verifies the deterministic reviewed artifact at:

```text
data/staging/code_indexes/<snapshot-id>/code_index_contract_v5/code_index_artifact.json
```

It remains local: no OpenAI call, Qdrant write, or activation occurs.

## 5. Embed, index, and verify a new isolated code generation

Choose a new, unused collection name that matches the snapshot, for example
`code_custom_r3_v1`:

```powershell
.\scripts\run_code_generation.ps1 -SnapshotRequest fci-custom-r3 -Stage embed-index `
  -CollectionName code_custom_r3_v1
```

The launcher requires the exact `APPROVE` acknowledgement before disclosing
prepared internal PL/SQL to OpenAI. It may incur cost. It then creates and
exactly verifies the isolated collection; it does not activate it.

Content-identical embedding inputs can reuse the existing cache. Changed files
produce changed occurrence identities, and the new generation remains complete
and independently reproducible.

Use `code_custom_smoke_*` only for a disposable local smoke collection. A
candidate intended for later activation needs a normal versioned collection
name such as `code_custom_r<revision>_v1` and its own reviewed evidence.

## 6. Evaluate

```powershell
.\scripts\run_code_generation.ps1 -SnapshotRequest fci-custom-r3 -Stage evaluate
```

The default launcher evaluation is lexical and makes no query-embedding call.
Dense or hybrid evaluation requires a separately reviewed precomputed query
vector artifact and the isolated collection name; the launcher will not create
query embeddings implicitly.

Combined FDD/code retrieval also requires the approved current FDD generation
and a reviewed FDD-to-code lineage artifact. Do not treat code-only evaluation
as proof of combined retrieval, citation, or answer quality.

For new or changed reviewed evaluation cases, record the expected code path as
the full logical path preserved from the imported source tree, for example
`BACKEND/LOB/SQL/pkgamlaintegration_p_custom.sql`. Older filename-only
expectations are a compatibility bridge only: they match an imported path only
when that basename is unique. A duplicate basename fails closed and the
reviewed manifest must be updated with the full logical path.

Every evaluation report is no-overwrite. Keep a failed report as historical
evidence; after a deterministic evaluator repair, rerun the evaluation to
create a new report rather than changing the earlier result.

## 7. Prepare the separate runtime activation contract

Index creation is not activation. Before changing `.env`, complete the
approved retrieval, citation, grounded-answer, SME, readiness, and rollback
gates.

> **Important -- this launcher does not promote a code generation.** `-Stage
> activate` changes only the `CODE_MODES_ENABLED` feature flag. It does not
> write `CODE_INDEX_ARTIFACT_PATH`, `CODE_ANALYSIS_DIRECTORY`,
> `CODE_QDRANT_COLLECTION_NAME`, or a lineage artifact selection. Do not try
> to promote a staged collection by manually editing `.env` or by running
> `activate` against a smoke snapshot. A safe, approval-bound generation
> promotion mechanism is a separate prerequisite for a future live candidate.

The activation procedure below is therefore only for enabling an already
selected, hash-bound runtime configuration that starts with
`CODE_MODES_ENABLED=false`. Create a hash-bound request and its initial
preflight from that approved configuration:

```powershell
.\.venv\Scripts\python.exe scripts\prepare_code_mode_activation.py `
  --readiness-report data\exports\activations\code\<readiness-report>.json `
  --requested-by Pavan `
  --request-output data\exports\activations\code\<snapshot-id>-activation-request.json `
  --preflight-output data\exports\activations\code\<snapshot-id>-activation-preflight.json
```

The output includes an identity like:

```text
request_identity_sha256=f927d16dde75bdf6ef3fc8d96ad07279ae22185b1ea6c3aea030e57b44692fff
ready_to_apply=false
approval_required=true
```

The first preflight is expected to be blocked until a separate human approval
artifact exists. Record that approval without overwriting the request:

```powershell
.\.venv\Scripts\python.exe scripts\record_code_mode_activation_approval.py `
  --request data\exports\activations\code\<snapshot-id>-activation-request.json `
  --approved-by Pavan `
  --output-file data\exports\activations\code\<snapshot-id>-activation-approval.json `
  --confirm-approved
```

Add `--authorize-paid-smoke` and
`--authorize-internal-evidence-disclosure` only when those separate operations
are explicitly approved. They are not needed merely to switch the feature flag.

## 8. Feature-flag activation preflight and atomic `.env` update

The launcher now performs the existing approval-bound switch. It changes
**only** `CODE_MODES_ENABLED=false` to `CODE_MODES_ENABLED=true`, atomically,
after validating the request, approval, readiness report, runtime hashes, and
the disabled starting state. It never overwrites a duplicate, invalid, or
already-enabled flag.

Run a no-write preflight first:

```powershell
.\scripts\run_code_generation.ps1 -SnapshotRequest fci-custom-r3 -Stage activate `
  -ActivationRequest data\exports\activations\code\<snapshot-id>-activation-request.json `
  -ActivationApproval data\exports\activations\code\<snapshot-id>-activation-approval.json `
  -ActivationReadinessReport data\exports\activations\code\<readiness-report>.json
```

When that passes and the human authorization is in place, apply it:

```powershell
.\scripts\run_code_generation.ps1 -SnapshotRequest fci-custom-r3 -Stage activate `
  -ActivationRequest data\exports\activations\code\<snapshot-id>-activation-request.json `
  -ActivationApproval data\exports\activations\code\<snapshot-id>-activation-approval.json `
  -ActivationReadinessReport data\exports\activations\code\<readiness-report>.json `
  -ApplyActivation
```

Type exactly `ACTIVATE <snapshot-id>` when prompted. On success, `.env` has
`CODE_MODES_ENABLED=true`. The launcher does not restart services. Restart
FastAPI and Streamlit if they are running, and toggle the Desktop-owned MCP
server off/on so its child process reads the new `.env`.

The code collection/artifact/analysis and lineage identities are bound into the
activation request. The feature-flag launcher deliberately does not substitute
or rewrite those selected generation values during activation; a changed
candidate requires a new request and approval. If code modes are already true,
this activation command fails closed rather than claiming to have promoted a
different collection.

If any later runtime gate fails, use the approved rollback path:

```powershell
.\.venv\Scripts\python.exe scripts\switch_code_modes.py rollback `
  --request data\exports\activations\code\<snapshot-id>-activation-request.json `
  --approval data\exports\activations\code\<snapshot-id>-activation-approval.json `
  --readiness-report data\exports\activations\code\<readiness-report>.json `
  --apply
```

Restart the serving processes and verify code/combined readiness after rollback.

## Recovery rules

- A failed local import or parse does not call OpenAI. Before snapshot
  publication, correct the request/intake and retry. After publication, use a
  new request only when the source must change; use a new parser generation
  when the immutable source is already correct and parser behaviour is fixed.
- Do not overwrite any published snapshot, packet, ledger, collection,
  evaluation, request, approval, or report.
- Never edit archived snapshot source; publish a new complete snapshot from the
  current SVN working copy.
- A successful `embed-index` collection is staged only. It cannot become live
  until a separate generation-promotion implementation and activation contract
  both pass.
