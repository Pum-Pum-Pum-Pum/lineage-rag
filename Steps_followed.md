# Phase 2 progress

## Step 150 - Define the custom-code snapshot contracts

Implemented frozen, strict Pydantic contracts in
`app/code_ingestion/snapshot_models.py` for:

- snapshot requests and optional compiler context;
- validated source-file metadata;
- deterministic snapshot differences;
- immutable snapshot manifests.

The request accepts a module set, numeric SVN revision, application build,
reviewer, optional base snapshot, optional expected package list, and optional
Oracle compiler context. Relative paths are normalized to POSIX form and path
traversal, absolute paths, duplicate case-insensitive expectations, and unknown
schema fields fail closed.

Python example:

```python
request = SnapshotRequest.model_validate_json(request_path.read_text("utf-8"))
```

Production interpretation: these schemas establish a stable, reviewable input
contract before parsers, embeddings, or vector stores are introduced. Frozen
models prevent accidental in-process mutation; persisted immutability is
enforced by Step 152.

Failure-mode tests reject unsafe paths, duplicate paths, unknown fields, and
post-validation model assignment.

## Step 151 - Add streaming intake validation

Implemented `validate_code_intake(...)` in
`app/code_ingestion/intake_validation.py` with a case-insensitive `.sql`,
`.prc`, `.fnc`, and `.ddl` allowlist. It streams exact hashing, encoding
validation, newline-normalized hashing, binary checks, and privacy-safe secret
category detection. UTF-8, BOM-marked UTF-16, and Windows-1252 are recorded
explicitly.

Python example:

```python
report = validate_code_intake(Path("data/raw_code/fci-custom-r12345/source"))
```

Files above 5 MiB receive a `large_file` warning but are accepted. This keeps
large enterprise packages available while making later isolated parser limits
visible. Potential secrets and binary or non-allowlisted files fail the entire
intake before local publication.

Failure-mode tests cover binary bytes, secret assignments without value
leakage, non-allowlisted files, legacy encoding, chunk-boundary newline
normalization, and accepted oversized files.

## Step 152 - Publish immutable snapshots and compute deterministic diffs

Implemented `build_code_snapshot(...)` and the local launcher
`scripts/build_code_snapshot.py`. Publication validates the complete intake,
derives a content-addressed ID, copies to a temporary directory, verifies every
copied source hash, then atomically promotes it beneath `data/code_snapshots/`.
An existing ID is never overwritten.

Python example:

```python
manifest = build_code_snapshot(intake_directory, snapshot_root)
```

Complete manifests—not a user-entered list—determine added, modified, deleted,
unchanged, formatting-only, and unambiguous exact-rename states. Optional
expected filenames produce missing/unexpected review signals. Git exclusions
protect proprietary raw code and generated snapshots from accidental commits.

Production interpretation: a snapshot is a reproducible evidence boundary,
not an active index. These steps make no OpenAI calls and no Qdrant writes.

Failure-mode tests prove no-overwrite behavior, tamper detection, failure on an
intake mutation between validation and copy, full deletion detection, exact
rename handling, and validate-only operation with no writes.

Runbook: `docs/Steps_for_Code_Snapshot_Ingestion.md`.

Verification completed after Step 152:

```text
Focused Phase 2 tests: 21 passed
Full regression suite: 414 passed, 1 existing Starlette TestClient warning
uv lock --check: passed with a workspace-local uv cache
git diff --check: passed
```

Pytest discovery is now constrained to `tests/` in `pyproject.toml`. This
prevents generated or access-restricted local artifacts under `data/tmp/` from
being mistaken for test collections; no generated data was deleted.

### Step 152 remediation - Centralized source-extension policy

Removed the code-lane extension constant and centralized FDD/code mappings in
`config/ingestion_sources.toml`. `app/core/ingestion_policy.py` validates the
policy schema, extension syntax, duplicates, lane, and implemented handler.

Python example:

```python
policy = load_ingestion_source_policy("config/ingestion_sources.toml")
report = validate_code_intake(source_directory, source_policy=policy)
```

Extensions such as `.pkb` can be mapped to the existing `plsql` handler without
a Python edit. An unknown handler such as `pdf` fails closed because an
allowlist must not claim an extractor capability that does not exist. Phase 1
DOCX discovery and extraction now consult the same policy.

The normalized policy SHA-256 and selected per-file handler are stored in code
snapshot manifests. A base/current policy change is reported separately from
file changes, preventing identical bytes from silently receiving a different
interpretation contract. The native deployment bundle now includes the policy.

Failure-mode tests cover wildcard extensions, unimplemented handlers, semantic
policy hashing, configuration-only FDD/code extensions, snapshot policy drift,
and deployment packaging. Focused remediation verification passed 36 tests.
The full regression suite passed 420 tests with the existing non-failing
Starlette TestClient warning, and `uv lock --check` passed without changing the
lockfile.

## Step 153 - Pin the PL/SQL grammar and preserve conditional compilation

Pinned `antlr4-python3-runtime==4.13.2` and vendored the grammars-v4 PL/SQL
lexer/parser sources at commit
`a7704d4c029c33a89818ac103f758f7c72d8d16c`, with source hashes and the ANTLR
generator hash recorded under `app/code_ingestion/grammar/plsql/`. Generated
Python parser sources are local runtime artifacts; no live grammar download is
required during ingestion.

Python example:

```python
view = build_conditional_parse_view(
    source_text,
    source_path="packages/pkg_customer.sql",
    compiler_context=snapshot.request.compiler_context,
)
```

The parser attempts immutable original source first. A separate
offset/newline-preserving parse view may mask only directive syntax when it
produces fewer grammar errors. `$IF`, `$ELSIF`, `$ELSE`, `$END`, `$ERROR`, and
`$$...` regions remain mapped to original lines. Known compiler context can
mark branches active/inactive; missing or unsupported context remains
`conditional_unknown` or `unresolved`.

Production interpretation: parser convenience never rewrites the citation
source or turns mutually exclusive branches into unconditional deployed facts.
Failure tests cover nested directives, comments/string lookalikes, `$ERROR`,
unclosed regions, known/unknown compiler context, grammar gaps, pinned hashes,
runtime versions, and the documented generated-code target fix.

## Step 154 - Add resource-isolated full, segmented, and fallback parsing

Implemented full-file ANTLR parsing, token-aware routine segmentation, bounded
original-line fallback chunks, and exact source maps. Each snapshot file is
parsed in a separate subprocess through `parse_file_isolated(...)`; the parent
enforces configurable timeout and memory limits and records duration, observed
peak memory, state, and diagnostics.

Python example:

```python
artifact = parse_file_isolated(
    source_file,
    snapshot_id=manifest.snapshot_id,
    source_path=entry.path,
    source_sha256=entry.sha256,
    encoding=entry.encoding,
    compiler_context=manifest.request.compiler_context,
    work_root=worker_root,
    timeout_seconds=settings.code_parse_timeout_seconds,
    memory_limit_bytes=settings.code_parse_memory_limit_mib * 1024 * 1024,
)
```

States are `full_parse`, `segmented_parse`, `fallback_parse`, and `failed`.
Files over 5 MiB remain eligible. A timeout or memory breach retains bounded
original source only when the immutable hash still matches; changed bytes fail
closed. Standalone `CREATE PROCEDURE` and `CREATE FUNCTION` files are extracted
as well as package routines.

Production interpretation: one pathological package cannot silently exhaust
or stop the entire parser process, and degraded evidence never masquerades as a
clean structural parse. Failure tests cover timeout, memory exhaustion, source
hash mismatch, invalid limits, nested block/string boundaries, full-parse
failure with successful segments, complete fallback, and a file above 5 MiB.

## Step 155 - Build selective context-enriched code retrieval units

Implemented exact-source retrieval artifacts for routines, types, constants,
package globals, cursors, and declared fallback chunks. Routine units link only
the package declarations referenced by their lexer-visible source tokens.
Compact derived context is bounded, deduplicated, and marked
`DERIVED RETRIEVAL CONTEXT - NOT A CITATION SOURCE`.

Python example:

```python
retrieval = build_code_retrieval_artifact(
    parse_artifact,
    source_text,
    verified_source_sha256=entry.sha256,
)
```

Added `scripts/parse_code_snapshot.py` to verify one immutable snapshot and
atomically publish no-overwrite parse/retrieval artifacts beneath
`data/staging/code/<snapshot-id>/plsql_antlr_4_13_2_v1/`. Stage manifests
account for every file and expose `complete`, `complete_with_degradation`, or
`failed`. The launcher performs no OpenAI or Qdrant calls.

Production interpretation: embeddings can later use compact enriched text,
while answers and citations must use original `text` plus exact `source_map`.
The full package header is not repeated across every procedure, controlling
cost and provenance noise. Failure tests cover unused declarations, context
caps/deduplication, deterministic IDs, incorrect source hashes, fallback
preservation, inconsistent manifest counts, snapshot tampering, atomic
publication, and no-overwrite behavior.

Runbook: `docs/Steps_for_Code_Snapshot_Ingestion.md`.

Verification completed after Step 155:

```text
Focused Steps 153-155 tests: 37 passed
Full regression suite: 457 passed, 1 existing Starlette TestClient warning
uv lock --check: passed with a workspace-local uv cache
git diff --check: passed (line-ending notices only)
```

No OpenAI API call, embedding operation, Qdrant write, or paid operation was
performed in Steps 153-155.

## Step 156 - Add overload-safe Oracle symbol identity

Implemented Oracle-aware identifiers, deterministic parameter contracts,
overload discriminator hashes, full declaration signature hashes, symbol keys,
and per-source occurrence IDs. Unquoted identifiers canonicalize to uppercase;
quoted identifiers retain exact spelling and quotes. Nested routines carry
their enclosing routine scope.

Python example:

```python
symbols = extract_symbols(parse_artifact, module_id="fci-custom")
diagnostics = diagnose_symbol_groups(symbols)
```

The overload discriminator uses ordered parameter names, quoted-name state,
canonical declared types, and type families. Modes, defaults, `NOCOPY`, and
function return type remain in the separate declaration hash because they
cannot safely distinguish an Oracle overload. Declaration and implementation
occurrences may share one symbol key without overwriting each other.

Production interpretation: the stable symbol key identifies one logical
overload, while occurrence IDs preserve exact source citations. Mode-only,
return-only, duplicate-role, and incompatible declaration/implementation cases
fail closed through diagnostics. Default-only spec/body differences remain
visible without being falsely treated as incompatibility.

Failure-mode tests cover numeric/character overloads, mode-only and return-only
collisions, quoted/unquoted names, declaration/body pairing, default metadata,
and nested local-routine qualification.

## Step 157 - Extract dependencies and explicit unavailable boundaries

Added snapshot-scoped call and table resolution, package declaration
references, cursor-query dependencies, dynamic-SQL detection, external package
classification, and configured hidden-kernel boundaries.

Python example:

```python
dependencies = extract_dependencies(
    source_text,
    parse_artifact,
    file_symbols=file_symbols,
    all_symbols=all_snapshot_symbols,
    schema_objects=all_snapshot_schema_objects,
    policy=analysis_policy,
)
```

Every edge records exact source range, extraction method, confidence, target,
resolution state, and all plausible symbol candidates. Ambiguous overloads are
never collapsed to one arbitrary target. `EXECUTE IMMEDIATE` and dynamic
`OPEN ... FOR` become `dynamic_unknown`; configured kernel calls become
`kernel_unavailable`.

The versioned `config/code_analysis.toml` stores external package prefixes,
ignored built-ins, and SME-reviewed kernel prefixes. Kernel prefixes are empty
by default so the system does not guess this business boundary. Its normalized
SHA-256 is stored in analysis artifacts and the stage manifest.

Production interpretation: these edges are evidence for later impact analysis,
not proof of runtime execution or root cause. Failure tests cover ambiguous
overloads, unresolved calls, kernel/external boundaries, static reads/writes,
dynamic SQL, package declarations, cursor SQL, comma joins, and preventing a
nested routine body from being attributed to its parent.

## Step 158 - Extract DDL and resolve synonyms across the snapshot

Implemented structural artifacts for tables, columns, defaults, constraints,
views, sequences, indexes, object/collection types, and synonyms. Synonyms are
resolved only after all approved snapshot files have been analyzed.

Python example:

```python
objects, synonyms, diagnostics = extract_ddl_structures(source_text, parse_artifact)
resolved_synonyms = resolve_synonyms(all_snapshot_objects, all_snapshot_synonyms)
```

Resolution states are `resolved_in_snapshot`, `external_schema`,
`database_link`, `ambiguous`, and `cyclic`. Database links are recorded but
never followed. A degraded parse emits no schema claim. Duplicate symbol or
schema identities remain in diagnostic artifacts and set the analysis stage to
`failed` rather than permitting last-file-wins behavior.

The expanded evidence contract publishes atomically beneath
`data/staging/code/<snapshot-id>/plsql_antlr_4_13_2_analysis_v1/`, preserving
the Step 155 generation independently. Each file receives parse, retrieval,
and static-analysis artifacts; cross-file resolution occurs before publication.

Production interpretation: static DDL supports code understanding but cannot
prove live Oracle ownership, editions, synonym state, privileges, or database
link targets. Those remain Phase 3 metadata responsibilities.

Failure-mode tests cover constraints, quoted objects, same-snapshot synonym
chains, external targets, database links, cycles, ambiguous names, duplicate
schema identities, cross-file resolution, degraded parsing, analysis-stage
failure, atomic publication, and policy validation.

Verification completed after Step 158:

```text
Focused Steps 156-158 tests: 32 passed
Full regression suite: 482 passed, 1 existing Starlette TestClient warning
```

No real custom packages were required. No OpenAI API call, embedding operation,
Qdrant write, or paid operation was performed in Steps 156-158.

## Pre-Step 159 real-corpus readiness gate

### Practical substep 1 - Validate and publish the first curated snapshot

Accepted the reviewer-supplied request for SVN revision `1`, application build
`Code1`, reviewer `AIAgentSmith`, and no base snapshot. Renamed the intake
directory from `sources/` to the required `source/` only after resolving the
exact paths and confirming the target was absent. Added the real `.spc` package
specification extension to the versioned source policy.

Python command:

```powershell
& .\.venv\Scripts\python.exe scripts\build_code_snapshot.py `
  data\raw_code\fci-custom-r1
```

Published immutable snapshot `fci-custom-r1-a47f5d4d54e1`: two UTF-8 files,
36,823 total lines, no intake warnings, no binary/secret/size rejection, and no
external call. This is a complete curated module snapshot, not a changed-file
patch.

Production interpretation: request metadata and hashes establish what bytes
were approved for analysis. They do not prove parsing, retrieval, code
behavior, or deployment. Failure checks covered wrong directory shape,
duplicate targets, non-allowlisted extensions, encoding/binary/secret
violations, and immutable no-overwrite publication.

### Practical substep 2 - Recover structure within measured parser bounds

The first real package-body parse exposed a resource-path defect: the full
ANTLR worker timed out and immediately produced line fallback. Added a separate
bounded segmented worker attempt. Real measurement then showed that some
individual routines still produce pathological Python-ANTLR cost. Added the
configurable boundary:

```text
CODE_PARSE_MAX_SEGMENT_CHARACTERS=1000
```

Routines above the boundary retain lexer-proven names, exact source ranges,
original text, and `token_structural` extraction state. Smaller routines still
receive ANTLR parsing. Unparsed segments are explicitly retained rather than
disappearing when another segment succeeds.

Python command:

```powershell
& .\.venv\Scripts\python.exe scripts\parse_code_snapshot.py `
  fci-custom-r1-a47f5d4d54e1
```

Production interpretation: a resource-heavy routine cannot consume the entire
segmented budget or be silently dropped. Token-structural identity is useful
for bounded retrieval and dependency scanning but is not equivalent to a full
grammar parse. Changing the threshold changes the evidence contract and needs
a new immutable generation and benchmark.

Failure tests covered full and segmented timeouts, memory boundaries,
source-hash drift, oversized structural recovery, exact source mapping,
retention of degraded units, invalid limits, and atomic no-overwrite output.
Diagnostic generations `analysis_v1` and `analysis_v2` remain immutable; the
improved result is `analysis_v3`.

### Practical substep 3 - Review real parser and analysis coverage

The `analysis_v3` stage completed with declared degradation:

```text
specification: full_parse
package body: segmented_parse
fallback files: 0
failed files: 0
body routine segments: 19
ANTLR-parsed body routines: 11
token-structural body routines: 8
specification symbol keys matched in body: 7/7
external calls: none
```

The review found two blockers before Steps 159-161:

- the largest exact routine unit is about 93.9 KB and must become bounded child
  units before embedding or prompt packing;
- the conservative analyzer emitted 1,795 unresolved call candidates, with
  substantial SQL/table-syntax false positives that must be reduced without
  suppressing legitimate unknown dependencies.

Production interpretation: `complete_with_degradation` proves complete source
retention and explicit parser confidence, not index readiness. No OpenAI call,
embedding, Qdrant write, or active retrieval change occurred. Steps 159-161
remain blocked until bounded routine chunking and call classification pass
deterministic tests and a renewed real-corpus review.

Verification:

```text
Focused parser/recovery tests: 31 passed
Full regression suite: 484 passed, 1 existing Starlette TestClient warning
git diff --check: passed (line-ending notices only)
uv lock --check: not run because uv is not installed in the project venv/PATH
```

## Step 159A - Create deterministic bounded retrieval children

Separated retrieval chunk limits from the ANTLR segment limit. Retrieval units
now have a 6,000-character hard bound and up to 400 characters of deterministic
line-aware overlap. Oversized routine units become child units that record:

```python
CodeRetrievalUnit(
    parent_unit_id=parent_id,
    parent_source_map=parent_map,
    chunk_index=index,
    chunk_count=len(ranges),
    source_map=exact_child_map,
    text=original_source[start:end],
)
```

The bound applies to `retrieval_text`, including derived context, while `text`
remains an exact slice of immutable source. Child IDs hash parent ID, ordered
index, and exact offsets. Artifact validation rejects missing indexes, gaps,
inconsistent parent provenance, oversized units, and invalid overlap.

Production interpretation: large routines can be embedded and packed without
losing their parent routine or citation lines. An overlapping hit must still be
deduplicated during later evidence packing; bounded chunks do not prove
retrieval relevance or semantic completeness.

Failure tests cover invalid bounds, deterministic rebuilds, exact source text,
complete ordered child indexes, no gaps, controlled overlap, and full parent
range coverage.

## Step 159B - Reduce false routine calls without deleting unknowns

Restricted callable names to Oracle identifier tokens and added context checks
for `INSERT` column lists and collection/record indexed access. The classifier
no longer treats SQL operators, table column lists, or uses such as
`tblBundleRpt(index).field` and `VALUES tblBundleRpt(index)` as routine calls.

Python evaluation:

```powershell
& .\.venv\Scripts\python.exe `
  scripts\evaluate_code_dependency_classifier.py
```

Real calls, unresolved package calls, dynamic SQL, configured kernel calls,
external package calls, and table edges remain distinct. The draft labeled
fixture produced precision `1.0`, recall `1.0`, 3/3 correct calls, and 3/3
correct unknown-boundary states. Because the fixture is small and not yet
SME-reviewed, these numbers validate the targeted mechanism only.

Production interpretation: fewer false edges improves impact-analysis signal
and storage cost, but broad suppression would create dangerous false
negatives. Failure tests therefore include both forbidden SQL-noise targets and
required unresolved/kernel/external/dynamic targets.

## Step 159C - Rerun the real-corpus pre-index gate

Added `scripts/check_code_preindex_gate.py` to reconstruct retrieval artifacts
from the immutable source and verify deterministic identity, exact source maps,
bounds, parent coverage, routine retention, known false-call absence, parser
states, and specification/body symbol matching.

The old snapshot archive developed an OS ACL read failure. The readable raw
intake was revalidated into a separate local verification root and reproduced
the exact snapshot ID `fci-custom-r1-a47f5d4d54e1`, content hash, two file
hashes, and policy hash. The inaccessible original was not modified.

Runtime variance caused `analysis_v4` to fall back at a 1,000-character ANTLR
fragment limit. A measured 500-character limit completed reliably. `v5`
exposed verifier newline handling and remaining collection-access noise; both
were fixed under immutable `analysis_v6` rather than overwriting results.

Final `analysis_v6` evidence:

```text
full_parse files: 1
segmented_parse files: 1
fallback/failed files: 0
body routines retained: 19/19
specification/body matching symbol keys: 7/7
body retrieval units: 86
bounded child units: 73 across 6 parents
maximum retrieval_text: 6,000 characters
routine-call candidates: 1,439 -> 229
unresolved routine calls: 90
known false-call targets: 0
table edges retained: 488
deterministic rebuild and exact source mapping: pass
```

Local reports:

```text
data/exports/code_analysis/fci-custom-r1-a47f5d4d54e1-analysis-v6-preindex-gate.json
data/exports/code_analysis/dependency-classifier-v1-eval.json
```

Production interpretation: the operational pre-index mechanism gate passes,
but the draft precision/recall labels still require SME review before they can
become an activation-quality threshold. No OpenAI, embedding, Qdrant, or active
retrieval operation occurred.

Verification:

```text
Focused remediation tests: 42 passed
Full regression suite: 489 passed, 1 existing Starlette TestClient warning
git diff --check: passed (line-ending notices only)
```

## Step 159 - Build deterministic code index and cache contracts

Added a code-specific index schema rather than forcing PL/SQL evidence into the
FDD release/document contract. Each record contains snapshot, module, path,
routine/chunk, exact source map, parent identity, parser confidence, conditional
state, original citation text, derived embedding text, embedding model, content
hash, cache key, and deterministic Qdrant point ID.

Python preparation command:

```powershell
& .\.venv\Scripts\python.exe scripts\prepare_code_index_artifacts.py `
  fci-custom-r1-a47f5d4d54e1 `
  --parse-generation plsql_antlr_4_13_2_analysis_v6
```

The real local `code_index_artifact_v2` contains 96 records and 96 unique
embedding inputs. Its identity is:

```text
922a253dd07e6b7818b4180c1d3573fa92fa780099c0bc9b04f4f9d164e3e75c
```

Cache identity combines normalized embedding text, model, and
`code_embedding_input_v1`; point identity combines snapshot and source-unit
identity. Identical text may reuse a vector but never collapses distinct
source occurrences or citations.

Production interpretation: deterministic contracts make cost estimation,
cache reuse, and exact downstream verification possible. They do not prove
that an embedding is relevant or safe to disclose externally. Failure tests
cover record/point collisions, inconsistent vectors, reordered provider
responses, missing provider indexes, no-overwrite publication, and cache
conflicts.

## Step 160 - Add isolated code lexical and Qdrant generation tooling

Added local code lexical search over the prepared contract and verified the
real query `spPNBRPT006 branch report` returns the expected procedure children.
The code lane uses its own artifact root and requires Qdrant collection names
to start with `code_custom_`; it never writes `functional_specs_v4`.

Python commands:

```powershell
& .\.venv\Scripts\python.exe scripts\query_code_lexical.py `
  <prepared-code-index-artifact> "spPNBRPT006 branch report"

& .\.venv\Scripts\python.exe scripts\index_code_qdrant.py `
  <embedded-code-index-artifact> `
  --qdrant-path data\qdrant_code_local `
  --collection-name code_custom_r1_v1
```

Indexing accepts only a complete embedded artifact, refuses existing
collections, writes a new isolated namespace, and leaves prior code/FDD
collections unchanged. The real Qdrant command was not run because vectors do
not exist yet.

Production interpretation: separate lanes prevent mixed FDD/code scores,
payloads, and citations. A newly created collection is only staged state; it is
not active or correct until exact verification and later retrieval evaluation
pass. Failure tests cover invalid collection names, existing-target refusal,
partial/extra points, and rollback-collection preservation.

## Step 161 - Add exact verification and rollback isolation

Added exact collection verification for total point count, every deterministic
point ID, all provenance payload fields, and vector dimension. Local in-memory
tests prove that an extra point fails verification and that building a new code
generation does not modify the prior rollback collection.

Python command:

```powershell
& .\.venv\Scripts\python.exe scripts\verify_code_qdrant.py `
  <embedded-code-index-artifact> `
  --qdrant-path data\qdrant_code_local `
  --collection-name code_custom_r1_v1
```

The paid embedding launcher requires two independent gates: a persisted
`dependency_review_status="reviewed"` contract and the exact one-time operator
authorization acknowledging OpenAI disclosure and cost. The current real
contract is persistently `draft`. Its dry run reports 96 records/inputs and
`external_calls_performed=false`.

Production interpretation: Steps 159-161 engineering is implemented and
deterministically tested, but real vector construction and Qdrant verification
remain unexecuted. SME label review and explicit code-disclosure/cost approval
are required first. Retrieval, citation, grounded-answer, SME, activation, and
rollback-rehearsal gates remain Steps 162-170.

Verification:

```text
Focused code-index/vector/lexical tests: 35 passed
Final full regression suite: 496 passed, 1 existing Starlette warning
Real prepared records: 96
Real unique embedding inputs: 96
Real OpenAI calls: 0
Real Qdrant writes: 0
git diff --check: passed (line-ending notices only)
```

Re-verification on 2026-08-13:

```text
Initial focused-test attempt: not executed because the default pytest user
temp directory was denied by Windows ACLs.
Workspace-local isolated pytest temp root: 7 passed.
Real lexical probe: spPNBRPT006 retrieved the expected package artifact first.
Embedding disclosure dry run: 96 records, 96 unique inputs, 0 external calls.
Real OpenAI calls: 0
Real Qdrant writes: 0
```

Production interpretation: restricted service identities may not be able to
use the interactive user's default temp directory. Test and batch runners must
use an explicitly provisioned writable temporary path. This environmental
failure must not be reported as a code-test failure, but it also must not be
silently ignored.

## Step 161A - Configure the custom program-unit convention

Recorded `_CUSTOM` package and standalone-function suffixes in
`config/code_analysis.toml` under versioned `code_analysis_policy_v2`. The
policy also explicitly enables inference of qualified non-custom package calls
as unavailable kernel boundaries. Configuration values are normalized and
included in the deterministic policy SHA-256.

Production interpretation: future naming-policy changes do not require hunting
through analyzer code. A policy change intentionally invalidates derived
analysis identity and requires a new immutable generation.

Failure testing covers blank/duplicate normalized configuration and preserves
unqualified calls as unresolved when tokens cannot prove the owner kind.

## Step 161B - Apply owner-package classification without filtering tables

Qualified calls use the component immediately before the routine as the owner
package. `*_CUSTOM` owners remain custom unresolved calls when their source is
absent; non-custom owners become medium-confidence `kernel_unavailable`
boundaries unless an explicit external prefix applies. Resolved in-snapshot
symbols retain their stronger identity.

A called unit ending `_CUSTOM` also prevents inferred-kernel classification.
This preserves schema-qualified standalone custom functions conservatively
because two-part call tokens can also represent `PACKAGE.ROUTINE`.

All table reads/writes remain extracted independently of suffix. Tests include
schema-qualified custom/kernel calls and both ordinary and `_CUSTOM` table
names.

Production interpretation: answers may distinguish visible custom dependencies
from hidden kernel boundaries without pretending that table access is limited
to custom tables. Medium confidence preserves the fact that static naming is a
convention rather than runtime proof.

## Step 161C - Re-evaluate the dependency policy and detect stale generations

Updated the dependency fixture to represent custom and kernel owners under the
approved rule. The focused dependency/static-analysis suite passed `14/14`.
The draft classifier report passed with precision `1.0`, recall `1.0`, two
expected routine calls, and four correct boundary labels.

The current policy hash is
`b0283dcedf6c6d511a9cb55b3f898f8a17a497dea4db81cf2ebc150930173370`.
Existing `analysis_v6` artifacts record the older policy hash
`141620341f88f0b447c42628324f2c1a052d682f3b730a4e10030d61f9f1aabd`.
They remain historical artifacts and are not eligible for reviewed promotion.

Failure-mode result: the first fixture run failed because one old case still
labeled a non-custom package as a custom unresolved call. Correcting it to a
`*_CUSTOM` owner made the new contract explicit rather than weakening the
classifier. No OpenAI call or Qdrant write occurred.
