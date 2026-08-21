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

## Step 161D - Publish immutable policy-v2 analysis generation

Introduced `plsql_antlr_4_13_2_analysis_v7` while keeping `analysis_v6`
manifests readable. The real snapshot was rebuilt from the previously verified
recovery copy because the original archive remains ACL-constrained. Snapshot
ID, source hashes, and content hash were verified before parsing.

Real result:

```text
status: complete_with_degradation
files: 2
full_parse: 1
segmented_parse: 1
fallback_parse: 0
failed: 0
analysis_policy_sha256: b0283dcedf6c6d511a9cb55b3f898f8a17a497dea4db81cf2ebc150930173370
external calls: 0
```

Production interpretation: the large body used the declared bounded segmented
path; degradation remains explicit rather than being hidden. `analysis_v6`
was not modified. Two deliberately short wrapper timeouts killed unpublished
attempts; one exact `.code-parse-*` directory was validated and removed before
the successful atomic run.

## Step 161E - Gate the real corpus and export focused SME review

Added `dependency_review.py` and `export_code_dependency_review.py`. Review
cases are deterministically grouped by target, proposed dependency kind,
resolution state, and confidence, with up to three exact local source excerpts.
Tables are excluded from this ambiguity review while remaining present in the
analysis graph.

Real pre-index result:

```text
status: pass
routine segments retained: 19/19
spec/body matching symbol keys: 7
retrieval units: 96 total
body child units: 73 across 6 parents
routine calls: 189
unresolved routine calls: 50
table edges: 488
known false calls: 0
```

The draft SME packet groups 90 occurrences into 39 cases: 24 unresolved custom
routine targets covering 50 occurrences and 15 medium-confidence inferred
kernel targets covering 40 occurrences. Packet identity is
`f7b3143766c83791172f36f1a95a7c8b17f86984085101efc81b716785c025ed`.

Production interpretation: the reviewer assesses repeated dependency identity
once rather than reviewing every occurrence. A draft packet is diagnostic
evidence only; it does not set `dependency_review_status="reviewed"`.

## Step 161F - Prepare and independently verify policy-bound index contract

Bumped the no-overwrite prepared output directory to
`code_index_contract_v3` and added `verify_prepared_code_index.py`. The verifier
loads the artifact, checks the approved policy hash, rebuilds it from
`analysis_v7`, and requires exact model equality.

Real result:

```text
status: prepared and verified
records: 96
unique embedding inputs: 96
unique point IDs: 96
dependency_review_status: draft
artifact identity: 7be1370717d3400d130ca24c25b157c265e608946f9cdae0b68f1fa45dc52a61
old v2 identity retained: 922a253dd07e6b7818b4180c1d3573fa92fa780099c0bc9b04f4f9d164e3e75c
OpenAI calls: 0
Qdrant writes: 0
```

Failure tests reject policy mismatch, artifact tampering, duplicate generation
publication, invalid review inputs, and paid embedding while review remains
draft. Prepared means reproducible local input—not semantically reviewed,
embedded, indexed, retrieved, or activated.

Final verification: `499 passed` with the one existing Starlette/HTTPX
deprecation warning. The v3 embedding dry run reported 96 records and 96 unique
inputs with `external_calls_performed=false`. `git diff --check` passed with
line-ending notices only.

## Step 161G - Enforce declared custom program-unit ownership

Upgraded `code_analysis_policy_v3` to accept top-level packages, standalone
functions, and standalone procedures ending `_CUSTOM` or `_MAIN`. Added
`program_unit_validation.py`: the declared canonical program-unit name must
match the filename stem, while all routines contained inside an accepted
package inherit the package's available-source status. `.ddl` remains exempt.

Production interpretation: filename conventions are intake assertions, not
the source of truth. A renamed or kernel package cannot enter the custom lane
silently, and ordinary package members do not need custom suffixes.

Failure tests reject non-custom top-level units, declaration/filename mismatch,
multiple top-level owners, and missing extracted owners while accepting package
members with ordinary names and unrestricted table DDL.

## Step 161H - Correct custom-missing and package-only kernel states

Added `custom_source_missing` and made resolution availability-first. Uploaded
symbols resolve first; absent `_CUSTOM`/`_MAIN` owners become missing custom
source; exact configured kernel package names/prefixes become
`kernel_unavailable`; everything not proven remains unresolved. Blanket
non-suffix kernel inference is disabled.

The first v8 analysis exposed why: 15 supposed kernel cases included aliases
such as `ALC.TRANSACTIONNUMBER` and `F.FUNDID`. v8 was retained as historical
failed-quality evidence and was not promoted. The corrected v9 packet contains
no unconfigured kernel assertions: one `custom_source_missing` case covers
seven occurrences and the remaining 38 cases require unresolved-target review.

Production interpretation: an increased unresolved count is safer than a
lower count achieved by false kernel claims. Reviewed kernel package names can
be added explicitly later, producing a new policy hash and generation.

## Step 161I - Add incremental reuse and indexed symbol lookup; publish v9/v4

Added source/parser-contract reuse keys and manifest-level reuse records.
Unchanged parse/retrieval artifacts may be copied into a new generation only
when snapshot identity, source hash, encoding, compiler context, grammar, and
resource/chunk boundaries match. Static dependency analysis is always rebuilt
under the current policy. Added a canonical-name symbol lookup so call
resolution reads small indexed candidate buckets instead of scanning every
symbol; a 4,000-symbol fixture preserves exact resolution.

Real v9 result:

```text
generation: plsql_antlr_4_13_2_analysis_v9
policy SHA-256: fcf67d689732da8a0ad6041848c64af6eff70a26a0c17d656454380e260415eb
reused parse/retrieval files: 2/2 from v8
publication time: about 4.6 seconds
full/segmented/fallback/failed: 1/1/0/0
routine segments retained: 19/19
retrieval units: 96
routine calls: 229
unresolved routine calls: 83
table edges: 488
known false SQL calls: 0
```

Prepared `code_index_contract_v4` contains 96 unique point/cache identities,
remains `dependency_review_status="draft"`, and passed exact deterministic
rebuild verification. Artifact identity:
`29320a23993c9ea8bafd8b39fbacc48a2061a3876511fa43075044abfe8097c7`.
No OpenAI call or Qdrant write occurred.

Final verification: `510 passed` with the one existing Starlette/HTTPX
deprecation warning. The v4 embedding dry run reported 96 records and 96 unique
inputs with `external_calls_performed=false`; `git diff --check` passed with
line-ending notices only.

## Step 161J - Add an independent routine-declaration inventory

Added `inventory_routine_declarations()` as a lexer-only inventory of every
PL/SQL `PROCEDURE` and `FUNCTION` declaration start. Added
`uncovered_routine_declarations()` to compare that inventory with independently
constructed routine segments. This closes the previous circular gate, which
could only prove that already-detected segments reached retrieval artifacts.

```python
declarations = inventory_routine_declarations(source_text, source_path=path)
uncovered = uncovered_routine_declarations(declarations, parsed.segments)
if uncovered:
    raise RuntimeError("Routine declaration coverage failed")
```

Production interpretation: a top-level routine can no longer disappear from
the code knowledge lane merely because the segmenter failed to return it.
Nested declarations remain covered by their retained parent routine. The real
body inventory initially exposed `spPNBRPT023` at line 32499 as missing.

Failure-mode tests deliberately remove one routine segment and prove the
independent inventory reports the omitted declaration and original line.

## Step 161K - Repair SQL CASE-aware structural segmentation

Updated the token-aware end detector to maintain a separate `CASE` depth.
`END AS alias` from a SQL CASE expression is no longer mistaken for the end of
the PL/SQL routine. The detector also avoids aborting its scan when an apparent
block end is not followed by a nearby semicolon.

```python
if token.type == PlSqlLexer.CASE:
    case_depth += 1
elif token.type == PlSqlLexer.END and case_depth:
    case_depth -= 1
```

Production interpretation: this repairs the mechanism rather than manually
marking one dependency edge as resolved. On the real 1.66 MB package, the
result changed from 19 detected body routines to 20, with `spPNBRPT023`
retained at exact source lines 32499-33237 and no uncovered declarations.

Failure-mode coverage includes a SQL CASE expression followed by `END AS`, a
second routine after it, comments/string false declarations, and legacy-parser
reuse rejection.

## Step 161L - Fail closed, publish v10, and verify the corrected edge

Added parser contract `plsql_parser_contract_v2`, bumped the immutable
generation to `plsql_antlr_4_13_2_analysis_v10`, and prevented v9 parse reuse.
The pipeline now requires routine segments to have extracted nodes, citeable
retrieval units, and symbol occurrences before atomic publication. The
pre-index report is now `code_preindex_gate_v2` and includes declaration counts
and uncovered declaration details.

The original archive remains ACL-inaccessible. A recovery snapshot rebuilt
from the unchanged intake bytes reproduced snapshot ID
`fci-custom-r1-a47f5d4d54e1` and the exact content hash before v10 parsing.

Real v10 result:

```text
status: complete_with_degradation
parser contract: plsql_parser_contract_v2
full/segmented/fallback/failed: 1/1/0/0
body declarations/segments/retained: 20/20/20
uncovered declarations: 0
retrieval units: 111 total
SPPNBRPT023 symbol: implementation, lines 32499-33237
SPPNBRPT023 call: resolved_in_snapshot, one candidate, line 36417
external calls: 0
```

The strengthened pre-index gate passed. A new draft v10 SME packet contains 40
cases covering 121 ambiguous occurrences and correctly contains zero
`SPPNBRPT023` review hits. Packet identity:
`41f9dca34556548eb8be1cffa5a7ee76ae71eac4efa59eadf05b274b8aedf0c3`.

Failure behavior: old parser-contract reuse is rejected before publication;
missing declarations, nodes, retrieval units, or symbols fail closed; the
active FDD collection and historical v9 artifacts remain unchanged. No OpenAI
call, embedding, Qdrant write, prepared-contract promotion, or activation was
performed.

Verification: targeted parser/pipeline tests passed `19/19`; Python compilation
passed; the full suite passed `513` tests with the one existing non-failing
Starlette/HTTPX deprecation warning. Ruff was unavailable in the project
environment, so no Ruff claim is made.

## Step 161M - Preserve the received mixed SME submission

The reviewed Markdown in the original export location was left unchanged after
validation found that its header identified v10 while its case body mixed v9
and v10 content. The file remains available as received for diagnosis and
comparison; no verdict text was silently migrated or discarded.

```python
original_submission = Path("data/exports/code_analysis/...analysis_v10-dependency-review.md")
assert original_submission.is_file()
```

Production interpretation: SME input is evidence and must not be overwritten
when its generation binding is uncertain. Preserving it allows later decisions
to be reconciled by stable review ID rather than case number.

Failure check: target-set validation found missing `ALCS.TRANSACTIONNUMBER`, an
extra stale `SPPNBRPT023`, three placeholder verdicts, and the first ordering
drift at case 2. The mixed packet was therefore not approved.

## Step 161N - Regenerate a canonical v10 review packet

Ran the existing deterministic local exporter against immutable
`plsql_antlr_4_13_2_analysis_v10` artifacts and the verified recovery snapshot,
writing to the separate `data/exports/code_analysis/canonical/` namespace.

```powershell
& .\.venv\Scripts\python.exe scripts\export_code_dependency_review.py `
  fci-custom-r1-a47f5d4d54e1 `
  --snapshot-root data\tmp\snapshot-recovery-step161j `
  --generation plsql_antlr_4_13_2_analysis_v10 `
  --output-root data\exports\code_analysis\canonical
```

Production interpretation: a separate no-overwrite namespace prevents a valid
canonical packet from destroying the SME submission. The exporter reconstructs
cases from the v10 analysis rather than trusting edited Markdown headings.

Failure behavior: the exporter refuses if either canonical output already
exists, preventing accidental replacement of a packet under review. No external
API or vector-store operation is part of this command.

## Step 161O - Verify canonical identity and review scope

Compared the regenerated JSON with the original v10 JSON using SHA-256 and
checked the rendered Markdown scope.

```text
JSON SHA-256: B0D7592C386C982415A4A0412030B1D6D370EC005B9ED00580959A4BB7C3FC13
JSON bytes equal: true
packet identity: 41f9dca34556548eb8be1cffa5a7ee76ae71eac4efa59eadf05b274b8aedf0c3
headings/placeholders: 40/40
ALCS.TRANSACTIONNUMBER: present once
SPPNBRPT023: absent
external calls: 0
```

Production interpretation: the regenerated review file is bound to the exact
v10 case contract and is safe for a new SME pass. Structural identity does not
approve any verdict; all 40 canonical placeholders still require review.

Failure checks compare hashes, target counts, required/forbidden targets, and
placeholder counts. The original mixed submission remains non-promotable.

## Step 161P - Version approved infrastructure-utility policy

Upgraded `code_analysis_policy_v3` to `code_analysis_policy_v4` and added the
normalized `infrastructure_utility_calls` boundary. The approved exact calls
are `DEBUG.PR_DEBUG`, `GLOBAL.PR_INIT`, `ISDEBUG.WRITELINE`,
`PKGGLOBAL.PR_INIT`, and `PR_DEBUG`. Added `infrastructure_utility` as a
first-class dependency kind.

```toml
infrastructure_utility_calls = [
  "DEBUG.PR_DEBUG",
  "GLOBAL.PR_INIT",
  "ISDEBUG.WRITELINE",
  "PKGGLOBAL.PR_INIT",
  "PR_DEBUG"
]
```

Production interpretation: logging and initialization procedures remain
visible execution-flow dependencies, but approved utility calls no longer
pollute business-dependency SME review. Exact configuration avoids treating all
non-custom packages as kernel or infrastructure.

Failure tests normalize case, reject duplicate/unknown policy entries, preserve
real routine-call syntax, and exclude utility dependencies from the ambiguity
packet without deleting them from static analysis.

## Step 161Q - Separate Oracle outer joins and degraded cursor references

The call extractor now rejects the exact Oracle legacy outer-join marker `(+)`
before argument counting. `CURSOR name(...)` declarations are excluded as
calls, while cursor invocations are emitted as resolved `cursor_reference`
edges. A full-file token inventory supplies cursor identities when segmented
parsing cannot produce ANTLR cursor declaration nodes. Keyword owners such as
`GLOBAL.PR_INIT` are retained in qualified names.

```python
if len(argument_tokens) == 1 and argument_tokens[0].type == PlSqlLexer.PLUS_SIGN:
    continue
if final_name in declared_cursor_names:
    state, kind = "resolved_in_snapshot", "cursor_reference"
```

Production interpretation: SQL columns, cursor execution, and routine calls
now have different dependency semantics. This improves impact analysis without
silently discarding cursors or utility calls.

The first immutable v11 run exposed that per-routine cursor discovery was too
narrow for package-level declarations in degraded parsing. v11 was retained as
failed-quality evidence. A deliberate test removed ANTLR cursor nodes and
failed until full-file cursor inventory was implemented for v12.

## Step 161R - Publish and gate immutable dependency analysis v12

Published `plsql_antlr_4_13_2_analysis_v12` under policy hash
`a052206a131a7402afa8c0765a631e3d263973fcf49ceb68c5789244dbe18129`.
Parse/retrieval artifacts were safely reused because source bytes and parser
contract v2 were unchanged; static analysis was rebuilt under policy v4.

```text
status: complete_with_degradation
parse/retrieval reuse: 2/2 from v11
body declarations/segments retained: 20/20
uncovered declarations: 0
routine calls resolved/custom-missing/unresolved: 174/9/0
cursor references resolved: 19
infrastructure utility occurrences: 19
table edges: 544
SME cases/occurrences: 1/9
external calls: 0
```

The only remaining SME case is
`PKGPAYINSLIP_P_CUSTOM.FNGETPAYINSLIPNUMBER`, correctly proposed as
`routine_call / custom_source_missing` because its package follows the approved
custom convention but was not included in this snapshot. Packet identity:
`cb26f0af0185abf408c379b478a1aa53579e8e2719f67d236ab16e9d718dbbb6`.

Failure behavior: v10 and v11 remain immutable; policy changes produce a new
hash and generation; the pre-index v2 gate passed; no embedding, OpenAI call,
Qdrant write, prepared-contract promotion, or activation occurred.

Verification: focused tests passed `23/23`, Python compilation passed, and the
full regression passed `514` tests with the one existing non-failing
Starlette/HTTPX deprecation warning. `git diff --check` is run separately.

## Step 161S - Import the accepted v12 SME decision into a hash-bound ledger

Added a strict local importer that validates the canonical packet identity,
the reviewed Markdown header, exact case order and review IDs, allowed verdicts,
and non-empty rationale before producing a no-overwrite JSON ledger.

```powershell
& .\.venv\Scripts\python.exe scripts\import_code_dependency_review.py `
  data\exports\code_analysis\canonical\fci-custom-r1-a47f5d4d54e1-plsql_antlr_4_13_2_analysis_v12-dependency-review.json `
  data\exports\code_analysis\canonical\fci-custom-r1-a47f5d4d54e1-plsql_antlr_4_13_2_analysis_v12-dependency-review.md `
  --reviewer project-sme `
  --output data\exports\code_analysis\reviews\fci-custom-r1-a47f5d4d54e1-analysis-v12-dependency-review-ledger.json
```

The ledger records one accepted `routine_call / custom_source_missing`
decision for `PKGPAYINSLIP_P_CUSTOM.FNGETPAYINSLIPNUMBER`. It is bound to
packet identity `cb26f0af...718dbbb6`, the exact packet JSON and reviewed
Markdown hashes, policy hash `a052206a...be18129`, and ledger identity
`d26b5ffa...17f0cb`.

Production interpretation: human approval is now a reproducible input to later
indexing rather than an informal chat state. Acceptance preserves the missing
source boundary; it does not assert that the absent implementation is known.

Failure tests reject packet-identity tampering, reordered/missing cases,
placeholder or invalid verdicts, blank rationale, header-generation mismatch,
post-import ledger mutation, and attempts to overwrite an existing ledger.
No external call occurred.

## Step 161T - Build the reviewed v12 code-index contract

Upgraded the prepared-artifact directory contract to `code_index_contract_v5`.
A reviewed artifact must now contain both the dependency packet identity and
the reviewed ledger identity. Draft contracts must contain neither, preventing
an unreviewed artifact from presenting review hashes selectively.

```powershell
& .\.venv\Scripts\python.exe scripts\prepare_code_index_artifacts.py `
  fci-custom-r1-a47f5d4d54e1 `
  --parse-generation plsql_antlr_4_13_2_analysis_v12 `
  --dependency-review-ledger data\exports\code_analysis\reviews\fci-custom-r1-a47f5d4d54e1-analysis-v12-dependency-review-ledger.json
```

The immutable prepared result contains 111 records, 111 unique embedding
inputs, and artifact identity
`fd2285b3e3f0cfa39c4b53f9be87fab046e94b7a7cf81d58fcdbcf24746762dd`.
Its status is `prepared` and dependency-review status is `reviewed`.

Production interpretation: the artifact defines exactly which source units
would be disclosed and indexed, while retaining snapshot, parser, policy,
packet, ledger, path, symbol, and line provenance. Prepared does not mean
embedded, indexed, retrievable, or active.

Failure tests reject a ledger from another snapshot, parser generation,
analysis policy, or packet; inconsistent draft/reviewed hash combinations;
artifact-identity tampering; and overwrite of an existing contract.

## Step 161U - Verify exactness and stop at the paid-operation boundary

The verifier recomputed the v12 artifact identity and confirmed 111 records,
111 unique point IDs, 111 unique cache keys, the expected policy, and exact
packet/ledger bindings. The embedding launcher was then exercised only in dry
run mode.

```text
status: dry_run
records / unique embedding inputs: 111 / 111
embedding model: text-embedding-3-large
external_code_would_be_sent: true
external_calls_performed: false
```

Production interpretation: the disclosure surface and paid-work size are now
reviewable before any code leaves the local environment. A real run still
requires the exact, deliberate disclosure-and-cost authorization token.

Failure testing invoked the non-dry command without authorization; it failed
closed with `PermissionError` before creating embeddings or writing Qdrant.
Compilation passed, focused tests passed `11/11`, `git diff --check` reported no
whitespace errors, and the full regression passed `517` tests with the one
existing non-failing Starlette/HTTPX deprecation warning. No OpenAI call,
embedding, Qdrant write, retrieval evaluation, or activation occurred.

## Step 161V - Embed the explicitly authorized reviewed code contract

The user explicitly authorized disclosure of the 111 prepared internal PL/SQL
embedding inputs to OpenAI and accepted the associated provider cost. Ran the
reviewed `code_index_contract_v5` artifact through the pinned
`text-embedding-3-large` embedding path.

```powershell
& .\.venv\Scripts\python.exe scripts\embed_code_index_artifacts.py `
  data\staging\code_indexes\fci-custom-r1-a47f5d4d54e1\code_index_contract_v5\code_index_artifact.json `
  --output-root data\staging\code_embeddings `
  --authorization I_AUTHORIZE_OPENAI_CODE_DISCLOSURE_AND_COST
```

```text
status: embedded
records / unique inputs: 111 / 111
cached / newly embedded: 0 / 111
OpenAI requests: 4
vector dimension: 3072
external calls performed: true
```

Production interpretation: every approved source occurrence retains a distinct
record and future Qdrant point, while identical semantic inputs could reuse a
cache vector in later compatible generations. The recorded request count proves
external work occurred but does not prove the provider's final monetary charge;
billing remains provider-authoritative.

Failure controls required reviewed packet/ledger bindings, the exact deliberate
authorization acknowledgement, a new no-overwrite output generation, complete
vectors, and one consistent dimension. The earlier unauthorized invocation had
already proven fail-closed behavior.

## Step 161W - Index an isolated code-only Qdrant generation

Created the new local code vector-store path and collection
`code_custom_r1_v1` from the completed embedded artifact. The `code_custom_`
prefix and separate `data/qdrant_code_local` path keep code evidence isolated
from the active FDD collection `functional_specs_v4`.

```powershell
& .\.venv\Scripts\python.exe scripts\index_code_qdrant.py `
  data\staging\code_embeddings\fci-custom-r1-a47f5d4d54e1\code_index_text_embedding_3_large_v1\code_index_artifact.json `
  --qdrant-path data\qdrant_code_local `
  --collection-name code_custom_r1_v1
```

Production interpretation: this is an isolated indexed generation, not an
active application knowledge lane. The current FDD API/UI configuration was not
changed, and no combined retrieval or automatic routing was enabled.

Failure testing repeated the indexing command against the existing collection.
It failed with `FileExistsError` before modification, demonstrating immutable
generation behavior instead of silently upserting into a previously verified
namespace.

## Step 161X - Verify exact code points and update the runbook

Ran the independent verifier before and after the rejected duplicate-index
attempt. Both passes confirmed exact equality between the embedded artifact and
the isolated collection.

```text
expected / verified points: 111 / 111
vector dimension: 3072
artifact identity: fd2285b3e3f0cfa39c4b53f9be87fab046e94b7a7cf81d58fcdbcf24746762dd
external calls during indexing/verification: false
```

The verifier checks exact count, deterministic point IDs, provenance payloads,
and vector dimensions separately. This proves storage exactness, not semantic
retrieval relevance, citations, grounded answers, or activation readiness.

Updated `docs/Steps_for_Code_Snapshot_Ingestion.md` from the obsolete v9/v4
examples to the v12 analysis, hash-bound review-ledger import, reviewed
`code_index_contract_v5`, exact ledger verification, and explicit disclosure
acknowledgement used by the controlled embedding command.

Failure-mode and regression verification passed `10/10` focused indexing,
embedding-boundary, and native-package tests. `git diff --check` reported no
whitespace errors. The immediately preceding complete suite remains `517`
passing tests with one existing non-failing Starlette/HTTPX warning.

## Step 161Y - Publish a five-file successor snapshot without mutating r1

Validated the expanded `data/raw_code/fci-custom-r1/source/` intake. It contains
the two original branch-report files plus three additions:
`pkgamlaintegration_p_custom.spc`, `pkgamlaintegration_p_custom.sql`, and
`pkgutils_custom.sql`. The request remains SVN revision `1` and build `Code1`,
because this is expanded coverage of the same curated revision, and now names
the immutable r1 snapshot as its base with the three expected additions.

```powershell
& .\.venv\Scripts\python.exe scripts\build_code_snapshot.py `
  data\raw_code\fci-custom-r1 --validate-only
```

The existing r1 archive had the previously observed Windows ACL defect. An
approved inheritance reset was attempted only on that exact directory, but
Windows denied it. Ownership was not taken and permissions were not weakened.
The successor was therefore built against the byte-verified recovery copy,
then copied as a new sibling and fully revalidated from the normal archive.

```text
successor: fci-custom-r1-b1c79c6dc2c5
files: 5
added / unchanged: 3 / 2
modified / deleted / renamed: 0 / 0 / 0
missing expected / unexpected: 0 / 0
warnings: 0
external calls: 0
```

Production interpretation: the old snapshot and `code_custom_r1_v1` remain
immutable. The new content is a successor generation rather than an in-place
append, preserving rollback, diff evidence, citations, and snapshot identity.

Failure controls included allowlist/encoding/binary/secret validation, exact
source hashes, base diff assertions, no-overwrite publication, target-absence
checks, and complete post-copy manifest/source verification.

## Step 161Z - Parse five files with measured resource-bound failures

The first correct 120-second parser run was terminated by an insufficient
outer shell limit and published nothing. A 30-second experiment then failed
closed on the branch-report body because it could no longer prove a top-level
program unit. The normal 120-second run later proved that `pkgutils_custom.sql`
was the remaining parser-resource boundary, despite being only about 112 KB.

An isolated utility-package diagnostic showed that the valid package can be
recovered as `segmented_parse`, requiring approximately 300 seconds for the
failed full attempt plus 186 seconds for segmented recovery. The final run used
a recorded 220-second per-attempt boundary and sufficient orchestration time:

```powershell
& .\.venv\Scripts\python.exe scripts\parse_code_snapshot.py `
  fci-custom-r1-b1c79c6dc2c5 --timeout-seconds 220
```

```text
status: complete_with_degradation
files: 5
full / segmented / fallback / failed: 2 / 3 / 0 / 0
policy: a052206a...be18129
wall time: approximately 17 minutes
external calls: 0
```

Production interpretation: grammar complexity, not source size alone, drives
ANTLR cost. Seventeen minutes is tolerable for this controlled five-file run
but is not a viable 4,000-file ingestion design. Verified content-hash reuse and
bounded concurrency are required before bulk scaling.

Failure behavior was deliberately exercised: both shell-timeout and reduced
parser-timeout attempts left no published generation; temporary stages were
removed only after exact path validation. The final stage still records all
degradation and resource-policy inputs in its immutable manifest.

## Step 161AA - Pass the expanded pre-index gate and export new SME scope

Ran the independent real-corpus gate against all five source files and the new
v12 generation.

```powershell
& .\.venv\Scripts\python.exe scripts\check_code_preindex_gate.py `
  fci-custom-r1-b1c79c6dc2c5 `
  --snapshot-root data\code_snapshots `
  --generation plsql_antlr_4_13_2_analysis_v12
```

The gate passed with exact deterministic rebuild and source mapping for every
file. AMLa retained 41/41 body declarations and 91 retrieval units; branch
reports retained 20/20 and 101 units; utilities inventoried 29 declarations,
retained 28 top-level routine segments, and produced 39 retrieval units. No
declaration was uncovered and no known SQL false-call marker reappeared.

The canonical dependency exporter produced a new draft packet:

```text
review cases: 33
occurrences: 313
packet identity: cda3ab782c452c0f2312bf7f428b27e9aa2670b525079b710cb6c9199c508741
external calls: 0
```

Production interpretation: structural preparation passed, but the previous
one-case SME ledger cannot approve dependencies introduced by the expanded
snapshot. The 33 new cases require SME decisions before a reviewed v5 index
contract can be prepared.

Failure boundary: no embedding was attempted. The prior authorization was tied
to the old 111-input contract and does not authorize disclosure or cost for this
new artifact. Focused snapshot, parsing, review-packet, and ledger tests passed
`28/28`; `git diff --check` reported no whitespace errors.

## Step 161AB - Separate kernel suffixes, object methods, and collection access

Applied the SME's naming rule as versioned `code_analysis_policy_v5`: only the
qualified owner package ending `_KERNEL` is classified as a retained
`kernel_boundary`. Exact configured kernel names/prefixes remain available but
are empty in the active policy. Other syntactically valid calls remain visible
as resolved, `custom_source_missing`, or `unresolved`; uncertainty is not
rewritten as kernel behavior.

```toml
kernel_program_unit_suffixes = ["_KERNEL"]
external_object_type_names = ["JSON_ARRAY_T", "JSON_ELEMENT_T", "JSON_OBJECT_T"]
```

Added first-class `object_method_call` and `collection_reference` dependency
kinds. Declared Oracle JSON object receivers and fluent method chains remain in
the graph without becoming ordinary PL/SQL routine calls. Repeated-parenthesis
access such as `ipTxnData.Desc_Fields(...)(1)(...)` is retained as collection
access instead of a procedure/function call.

Production interpretation: `GET_STRING` was not dropped as
`not_routine_call`; it is a real JSON object-method relationship. The
`IPTXNDATA.DESC_FIELDS` correction becomes a collection relationship. Both are
technical evidence but no longer business-dependency SME noise.

The review contract now asks SMEs about dynamic SQL, ambiguous overloads, and
non-high-confidence kernel inference. Explicit unresolved and missing-source
routine edges remain visible unknowns but do not require repetitive approval.

Failure tests prove `_KERNEL` owner matching without classifying an unsuffixed
package as kernel, preserve JSON receiver/fluent methods, separate collection
index chains, retain unresolved calls, and keep ambiguous/dynamic cases
reviewable. The first focused run exposed and fixed a fluent-method interaction
with the previous indexed-access heuristic.

## Step 161AC - Rebuild immutable v13 and bind the reduced SME review

Published `plsql_antlr_4_13_2_analysis_v13` with policy hash
`f707e1f5f9bcc63db8ad48f9d0448d4904e1f28fd53947cc39b427b1e67cd5fc`.
All five parse and retrieval artifacts were reused from v12 because source,
parser contract, timeout, memory, and chunk inputs matched; dependency analysis
was rebuilt under policy v5.

```text
parse/retrieval reused: 5/5
full / segmented / fallback / failed: 2 / 3 / 0 / 0
v13 publication time: 13 seconds
pre-index gate: pass
external calls: 0
```

Routine-call noise in `pkgutils_custom.sql` fell from 391 to 248 and unresolved
routine calls from 153 to 10 because JSON methods and collection accesses now
have correct edge kinds. The v13 review packet contains only the two literal
dynamic-SQL cases already reviewed by the SME: two cases and three occurrences.

Before migrating those acceptances, exact equality was verified for target,
proposed kind/state, occurrence count, source hash, path, line range, and
excerpt. Only policy-bound review IDs changed. The resulting reviewed ledger is
bound to packet `ea9c4b95...568df3` and has identity
`d5544146...17f08e`.

Production interpretation: compatible parser work was reused, but no v12
dependency classification or ledger identity was relabeled as v13. The two
accepted dynamic-SQL decisions are traceable to identical reviewed evidence.

Failure controls reject policy/hash mismatch, stale packet IDs, missing
rationales, partial case sets, and incompatible parse reuse.

## Step 161AD - Prepare the 279-input successor and plan cache reuse

Prepared and exactly verified the reviewed successor code-index contract.

```text
status: prepared
records / point IDs / cache keys: 279 / 279 / 279
artifact identity: 386def0beab98e14f67a23029cd7bc96d861f2dec26ad197927e47045ee3f5eb
compatible cached vectors from v1: 111
new paid embedding inputs: 168
expected OpenAI requests at batch size 32: 6
external calls performed: false
```

Production interpretation: the two unchanged files still create new
source-occurrence points for the successor snapshot, while their compatible
semantic vectors can be reused. Only 168 new inputs need external embedding;
the old 111-input artifact remains the immutable cache source.

The dry run confirms the expanded internal code would be disclosed during a
real call but sends nothing. The earlier authorization covered only the old
111-input artifact, so the 168 new paid inputs require a new explicit
disclosure-and-cost authorization.

Verification: compilation passed, the final focused indexing fixture passed
`6/6`, and the full regression passed `519` tests with the one existing
non-failing Starlette/HTTPX warning. `git diff --check` reported no whitespace
errors. No OpenAI call, new embedding artifact, Qdrant v2 collection, retrieval,
or activation occurred.

## Step 161AE - Embed the authorized successor with exact cache reuse

The user explicitly authorized disclosure and provider cost for the 168 new
internal PL/SQL embedding inputs, with reuse of 111 compatible vectors from the
immutable v1 embedded artifact.

```powershell
& .\.venv\Scripts\python.exe scripts\embed_code_index_artifacts.py `
  data\staging\code_indexes\fci-custom-r1-b1c79c6dc2c5\code_index_contract_v5\code_index_artifact.json `
  --output-root data\staging\code_embeddings `
  --cache-artifact data\staging\code_embeddings\fci-custom-r1-a47f5d4d54e1\code_index_text_embedding_3_large_v1\code_index_artifact.json `
  --authorization I_AUTHORIZE_OPENAI_CODE_DISCLOSURE_AND_COST
```

```text
status: embedded
records / unique inputs: 279 / 279
cached / newly embedded: 111 / 168
OpenAI requests: 6
vector dimension: 3072
external calls performed: true
```

Production interpretation: reuse was based on the exact embedding cache key,
model, content hash, input text, and vector consistency—not filename or source
position. Every successor source occurrence retains its own record and future
point identity even when its semantic vector is reused.

Failure controls required the reviewed v13 ledger, exact scoped authorization,
compatible embedded cache status/model, conflict-free cached vectors, complete
provider response indexes, and a single non-zero vector dimension. Actual cost
remains provider-billing authoritative.

## Step 161AF - Create isolated Qdrant generation code_custom_r1_v2

Indexed only the completed successor embedded artifact into the new collection
`code_custom_r1_v2` under the separate local code Qdrant path.

```powershell
& .\.venv\Scripts\python.exe scripts\index_code_qdrant.py `
  data\staging\code_embeddings\fci-custom-r1-b1c79c6dc2c5\code_index_text_embedding_3_large_v1\code_index_artifact.json `
  --qdrant-path data\qdrant_code_local `
  --collection-name code_custom_r1_v2
```

Production interpretation: v2 is an indexed candidate generation containing
all five files. It does not modify the FDD collection, replace v1, configure the
API/UI, enable combined retrieval, or become active merely because indexing
succeeded.

Failure testing repeated the indexing command against v2. It failed closed
with `FileExistsError` and performed no upsert or mutation into the verified
namespace.

## Step 161AG - Verify v2 exactness and preserve v1 rollback

Independent verification checked the embedded artifacts against both
collections:

```text
code_custom_r1_v2: 279 expected / 279 verified, dimension 3072
v2 artifact: 386def0beab98e14f67a23029cd7bc96d861f2dec26ad197927e47045ee3f5eb
code_custom_r1_v1: 111 expected / 111 verified, dimension 3072
v1 artifact: fd2285b3e3f0cfa39c4b53f9be87fab046e94b7a7cf81d58fcdbcf24746762dd
```

V2 was verified again after the rejected duplicate-index attempt and remained
exact. Count, deterministic IDs, provenance payloads, and dimensions were
checked separately.

Production interpretation: v1 is a verified rollback generation and v2 is a
verified candidate. Neither exact storage nor vector creation proves semantic
retrieval, source-line citations, grounded code explanations, impact analysis,
unknown handling, or activation readiness.

Focused cache/indexing and authorization-boundary tests passed `9/9`.
`git diff --check` reported no whitespace errors. The preceding complete suite
remains `519` passing tests with one existing non-failing Starlette/HTTPX
warning. Retrieval and activation remain deliberately deferred.

## Step 162 - Add explicit isolated code retrieval

Added `app/code_retrieval/service.py`, typed retrieval models, and
`scripts/query_code_index.py`. The retrieval boundary exposes only explicit
`lexical`, `dense`, or `hybrid` modes. Dense/hybrid calls require an explicitly
supplied query vector, Qdrant client, and `code_custom_*` collection; retrieval
does not silently make an OpenAI call.

```powershell
& .\.venv\Scripts\python.exe scripts\query_code_index.py `
  data\staging\code_embeddings\fci-custom-r1-b1c79c6dc2c5\code_index_text_embedding_3_large_v1\code_index_artifact.json `
  "spPNBRPT006 branch report" --mode lexical --limit 2
```

The real-corpus lexical smoke test selected `spPNBRPT006` first from
`pkgpnbbranchreports_p_custom.sql`, with exact snapshot, symbol, and line
provenance. Production interpretation: code retrieval is isolated from the FDD
lane and one immutable embedded artifact is the identity authority. Qdrant
filters are defense in depth; every returned unit, point, path, symbol, hash,
and line range is checked against that artifact before becoming evidence.

Failure tests reject blank/invalid modes, unreviewed or unembedded artifacts,
missing collections, non-code namespaces, missing/wrong-dimension vectors,
unknown units, non-code payloads, point-ID mismatches, and tampered provenance.
The smoke test proves lexical reachability only, not dense relevance or answer
quality.

## Step 163 - Add exact source-line citation contracts

Added a separate code citation model containing snapshot, path, symbol, source
kind, exact original line range, score, and a bounded preview. Citeable text is
always immutable `citation_text`; derived `embedding_text` is retrieval-only.

```python
citation = CodeCitation(
    citation_id="C1",
    source_path="pkg_claim.sql",
    display_name="process_claim",
    start_line=10,
    end_line=12,
    # remaining identity fields omitted here for brevity
)
```

Production interpretation: a user or reviewer can trace a claim to one exact
source occurrence without mistaking search enrichment for literal PL/SQL.
Dense and lexical candidate summaries are retained for diagnosis but exclude
full source text to control trace size and avoid duplicating evidence.

Failure tests reject missing source/text, invalid line ranges, answered content
without citations, and citation IDs outside the bounded evidence set.

## Step 164 - Add fail-closed grounded code-answer and impact contracts

Added `finalize_code_answer`, which validates a future generated response
without making an LLM call. Content must begin with `DECISION: ANSWER` or
`DECISION: REFUSE`; an answer requires valid `[C#]` references. The response
retains structured parser, conditional-compilation, kernel, dynamic-SQL, and
external-schema unknowns. Impact analysis carries the explicit limitation that
reported locations are candidates in visible custom code, not proven root
causes. Patch generation remains disabled.

```python
response = finalize_code_answer(
    query="How is the claim processed?",
    generated_content="DECISION: ANSWER\nThe visible routine performs ... [C1].",
    evidence=retrieved_evidence,
    analysis_kind="explanation",
)
```

Production interpretation: helpfulness does not overwrite grounding state.
Malformed output, no evidence, invalid citations, or patch-like output becomes
a machine-readable safe refusal. Degraded parsing and unresolved conditional or
unavailable behavior remain visible qualifications rather than invented facts.

Focused retrieval/citation/answer tests passed `15/15`, including lexical,
dense, and weighted-RRF hybrid mechanics, tampered-payload rejection, exact
line citations, no-evidence refusal, conditional/parser unknowns, bounded impact
language, and patch rejection. No OpenAI call, API/UI routing change, collection
mutation, or activation occurred in Steps 162-164. The full regression suite
passed `528` tests with the one existing non-failing Starlette/HTTPX deprecation
warning.

## Step 164D - Ingest and exactly verify three Neo AML FDDs

The user explicitly authorized OpenAI embedding disclosure and cost for three
new internal FDDs. A process-local intake generation prevented writes to the
active v4 pair:

```powershell
$env:QDRANT_COLLECTION_NAME='functional_specs_v5_intake'
$env:INGESTION_OUTPUT_DIR='data/staging/functional_specs_v5_intake/processed'
uv run --locked python scripts/master_ingestion_embedding_docs.py `
  --request-batch-size 32
```

Extraction produced 188 retrieval units across the three sources: 79 for the
R22 document, 41 for R24 Day2, and 68 for R24 Day2 Part2. Intake embedding
reused 16 compatible vectors and generated 172. Exact Qdrant verification
confirmed all 188 intended source occurrences before the DOCX files moved from
`data/raw_specs` to `data/docs_embedded`.

Production interpretation: successful archival means each new source passed
extraction, all-unit embedding, indexing, and identity verification. It does
not establish retrieval relevance, release-lineage interpretation, citation
entailment, or answer correctness. Failure before exact verification would have
left the affected DOCX in intake for a safe retry.

## Step 164E - Build the complete immutable functional_specs_v5 generation

Built v5 from all 11 archived FDDs rather than appending only the three new
documents to the live collection:

```powershell
uv run --locked python scripts/stage_archived_fdd_rebuild.py `
  --source-directory data/docs_embedded `
  --stage-directory data/staging/functional_specs_v5 `
  --collection-name functional_specs_v5 `
  --index-generation functional_specs_v5
```

The verified stage manifest records 1,125 records, 667 cache hits, 458 newly
embedded inputs, 1,125 upserted points, and 1,125 exactly verified points at
dimension 3,072. Eleven source names, byte sizes, and SHA-256 hashes are bound
to the generation.

Production interpretation: v5 is independently complete and v4 remains a
rollback generation. The unexpectedly high 458 embedding count is important:
188 units came from the new sources, while 270 older retrieval inputs were not
available under a compatible identity in the central seed cache and were sent
again. A completed request proves provider response success, not authoritative
billing or semantic quality. Future rebuild planning must inspect cache scope,
not assume every vector in an older Qdrant generation is reusable from the
configured seed directory.

Failure controls rejected existing stage/collection names, validated every
vector dimension, published a failed manifest on exceptions, and required exact
artifact-to-point verification before `status=verified`.

## Step 164F - Promote and activate the paired v5 retrieval generation

Promoted the verified lexical artifacts to
`data/indexes/functional_specs_v5/processed`. All 33 promoted files matched the
staged files by relative path and SHA-256. Updated the active pair together:

```text
QDRANT_COLLECTION_NAME=functional_specs_v5
PROCESSED_DIR=data/indexes/functional_specs_v5/processed
RETRIEVAL_MODE=hybrid
HYBRID_DENSE_WEIGHT=0.40
HYBRID_LEXICAL_WEIGHT=0.60
```

A fresh settings process confirmed the effective pair. Active Qdrant inspection
confirmed 1,125 points, cosine distance, and dimension 3,072. The ingestion
runbook and current evaluation defaults now identify v5 as active and use v6 as
the next isolated example. V4 was retained for rollback.

Local lexical smoke checks reached all three new document IDs. The broad R22
Neo AML query ranked a closely related R24 document first and the intended R22
document second; this is a known cross-release confusion risk. The user
explicitly deferred semantic evaluation, so this local activation is not a
claim of production readiness or an evaluation pass. Focused configuration and
runbook tests passed `7/7`. The complete regression suite passed `528` tests
with the one existing non-failing Starlette/HTTPX deprecation warning.

## Step 165 - Add reviewed FDD-to-code lineage contracts

Added `app/fdd_code_lineage/models.py` with immutable candidate/reviewed mapping
artifacts bound to one FDD generation, exact code snapshot, and code artifact
identity. Targets support broad file scope, all overloads of one symbol, or one
exact overload. Exact selectors require canonical qualified name, symbol kind,
and overload discriminator hash; unknown documents, paths, modules, symbols,
overloads, or snapshot identities fail validation.

```python
target = FddCodeTarget(
    module_id="fci-custom",
    path="pkgamlaintegration_p_custom.sql",
    selector_scope="file",
    rationale="Broad candidate; exact symbols require SME selection.",
)
```

Created a local candidate artifact for the three Neo AML FDDs with six broad
file targets covering the AML package specification and body. Its identity is
`bc21581f77b7e883d936762280cf61be61c355c93d54d37ea11a01f3bd766565`.
The human review packet is
`data/exports/code_analysis/neo-aml-fdd-code-lineage-v1-review.md`.

Production interpretation: user-supplied alignment is useful candidate
evidence, but it is not proof that every routine in the package implements each
FDD. Only `reviewed` mappings can influence combined retrieval. No-overwrite
publication and deterministic identities preserve the exact review subject.

Failure tests reject missing FDDs, paths, symbols, overload hashes, mismatched
snapshot/artifact identities, illegal selector shapes, and overwrite attempts.

## Step 166 - Add independent and mapping-aware combined retrieval

Added `retrieve_combined_evidence` and `scripts/query_combined_index.py`. FDD
and code retrieval run independently and retain separate evidence and scores.
Code hybrid mode continues to use weighted RRF inside the code lane only. A
second bounded code search may follow exact unit IDs resolved from reviewed
lineage mappings; candidate mappings are never followed.

```python
combined = retrieve_combined_evidence(
    query=query,
    fdd_results=fdd_results,
    code_artifact=code_artifact,
    lineage_artifact=lineage_artifact,
    code_mode="hybrid",
    # explicit clients/vectors omitted here
)
```

Production interpretation: a combined question can retrieve useful visible
code even before mappings are reviewed, but the response must describe it as
directly retrieved candidate evidence—not as proven implementation of the FDD.
Mappings add provenance and bounded expansion; they do not merge FDD and code
score scales.

The real local Neo AML smoke test retrieved R24 and R22 FDD units plus
`spBatchTxnEndPoint` units from `pkgamlaintegration_p_custom.sql`. Because the
artifact is still `candidate`, `mapped_code_evidence` and `reviewed_lineage`
remained empty and the result explicitly recorded that no reviewed mapping
applies. No OpenAI call occurred.

Failure tests reject unknown unit filters, cross-snapshot or cross-artifact
mappings, malformed FDD identities, and candidate mappings used as authority.

## Step 167 - Add four-section combined answer contracts

Added a deterministic combined-answer validator with these independent
sections:

```text
Documented functionality
Visible custom implementation
Impact and likely change locations
Unknown or unavailable behavior
```

Documented claims accept only FDD `[F#]` citations. Implementation and impact
claims accept only code `[C#]` citations with exact path/symbol/line provenance.
An invalid or cross-lane citation refuses only the affected section. Unknown
mapping, kernel, conditional, parser, dynamic-SQL, or external-schema boundaries
remain explicit. Patch-like output is rejected and impact locations remain
candidates rather than proven root causes.

Production interpretation: combined mode does not blur documentation into code
or code into documented requirements. Partial knowledge stays useful while each
unsupported section fails safely with a machine-readable reason.

Focused lineage, combined retrieval, and answer-contract tests passed `20/20`,
including exact overload resolution, candidate non-authority, reviewed mapping
expansion, lane-specific citations, independent section refusal, and patch
rejection. No API/UI routing, paid generation, mapping approval, or combined-mode
activation occurred. The complete regression suite passed `533` tests with the
one existing non-failing Starlette/HTTPX deprecation warning.

## Step 167A - Import and bind the approved Neo AML lineage review

The SME completed all three mapping decisions as `reviewed`, each with the
rationale `Correct mapping`. The importer verified the candidate artifact ID,
exact mapping-ID set, nonblank verdicts/rationales, FDD document IDs, code
snapshot/artifact identity, and all six file targets before publishing a
separate reviewed artifact.

```powershell
& .\.venv\Scripts\python.exe scripts\import_fdd_code_lineage_review.py `
  data\staging\fdd_code_lineage\neo_aml_v1\candidate_lineage_artifact.json `
  data\exports\code_analysis\neo-aml-fdd-code-lineage-v1-review.md `
  --reviewer AIAgentSmith `
  --code-artifact data\staging\code_embeddings\fci-custom-r1-b1c79c6dc2c5\code_index_text_embedding_3_large_v1\code_index_artifact.json `
  --analysis-directory data\staging\code\fci-custom-r1-b1c79c6dc2c5\plsql_antlr_4_13_2_analysis_v13\analysis `
  --fdd-processed-directory data\indexes\functional_specs_v5\processed `
  --output data\staging\fdd_code_lineage\neo_aml_v1\reviewed_lineage_artifact.json
```

```text
review packet SHA-256: 47c7a8f0bb18167a4810be873f8d37a75cdeda643baac21fd2e0cb4a44ec20e1
reviewed artifact identity: 85f1623e298b73858abbd68596c66aab36c4409739eb48d9ab07f29998f9d738
reviewed mappings: 3
reviewed targets: 6
```

Production interpretation: review authority is now machine-readable and bound
to the exact packet and candidate generation. The candidate files remain
immutable; their status was not edited in place. A local replay followed only
the reviewed artifact and produced three mapping-bounded code units with no
mapping unknowns. Two mappings applied because only their FDDs entered the
top-three evidence set; a reviewed mapping does not force an irrelevant FDD
into retrieval.

Failure behavior rejects packet/artifact identity mismatch, missing or duplicate
mapping IDs, blank rationale, non-reviewed verdicts, unknown sources/targets,
and overwrite attempts. Focused lineage tests passed `5/5`; no OpenAI call or
combined-mode activation occurred. The complete regression suite remained
`533` passing tests with the one existing non-failing Starlette/HTTPX
deprecation warning.

## Step 168 - Draft code-only and combined evaluation manifests

Added a strict `CodeCombinedEvalCase` contract and two versioned draft JSONL
manifests:

```text
data/evaluations/code_grounded_eval_v1_draft.jsonl
data/evaluations/combined_grounded_eval_v1_draft.jsonl
```

The ten release-free user questions cover exact code explanations, impact
analysis, reviewed FDD-to-code lineage, and two unavailable-kernel abstention
cases. Code-only cases cannot declare FDD authority. Answered combined cases
must declare both expected FDD documents and code paths. Abstention cases cannot
declare positive evidence expectations, and a case cannot claim reviewed status
without `sme_reviewed=true`.

```python
case = CodeCombinedEvalCase(
    case_id="combined-aml-transaction-flow-001",
    mode="combined",
    question="How does the system integrate FCIS transactions with FlagRight?",
    expected_code_paths=("pkgamlaintegration_p_custom.sql",),
    expected_fdd_document_ids=("exact-document-id",),
    require_reviewed_lineage=True,
    rationale="Checks reviewed cross-lane lineage.",
)
```

Generated the no-overwrite SME packet
`data/exports/evaluations/code-combined-eval-v1-sme-review.md`, bound to both
manifest hashes. The packet was created locally with zero external API calls.

Production interpretation: these are draft expectations, not a quality gate.
They prevent release labels from leaking into ordinary questions while keeping
source identity as hidden evaluation metadata. SME review must validate the
expected routines, documents, answer state, and unknown boundary before a paid
run.

Failure tests reject cross-lane case definitions, positive evidence on an
abstention case, inconsistent SME status, duplicate case IDs, and review-packet
overwrite.

## Step 169 - Deterministic code/combined retrieval and failure gates

Added `scripts/run_code_combined_retrieval_eval.py` and immutable retrieval
reports. The runner supports local lexical retrieval and dense/hybrid retrieval
only when precomputed query vectors are explicitly supplied. It never creates
query embeddings or calls an LLM. Hybrid code retrieval retains weighted RRF
with `0.40` dense and `0.60` lexical weights; FDD and code scores remain separate.

```powershell
& .\.venv\Scripts\python.exe scripts\run_code_combined_retrieval_eval.py `
  --eval-file data\evaluations\code_grounded_eval_v1_draft.jsonl `
  --eval-file data\evaluations\combined_grounded_eval_v1_draft.jsonl `
  --code-mode lexical --allow-unreviewed
```

The bounded lexical draft run produced
`data/exports/evaluations/code-combined-v1-draft-lexical-bounded-20260820.json`.
All four code-only positive cases passed. Two of four combined positives passed;
the other two retrieved the expected FDD and code file but did not retain the
expected exact routine. The positive pass rate was `6/8 = 0.75`, below the
configured `0.90` threshold. Two kernel cases remained abstention diagnostics.
Because the manifest is unreviewed, release-gate eligibility is false regardless
of score.

The combined merger was corrected to cap final code evidence at `code_limit`;
direct and mapping-bounded searches may each produce candidates, but their union
cannot silently double prompt size. The static-analysis loader now selects
`code_static_analysis_v1` artifacts by schema and accepts either the generation
root or its `analysis/` child, avoiding accidental parsing of the stage manifest.

Production interpretation: finding the correct package is insufficient for a
code claim. The exact routine must survive ranking and bounded evidence packing.
The two failures are retrieval/evidence-selection findings to review before
generation, not reasons to tune the LLM.

Failure tests cover wrong-but-nearby symbols, missing FDD evidence, absent
reviewed mappings, query/case mismatch, unreviewed release-gate exclusion,
unbounded merged evidence, incompatible query-vector input, and immutable report
publication.

## Step 170 - Paid answer-evaluation authorization boundary

Added `scripts/prepare_code_combined_answer_eval.py`. It hash-binds the exact
evaluation manifests and retrieval report, requires exact case-set equality,
requires a passing retrieval threshold, and records the intended disclosure and
request counts without calling OpenAI.

```python
if not report["summary"]["retrieval_threshold_passed"]:
    raise ValueError("Paid answer evaluation is blocked by the retrieval gate")
```

The real draft report intentionally triggered this failure because its pass rate
was `0.75`. Consequently no paid answer plan was published, no question or
internal evidence was sent externally, and no OpenAI cost was incurred. Even
after retrieval passes, explicit disclosure/cost authorization and SME answer
review remain separate gates; activation is not automatic.

Production interpretation: the cheapest safe failure is before generation.
Spending on an LLM when exact evidence is already known to be missing would add
cost and nondeterminism without repairing grounding.

Failure tests prove that a failed retrieval gate blocks preparation, report and
manifest case-set drift is rejected, a valid plan records zero external calls,
and an existing plan cannot be overwritten.

Focused Steps 168-170 tests passed `14/14`. The complete regression suite passed
`542` tests with the one existing non-failing Starlette/HTTPX deprecation
warning. Steps 168-169 mechanisms and the Step 170 authorization boundary are
implemented; the SME manifest review, two combined retrieval gaps, paid answer
run, SME answer review, rollback proof, and deliberate activation remain open.

## Step 168A - Import the reviewed code/combined benchmark

Added `scripts/import_code_combined_eval_review.py` to validate the exact
ten-case review scope, accepted verdicts, absence of unapplied corrections,
manifest hash bindings, and nonblank per-case or explicit global rationale. It
published separate reviewed manifests rather than changing the draft evidence:

```text
data/evaluations/code_grounded_eval_v1_reviewed.jsonl
data/evaluations/combined_grounded_eval_v1_reviewed.jsonl
data/evaluations/code_combined_eval_v1_review_20260821.json
```

```text
reviewed cases: 10
review packet SHA-256: 7f06d2d275989e218d339f796ea3eb99511b2688fa7f33374ca2548219b1d913
review ledger identity: 76a09f9781ba0a34cf7febf19c3845b1ecb4632a71c2065ece5883f813a39592
```

Three packet sections had accepted verdicts but blank local rationales. The
importer bound those decisions to the explicit global approval note supplied for
the whole packet and recorded `rationale_source=global_approval_note`; it did not
invent source-specific reasoning.

The reviewed lexical gate reproduced the draft result: `6/8` positive cases
passed (`0.75`), while both abstention cases remained diagnostics. SME approval
therefore established benchmark authority but did not mask the two exact-symbol
retrieval failures. The paid-answer preparation boundary continued to fail
closed, and no OpenAI call or cost occurred.

Production interpretation: human approval determines whether the expectations
are legitimate; it does not turn a failing system result into a pass. Failure
tests reject duplicate cases, non-accepted verdicts, unapplied corrections,
packet/manifest hash drift, scope drift, blank rationale without an explicit
global approval note, and overwrite of reviewed outputs.

## Step 171 - Localize combined exact-symbol evidence loss

Extended combined retrieval and evaluation traces to retain direct and
mapping-bounded dense/lexical candidate summaries, including `parent_unit_id`,
without copying full source text into diagnostic summaries.

```python
class CombinedRetrievalResult(FrozenModel):
    direct_lexical_candidates: tuple[CodeCandidateSummary, ...] = ()
    mapped_lexical_candidates: tuple[CodeCandidateSummary, ...] = ()
```

The reviewed failures were localized before generation:

```text
combined-aml-batch-send-002: spSendBatchTxnEndData lexical rank 12
combined-aml-offline-impact-004: spOfflineParallelUserEnd lexical rank 13
```

Both expected routines were valid candidates within the configured candidate
budget of 30. They were lost when repeated bounded children from higher-ranked
parent routines consumed the final top-10 evidence budget. This ruled out
ingestion, embedding absence, code-file filtering, FDD selection, reviewed
mapping, and LLM generation as root causes.

Production interpretation: candidate-lane traces identify the first stage where
evidence disappears. Full source is excluded from these diagnostics to control
storage and protect citation provenance.

Failure tests preserve query/case identity, source path, symbol, parent, rank,
and line ranges independently so nearby code cannot masquerade as the expected
routine.

## Step 172 - Add parent-first code evidence diversity

Added configurable `max_units_per_parent` selection, defaulting to two, across
lexical, dense, hybrid, direct, mapped, and final merged code evidence. Selection
uses parent-first round-robin ordering:

```python
for occurrence in range(max_units_per_parent):
    layer = each_parent_at_occurrence(occurrence)
    add_in_original_rank_order(layer)
```

The first attempted greedy cap still allowed a second child from an early parent
to displace the first child of a later parent. Its real-corpus replay continued
to fail `6/8`, so it was not accepted as the fix. Parent-first round-robin takes
the best child from each distinct parent before admitting second children. It
does not alter embeddings, Qdrant points, reviewed mappings, lexical scoring, or
the weighted-RRF weights.

Production interpretation: bounded child chunks are necessary for large source
files, but chunk multiplicity must not become an accidental ranking advantage.
Parent diversity improves symbol coverage while the configurable second round
still permits extra context when budget remains.

Failure tests cover an unseen parent competing with a higher-ranked second child,
invalid zero limits, exact parent metadata propagation, final merge bounds, and
candidate-trace retention. Focused retrieval/lineage/evaluation tests passed
`30/30`.

## Step 173 - Re-run the reviewed gate and prepare paid evaluation

The reviewed deterministic lexical gate passed every positive case after the
parent-first correction:

```text
positive cases: 8/8
positive pass rate: 1.0
minimum required: 0.90
reviewed manifest: true
release-gate eligible retrieval result: true
abstention diagnostics: 2
external API calls: 0
```

The immutable report is
`data/exports/evaluations/code-combined-v1-reviewed-lexical-parent-round-robin-20260821.json`
with SHA-256
`2bbe56454a2c1a8bf97ae391b5aa1783fbfa3f614a8c2f6e6d1337fb3692c8bb`.
The two repaired final evidence sets now contain `spSendBatchTxnEndData` and
`spOfflineParallelUserEnd` respectively.

Prepared, but did not execute, the hash-bound paid evaluation plan at
`data/exports/evaluations/code-combined-answer-eval-plan-v1-20260821.json`
(SHA-256
`8ec0b81d8081e07ea032e085cf3debfb9c4aa1f0ccfd76734960e3762894f988`).
It covers ten answer-generation requests and ten query-embedding inputs and
states that evaluation questions plus retrieved internal FDD and PL/SQL excerpts
would be disclosed to OpenAI.

Production interpretation: deterministic retrieval is now eligible to proceed
to paid evaluation, but it is not proof of answer correctness or activation.
Explicit disclosure/cost authorization, paid traces, citation/entailment checks,
SME answer review, rollback verification, and deliberate activation remain open.

Failure behavior still blocks unreviewed manifests, case-set/report drift,
failed retrieval thresholds, plan overwrite, and paid execution without explicit
authorization. No OpenAI call or cost occurred in Steps 171-173. The complete
regression suite passed `549` tests with the one existing non-failing
Starlette/HTTPX deprecation warning.

## Step 174 - Harden the paid code/combined evaluation boundary

Added `app/fdd_code_lineage/paid_evaluation.py` and
`scripts/run_code_combined_paid_answer_eval.py`. The runner hash-validates the
ten reviewed cases and passing retrieval report, requires an explicit disclosure
flag, performs one query embedding per case for both FDD and code lanes, disables
automatic OpenAI retries, retains exact prompts/evidence and provider request and
usage metadata, validates lane-specific citation contracts, and publishes a
separate SME review packet. An explicit query vector can now be supplied to the
FDD retrieval service so it cannot silently create a duplicate embedding call.

Production interpretation: structural validation and request accounting are
release evidence, not semantic approval. Generated results remain inactive and
require SME entailment review, rollback evidence, and deliberate activation.

Failure tests cover explicit-vector embedding bypass, malformed or cross-lane
citations, exact case/hash drift, missing authorization, collection absence,
output overwrite, and fail-closed partial-run preservation. Focused tests passed
`21/21`; the complete regression suite passed `550` tests with the existing
non-failing Starlette/HTTPX deprecation warning.

## Step 175 - Paid run attempted and stopped before generation

The authorized run made one successful query-embedding request for
`code-aml-batch-send-001`, then failed closed before retrieval or answer
generation because the command incorrectly used the FDD Qdrant path for the
separate `code_custom_r1_v2` collection. The immutable partial run is retained at
`data/exports/evaluations/code-combined-paid-answer-20260821/`:

```text
embedding requests completed: 1
answer requests completed: 0
failed case: code-aml-batch-send-001
failure: code_custom_r1_v2 absent from data/qdrant_local
```

The runner now accepts separate FDD and code Qdrant paths and validates both
required collections before any external call. A corrected dry run succeeded
with `data/qdrant_local/functional_specs_v5` and
`data/qdrant_code_local/code_custom_r1_v2`, making zero external calls.

Production interpretation: separate knowledge lanes also have separate physical
stores. Every paid workflow must complete all local collection/readiness checks
before consuming external cost. The lost in-memory query vector was not persisted,
so completing all ten cases requires one replacement embedding call; it will not
be retried without explicit additional authorization.

The user authorized the required replacement embedding. The first corrected run
completed all five code-only cases with structural passes, then failed closed on
the first combined case after its query embedding because `TemporalQueryPlan` is
a dataclass rather than a Pydantic model. It made six embeddings and five answer
calls. The completed five cases were preserved and were not regenerated.

## Step 176 - Resume, finish, and structurally evaluate the paid run

Corrected temporal-plan serialization with `dataclasses.asdict` and added a
hash-validated `--resume-from` mechanism. Resume accepts only a `failed_closed`
run bound to the same authorization plan, validates completed case IDs and trace
presence, skips completed cases, and reports prior/current request counts
separately. Its dry run proved that only the five unfinished combined cases
would create new requests.

The continuation completed all five combined cases. Across the final run chain:

```text
reviewed cases completed: 10/10
structural passes: 8/10
query embeddings in final run chain: 11 (one replacement after the local failure)
answer generations: 10
successful-case embedding tokens: 207
answer input tokens: 150,560
answer output tokens: 28,165
activation authorized: false
```

Including the earlier wrong-Qdrant-path attempt, the complete operational history
contains 12 successful embedding requests and 10 answer requests. Automatic API
retries remained disabled throughout.

The two structural findings are:

- `combined-aml-offline-impact-004` answered with several cited candidate change
  locations but did not cite the benchmark's expected `spOfflineParallelUserEnd`
  symbol.
- `combined-kernel-http-negative-005` correctly refused the exact hidden-kernel
  fact in its unknown section, but also answered documented/visible-code sections
  with nearby connection-handling evidence, while the deterministic benchmark
  expected every material section to refuse.

These are SME-review findings, not automatic proof that either answer is
semantically unacceptable. The immutable report and review packet are:

```text
data/exports/evaluations/code-combined-paid-answer-20260821-retry2/run-state.json
SHA-256: 17cf1836e3f05343fba7c6fd6bdb7e94335321b8575bd032cfa008fac69fa832

data/exports/evaluations/code-combined-paid-answer-20260821-retry2/sme-review.md
SHA-256: 952989539ed945134e315421a8c3b21b551c6b0405b4b727956bdef12d11fd3f
```

Production interpretation: a structural gate should expose missing expected
citations and refusal-state disagreements, but the SME decides whether the
benchmark is too strict or the answer is materially unsafe. No collection,
runtime configuration, API/UI route, or activation state changed.

Failure-mode testing covered missing/wrong physical collections before external
calls, no automatic retries, immutable partial traces, same-plan resume binding,
completed-case deduplication, dataclass serialization, citation-lane validation,
expected-symbol citation checks, and strict abstention behavior.
