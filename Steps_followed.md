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
