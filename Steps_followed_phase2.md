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

## Step 177 - Validate the submitted paid-answer SME packet

Compared the edited Markdown packet with the immutable `run-state.json` and
case traces before importing any verdict. The packet SHA-256 is now
`75c2a641307bbff970eba1d79739e8ae167815e11660c13c63b0de91b9194c1e`.

Eight ordinary accepted cases retain their generated structural results. Two
material cases require normalization:

```text
combined-aml-offline-impact-004
machine result: fail (expected spOfflineParallelUserEnd was not cited)
edited packet display: pass
SME verdict: accepted, rationale blank

combined-kernel-http-negative-005
machine result: fail (nearby sections answered; exact kernel fact refused)
SME verdict: corrected
SME intent: accept the answer and close the case
```

The machine result is immutable observed evidence and cannot be changed by an
SME verdict. An SME may instead override the benchmark expectation or accept a
semantic answer despite a structural miss; that decision must be stored as a
separate reviewed field with rationale. The final case also requires scope
clarification: visible custom PL/SQL is available, but the exact hidden kernel
implementation and a proven defect line remain unavailable.

Production interpretation: structural execution results, SME semantic verdicts,
and benchmark corrections are three different facts. Collapsing them into one
`pass` field destroys auditability and prevents meaningful regression analysis.
No verdict ledger was published and activation remains unchanged while the two
normalization details are unresolved.

## Step 178 - Import the semantic SME answer ledger

Added `scripts/import_code_combined_answer_review.py`. It hash-binds the paid
run state, edited SME packet, and every exact case trace while storing machine
structural results separately from semantic verdicts. It detected the one edited
structural display drift and retained the immutable observed value.

```text
observed structural passes: 8/10
SME semantic acceptances: 10/10
packet structural display drifts: 1
ledger identity: 20b432aaa01884e6e6e58b153ffde2b6c35338d020092a055b97182484988edf
```

The ledger is `data/evaluations/code_combined_answer_review_20260822.json`.
The final-case `corrected` packet label was normalized to `accepted` using the
explicit chat confirmation; this override source is recorded. No historical
trace or machine result was modified.

Production interpretation: execution evidence and human acceptance answer
different questions. Retaining both allows later teams to improve the benchmark
without falsely claiming the old implementation passed a rule it did not pass.

Failure handling rejects scope drift, duplicate or unknown case IDs, missing
traces, authorization-plan hash drift, blank rationale without an explicit global
approval note, non-accepted semantic scope, and ledger overwrite.

## Step 179 - Correct the combined evaluation contract

Added a backward-compatible v2 evaluation contract. Exact expected code symbols
now support `all`, `any`, or `advisory` policies. The offline impact case retains
`spOfflineParallelUserEnd` as a useful diagnostic but makes it advisory, so other
grounded candidate locations do not become an automatic semantic failure.

Combined answers now expose two independent states:

```python
requested_claim_supported: bool
related_grounded_context_provided: bool
```

A hidden-kernel request can therefore be marked unsupported while separately
providing cited visible custom-code context. The future abstention evaluator
checks the global requested-claim state rather than requiring every helpful
section to refuse.

Published immutable artifacts:

```text
data/evaluations/combined_grounded_eval_v2_reviewed.jsonl
SHA-256: 902c22780af4fb870f7e50dcf8304704e3070335dc8463e34f7c4b93c9039edb
data/evaluations/combined_grounded_eval_v2_contract_20260822.json
ledger identity: b65725e182c02b0b10fa9651ecdc4bb02fb671e18c4d0dc4c397fffe5e2da156
```

Production interpretation: this changes evaluation semantics, not retrieval
weights, embeddings, indexed evidence, or the historical paid answers. Old v1
manifests remain reproducible.

Failure tests cover missing all-of symbols, advisory diagnostics, helpful context
with an unsupported requested claim, cross-lane citations, and patch rejection.

## Step 180 - Assess activation readiness without cutover

Added `scripts/assess_code_combined_activation.py` and published
`data/exports/evaluations/code-combined-activation-readiness-20260822.json`
(identity `5a5a761843a4a0cfee289f5549c61885ba9eac5d9e027b8927f6f1e11b869f5e`).

Four of nine checks passed: semantic review, corrected evaluation contract, FDD
v5 with v4 rollback, and code v2 with v1 rollback. The retained collection sizes
are:

```text
functional_specs_v5: 1125 points; rollback v4: 937
code_custom_r1_v2: 279 points; rollback v1: 111
```

Activation correctly remains blocked because the public query contract has no
explicit `fdd`/`code`/`combined` mode, settings do not identify the active code
artifact/store, readiness checks only FDD dependencies, combined generation is
evaluation-only rather than runtime orchestration, and API-level rollback has
not been rehearsed.

Production interpretation: the three knowledge capabilities work as evaluated
mechanisms, but only FDD mode currently serves users. The next batch should add
explicit runtime mode contracts, orchestration/readiness, and rollback tests
before requesting a deliberate cutover.

Focused tests passed `22/22`; the complete suite passed `551` tests with the one
existing non-failing Starlette/HTTPX deprecation warning. No external API call,
Qdrant mutation, `.env` change, or runtime activation occurred in Steps 178-180.

## Step 181 - Add explicit feature-gated API and configuration contracts

Extended `QueryRequest` with independent `knowledge_mode` (`fdd`, `code`, or
`combined`) and `analysis_kind` fields. `retrieval_mode` remains independently
responsible for dense/lexical/hybrid search. Responses now support code citations
with snapshot/path/symbol/line identity, combined section states, and global
requested-claim/related-context states while retaining backward-compatible FDD
defaults.

Added configuration for the code Qdrant path/collection, embedded artifact,
analysis generation, reviewed lineage artifact, and FDD generation. The exposure
gate defaults to:

```text
CODE_MODES_ENABLED=false
```

Production interpretation: validated offline capability does not silently become
a served capability. Existing API/UI/conversation callers continue to use FDD
mode unless they explicitly request another mode after activation.

Failure tests prove invalid knowledge modes fail schema validation, legacy FDD
payloads remain compatible, and code/combined requests receive a safe `503`
without invoking retrieval or OpenAI while the flag is disabled.

## Step 182 - Add shared-query code/combined runtime orchestration

Added `app/services/knowledge_mode_orchestration.py`. The service loads the exact
reviewed code artifact, creates one query embedding, reuses it for code and FDD
dense retrieval, retains independent weighted-RRF lanes, follows only reviewed
lineage in combined mode, applies the code/combined answer contracts, and writes
a local trace containing request usage, exact evidence, prompts, response, and
citations.

```python
vector = embed_query_once(query)
code = retrieve_code(query_vector=vector)
fdd = retrieve_fdd(query_vector=vector)  # combined mode only
```

Production interpretation: embedding reuse controls cost without merging the two
knowledge authorities. FDD citations describe documented functionality; code
citations describe visible implementation. Hidden kernel/runtime behavior remains
unsupported even when related custom-code context is returned.

Failure tests verify one embedding call, one retrieval/generation path, explicit
vector reuse, unavailable collection failure, reviewed lineage compatibility,
immutable trace writes, cross-lane citation rejection, and required global claim
state in combined model output.

## Step 183 - Add mode-aware readiness and rehearse rollback

`GET /ready` now accepts an explicit `knowledge_mode`. Code readiness checks the
feature gate, model configuration, reviewed embedded artifact, and configured
code collection/artifact count. Combined readiness additionally checks FDD
retrieval artifacts, FDD collection, and reviewed lineage presence. These bounded
runtime checks complement, rather than repeat, the heavier exact point verifier
used at indexing/activation gates.

The API feature flag was tested through a deterministic false → true → false
sequence: requests failed closed before enablement, reached orchestration only
while enabled, and immediately failed closed after rollback. No collection or
`.env` mutation was involved.

The final readiness artifact is
`data/exports/evaluations/code-combined-activation-readiness-v3-20260822.json`
with identity
`654c3c7907f97755f8a08c6479f2c4470c1dd62a541d0f23fb37ebdb79488d93`:

```text
activation readiness checks: 9/9
activation ready: true
activation performed: false
```

Production interpretation: the API capability is ready for a deliberate local
activation and smoke test, but remains inactive. The Streamlit UI has not yet
been given a mode selector, and no live code/combined request was made in this
batch.

Focused runtime tests passed `51/51`; the complete regression suite passed `556`
tests with the one existing non-failing Starlette/HTTPX deprecation warning. No
external API call, Qdrant write, `.env` change, or activation occurred.

# Active progress record

The complete Phase 2 implementation history for Steps 150-183 is archived in
`Steps_followed_phase2.md`.

## Current handoff after Step 183

- Explicit `fdd`, `code`, and `combined` knowledge-mode API contracts exist.
- Knowledge mode remains separate from dense, lexical, or hybrid retrieval mode.
- Combined mode reuses one query embedding while retrieving and ranking the FDD
  and code lanes independently with separate evidence and citations.
- Mode-aware readiness and deterministic feature-flag rollback are implemented.
- Activation-readiness report:
  `data/exports/evaluations/code-combined-activation-readiness-v3-20260822.json`
- Readiness identity:
  `654c3c7907f97755f8a08c6479f2c4470c1dd62a541d0f23fb37ebdb79488d93`
- Readiness result: **9/9; activation-ready; activation not performed**.
- `CODE_MODES_ENABLED` remains `false`; `.env` was not changed.
- Latest complete regression result: **556 passed**, with one existing
  non-failing Starlette/HTTPX deprecation warning.

The next batch selected for implementation was Steps 184-186: Streamlit
knowledge-mode selection, lane-specific evidence rendering, and fail-closed UX.

## Step 184 - Wire explicit knowledge modes through durable conversations

Extended `ConversationMessageRequest` with validated `knowledge_mode` and
`analysis_kind` fields, then forwarded both through the conversation endpoint
into the existing `QueryRequest` contract. The typed UI client now requests
readiness for the exact selected knowledge mode.

The code/combined runtime receives bounded conversation memory for reference
resolution. It enriches the single shared retrieval query and includes an
explicit prompt instruction that conversation memory is not source evidence and
must never support or receive citations.

```python
class ConversationMessageRequest(BaseModel):
    content: str
    knowledge_mode: Literal["fdd", "code", "combined"] = "fdd"
    analysis_kind: Literal["explanation", "impact_analysis"] = "explanation"

readiness = api.get_readiness(knowledge_mode=request.knowledge_mode)
if readiness.knowledge_mode != request.knowledge_mode:
    raise UiApiError(code="readiness_mismatch", ...)
```

Production interpretation: a Streamlit selection now reaches the authoritative
runtime lane instead of being silently dropped by the conversation API. A
mode-mismatched readiness response fails closed. Failure testing verified field
forwarding, mode-specific readiness, and bounded context propagation without an
external API call.

## Step 185 - Add lane-aware Streamlit controls and evidence rendering

Added explicit sidebar controls for Functional documents, Visible custom code,
and Documents + custom code. Explanation and impact-analysis intent are exposed
only where meaningful. FDD-only request filters are hidden for code/combined
modes because the current extended runtime does not honor those per-request
filters; displaying them would create a false control.

The evidence panel now reports the global requested-claim state, related-context
state, independent combined-section states, `[F#]` document citations, and
`[C#]` code citations with snapshot, path, symbol, and exact line ranges.

```python
if response.combined_sections:
    for name, section in response.combined_sections.items():
        marker = "SUPPORTED" if section.status == "answered" else "REFUSED"

for index, citation in enumerate(response.code_citations, start=1):
    render(f"C{index}: {citation.source_path}:{citation.start_line}-{citation.end_line}")
```

Production interpretation: users can distinguish documented intent from visible
implementation and can see when only one section is supported. Hidden kernel,
dynamic SQL, and external-schema limits remain explicit. Failure-mode UX directs
users to FDD mode when code modes are disabled instead of silently falling back.

## Step 186 - Verify disabled-mode UX and reversible rollback behavior

Added deterministic tests for explicit mode payloads, conversation forwarding,
mode-aware readiness, readiness-identity mismatch, lane-rendering contracts, and
a false -> true -> false feature-gate sequence. Only the enabled middle request
may reach message submission.

```python
with pytest.raises(UiApiError):
    run_code_turn()          # disabled
assert run_code_turn()      # enabled
with pytest.raises(UiApiError):
    run_code_turn()          # rolled back
assert api.submit_count == 1
```

Focused verification passed **62/62**. The complete regression suite passed
**559 tests** with the one existing non-failing Starlette/HTTPX deprecation
warning. Python Ruff was not installed in the project environment, so no
dependency was added; compilation and whitespace checks are recorded separately.

Production interpretation: the serving control is reversible in the tested
single-process contract. This does not prove multi-worker propagation, in-flight
request behavior, live answer quality, or disaster recovery. `CODE_MODES_ENABLED`
remains `false`; no `.env` change, service restart, paid API call, Qdrant write,
or activation occurred.

## Step 187 - Define an immutable activation and approval contract

Added frozen Pydantic contracts for an activation request, a separate approval,
preflight checks, and the config-switch result. The request binds the exact FDD
and code collections, processed/index paths, code artifact and lineage hashes,
the accepted readiness report, and a deterministic hash of the runtime query,
readiness, conversation, orchestration, and schema source files.

```python
request = build_activation_request(
    settings=settings,
    readiness_report_path=readiness_report,
    requested_by="AIAgentSmith",
)
assert request.status == "pending_approval"
assert request.target_configuration["CODE_MODES_ENABLED"] == "true"
```

Production interpretation: activation now identifies exactly what is proposed;
directory order and implicit latest files are not authority. Request and approval
hashes detect accidental or silent changes, but they are integrity bindings—not
user authentication or cryptographic signatures. A real approver remains an
organizational control.

Failure testing caught and fixed inconsistent datetime canonicalization that
initially made an unchanged request fail its own identity verification.

## Step 188 - Produce a fail-closed offline activation preflight

Implemented local preflight checks for request integrity, readiness-report byte
identity, runtime configuration drift, safe disabled start state, and a matching
approval. Reports contain no API key names or secret values.

```python
preflight = evaluate_activation_preflight(
    request=request,
    settings=settings,
    readiness_report_path=readiness_report,
    approval=None,
)
assert preflight.ready_to_apply is False
```

Generated pending artifacts:

- `data/exports/activation/code-modes-activation-request-20260822.json`
- `data/exports/activation/code-modes-preflight-pending-20260822.json`
- request identity:
  `a892169b2d50825912b80ee0572055928e701bf3a68732fe20411650e0d0c145`
  (initial request, superseded by the Step 191 regeneration)

The real preflight passes four of six checks and intentionally fails both the
explicit-key and approval checks, so `ready_to_apply=false`. `.env` currently
relies on the safe application default and does not contain a switchable
`CODE_MODES_ENABLED=false` entry. Production interpretation: evaluation success
cannot silently become serving authority, and the activation mechanism will not
invent or append a control key. Configuration or readiness drift also blocks the
switch and requires a new reviewed request.

## Step 189 - Add approval-bound atomic switch and rollback mechanics

Added scripts to record an explicit approval and to dry-run or apply one atomic
change to `CODE_MODES_ENABLED`. The switch requires a matching approved request,
the requested action, and a passing current preflight. It refuses missing or
duplicate `.env` keys, unexpected starting state, rejected/stale approvals, and
failed preflight.

```python
result = switch_code_modes(
    env_path=env_path,
    action="activate",
    request=request,
    approval=approval,
    preflight=preflight,
    apply=False,
)
assert result.applied is False
```

The apply path writes a same-directory temporary file, flushes and `fsync`s it,
then uses `os.replace`; it never prints the `.env` contents. Tests prove dry-run
non-mutation and a false -> true -> false file-level sequence while preserving an
unrelated secret line.

Production interpretation: this is an atomic configuration-file switch, not an
atomic multi-process deployment. It deliberately does not restart FastAPI or
Streamlit, drain in-flight requests, call readiness, or run a paid smoke test.
Those remain explicit activation operations. Focused activation tests passed
**5/5**. The exact final-state regression passed **564 tests** with the one
existing non-failing Starlette/HTTPX deprecation warning. Python compilation and
`git diff --check` also passed.

## Step 190 - Establish a durable explicit-disabled baseline

Added a dry-run-by-default initializer that appends only
`CODE_MODES_ENABLED=false` when the key is absent. It refuses an enabled value,
duplicates, and invalid values while preserving every unrelated `.env` line.

```python
result = initialize_disabled_baseline(env_path=Path(".env"), apply=False)
assert result.before == "missing"
assert result.applied is False
```

The apply path uses the same temporary-file, file-`fsync`, and atomic-replace
mechanism. It now attempts parent-directory `fsync` where the platform exposes
`O_DIRECTORY` and reports whether that durability step succeeded. On this
Windows filesystem it reported `parent_directory_fsynced=false`, making the
remaining durability boundary explicit rather than claiming it was guaranteed.

The initializer was run against the real `.env`: dry-run first, then apply.
`CODE_MODES_ENABLED=false` now appears exactly once. No `true` transition or
process reload occurred.

## Step 191 - Regenerate the approval-pending preflight

Added a reusable preflight command that evaluates an existing immutable request
without overwriting it and optionally consumes a separate approval. The final
request also binds the activation implementation source hash, so changes to the
switching mechanism invalidate the approval target.

The request and preflight were regenerated after the explicit disabled baseline:

- final request identity:
  `d1de15f2a9c83a33666bed88bd6ce7cac21d62fa464dec0796d6e6b5325be23d`
- preflight result: **5/6 passed**;
- sole failed check: `approval`;
- `ready_to_apply=false`.

Production interpretation: every technical prerequisite currently represented
by the offline preflight passes, but technical readiness still cannot authorize
activation.

## Step 192 - Define complete activation execution evidence

Added an immutable execution-evidence contract bound to the request and approval.
It records configuration application, process restart, effective runtime state,
separate code and combined readiness, smoke trace IDs, rollback owner, and
rollback rehearsal.

```python
evidence = build_execution_evidence(
    request=request,
    approval=approval,
    configuration_applied=True,
    service_restart_confirmed=True,
    effective_code_modes_enabled=True,
    code_readiness_passed=True,
    combined_readiness_passed=True,
    smoke_trace_ids=("trace-code", "trace-combined"),
    rollback_owner="operator",
    rollback_rehearsed=True,
)
```

`activation_complete` can become true only when every operational field passes
and the approval separately authorizes both paid smoke calls and internal-evidence
disclosure. Tests prove that activation-only approval remains incomplete even
when fabricated runtime fields are all true. Focused activation tests passed
**7/7** before the final regression run.

Production interpretation: the ledger is a validated evidence container, not a
source of truth for fabricated claims. Real process, readiness, trace, and
rollback evidence must be collected by the controlled activation operation.
The exact final-state regression passed **566 tests** with the one existing
non-failing Starlette/HTTPX deprecation warning. Python compilation and
`git diff --check` passed.

## Step 193 - Bind approval and pass the final activation preflight

Recorded the user's approval as a separate immutable artifact bound to request
`d1de15f2a9c83a33666bed88bd6ce7cac21d62fa464dec0796d6e6b5325be23d`.
The approval authorizes activation and rollback, two paid smoke requests, and
disclosure of their retrieved internal FDD/PLSQL excerpts.

```python
approval = build_activation_approval(
    request=request,
    approved_by="AIAgentSmith",
    paid_smoke_authorized=True,
    internal_evidence_disclosure_authorized=True,
)
```

The approval identity is
`f1515883c9903c6b50d66380ebdd4ad9ab83cd7668b97207620470dffd3f8694`.
The real preflight then passed **6/6**, and an approval-bound dry-run confirmed
the intended `false -> true` transition without mutation. Failure tests retain
the existing controls: stale/mismatched approval, configuration drift, an
unexpected start state, or an ambiguous `.env` still blocks activation.

Production interpretation: technical readiness and human authority were both
present for this exact request. Neither the approval nor the dry-run alone
claimed that the running processes had activated the feature.

## Step 194 - Apply activation, restart, and verify mode readiness

Applied the atomic `.env` transition, restarted local FastAPI and Streamlit, and
checked all requested modes before any paid smoke request.

```python
result = switch_code_modes(
    env_path=Path(".env"), action="activate",
    request=request, approval=approval,
    preflight=preflight, apply=True,
)
assert result.after == "true"
```

Observed runtime state before smoke:

- `/health`: healthy;
- FDD readiness: ready;
- code readiness: ready;
- combined readiness: ready;
- Streamlit: HTTP 200.

The switch reported `parent_directory_fsynced=false` on Windows, preserving the
previously documented crash-durability boundary. Production interpretation:
configuration and bounded dependency readiness passed, but semantic serving was
still conditional on both paid smoke cases. A stale process was deliberately
tested during rollback later; the port-binding conflict demonstrated why a file
change alone cannot prove effective runtime state.

## Step 195 - Run two paid smokes and fail closed with verified rollback

Added `scripts/run_code_modes_activation_smoke.py`, an approval-bound runner with
exactly two reviewed cases, no automatic retries, lane-specific support/citation
checks, and immutable report output. It refuses missing cost or disclosure
authority before any HTTP request. It now also records non-2xx response bodies
and status codes rather than raising before evidence persistence.

```python
report = run_smokes(
    client=client,
    base_url="http://127.0.0.1:8000",
    request=request,
    approval=approval,
)
assert report["automatic_retries"] == 0
```

Exactly two local smoke requests were attempted. The code-only request returned
HTTP 200 and produced grounded trace
`7055d845-2e79-433f-afc4-89c497004d42`: claim support was true, citations C2 and
C4 resolve to the exact `spSendBatchTxnEndData` specification/body ranges, the
query embedding used 24 tokens, and the answer call used 11,449 total tokens.
The combined request returned HTTP 400 and produced no durable answer trace.
Because the initial runner raised before preserving that response body, the
precise ValueError detail is not recoverable; the runner was corrected locally
for future attempts. No retry or third request was made.

The failure triggered the approved rollback. The first restart attempt exposed a
stale process still owning port 8000; exact port owners were then stopped and the
services restarted again. Final observed rollback state is:

- `.env`: `CODE_MODES_ENABLED=false`;
- FDD readiness: HTTP 200 and ready;
- code readiness: HTTP 503, explicitly disabled by configuration;
- Streamlit: HTTP 200;
- activation complete: false.

The local attempt record is
`data/exports/activation/code-modes-activation-attempt-20260822.json`; the
successful code trace remains under `data/exports/answer_runs/`. This proves the
rollback gate works and prevents a partial semantic success from becoming
serving authority. It does not establish the root cause of the combined 400;
diagnosis must occur without another paid request, followed by a new hash-bound
activation request because any runtime fix changes the approved contract.

Focused activation/smoke tests passed **11/11**. The full regression passed
**570 tests** with the one existing non-failing Starlette/HTTPX deprecation
warning. `git diff --check` passed; Ruff is not installed in the project virtual
environment.

## Step 196 - Diagnose the combined HTTP 400 without another paid call

Traced the local execution boundary from `generate_grounded_answer` through the
FastAPI query route. Combined answers required model output to parse as one exact
JSON/Pydantic contract. A parse or schema `ValueError` escaped generation, the
API converted it to HTTP 400, and orchestration wrote its durable answer trace
only after generation returned successfully.

```python
draft = CombinedAnswerDraft.model_validate(_parse_json_object(raw))
# Before this step, ValueError escaped and trace persistence was never reached.
```

The stored code-only trace and API access log prove the observed boundary: the
first request completed and persisted; the combined request returned 400 and no
combined trace exists. Because the original smoke runner discarded the HTTP
body, the exact malformed response and exact exception text cannot be recovered.
The diagnosis therefore establishes the failure class and trace-loss mechanism,
not the exact historical payload. No OpenAI call was made for diagnosis.

Production interpretation: structurally invalid model output is expected
nondeterminism at an LLM boundary and must be handled as an unsupported answer,
not as an untraceable client error.

## Step 197 - Convert invalid combined contracts into grounded safe refusals

Added `build_combined_contract_refusal` and changed paid combined-answer parsing
to catch only local answer-contract `ValueError`s after a successful model call.
Provider, transport, and empty-response failures remain exceptions; they are not
misrepresented as contract failures.

```python
try:
    answer = finalize_combined_answer(
        retrieval=retrieval,
        draft=CombinedAnswerDraft.model_validate(_parse_json_object(raw)),
    )
except ValueError as exc:
    answer = build_combined_contract_refusal(retrieval=retrieval)
    contract_valid = False
    contract_error = type(exc).__name__
```

The safe response sets `requested_claim_supported=false`, refuses documented,
implementation, and impact sections, returns no FDD/code citations, and explains
only that the generated contract could not be validated. It does not expose the
malformed output as functional evidence. Call metadata, prompts, evidence, raw
model output, `contract_valid=false`, and the error class can now reach the normal
local trace writer.

Production interpretation: users receive a semantically honest abstention with
HTTP 200, while operations retain enough evidence to diagnose provider/model
contract instability. This change affects runtime-bound source and therefore
invalidates the prior activation request; a new request will be required.

## Step 198 - Prove graceful failure and activation diagnostics offline

Added deterministic tests with a fake chat response containing malformed JSON.
They prove the response becomes a safe refusal, paid-call metadata remains
available for tracing, and the API returns HTTP 200 with an explicit unsupported
claim rather than HTTP 400.

```python
assert answer.requested_claim_supported is False
assert call["contract_valid"] is False
assert call["contract_error"] == "JSONDecodeError"
assert response.status_code == 200
```

The approval-bound smoke runner also records non-2xx status and response content
in future attempt reports and still performs no automatic retry. Tests cover
missing disclosure authority, exact two-request bounds, missing lane evidence,
and HTTP failure capture. Focused tests passed **26/26**. The full regression
passed **572 tests** with the one existing non-failing Starlette/HTTPX deprecation
warning. Python compilation and `git diff --check` passed. No OpenAI request,
embedding, Qdrant write, configuration switch, or service activation occurred.

The next activation path must regenerate readiness/request identities for the
changed runtime contract, receive a new approval, and repeat both controlled
smokes. Code/combined serving remains disabled meanwhile.

## Step 199 - Expand the hash-bound activation runtime surface

Replaced the private seven-file runtime tuple with the explicit
`ACTIVATION_RUNTIME_FILES` contract and added the files that governed the failed
live boundary:

- `app/fdd_code_lineage/paid_evaluation.py`;
- `app/fdd_code_lineage/combined_answer.py`;
- `scripts/run_code_modes_activation_smoke.py`.

```python
"runtime_contract_sha256": _canonical_sha256({
    name: _file_sha256(root_dir / name)
    for name in ACTIVATION_RUNTIME_FILES
})
```

Tests assert these files remain in the bound surface. A change to parsing,
refusal semantics, paid-call handling, or smoke validation now changes the
request identity. Production interpretation: approval covers the actual semantic
and operational gate, not merely the API entry point. Failure testing confirms
test request construction also fails if a required bound source is absent rather
than silently omitting it.

## Step 200 - Regenerate offline readiness after containment

Regenerated the activation-readiness artifact without an OpenAI call or runtime
switch. All **9/9** checks passed, including reviewed semantic evidence, corrected
evaluation contract, current/rollback FDD and code collections, explicit API
modes, mode readiness, orchestration, and rollback rehearsal.

```text
report_identity_sha256 =
cf563f5c2b958839465f3b2823b3c2425da9453f8f89220c161f20f45a3aae70
```

The artifact is
`data/exports/evaluations/code-combined-activation-readiness-v4-20260822.json`.
Production interpretation: existing reviewed quality evidence and retained
generations remain compatible with the containment change. This offline result
does not prove live-model stability or authorize activation.

## Step 201 - Create a new approval-pending request and reject stale authority

Created a new immutable activation request bound to readiness v4, the expanded
runtime contract, current configuration, collections, artifacts, and lineage.

```text
request_identity_sha256 =
36e3c1c244def6be37d8b1ff9ee02cbc3f072623a24e16b42f65af05761a0fd4
runtime_contract_sha256 =
cf9d3b045468bec994a0cf2e95a332c03b6ea9877d52415aab2f56689716fa32
```

Artifacts:

- `data/exports/activation/code-modes-activation-request-v2-20260822.json`;
- `data/exports/activation/code-modes-preflight-pending-v2-20260822.json`;
- `data/exports/activation/code-modes-preflight-stale-approval-v2-20260822.json`.

The pending preflight is **5/6**, failing only `approval`. Supplying the old
approval intentionally also remains blocked because it references request
`d1de15f...`, not the new `36e3c1c...` identity. This proves authorization cannot
carry across changed runtime semantics.

Focused tests passed **32/32**. The full regression passed **572 tests** with the
one existing non-failing Starlette/HTTPX deprecation warning. Code/combined mode
remains disabled; FDD readiness remains healthy. No paid call, embedding, Qdrant
write, service restart, configuration mutation, or activation occurred.

## Step 202 - Approve and preflight activation request v2

Recorded explicit approval for request
`36e3c1c244def6be37d8b1ff9ee02cbc3f072623a24e16b42f65af05761a0fd4`
under `AIAgentSmith`. The separate approval authorizes activation, rollback, two
paid smokes, internal FDD/PLSQL evidence disclosure, and the associated cost.

```python
approval = build_activation_approval(
    request=request,
    approved_by="AIAgentSmith",
    paid_smoke_authorized=True,
    internal_evidence_disclosure_authorized=True,
)
```

Approval identity:
`1870f7fba579009ecd934b117de120a31874f68c1a49f61deb60e37d0214cb86`.
The approval-bound preflight passed **6/6** and the switch dry-run proved the
intended `false -> true` transition without mutation. Production interpretation:
the previously missing authority was supplied for this exact expanded runtime
contract; it did not itself claim a live transition.

## Step 203 - Activate, restart exact services, and verify all modes

Applied the atomic feature-flag switch and replaced the exact Python processes
owning ports 8000 and 8501. The restart refused unexpected process types and
verified that the old listeners released both ports before launching replacements.

```python
result = switch_code_modes(
    env_path=Path(".env"), action="activate",
    request=request, approval=approval,
    preflight=preflight, apply=True,
)
assert result.after == "true"
```

Before the paid calls, `/health`, FDD readiness, code readiness, combined
readiness, and Streamlit all returned HTTP 200. Code and combined checks verified
the feature gate, exact code point count, reviewed lineage, FDD collection, and
retrieval artifacts. Production interpretation: the new process had loaded the
enabled state and its bounded dependencies were available, but answer-contract
behavior still required live smoke evidence.

## Step 204 - Preserve the second paid failure and roll back safely

Ran exactly two approval-bound HTTP smokes with zero automatic retries. The
immutable smoke report identity is
`3727857d6ef6f487f6ced3dc747e5cd0b3c130562c3315c03442795513ddf73b`.

The code smoke passed:

- trace: `c480b670-3fb1-4628-a4dd-c0faefda7086`;
- embedding tokens: 24;
- answer tokens: 11,355;
- contract valid: true;
- exact code citations: 2.

The combined smoke returned HTTP 200 but correctly failed the activation gate:

- trace: `93deeb26-824a-46f5-a2ce-2ad6fe34a1a5`;
- embedding tokens: 22;
- answer tokens: 22,738;
- retrieval: 8 FDD units, 8 code units, 3 reviewed lineage links;
- `requested_claim_supported=false`;
- all functional sections safely refused;
- citations returned: zero;
- contract diagnostic: `ValidationError`.

The raw response is now durably available in the restricted trace and reveals
the exact defect: the model returned `requested_claim_supported` as a section
object containing `status` and `text`, while `CombinedAnswerDraft` requires a
boolean. The system prompt is ambiguous because it lists the boolean field and
then says “Each value must be an object.” This is a prompt/schema alignment bug,
not a retrieval or lineage failure.

The failed smoke immediately triggered rollback. The exact active port owners
were stopped, disabled processes were restarted, and final checks prove:

- `.env`: `CODE_MODES_ENABLED=false`;
- health/FDD/UI: HTTP 200;
- code readiness: HTTP 503 with `disabled by configuration`;
- activation retained: false.

Production interpretation: the new containment path worked as designed—the
second paid failure became a traceable safe refusal instead of an HTTP 400—but a
safe refusal does not satisfy a positive combined-mode smoke. The stage is not
activated. No third call or retry was attempted. The next fix should remove the
prompt/schema contradiction and add deterministic exact-schema fixtures before
seeking another activation approval. No Python source changed during Steps
202-204, so the already completed **572-test** regression remains the code-state
verification; post-operation configuration, process ownership, readiness, trace
hashes, request count, and `git diff --check` were verified separately.

## Step 205 - Align the combined prompt with an enforced response schema

Removed the contradictory instruction that treated every top-level response
value as a section object. The combined prompt now states unambiguously that
`requested_claim_supported` is a JSON boolean and that only the four named
answer sections contain `status` and `text`.

```python
response_format = {
    "type": "json_schema",
    "json_schema": {
        "name": "combined_grounded_answer",
        "strict": True,
        "schema": CombinedAnswerDraft.model_json_schema(),
    },
}
```

The combined Chat Completions call now supplies this Pydantic-derived schema as
strict Structured Outputs and records the response-format kind plus canonical
schema SHA-256 in the answer trace. This follows OpenAI's documented contract
that `json_schema` Structured Outputs constrains model output to the supplied
schema. Code-only generation retains its existing text contract. Production
interpretation: the provider receives one machine-enforced contract instead of
having to infer types from prose, while local Pydantic validation remains a
defense-in-depth boundary. Failure mode: a provider/SDK that rejects the schema
fails the paid operation rather than silently falling back to unconstrained JSON.

## Step 206 - Add exact-schema and fail-closed regression coverage

Added deterministic checks for the strict schema, prompt alignment, trace schema
identity, malformed JSON, and the exact historical wrong-type shape where
`requested_claim_supported` is an object. Both malformed cases produce a safe
refusal with `requested_claim_supported=false` and no citations.

```python
assert schema["properties"]["requested_claim_supported"]["type"] == "boolean"
assert "Each value must be an object" not in COMBINED_SYSTEM_PROMPT
```

The readiness assessor now includes a `combined_structured_output_contract`
check, so future prompt/schema drift blocks readiness before paid execution.
Production interpretation: a local deterministic gate catches the previously
observed defect without disclosing evidence or consuming tokens. Failure tests
prove invalid JSON and schema-valid JSON with the wrong field type remain
distinguishable (`JSONDecodeError` versus `ValidationError`) while both fail
closed. Focused verification passed **16/16**.

## Step 207 - Rebuild readiness and activation authority offline

The full regression passed **574 tests** with the one existing non-failing
Starlette/HTTPX deprecation warning. Offline readiness then passed **10/10**,
including the new structured-output contract check:

```text
readiness identity = 8e598067ed06eadbf997a95986be6bac5c3a19dacf8eaf39d86016d916139775
```

Created a new immutable, approval-pending request:

```text
request identity = f927d16dde75bdf6ef3fc8d96ad07279ae22185b1ea6c3aea030e57b44692fff
runtime contract = 44218850a432d022dcb13af73600c00c28fb602b5c498f963c3e9865ed279c7d
```

Artifacts:

- `data/exports/evaluations/code-combined-activation-readiness-v5-20260822.json`;
- `data/exports/activation/code-modes-activation-request-v3-20260822.json`;
- `data/exports/activation/code-modes-preflight-pending-v3-20260822.json`;
- `data/exports/activation/code-modes-preflight-stale-approval-v3-20260822.json`.

The pending preflight is **5/6**, failing only approval. The prior v2 approval
was tested against v3 and rejected. Production interpretation: reviewed
knowledge artifacts remain usable, but changed prompt/schema behavior creates a
new serving contract that requires fresh authority. `CODE_MODES_ENABLED` remains
`false`; no OpenAI request, service restart, `.env` mutation, embedding, Qdrant
write, or activation occurred. Generated pytest runtime data is now explicitly
ignored under `data/test_runtime/`.

## Step 208 - Bind approval and complete activation preflight

Recorded `AIAgentSmith` approval for request
`f927d16dde75bdf6ef3fc8d96ad07279ae22185b1ea6c3aea030e57b44692fff`.
The approval separately authorizes activation, rollback, two paid smokes,
internal FDD/PLSQL evidence disclosure, and the associated OpenAI API cost.

```text
approval identity = 6a2d0d3c54639cbdf31400827028386c59f6a9b4f8c2c7d47bdb44512dd8a394
preflight = 6/6
```

The switch dry-run first proved the intended `false -> true` transition without
mutation. Production interpretation: the authorization is bound to the repaired
runtime contract and cannot be reused for another request. Failure testing still
rejects absent, stale, or hash-invalid approval before configuration changes.

## Step 209 - Activate and establish unambiguous runtime ownership

Applied the atomic `.env` transition to `CODE_MODES_ENABLED=true`. Initial
process startup discovered a stale FastAPI listener on port 8000 and duplicate
Streamlit ownership. The new FastAPI process could not bind, so readiness and
paid execution remained blocked.

The exact Python commands and parent/child process identities were inspected.
Only the verified FastAPI and Streamlit service trees were stopped. Both ports
were confirmed released before launching fresh virtual-environment services.
Final owners are:

```text
FastAPI  port 8000 -> PID 46412, parent PID 28968
Streamlit port 8501 -> PID 24324, parent PID 42728
```

Health, FDD readiness, code readiness, combined readiness, and Streamlit then
returned HTTP 200. Production interpretation: process ownership is part of the
activation evidence; merely starting a process does not prove it serves traffic.
Failure-mode handling prevented paid calls against a stale process that might
have loaded the prior disabled configuration.

## Step 210 - Pass both paid smokes and retain rollback capability

Ran exactly the two approval-bound cases with zero automatic retries. Both
passed:

```text
code trace     = 642a27b6-6408-48ab-b417-587fb4df43a7
code tokens    = 11,704
code citations = 5

combined trace     = 8726d6a7-2cfd-44a5-b751-fcfb06627268
combined tokens    = 22,342
FDD citations      = 5
code citations     = 6
```

The combined trace records `contract_valid=true`, `response_format=json_schema`,
and schema SHA-256
`3f31aa0e73dce41327b21dc8c4be916bc466ce5f2cc6f74cd5993cb4363e9e9d`.
Smoke report identity:
`4f15ed0ebbb55205cc73e94a0913545087f13c23cd723b4146b64c76aa8412ff`.

Activation execution evidence is complete with identity
`859d9499ed4d5e4d34b05a82cbc67eb2b5b892662d2e006f133a6f002300f306`.
Post-smoke health/readiness remains HTTP 200 and
`CODE_MODES_ENABLED=true`. A rollback preflight and dry-run passed without
changing the active state, proving the current request remains immediately
rollback-capable. The prior FDD/code generations are retained.

Production interpretation: the local code and combined capability is now
deliberately active and has passed the defined activation gate. This does not
prove production concurrency, long-duration availability, latency SLOs,
centralized identity/access controls, or disaster recovery. Failure policy
remains fail closed and apply the already authorized rollback if a required
runtime gate later fails. The previously completed **574-test** regression
remains the source-state verification; no Python source changed during this
activation batch.

## Step 211 - Add an explicit bounded FDD-search tool contract

Added the versioned `config/agentic_tools.toml` policy and typed explicit-plan
models under `app/agentic_tools`. The policy hash is:

```text
dcdd5b90790c77e913ddcfeaa619715311b8cdca9491d8b78a806472e5b789e2
```

The default contract allows at most three calls, eight results per call, and 16
total retrieved evidence units. `automatic_routing=false` is enforced as a
literal value. FDD mode exposes only `fdd_search`.

```python
plan = create_explicit_tool_plan(
    knowledge_mode="fdd",
    invocations=(("fdd_search", question, 8),),
)
policy.validate_plan(plan)
```

The FDD tool requires complete unit, document, family, release, source-kind, and
text identity before returning evidence. It consumes at most `limit + 1` items
even if a buggy retriever returns an endless iterator. Production
interpretation: the tool is a read-only adapter over the existing grounded FDD
retrieval lane; it does not grant an agent arbitrary search scope or make memory
authoritative. Failure tests reject automatic routing, mode/tool mismatch,
tampered plan identities, over-budget plans, and incomplete citation identity
before downstream claims are possible.

## Step 212 - Add a bounded custom-code search tool contract

Added a separate `code_search` tool that accepts an injected read-only code
retriever and returns the existing source-grounded `CodeEvidence` contract. Code
mode exposes only this tool.

```python
retrieval = search_runner(invocation.query, invocation.limit)
if retrieval.query != invocation.query:
    raise RuntimeError("Code retriever returned evidence for a different query")
```

The output preserves snapshot, path, symbol, exact line range, parser state,
conditional state, and source text. It caps returned units to the approved call
limit and does not perform embeddings, generation, writes, or patch creation.
Production interpretation: code evidence remains independently ranked and
citeable; the tool cannot silently substitute evidence retrieved for another
question. Failure testing verifies truncation and fails closed on query/result
identity mismatch rather than attaching unrelated code to the active plan.

## Step 213 - Add reviewed-lineage impact graph and explicit orchestration

Added a one-hop `impact_graph` tool. It creates FDD-to-code edges only from
`ReviewedLineageUse` records already produced by validated combined retrieval.
It also adds static dependency edges whose source lines overlap selected code
evidence, retaining resolution states such as `kernel_unavailable` instead of
claiming hidden behavior.

```python
execution = execute_explicit_tool_plan(
    plan=caller_authored_plan,
    policy=policy,
    handlers=handlers_by_invocation_id,
)
assert execution.trace.automatic_routing_used is False
```

Graph nodes and edges are capped during construction, not merely sliced after
building an unbounded graph. The executor runs only the caller-authored plan in
sequence, stops after the first blocked/failed call, and records policy, plan,
call, result-count, and evidence identities. Citeable outputs are kept separate
from the privacy-safer trace so source text need not be duplicated in operational
logs.

Production interpretation: this is bounded deterministic tool orchestration,
not autonomous agent routing. Missing reviewed lineage becomes an explicit
unknown, missing handlers become a blocked trace, and handler exceptions record
only their type before later calls are suppressed. Focused and related suites
passed **38/38**; the full regression passed **584 tests** with the existing
non-failing Starlette/HTTPX deprecation warning. No OpenAI call, Qdrant write,
API/UI route change, feature-flag change, or runtime activation occurred. The
existing local FDD/code/combined service remains active, but these new tools are
not yet exposed to users; deterministic tool-level evaluation is the next gate
before runtime integration.

## Step 214 - Define the bounded-tool evaluation contract

Created `app/agentic_tools/evaluation.py` and the six-case draft manifest
`data/evaluations/bounded_agentic_tools_v1_draft.jsonl`. The contract validates
unique case IDs, explicit review state, expected source identities, and the exact
tool sequence for each knowledge mode: FDD uses `fdd_search`, code uses
`code_search`, and combined uses `fdd_search`, `code_search`, then
`impact_graph`. Draft execution requires the explicit `--allow-draft` switch.

Production interpretation: previously reviewed answer questions are useful seeds,
but they do not automatically approve new tool-level expectations. Tool order is
part of the bounded execution contract because changing it can change the inputs
available to later tools. Failure tests cover duplicate IDs, invalid mode/tool
contracts, unreviewed cases, and missing draft authorization. The manifest
SHA-256 is `f38e19ec5942c8b24754fca304629b302a7c5378b5df2c68bf85e46949514aef`.

## Step 215 - Run local lexical evaluation against promoted artifacts

Added `scripts/run_bounded_tool_eval.py`. It loads the promoted FDD v5 lexical
artifacts, the prepared code artifact, and reviewed lineage locally, then runs the
fixed plans without OpenAI, query embeddings, or answer generation. The code path
uses the existing lexical retrieval implementation, for example:

```python
evidence = retrieve_code_evidence(query=case.question, mode="lexical", limit=limit)
```

The real-corpus draft result was **5/6 positive cases**. The FDD cases and both
other combined cases passed. `tool-combined-aml-batch-send-004` retrieved the
correct FDD, code path, and reviewed lineage, but the expected
`spSendBatchTxnEndData` symbol did not enter the combined top-eight code evidence;
the code-only case retrieved that exact symbol at rank eight.

Production interpretation: this isolates a combined evidence-selection or
ranking gap without spending money or disclosing source externally. It does not
measure dense/hybrid retrieval, generated answers, citation entailment, or user
usefulness. The failing reviewed expectation is preserved before any tuning so a
benchmark defect can be distinguished from a retrieval defect.

## Step 216 - Preserve the result, safety evidence, and SME review packet

The evaluator now writes no-overwrite JSON reports and source-minimized SME
review packets. The report identity is
`a7ee4d4bf0a04c7a8d8b5dff8db8849da2fca5088dbfbd2480fd953f9c45758a`.
Safety checks passed **5/5**: automatic routing stayed disabled, over-budget plans
and missing handlers failed before execution, operational traces omitted source
outputs, and external API calls remained zero. Because the cases are draft and
one positive expectation failed, `release_gate_passed=false`.

The packet is
`data/exports/evaluations/bounded-agentic-tools-v1-draft-sme-review-20260823.md`.
It records identities, observed results, and SME fields without copying source
text. Unit fixtures cover capped unresolved static dependencies; the real-corpus
run did not load the protected dependency-analysis directory, so it does not
claim real-corpus dependency precision/recall. Focused tests passed **15/15** and
the full regression passed **589 tests** with the existing non-failing
Starlette/HTTPX deprecation warning.

Production interpretation: the bounded tools remain offline and are not exposed
through the API or UI. No OpenAI call, Qdrant write, configuration activation, or
runtime restart occurred. SME review of the six cases, especially the combined
batch-send expectation, and real-corpus dependency evaluation remain gates.

## Step 217 - Import the bounded-tool SME decision safely

Added `app/agentic_tools/review.py` and
`scripts/import_bounded_tool_eval_review.py`. The importer parses each packet
section by stable case ID and accepts only unchanged `accepted` verdicts. It
requires the packet to contain the exact draft-manifest SHA-256 and evaluated
report identity. A correction or `needs_more_context` verdict fails closed and
requires a new manifest instead of silently rewriting an evaluated expectation.

Production interpretation: the user's chat confirmation is recorded as the
approval source and durable rationale, while the exact packet remains the review
artifact. This separates human acceptance from the deterministic result. Tests
cover CRLF packets, nonaccepted verdicts, corrected expectations, scope/hash
mismatches, and missing approval notes.

## Step 218 - Promote a separate reviewed manifest and ledger

Created `data/evaluations/bounded_agentic_tools_v1_reviewed.jsonl` without
modifying the evaluated draft manifest. All six cases now have
`review_status=reviewed` and `sme_reviewed=true`. Created the hash-bound ledger
`data/evaluations/bounded_agentic_tools_v1_review_20260823.json` under reviewer
`AIAgentSmith`.

The reviewed manifest SHA-256 is
`5f55f5ab9edd073370b835584b95479160bd11257e7cdcac435e3db7faba89a7` and the
ledger identity is
`454cc7a316f4c11ca6a4cbbcb373a9dfdb939bc8c2b689f322eacfc361a29f41`.
Both reviewed outputs refuse overwrite.

Production interpretation: review promotion changes approval state, not observed
retrieval behavior. In particular, SME acceptance of case 4 confirms that its
exact-symbol expectation is valid; it does not turn its structural failure into
a pass.

## Step 219 - Rerun the reviewed deterministic release gate

Reran the local lexical evaluator without `--allow-draft` and wrote
`data/exports/evaluations/bounded-agentic-tools-v1-reviewed-20260823.json`.
The reviewed result is **5/6 positive**, **5/5 safety**, all cases reviewed, and
`release_gate_eligible=false`. The reviewed report identity is
`a66c71b303c88439730e4b597d618f9313a1083b4a9df92b63d092a0290e8b1e`.

The evaluator's exit code 1 is the intended gate signal: the approved case
`tool-combined-aml-batch-send-004` still lacks `spSendBatchTxnEndData` in the
bounded combined code evidence. No external API calls occurred. Focused review
and tool tests passed **18/18**; the full regression passed **592 tests** with the
existing non-failing Starlette/HTTPX deprecation warning.

Production interpretation: bounded-tool API/UI exposure remains blocked. The
next change should target the reviewed combined evidence-selection gap and prove
the fix locally, while preserving code-only behavior, budgets, reviewed lineage,
trace privacy, and the five safety controls.

## Step 220 - Repair the reviewed combined evidence-selection gap

Candidate inspection proved that `spSendBatchTxnEndData` existed at lexical rank
10 for the natural combined question while the bounded output limit was eight.
Added a combined-only identifier-affinity reservation in
`app/agentic_tools/tools.py`. It tokenizes routine names and query wording,
normalizes bounded aliases such as `txn` to `transaction` and `sent` to `send`,
and may reserve one slot when an already retrieved candidate matches at least
three identifier terms more strongly than the weakest selected item.

```python
evidence = select_identifier_affinity_evidence(
    query=invocation.query,
    evidence=retrieval.evidence,
    limit=invocation.limit,
)
```

Production interpretation: the repair does not create evidence, increase the
eight-unit budget, alter embeddings, or change global dense/lexical/RRF weights.
Code-only behavior is unchanged. Failure tests cover weak/no affinity, bounded
replacement, stable leading order, invalid limits, and exact case-4 selection.

## Step 221 - Close the reviewed deterministic gate

Reran the unchanged reviewed manifest and wrote
`data/exports/evaluations/bounded-agentic-tools-v1-reviewed-selector-v2-20260823.json`.
The result is **6/6 positive**, **5/5 safety**, all cases reviewed, zero external
calls, and `release_gate_eligible=true`. The report identity is
`09e52e3a6ecfb2de8a9b8362a97eafc286e0549896d9cc535b767d74836059ec`.

Production interpretation: this closes the deterministic lexical-tool gate for
the six reviewed cases. It does not prove dense/hybrid behavior, generated
answers, broader-corpus recall, or production serving properties. The full suite
passed **595 tests** with the existing non-failing Starlette/HTTPX warning.

## Step 222 - Add a local manual retrieval-UAT boundary

Added `app/agentic_tools/uat.py`,
`scripts/run_local_bounded_tool_uat.py`, and
`docs/Bounded_Tool_Manual_UAT.md`. The runner accepts an explicit mode and
question, enforces the configured result budget, runs only the fixed local
lexical plan, and writes a no-overwrite identity-bound JSON report. It requires
`--acknowledge-internal-evidence-output` because citeable outputs contain internal
FDD/PLSQL source text.

The first case-4 UAT report is
`data/exports/evaluations/bounded-tool-manual-uat-case4-20260823.json`, identity
`689577f9bd8d7bf91fdf645a3f47c3b0eec61ef6ebc991c85e86aceaf97a5a02`.
It completed three fixed calls with 16 total evidence units and included
`spSendBatchTxnEndData` among exactly eight code units.

Production interpretation: manual retrieval UAT can now begin locally without
OpenAI cost or disclosure. No API/UI route, feature activation, service restart,
or automatic routing was added. Generated-answer UAT and any API/UI exposure
remain separate, approval-bound later steps.

## Step 223 - Define a formal ten-case manual retrieval-UAT contract

Added `ManualToolUatCase` and related batch models in
`app/agentic_tools/uat.py`, plus the draft manifest
`data/evaluations/bounded_tool_manual_uat_v1_draft.jsonl`. The ten cases reuse
questions whose business expectations were reviewed previously, while keeping
the new tool-level UAT state explicitly `draft`. The scope covers FDD-only, four
code cases, four positive combined cases, impact analysis, and an unavailable
hidden-kernel request.

Production interpretation: prior SME review is provenance for the source
question, not automatic approval of a new bounded-tool run. The contract records
expected source identities, reviewed-lineage requirements, and whether the tool
should provide evidence or a qualified unknown. Duplicate IDs, invalid limits,
and unexpected schema fields fail validation.

## Step 224 - Preserve a source-aware local UAT batch

Added `scripts/run_local_bounded_tool_uat_batch.py`. It preflights all target
paths, executes fixed local lexical plans, writes one no-overwrite source-bearing
report per case, then writes a source-minimized batch index and SME packet. The
explicit internal-evidence acknowledgement is mandatory. No query embedding,
answer generation, or external call occurs.

The first immutable batch produced **9/10** diagnostics. All positive cases
passed. The hidden-Java-kernel negative case retrieved useful nearby PL/SQL and
reviewed lineage but did not emit an explicit unavailable-boundary state. This
was preserved as a real contract failure rather than overwritten.

Production interpretation: retrieval of nearby visible code is not itself wrong,
but it cannot satisfy a request for an exact hidden Java method or defect line.
The tool graph must retain visible evidence and independently qualify the
unavailable requested boundary.

## Step 225 - Qualify unavailable kernel detail and prepare SME review

Added a bounded unavailable-boundary detector to the impact tool. It requires a
boundary term (`kernel`, `Java`, or `J2EE`), an unavailable-scope term (`hidden`,
`unavailable`, or `internal`), and an implementation-detail term before adding
the unknown. Ordinary visible-code questions remain unaffected.

The new immutable batch passed **10/10 diagnostics**, remained unreviewed, and
made **0 external calls**. Its identity is
`0ecd0b9856b920dc127d2d9c2d10eb878642e317298998646a0ececdd9482b61`.
The review packet is
`data/exports/evaluations/bounded-tool-uat-v1-kernel-qualified-20260824-sme-review.md`.
Focused tests passed **16/16** and the full regression passed **598 tests** with
the existing non-failing Starlette/HTTPX deprecation warning.

Production interpretation: the ten diagnostic results are not a release gate
until SME-reviewed. Paid-use permission has been acknowledged, but it does not
authorize disclosure of retrieved internal FDD/PLSQL excerpts. Paid grounded-
answer execution therefore remains blocked until the UAT packet is accepted and
internal-evidence disclosure is explicitly authorized.

## Step 226 - Bind UAT acceptance, disclosure, and paid-request limits

Added `app/agentic_tools/uat_review.py` and
`scripts/import_manual_bounded_tool_uat_review.py`. The review ledger binds the
draft manifest, successful local batch, source-minimized packet, reviewed
manifest, reviewer/chat approval, paid-use permission, and internal-evidence
disclosure permission. It limits execution to ten answer requests, zero query-
embedding requests, and zero automatic retries.

The first preflight exposed a Windows newline defect: the ledger hashed LF
content while the persisted reviewed manifest contained CRLF bytes. It failed
before any OpenAI call. The writer now uses explicit newline-preserving output;
the original mismatched artifacts remain preserved. The final reviewed manifest
is `data/evaluations/bounded_tool_manual_uat_v1_reviewed_v3.jsonl`; its ledger is
`data/evaluations/bounded_tool_manual_uat_v1_review_20260824_v3.json`, identity
`49d5911a4ff8370216fc9c42e59a4571ce6f8d4d54249e740e73f5be6a46138f`.

Production interpretation: semantic equality is insufficient for a byte-bound
authorization contract. A platform newline transformation must invalidate the
preflight rather than be silently accepted. Tests now compare the recorded hash
with the exact persisted manifest bytes.

## Step 227 - Build the no-retry paid bounded-tool evaluator

Added `app/agentic_tools/paid_uat.py` and
`scripts/run_paid_bounded_tool_uat.py`. The runner reconstructs prompt evidence
from the already preserved local UAT reports, so it makes no new query-embedding
requests. Code mode uses citeable code units; FDD and combined modes use the
strict combined JSON response contract with separate FDD/code sections and
unknown-boundary notes.

The first evidence preflight exposed the same CRLF mismatch in individual UAT
report hashes and stopped with zero paid calls. All UAT writers were corrected,
and the identical ten-case local batch was regenerated under a new immutable
namespace. The corrected batch passed **10/10** with identity
`b5137e83db33f20118389ac4a2c02810d63303a59c001663320796ae2f6e9575`.

Production interpretation: the runner refuses changed manifests, ledgers, batch
reports, or source-bearing case reports; preserves partial state on failure; and
does not retry automatically. Prompts, evidence, raw provider output, request
metadata, typed answers, and structural results are retained locally for the
authorized SME review.

## Step 228 - Execute the authorized paid grounded-answer evaluation

Executed the ten authorized OpenAI answer-generation requests using the preserved
local evidence. Query-embedding requests were **0**, answer requests were **10**,
and automatic retries were **0**. The run completed as
`completed_pending_sme_review` with **8/10 structural passes**.

Two immutable findings require SME review:

- `uat-code-aml-offline-impact-005` safely refused instead of returning the
  expected visible-code impact guidance;
- `uat-combined-aml-unitholder-008` answered but did not cite the expected R22 FDD
  document.

The review packet is
`data/exports/evaluations/bounded-tool-paid-answer-v2-20260824/sme-review.md`.
Focused paid-UAT and serialization tests passed **19/19**. The full regression
passed **601 tests** with the existing non-failing Starlette/HTTPX deprecation
warning.

Production interpretation: 10/10 completion proves the authorized paid scope ran;
8/10 structural pass does not establish semantic acceptance or activation. The
original answers must be SME-reviewed before deciding whether either finding is a
benchmark issue, an acceptable safe refusal/citation choice, or a targeted product
gap. No retry, API/UI exposure, automatic routing, or activation occurred.

## Step 229 - Import the paid SME decisions without rewriting machine results

Added `scripts/import_paid_bounded_tool_uat_review.py`. It binds the paid run
state, edited SME packet, and all ten exact trace hashes while preserving each
immutable structural result. The ledger records **9 semantic acceptances** and
**1 correction required**, with activation false and no additional paid requests
authorized.

The ledger is
`data/evaluations/bounded_tool_paid_answer_v1_review_20260824.json`, identity
`7f642f1061f4711c1275f388849fbee82faee6cf48514a547daad8a565d3c31f`.

Production interpretation: case 8 remains structurally failed because its expected
R22 citation was absent, but the SME explicitly accepted its answer. Human
acceptance is recorded alongside—not substituted for—the machine result. Case 5
remains a remediation item and prevents semantic gate completion.

## Step 230 - Localize the case-5 failure to citation formatting

Inspected only the two structurally failed traces. Case 5 already retrieved
`spOfflineParallelUserEnd` as `[C2]`, and the provider generated useful impact
guidance naming that routine. However, the raw response used bare references such
as `C2` and `Evidence: C2` instead of the required bracketed form `[C2]`.
`finalize_code_answer` therefore failed citation validation and safely returned
`invalid_or_missing_citation`.

Production interpretation: this is not a retrieval or reranking failure. Changing
RRF or evidence selection would treat the wrong layer and could regress unrelated
queries. The correct repair belongs to the generation/citation-format contract.
Case 8 is not changed because the SME accepted its useful R24-grounded answer;
its original missing-R22 structural observation remains available for future
benchmark refinement.

## Step 231 - Strengthen exact code-citation syntax locally

Updated `CODE_SYSTEM_PROMPT` to require exact square-bracket citations such as
`[C1]` and explicitly forbid bare forms such as `C1` or `Evidence: C2`. Existing
citation validation remains authoritative and still fails closed if the provider
does not follow the instruction.

```text
Every citation must use exact square-bracket syntax such as [C1] or [C2].
Never write a bare citation such as C1, "Evidence: C1", or "Evidence: C2".
```

Focused review/prompt tests passed **15/15** and the full regression passed
**602 tests** with the existing non-failing Starlette/HTTPX deprecation warning.

Production interpretation: deterministic tests prove the revised prompt contract
is present and the validator remains fail-closed; they do not prove a live model
will comply. The original ten-request authorization is exhausted. No retry was
made. A new activation-bound prompt identity plus explicit authorization for one
paid case-5 request and its internal code evidence are required before live replay.

## Step 232 - Bind the one-case replay authorization

Added `app/agentic_tools/replay.py` and
`scripts/prepare_paid_bounded_tool_case_replay.py`. The immutable authorization
binds the exact reviewed case, reviewed manifest, prior SME ledger, prior paid
trace, preserved local UAT evidence, current code-system-prompt hash, and answer
model. Its executable limits are exactly one answer request, zero query-embedding
requests, and zero automatic retries.

```text
maximum_answer_requests=1
maximum_query_embedding_requests=0
automatic_retries=0
authorization_identity_sha256=3ce3a5a8374a1de8439fd2bfdf594166c7925592d14c00fe1d16517b1e452eee
```

Production interpretation: prior disclosure does not create an open-ended retry
right. The new authorization is limited to the corrected prompt and the exact
previously retrieved evidence; any content, model, prompt, or request-limit drift
invalidates that authorization.

Failure-mode testing: deterministic tests reject authorization tampering, missing
approval notes, expanded request/embedding/retry limits, disabled permissions, and
attempted authorization-file overwrite.

## Step 233 - Add and exercise the fail-closed replay runner

Added `scripts/run_paid_bounded_tool_case_replay.py`. It validates all bound hashes,
the exact reviewed case, configured model, current prompt identity, disclosure
permission, and a new output namespace before constructing the preserved evidence.
It uses the no-retry OpenAI client and makes no query-embedding request.

The first execution intentionally omitted the execution-only disclosure flag. The
preflight reported one planned answer request, zero embeddings, and zero retries,
then stopped with `PermissionError` before creating the output directory or calling
OpenAI.

Production interpretation: authorization artifacts and execution confirmation are
separate gates. A valid stored authorization cannot be executed accidentally by a
partial command, and a failed preflight consumes neither cost nor disclosure budget.

Failure-mode testing: the runner refuses changed evidence or approval artifacts,
prompt/model drift, missing or duplicate cases, missing disclosure confirmation,
and an existing result directory. Paid failures are persisted as `failed_closed`
and are never retried automatically.

## Step 234 - Execute the authorized one-call replay

Executed exactly one paid answer request for
`uat-code-aml-offline-impact-005` using its previously preserved PL/SQL evidence.
The run used zero query embeddings and zero retries. It completed with a structural
pass: the answer is marked answered, provides visible-code impact candidates, uses
valid bracketed citations including `[C2]`, and retains explicit limitations for
missing routine bodies and unproven call paths.

Artifacts:

- authorization:
  `data/evaluations/bounded_tool_case5_replay_authorization_20260824.json`;
- trace and run state:
  `data/exports/evaluations/bounded-tool-case5-replay-20260824/`;
- SME packet:
  `data/exports/evaluations/bounded-tool-case5-replay-20260824/sme-review.md`.

The run state is `completed_pending_sme_review`, with
`answer_requests_completed=1`, `query_embedding_requests_completed=0`,
`automatic_openai_retries=0`, `structural_passed=true`, and
`activation_authorized=false`. Focused tests passed **21/21**. The full regression
passed **611 tests** with the existing non-failing Starlette/HTTPX deprecation
warning.

Production interpretation: this result demonstrates that the targeted live model
followed the repaired citation contract for this one case. It does not prove broad
model stability, semantic SME acceptance, or activation readiness. The preserved
SME packet must be reviewed; no additional paid call is authorized.

Failure-mode testing: citation validation remains independent of the prompt and
will still convert malformed or bare citations into a safe refusal. The immutable
prior failure remains preserved for before/after diagnosis, and the new runner
cannot overwrite this replay result.

## Step 235 - Parse the replay SME verdict without field bleed

Added `app/agentic_tools/replay_review.py` with a line-bounded review parser. The
reviewed packet records `accepted` for `uat-code-aml-offline-impact-005`, and its
displayed structural result matches the immutable replay result `pass`.

```text
SME verdict: accepted
Structural result: pass
```

The parser deliberately uses horizontal whitespace around field values rather
than `\s*`. This prevents a blank `SME rationale:` line from consuming the next
`Required follow-up:` label. Because the packet rationale is blank, the ledger
uses the explicit chat-confirmed acceptance note and identifies that source.

Production interpretation: human review fields must be parsed as separate records;
an apparently harmless multiline-regex choice can corrupt rationale provenance.
The parser does not infer acceptance from structural success or from an empty
follow-up field.

Failure-mode testing: focused tests reject duplicate/missing packet scope, blank or
unsupported verdicts, and mismatches between the displayed and stored structural
result. A blank rationale is verified to remain blank instead of absorbing the
following field label.

## Step 236 - Import a hash-bound remediation-closure ledger

Added `scripts/import_paid_bounded_tool_case_replay_review.py`. The new ledger
binds the exact replay run state, trace, edited SME packet, one-call authorization,
and original ten-case SME ledger. It verifies that the original case verdict was
`corrected`, the replay verdict is `accepted`, and the original suite had exactly
one unresolved semantic correction.

The ledger is
`data/evaluations/bounded_tool_case5_replay_review_20260824.json`, identity
`d3d1ab100a1c44e0dd1ecd49c5ad86082cd9629ca75bf970c59e7b81eec63e82`.
It records effective semantic acceptance as **10/10**, closes only the case-5
remediation item, authorizes zero additional paid calls, and keeps activation
false.

Production interpretation: the original 9/10 ledger and failed trace remain
immutable. The closure ledger forms a provenance chain rather than rewriting the
historical failure into a pass.

Failure-mode testing: the importer rejects authorization/hash drift, a nonterminal
replay, request/embedding/retry count drift, a mismatched case, an absent prior
correction, a non-accepted replay verdict, and attempted ledger overwrite.

## Step 237 - Close the bounded-tool semantic remediation gate

The effective reviewed result for the bounded-tool paid UAT set is now **10/10
semantically accepted after targeted replay**. Case 5 changed from an immutable
structural failure and SME correction to a separately recorded structural pass and
SME acceptance. No other case result or benchmark expectation was modified.

Focused replay/review tests passed **13/13**. The full regression passed **615
tests** with the existing non-failing Starlette/HTTPX deprecation warning.

Production interpretation: the targeted semantic remediation gate is closed. This
does not authorize another OpenAI request, enable automatic routing, or independently
prove broad bounded-agent runtime quality. Any subsequent API/UI exposure or new
agentic capability still requires its own readiness, privacy, serving, monitoring,
and rollback decision.

Failure-mode testing: `git diff --check` remains the documentation/code whitespace
gate, the ledger is no-overwrite, and activation remains explicitly false even
after complete SME acceptance.

### Phase 2 scoped completion decision

The Steps 235-237 learner gate is accepted **9/9**. Together with the existing
local code/combined activation evidence, reviewed lineage, deterministic retrieval,
source-line citations, safe unknown handling, manual UAT, paid answer evaluation,
and the accepted targeted replay, the approved initial **Phase 2 PL/SQL scope is
complete**.

This closure does not redefine later work as already delivered. JavaScript corpus
support, bounded-tool API/UI exposure, automatic agent routing, larger-corpus
evaluation, concurrency/load evidence, and production security/operations controls
remain explicit future scopes with independent gates.

## Step 238 - Define MCP-ready retrieval configuration and source contracts

Added safe configuration defaults for `INTERFACE_MODE=fastapi` and
`MCP_EVIDENCE_DISCLOSURE_ENABLED=false`. Added `RETRIEVAL_INDEX_PATH` as a
compatibility alias for the existing FDD lexical `PROCESSED_DIR`; configuration
now fails closed if both resolve to different directories.

```python
@property
def fdd_retrieval_artifact_dir(self) -> Path:
    return _resolve_project_path(
        self.retrieval_index_path or self.processed_dir,
        self.root_dir,
    )
```

Added framework-neutral search/fetch result models and SHA-256 opaque source IDs
in `app/retrieval/knowledge_service.py`. Public source references never expose
absolute paths or an FDD internal unit ID.

Production interpretation: configuration identifies one reproducible FDD lexical
generation while preserving explicit code and Qdrant locations. The new models
allow FastAPI and MCP to serialize the same evidence without sharing transport
logic.

Failure-mode testing: conflicting lexical paths fail validation; duplicate active
source identities fail catalog construction; malformed and unknown opaque IDs fail
fetch lookup without resolving a path.

## Step 239 - Add the shared KnowledgeRetrievalService

Added `KnowledgeRetrievalService`, the framework-independent boundary for FDD,
code, and combined retrieval. It owns configured lexical/dense/hybrid selection,
Qdrant lifecycle, active code/FDD artifact validation, reviewed lineage checks,
safe source catalog lookup, and bounded result formatting.

```python
result = service.retrieve(
    query="AML batch processing",
    mode="combined",
    limit=5,
)
```

Combined dense/hybrid retrieval creates one query vector and passes that vector to
both FDD planned retrieval and code retrieval. It does not merge FDD and code
scores; combined output keeps the existing five-per-lane contract.

Production interpretation: FastAPI and MCP can use exactly the same source,
retrieval, ranking, lineage, and Qdrant controls. A future adapter cannot bypass
the approved retrieval contract with a simplified search path.

Failure-mode testing: focused tests use missing/duplicate catalog identities and a
dense FDD path with a fake vector store. A combined hybrid test proves one
embedding call is reused across both lanes and both Qdrant clients close.

## Step 240 - Let answer orchestration consume prepared retrieval

`run_grounded_answer_query(...)` now accepts an optional `planned_retrieval`
argument and rejects it if it belongs to a different query. Code/combined answer
orchestration accepts a prepared `KnowledgeRetrievalExecution` and generates and
traces from that evidence without a second retrieval call.

```python
run_code_or_combined_query(
    mode="combined",
    query=query,
    analysis_kind="explanation",
    settings=settings,
    retrieval_config=config,
    limit=5,
    correlation_id=request_id,
    retrieval_execution=prepared_execution,
)
```

Production interpretation: retrieval is now a reusable, independently testable
phase. Existing answer contracts, citations, safe refusals, conversation context,
and trace format remain owned by their established answer layers.

Failure-mode testing: a prepared retrieval for another query or knowledge mode is
rejected. Existing orchestration, answer-service, and retrieval-config tests
remain green.

Focused verification: **16 passed** across shared-service, existing orchestration,
answer-service, and retrieval-config tests. `git diff --check` found no whitespace
errors; only pre-existing Windows line-ending advisories were reported.

Gate status: **awaiting learner answers for Steps 238-240.** No MCP SDK, tunnel,
disclosure, live OpenAI call, or data-egress operation has occurred.

### Learner evaluation — Steps 238-240

**Accepted, 9/9.** The learner correctly explained fail-closed lexical-generation
selection, stable occurrence/content identity, portable source references,
per-lane ranking, shared query-vector reuse, duplicate-catalog rejection,
query-bound prepared evidence, the answer-orchestration/MCP boundary, and the
remaining stdio protocol test gap. The answers were precise and production-aware;
no remediation was needed.

## Step 241 — Add retrieval-only FastAPI search through the shared service

Added the constrained `POST /search` contract and switched `/query` to prepare
retrieval with `KnowledgeRetrievalService` before passing the same result into its
established answer orchestration. `/search` accepts only a nonblank `query` and
knowledge lane `fdd`, `code`, or `combined`; it does not accept paths, point IDs,
SQL, arbitrary filters, or a retrieval-strategy override.

```python
@router.post("/search", response_model=KnowledgeSearchResponse)
def search_evidence(request: SearchRequest) -> KnowledgeSearchResponse:
    return build_knowledge_retrieval_service(
        settings=get_settings(),
        retrieval_config=build_retrieval_runtime_config(get_settings()),
    ).search(query=request.query, mode=request.mode, limit=5)
```

Production interpretation: API retrieval and answer generation now share one
source/ranking/lineage implementation. Existing answer contracts still own model
generation, citations, refusals, conversation context, and traces.

Failure-mode testing: code/combined remains blocked by `CODE_MODES_ENABLED`;
missing retrieval dependencies return safe 503 responses; a prepared retrieval
cannot be used with another query. FastAPI now refuses startup when
`INTERFACE_MODE=mcp`, preserving an explicit interface boundary.

## Step 242 — Add the guarded, read-only MCP adapter and structured encoding

Added `app/mcp/adapter.py` and `app/mcp/server.py`, using the maintained `mcp`
2.1.0 SDK and its `MCPServer` stdio transport. The adapter calls
`KnowledgeRetrievalService` directly; it contains no HTTP client and never calls
FastAPI.

```python
def search(self, *, query: str, mode: KnowledgeMode) -> KnowledgeSearchResponse:
    settings = self._disclosure_enabled_settings()
    return self._service_factory(settings).search(query=query, mode=mode, limit=5)
```

`MCP_EVIDENCE_DISCLOSURE_ENABLED=false` is checked before service construction.
Disabled `search` and `fetch` return only `Evidence disclosure is disabled.` with
no structured content. Therefore they cannot load a catalog, open Qdrant, embed a
query, resolve an opaque ID, or disclose source identifiers/metadata.

`encode_mcp_result` validates the Pydantic result, makes one canonical dictionary,
uses it as MCP `structuredContent`, and derives fallback text from that same
dictionary. The server exposes only read-only, idempotent, closed-world `search`
and `fetch`; `fetch` accepts only the existing SHA-256 opaque-ID shape.

Production interpretation: the kill switch is an MCP-only emergency egress
control, not a change to FastAPI or Streamlit RAG behavior. The ChatGPT side gets
bounded evidence for interpretation; it does not receive an alternate retrieval
implementation.

Failure-mode testing: disabled calls perform zero service work and emit no source
payload; malformed/unknown fetch IDs fail without resolving paths; non-activated
code lanes fail safely; MCP startup refuses `INTERFACE_MODE=fastapi`; Streamlit
also refuses `INTERFACE_MODE=mcp`.

## Step 243 — Verify adapter equivalence and prepare stdio-safe logging

Configured MCP startup logging explicitly to `sys.stderr` with `force=True`,
captured Python warnings, and reset known HTTP/Qdrant/OpenAI/MCP dependency logger
handlers to propagate to that stderr handler. The implementation deliberately does
not replace `sys.stdout`, because the MCP SDK owns stdout for JSON-RPC frames.

```python
logging.basicConfig(
    level=getattr(logging, level.upper(), logging.INFO),
    handlers=[logging.StreamHandler(sys.stderr)],
    force=True,
)
logging.captureWarnings(True)
```

Focused equivalence tests confirm the FastAPI `/search` route and direct MCP
adapter request the same shared result for `fdd`, `code`, and `combined` modes.
They also validate canonical structured/text encoding, disclosure-disabled output,
read-only tool metadata, and interface-mode startup refusal.

Production interpretation: this establishes one local retrieval implementation
behind two transports. Dense/hybrid remains internal retrieval configuration;
combined execution reuses one query vector across FDD and code lanes, so no new
public strategy parameter or duplicate embedding path exists.

Failure-mode testing: 46 focused tests passed. `uv lock --check` passed after
pinning the maintained MCP SDK. `git diff --check` found no whitespace errors;
only pre-existing Windows line-ending advisories were emitted. Actual subprocess
JSON-RPC framing and stderr/stdout isolation remain the explicit Steps 244-246
gate; no tunnel, live OpenAI call, or evidence disclosure occurred.

Gate status: **awaiting learner answers for Steps 241-243.**

### Learner evaluation — Steps 241-243

**Accepted, 9/9.** The learner accurately distinguished knowledge lanes from
retrieval strategy, explained deterministic prepared evidence, separated
retrieval and generation failure classes, described true pre-retrieval egress
control, opaque-ID safety, independent FDD/code feature gates, canonical output
serialization, stdout ownership, and the remaining subprocess wire test. No
answer needed strengthening.

## Step 244 — Test the actual MCP stdio JSON-RPC transport

Added a raw subprocess protocol harness in `tests/test_mcp_stdio_protocol.py`.
It starts `python -m app.mcp.server`, sends `initialize`,
`notifications/initialized`, `tools/list`, and a disclosure-disabled
`tools/call`, then parses every stdout line as JSON-RPC.

```python
message = json.loads(line)
assert message["jsonrpc"] == "2.0"
```

The test also enables a test-only diagnostic injection. An `httpx` logger warning
and a Python `RuntimeWarning` are emitted during startup and asserted on stderr.
The test proves they do not contaminate stdout protocol frames.

Production interpretation: stdio is a protocol boundary, not a console. A single
log prefix, warning, or `print()` on stdout could break a ChatGPT/tunnel session;
the test exercises the real child transport rather than assuming object-level JSON
returns prove wire correctness.

Failure-mode testing: the disabled call is an MCP tool error with exactly the
generic message and no `structuredContent`; tools publish only `search` and
`fetch` with read-only, non-destructive, closed-world annotations. No source
catalog, Qdrant, embedding, tunnel, or OpenAI call is used.

## Step 245 — Add child-process MCP startup preflight

Added `app/mcp/preflight.py`. Before tools register, the MCP child validates the
effective interface mode, retrieval strategy, FDD lexical artifact directory, and
when code modes are enabled, the code artifact, analysis directory, and reviewed
lineage artifact. These checks are local-only and deliberately do not load a
catalog, open Qdrant, or call OpenAI.

```python
report = run_mcp_startup_preflight(settings)
if not report.passed:
    raise RuntimeError("MCP startup preflight failed: ...")
```

The preflight checks only whether `CONTROL_PLANE_API_KEY` is present in the child
environment; it never reads, logs, traces, or returns the value. Presence fails
closed, proving the tunnel credential was improperly inherited.

Production interpretation: a descriptive runtime manifest cannot inject or prove
child environment. The child must validate its actual configuration before it
advertises tools. This prevents an MCP process from starting with a mismatched
interface mode, stale lexical directory, invalid retrieval policy, or leaked
tunnel-control credential.

Failure-mode testing: tests reject `INTERFACE_MODE=fastapi`, unsupported retrieval
strategy, absent artifact directory, and inherited control-plane credential; the
safe error contains no secret value.

## Step 246 — Record tunnel-client process ownership

Added `scripts/run_mcp_stdio.ps1` and updated
`deployment/native_runtime.json`. The tunnel client is the process owner; it
launches the PowerShell wrapper through `--mcp-command`. The wrapper removes the
parent-only control-plane key before starting the project virtual-environment
Python MCP child.

```powershell
Remove-Item -LiteralPath "Env:CONTROL_PLANE_API_KEY" -ErrorAction SilentlyContinue
& $pythonPath -m app.mcp.server
```

The runtime manifest records the MCP command, working directory, allowed interface
modes, non-secret application environment names, parent-only control-key rule, and
the fact that it describes—not injects—environment.

Production interpretation: in `both` mode there will be exactly three terminals:
FastAPI, Streamlit, and `tunnel-client run`. The latter owns the stdio child; no
separate manually started MCP server is used for tunnel operation.

Failure-mode testing: tests assert the native metadata omits
`CONTROL_PLANE_API_KEY` from application environment requirements and that the
launcher removes it before Python starts. The focused protocol/preflight/runtime
suite passed **17/17**; `uv lock --check` and `git diff --check` passed (only
pre-existing Windows line-ending advisories were emitted).

Gate status: **awaiting learner answers for Steps 244-246.** No tunnel was
created, no external API request occurred, and no internal evidence was disclosed.

### Learner evaluation — Steps 244-246

**Accepted, 9/9.** The learner clearly separated in-process logic testing from
wire-level protocol proof, explained whole-stream validation and independent
diagnostic paths, justified bounded local preflight, described credential
presence-only isolation, fail-closed tool registration, tunnel lifecycle
ownership, least-privilege key removal, and the limits of local tests versus live
ChatGPT/tunnel evidence. No remediation was needed.

## Step 247 — Publish the Phase 1 operator runbook

Created `docs/ChatGPT_Secure_MCP_Tunnel_Phase1.md` and added a concise command
section to `README.md`. The runbook covers generation verification, FastAPI/UI
startup, direct MCP Inspector testing, explicit disclosure enablement,
tunnel-client profile initialization/doctor/run, ChatGPT Developer Mode
connection, three-terminal `both` operation, costs, safe rollback, and concrete
troubleshooting.

```python
# The documented child process is the actual application entry point.
def main() -> None:
    settings = get_settings()
    configure_mcp_stdio_logging(settings.log_level)
    create_mcp_server(settings=settings).run(transport="stdio")
```

Production interpretation: documentation makes operational authority visible.
The approved operator must set `MCP_EVIDENCE_DISCLOSURE_ENABLED=true` before MCP
retrieval testing; this is deliberate data egress, not a hidden convenience
setting. Dense/hybrid queries can also incur query-embedding cost.

Failure-mode testing: the troubleshooting table provides fail-closed actions for
disclosure-disabled, missing environment/artifacts/Qdrant, embedding, tunnel,
stdout-corruption, and unexpected tool-metadata failures. It explicitly rejects a
fourth manually started MCP terminal in tunnel mode.

## Step 248 — Document process ownership, interface selection, and key boundary

The runbook and README now make the selected runtime topology executable:

```text
Terminal 1: FastAPI          (only fastapi/both)
Terminal 2: Streamlit        (only fastapi/both)
Terminal 3: tunnel-client run → owns MCP stdio child (mcp/both)
```

`scripts/run_mcp_stdio.ps1` is the documented tunnel command. It removes the
parent-only `CONTROL_PLANE_API_KEY` before virtual-environment Python starts;
the application validates only its absence, never its value. The real key is
injected only into Terminal 3 by the approved secret mechanism and is not stored
in `.env`, emitted in documentation examples, logs, traces, or tool output.

Production interpretation: interface mode and disclosure state are separate.
`INTERFACE_MODE` controls which local processes may start; disclosure controls
whether the already-authorized MCP transport may return internal evidence.
Neither flag enables code/combined retrieval without the established
`CODE_MODES_ENABLED` activation control.

Failure-mode testing: metadata/launcher tests verify parent-only key handling and
tunnel-client ownership. The startup preflight blocks a child with incompatible
configuration before it can advertise tools.

## Step 249 — Final offline verification

Added/updated MCP adapter, protocol, preflight, runtime-metadata, API, and
documentation coverage. A first full suite revealed two regressions: legacy API
test doubles did not define the new `interface_mode` field, and test-time MCP
factory construction globally reset logging before unrelated audit assertions.
Both were corrected: an absent legacy field defaults safely to `fastapi`, and
stderr logging is configured only in the actual MCP `main()` entry point.

```python
if getattr(settings, "interface_mode", "fastapi") == "mcp":
    raise RuntimeError("FastAPI is disabled when INTERFACE_MODE=mcp.")
```

Production interpretation: Phase 1 preserves existing FastAPI/Streamlit behavior
while adding a private, read-only MCP transport. The full suite validates code
compatibility and fail-closed behavior; it does not create a tunnel, disclose
source evidence, call OpenAI, or prove an external ChatGPT session.

Failure-mode testing and final checks:

```powershell
uv lock --check
uv run --locked pytest
git diff --check
```

The final clean suite passed **637 tests in 127.46 seconds**. `uv lock --check`
passed and `git diff --check` reported no whitespace errors (only pre-existing
Windows line-ending advisories). The temporary test-output files used to capture
the long suite were removed from `data/tmp` after verification.

Gate status: **offline implementation accepted.** No automated test made a live
OpenAI call, created a tunnel, or disclosed internal evidence. Live Secure MCP
Tunnel and ChatGPT validation remains a separate, operator-authorized exercise.

### Learner answers and mentor evaluation

**Accepted, 9/9.** The learner correctly distinguished the three independent
controls, immutable generation activation, Inspector versus tunnel-client
ownership, the parent-only control-plane key, transport health versus retrieval
quality, safe default interface behavior, logging scope, and the limits of the
637-test offline result. Phase 1 implementation is complete; the next work is
the documented manual tunnel setup and evidence-grounding validation.

## Step 250 — Controlled FastAPI dense retrieval probe

After the local Codex experiment showed that a shell-enabled client can launch a
separate child with caller-supplied environment variables, dense retrieval was
tested through the established FastAPI path rather than through Codex.

```http
POST /search
Content-Type: application/json

{"query":"What is the AML batch processing behavior?","mode":"fdd"}
```

The temporary localhost FastAPI process used `RETRIEVAL_MODE=dense`, FDD mode,
and code mode disabled. The operator explicitly authorized one paid query
embedding. The result was `HTTP 200`, `retrieval_mode=dense`, five FDD results,
and no answer-generation call. Only aggregate outcome metadata was recorded;
no source excerpt was copied into this progress log. The temporary process was
stopped after the request.

Production interpretation: dense/hybrid retrieval should be exercised through a
single controlled retrieval runtime. A local embedded Qdrant store is not a
multi-process service: competing MCP/FastAPI children can conflict on its lock.
An environment-variable disclosure switch is a useful gate for the intended MCP
child, but is not an enforceable egress boundary against a client that has local
shell and filesystem access to the repository.

Failure-mode testing: two stale MCP children were identified by command line and
terminated only after operator approval. The dense probe then completed without
a Qdrant lock. No hybrid probe was run or authorized.

## Steps 251-253 — Local MCP/Qdrant runtime hardening

### Step 251 — Safe embedded-Qdrant lock errors

`app/vectorstore/qdrant_schema.py` now recognizes the local embedded-Qdrant
storage-lock failure and raises a safe application error instead of passing a
filesystem path through the retrieval stack:

```python
try:
    return QdrantClient(path=str(storage_path))
except RuntimeError as exc:
    if _looks_like_local_storage_lock(exc):
        raise LocalQdrantLockError(
            "Local Qdrant storage is in use by another process."
        ) from exc
    raise
```

Explanation: the underlying client can report the local storage location in a
lock exception. That detail is useful locally but should not be returned to an
MCP/Chat client.

Production interpretation: this is failure containment, not multi-process
support. Embedded Qdrant remains an exclusive local-store runtime.

Failure-mode test: a mocked Qdrant lock with an internal path raises only the
generic `LocalQdrantLockError`; `tests/test_qdrant_schema.py` proves the path is
not exposed.

### Step 252 — Single MCP-child launcher guard

`scripts/run_mcp_stdio.ps1` now acquires a Windows named mutex before launching
the stdio server:

```powershell
$mutex = New-Object System.Threading.Mutex($false, "Local\CullingBladeLineageMcpStdio")
if (-not $mutex.WaitOne(0, $false)) {
    $mutex.Dispose()
    throw "Culling Blade MCP server is already running. Stop the existing MCP child before starting another."
}
```

Explanation: this blocks a duplicate local MCP child before both processes try
to open the same embedded Qdrant files. The existing parent-only control-plane
key removal remains in the launcher.

Production interpretation: the mutex protects duplicate **MCP** children only.
It does not permit simultaneous FastAPI and MCP dense/hybrid access to embedded
Qdrant. That topology requires a separately selected shared Qdrant server.

Failure-mode test: `tests/test_mcp_stdio_launcher.py` verifies key stripping,
the mutex, non-blocking acquisition, the safe duplicate message, and the
expected MCP module command. PowerShell parsing was also checked without
starting a server.

### Step 253 — Explicit local-runtime boundary

`docs/ChatGPT_Secure_MCP_Tunnel_Phase1.md` now documents the actual laptop
contract:

```text
lexical MCP search: no application embedding API call
dense/hybrid MCP search: application query embedding; API usage/cost possible
embedded Qdrant: one process at a time for a given local store
```

Explanation: ChatGPT can generate the final answer from MCP evidence, but that
does not remove the application embedding cost when dense or hybrid retrieval
is configured. The local direct-stdio test is valid for one client process; it
is not the server topology for concurrent interfaces.

Production interpretation: for the current personal laptop testing path, run
one local MCP child and select lexical retrieval when zero application embedding
cost is required. Before concurrent FastAPI + MCP dense/hybrid use, choose and
validate a shared Qdrant server deployment, credentials, health checks, and
rollback plan.

Verification: focused hardening tests passed **9/9**. `uv lock --check` passed.
The complete suite was executed with the new tests collected (640 total); no
test failure was reported by the runner. `git diff --check` reported no
whitespace failure. No OpenAI call, tunnel, or evidence disclosure occurred in
these hardening steps.

### Learner answers and mentor evaluation

**Accepted, 9/9.** The learner correctly distinguished information containment
from concurrency support, identified the pre-start race prevented by the mutex,
explained the lexical versus dense/hybrid cost boundary, and described the
shared-Qdrant server, access-control, network, lifecycle, and monitoring work
required before concurrent interface serving. They also correctly stated that
direct stdio testing is not proof of tunnel, hostile-client, or service-isolated
security behavior.
