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
