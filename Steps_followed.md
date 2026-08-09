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
