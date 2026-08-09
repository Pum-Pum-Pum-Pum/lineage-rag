# Custom PL/SQL snapshot ingestion

This runbook covers the Phase 2 snapshot boundary only. It validates and
archives selected custom PL/SQL/DDL files and compares complete snapshots. It
does **not** parse code, call OpenAI, create embeddings, or write to Qdrant.

## 1. Prepare one complete curated snapshot

Create a uniquely named intake directory:

```text
data/raw_code/fci-custom-r12345/
|-- snapshot_request.json
`-- source/
    |-- packages/
    |   |-- pkg_customer.prc
    |   `-- fn_validate_customer.fnc
    `-- ddl/
        `-- customer_tables.ddl
```

The `source/` directory must contain the complete selected custom module set
for that revision—not only files believed to have changed. Initial extensions
are matched case-insensitively: `.sql`, `.prc`, `.fnc`, and `.ddl`.

Use this request contract:

```json
{
  "schema_version": "code_snapshot_request_v1",
  "module_set": "fci-custom",
  "svn_revision": "12345",
  "application_build": "14.7.1",
  "reviewer": "SME name",
  "base_snapshot_id": null,
  "expected_changed_packages": [],
  "compiler_context": {
    "oracle_version": null,
    "plsql_ccflags": null
  }
}
```

For a later snapshot, set `base_snapshot_id` to the exact prior snapshot ID.
`expected_changed_packages` is an optional review assertion. Enter relative
paths or filenames; do not enter line numbers or manually edited diffs.

## 2. Validate without writing

Run with the project interpreter:

```powershell
uv run --locked python scripts/build_code_snapshot.py `
  data/raw_code/fci-custom-r12345 `
  --validate-only
```

Validation streams file content and checks:

- the request schema and safe relative paths;
- the case-insensitive extension allowlist;
- binary/control-byte content;
- UTF-8, BOM-marked UTF-16, or Windows-1252 decoding;
- potential secret categories without printing their values;
- exact SHA-256 and normalized-text SHA-256;
- case-insensitive path collisions and symlinks.

A file above 5 MiB produces `large_file`, but remains valid. The warning tells
later parser stages to isolate resource use; it is not a rejection rule.

Expected successful output includes:

```json
{
  "status": "valid",
  "file_count": 3,
  "writes_performed": false,
  "external_calls_performed": false
}
```

Production interpretation: this proves the selected files are safe enough to
enter the local snapshot archive. It does not prove that PL/SQL parses, that
dependencies resolve, or that the code is correct.

## 3. Publish the immutable local snapshot

```powershell
uv run --locked python scripts/build_code_snapshot.py `
  data/raw_code/fci-custom-r12345
```

The default archive is:

```text
data/code_snapshots/<module-set>-r<revision>-<content-hash-prefix>/
|-- snapshot_manifest.json
`-- source/
```

Publication copies into a temporary directory, verifies copied hashes, and
atomically promotes the directory. Existing snapshot IDs are never
overwritten. `data/raw_code/` and `data/code_snapshots/` are excluded from Git
because they can contain proprietary source and generated local state.

The manifest records exact and normalized hashes, encoding, line count, size,
warnings, source identity, and the comparison with the requested base.

## 4. Interpret the diff

The complete current manifest is compared with the complete base manifest:

- same path and exact hash: `unchanged`;
- same path and different exact hash: `modified`;
- current-only path: `added`;
- base-only path: `deleted`;
- exactly one deleted and one added file with the same exact hash:
  `exact_renames`;
- multiple possible same-hash rename pairs: left added/deleted and reported as
  `ambiguous_rename_hashes`.

Line-ending or BOM-only changes remain `modified` because source bytes changed,
but are also identified in `formatting_only_modified` when normalized text is
identical.

`missing_expected_changes` means a reviewer named a package that did not
change. `unexpected_changed_files` means the deterministic comparison found a
change outside a non-empty expected list. These are review signals, not a
substitute for the manifest.

## 5. Failure and recovery rules

- Invalid request, unsupported file, binary content, a suspected secret, or a
  missing base snapshot fails before publication.
- An intake mutation between validation and copying fails copied-hash
  verification and does not publish a partial snapshot.
- Manual changes inside an archived snapshot make later integrity verification
  fail.
- A failed intake stays in `data/raw_code/` for correction and rerun.
- Never edit an archived snapshot. Correct the intake and publish a new
  content-addressed snapshot.
- No OpenAI or Qdrant cost is incurred in these steps.

Run the focused deterministic tests with:

```powershell
uv run --locked pytest `
  tests/test_code_snapshot_models.py `
  tests/test_code_intake_validation.py `
  tests/test_code_snapshot_builder.py `
  tests/test_build_code_snapshot_script.py -q
```

