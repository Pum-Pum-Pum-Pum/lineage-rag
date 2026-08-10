# Custom PL/SQL snapshot ingestion

This runbook covers the Phase 2 snapshot and structural-parsing boundaries. It
validates and archives selected custom PL/SQL/DDL files, compares complete
snapshots, then creates local parser and retrieval-unit artifacts. These steps
do **not** call OpenAI, create embeddings, or write to Qdrant.

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

The versioned policy is `config/ingestion_sources.toml`. To enable another
extension already handled as PL/SQL, add a reviewed mapping such as:

```toml
".pkb" = "plsql"
```

This requires no Python change. A new handler type still requires an
implementation; for example, configuring `".pdf" = "pdf"` fails closed until
a PDF handler exists. The normalized policy SHA-256 is stored in every code
snapshot so changes to source interpretation remain auditable.

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
- the configured extension-to-handler mapping and supported-handler ceiling;
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
warnings, source handler, ingestion-policy identity, source identity, and the
comparison with the requested base. A policy change from the base snapshot is
reported even when all source bytes remain unchanged.

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

## 6. Parse one verified immutable snapshot

Use the exact `snapshot_id` printed by Step 3:

```powershell
uv run --locked python scripts/parse_code_snapshot.py `
  fci-custom-r12345-<manifest-hash-prefix>
```

Defaults are centrally configurable without changing parser code:

```text
CODE_SNAPSHOTS_DIR=data/code_snapshots
CODE_STAGING_DIR=data/staging/code
CODE_PARSE_TIMEOUT_SECONDS=120
CODE_PARSE_MEMORY_LIMIT_MIB=1024
```

The command verifies the immutable snapshot before starting and atomically
publishes a new no-overwrite generation under:

```text
data/staging/code/<snapshot-id>/plsql_antlr_4_13_2_v1/
|-- parse_stage_manifest.json
|-- parse/
`-- retrieval/
```

Each file runs in an isolated Python worker. The parent enforces wall-clock
and resident-memory boundaries. Files above 5 MiB remain eligible and do not
receive a separate rejection rule.

Parser states mean:

- `full_parse`: the complete file parsed with the pinned grammar;
- `segmented_parse`: the full file failed, but token-aware routine segments
  parsed with exact maps to the original file;
- `fallback_parse`: bounded original-source chunks were retained because no
  structural parse was trustworthy or a resource boundary was reached;
- `failed`: source identity or another safety boundary prevented even a safe
  fallback.

`complete_with_degradation` is not a clean parser pass. Review its diagnostics
before later indexing. A `failed` stage must not proceed.

## 7. Conditional-compilation interpretation

The grammars-v4 PL/SQL grammar and ANTLR Python runtime are pinned to recorded
versions and hashes. The parser always tries immutable original source first.
If grammar handling of `$IF`, `$THEN`, `$ELSIF`, `$ELSE`, `$END`, `$ERROR`, or
`$$...` is incomplete, it may use a separate line-preserving parse view only
when that produces fewer syntax errors.

The citation source is never edited. Conditional regions retain exact source
ranges and are marked `active`, `inactive`, `unresolved`, or
`conditional_unknown`. When `oracle_version` or `plsql_ccflags` are absent,
the system must not claim which branch is deployed.

## 8. Retrieval-unit interpretation

Routine and package-declaration units retain exact original text, path, line
range, snapshot identity, parser state, and conditional state. A routine gets
only referenced package types, constants, globals, and cursors through
`related_unit_ids`; the entire package header is not copied into every unit.

`retrieval_text` contains a compact header explicitly marked:

```text
DERIVED RETRIEVAL CONTEXT - NOT A CITATION SOURCE
```

That header may improve later retrieval but must never be cited as source code.
Code citations must use each unit's exact `text` and `source_map`. Fallback
chunks contain only original source and remain visibly degraded. Successfully
parsed constructs whose structural extractor is intentionally deferred (for
example DDL until Step 158) are retained as exact `source_chunk` units rather
than disappearing from the artifact.

## 9. Parse-stage failure and recovery

- A tampered snapshot fails before a stage directory is published.
- Timeout or memory exhaustion produces bounded fallback units when the exact
  source hash still matches; it never silently drops the file.
- A source change during parsing fails closed instead of producing a fallback
  for unverified bytes.
- An existing parse generation is never overwritten.
- Partial temporary directories are removed on failure; the final namespace is
  exposed only after all artifacts and the stage manifest are written.
- The command is local and incurs no OpenAI or Qdrant cost.

Run the Steps 153-155 focused tests with:

```powershell
uv run --locked pytest `
  tests/test_plsql_grammar_toolchain.py `
  tests/test_conditional_compilation.py `
  tests/test_plsql_parser_core.py `
  tests/test_plsql_isolation.py `
  tests/test_code_retrieval_artifact.py `
  tests/test_code_parsing_pipeline.py `
  tests/test_parse_code_snapshot_script.py -q
```
