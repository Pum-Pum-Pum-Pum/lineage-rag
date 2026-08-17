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
are matched case-insensitively: `.sql`, `.spc`, `.prc`, `.fnc`, and `.ddl`.

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
CODE_PARSE_MAX_SEGMENT_CHARACTERS=500
CODE_RETRIEVAL_MAX_UNIT_CHARACTERS=6000
CODE_RETRIEVAL_OVERLAP_CHARACTERS=400
CODE_ANALYSIS_POLICY_PATH=config/code_analysis.toml
```

The command verifies the immutable snapshot before starting and atomically
publishes a new no-overwrite generation under:

```text
data/staging/code/<snapshot-id>/plsql_antlr_4_13_2_analysis_v9/
|-- parse_stage_manifest.json
|-- analysis/
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
chunks contain only original source and remain visibly degraded. DDL source is
retained both as exact source units and as separate structured schema evidence.

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

## 10. Interpret overload-safe symbols

Every procedure and function occurrence records exact source identity plus:

```text
language + module_id + canonical_qualified_name
+ symbol_kind + overload_discriminator_hash
```

Unquoted Oracle identifiers canonicalize to uppercase. Quoted identifiers keep
their quotes and exact case, so unquoted `FOO` never merges with quoted
`"FOO"`. Nested local routines include their parent routine scope.

The overload discriminator includes ordered parameter names, quoted-name state,
canonical declared types, and type families. It intentionally excludes modes,
defaults, and function return type because those cannot safely create a new
Oracle overload. The separate declaration signature hash retains modes,
`NOCOPY`, defaults, return type, and conditional state for change detection.

All source occurrences remain stored. A mode-only or return-type-only collision
produces `overload_symbol_collision`; it never becomes a last-file-wins entry.
A package declaration and implementation can share a key. Default expressions
may legitimately differ between their source forms, while incompatible modes,
types, or return types fail the analysis gate.

Production interpretation: symbol keys are stable lookup identities, while
occurrence IDs preserve exact declaration/implementation citations. They do
not yet prove that a call resolves to one overload when argument types are not
statically known.

## 11. Interpret dependencies and unavailable boundaries

Static-analysis artifacts contain calls, table reads/writes, package type,
constant, global, cursor references, dynamic SQL, external packages, and
configured kernel boundaries. Resolution states distinguish:

```text
resolved_in_snapshot
ambiguous
unresolved
dynamic_unknown
kernel_unavailable
external_schema
```

The versioned policy is `config/code_analysis.toml`. Its normalized SHA-256 is
stored in every analysis artifact and stage manifest. Kernel prefixes are empty
by default because the system must not guess which customer packages are hidden
kernel code. Add only SME-reviewed prefixes before processing the real corpus.
Oracle external-package prefixes and ignored built-in calls are also configured
there rather than hidden in middleware or retrieval code.

Call resolution retains every plausible overload candidate. Dynamic SQL is
detected but its runtime objects are never invented. Token-aware extraction
handles direct table references, joins, comma joins, and cursor queries; complex
runtime name construction and behavior behind external packages remain explicit
unknowns.

Production interpretation: these edges support later impact analysis, not a
claim that a suspected dependency is a proven runtime path or root cause.

## 12. Interpret DDL and synonyms

Full parses create structural units for tables, columns, defaults, constraints,
views, sequences, indexes, object types, collection types, and synonyms. A
degraded parse emits no schema claim; original source remains available for
review.

Synonyms are resolved across the complete approved snapshot, not independently
per file. States are:

```text
resolved_in_snapshot
external_schema
database_link
ambiguous
cyclic
```

Only a unique target definition present in the snapshot becomes
`resolved_in_snapshot`. Missing qualified targets remain `external_schema`,
database links are never followed, and cycles or duplicate identities fail
closed. Phase 3 Oracle metadata is still required to confirm live schemas,
synonyms, editions, and database-link targets.

The expanded contract uses the no-overwrite generation
`plsql_antlr_4_13_2_analysis_v9`. The recovery path gives a
full-file parser that reaches its resource boundary one separate, bounded,
token-aware segmented attempt before using original-source fallback chunks.
The full and segmented attempts each use the configured timeout and memory
boundary, so the maximum per-file parse time may approach twice the configured
timeout. Within segmented parsing, routines above
`CODE_PARSE_MAX_SEGMENT_CHARACTERS` retain lexer-proven names, exact source
ranges, and `token_structural` confidence without making an unbounded ANTLR
call. They remain explicitly degraded and require review before indexing.
Smaller routines continue through ANTLR. Older parser generations remain
immutable and isolated.

Oversized exact-source retrieval units are divided independently of parser
segmentation. Each child is bounded by
`CODE_RETRIEVAL_MAX_UNIT_CHARACTERS`, includes at most the configured overlap,
and records its parent unit ID, parent source range, chunk index/count, and
exact child line/offset range. The maximum applies to embedding
`retrieval_text`, including its derived header, not only to citeable source
text. Child IDs are deterministic over parent identity, index, and offsets.
Static-analysis errors are published for diagnosis but set stage status to
`failed`, preventing later indexing.

Run the Steps 156-158 focused tests with:

```powershell
uv run --locked pytest `
  tests/test_code_analysis_policy.py `
  tests/test_plsql_symbol_analysis.py `
  tests/test_plsql_dependency_analysis.py `
  tests/test_ddl_analysis.py `
  tests/test_code_static_analysis.py `
  tests/test_code_parsing_pipeline.py `
  tests/test_parse_code_snapshot_script.py -q
```

These tests use synthetic fixtures. Real packages do not need to be placed in
`data/raw_code/` until this interview gate is accepted and the curated snapshot
is ready for parser-coverage review. No OpenAI or Qdrant operation occurs here.

## 13. Prepare the isolated code index contract

### Custom program-unit boundary convention

The configured application convention is stored in `config/code_analysis.toml`:

```toml
custom_program_unit_suffixes = ["_CUSTOM", "_MAIN"]
infer_noncustom_qualified_packages_as_kernel = false
kernel_package_names = []
kernel_package_prefixes = []
```

For PL/SQL intake, the declared top-level package, standalone function, or
standalone procedure is authoritative. Its name must match the filename stem
and end `_CUSTOM` or `_MAIN`, case-insensitively. Every member routine inside
an accepted package is available custom source regardless of the member name.
`.ddl` schema sources are exempt from program-unit suffix validation.

Resolved uploaded symbols take precedence. An absent target whose package or
standalone unit ends `_CUSTOM`/`_MAIN` is `custom_source_missing`. A target is
`kernel_unavailable` only when its owning package matches an approved exact
kernel package name or prefix. Blanket non-suffix inference is disabled because
record fields, table aliases, and `SCHEMA.FUNCTION` syntax can resemble package
calls. Unqualified uncertainty remains unresolved.

This convention never filters tables or views. All statically visible table
reads/writes remain indexed whether or not their names end `_CUSTOM`.

Changing this policy changes `analysis_policy_sha256`. Never promote an older
analysis artifact under the new policy. Build a new immutable analysis and
code-index generation, then repeat the deterministic gates.

Run the local real-corpus gate and export the focused SME packet:

```powershell
& .\.venv\Scripts\python.exe scripts\check_code_preindex_gate.py `
  <snapshot-id> `
  --snapshot-root <verified-snapshot-root> `
  --generation plsql_antlr_4_13_2_analysis_v9 `
  --output data\exports\code_analysis\<snapshot-id>-analysis-v9-preindex-gate.json

& .\.venv\Scripts\python.exe scripts\export_code_dependency_review.py `
  <snapshot-id> `
  --snapshot-root <verified-snapshot-root> `
  --generation plsql_antlr_4_13_2_analysis_v9
```

The packet groups repeated occurrences by target, proposed kind, resolution
state, and confidence. It includes a small source excerpt with immutable path,
line, and source-hash provenance. Tables are not included merely because their
definitions are absent; the review is focused on unresolved/ambiguous routine
calls, inferred kernel boundaries, and dynamic SQL. Packet publication is
no-overwrite and performs no external call.

After the real-corpus parser gate passes, prepare deterministic code indexing
records without calling OpenAI:

```powershell
& .\.venv\Scripts\python.exe scripts\prepare_code_index_artifacts.py `
  fci-custom-r1-a47f5d4d54e1 `
  --parse-generation plsql_antlr_4_13_2_analysis_v9
```

The output is isolated beneath:

```text
data/staging/code_indexes/<snapshot-id>/code_index_contract_v4/
```

Each record preserves snapshot, module, file, routine/chunk, line/offset,
parent, parser, conditional, citation-text, and embedding-text identity. Cache
keys use content, embedding model, and `code_embedding_input_v1`. Qdrant point
IDs use snapshot plus source-occurrence unit identity, so duplicate semantic
text can reuse a vector without collapsing citations.

Run local lexical search with:

```powershell
& .\.venv\Scripts\python.exe scripts\query_code_lexical.py `
  data\staging\code_indexes\fci-custom-r1-a47f5d4d54e1\code_index_contract_v4\code_index_artifact.json `
  "spPNBRPT006 branch report"
```

Verify the prepared contract by rebuilding it from the immutable stage:

```powershell
& .\.venv\Scripts\python.exe scripts\verify_prepared_code_index.py `
  data\staging\code_indexes\<snapshot-id>\code_index_contract_v4\code_index_artifact.json
```

Verification checks the current approved policy hash and exact equality of the
entire deterministic prepared artifact. It does not mark dependency review as
complete and does not authorize embedding.

## 14. Paid code-embedding boundary

Dry-run first:

```powershell
& .\.venv\Scripts\python.exe scripts\embed_code_index_artifacts.py `
  <prepared-code-index-artifact> `
  --output-root data\staging\code_embeddings `
  --dry-run
```

The dry run reports exact records and unique embedding inputs but sends
nothing. A real run is fail-closed unless both conditions hold:

1. the index contract records `dependency_review_status="reviewed"` following
   SME review of representative dependency labels;
2. the operator supplies the exact disclosure/cost authorization token printed
   by `--help` and documented in the controlled run procedure.

Internal code excerpts are sent to OpenAI during a real embedding run. General
permission to implement Phase 2 does not authorize that disclosure. Never put
the authorization token into `.env` or source control.

## 15. Index and verify an isolated code collection

Only a complete embedded artifact may be indexed. Choose a new collection name
with the required `code_custom_` prefix:

```powershell
& .\.venv\Scripts\python.exe scripts\index_code_qdrant.py `
  <embedded-code-index-artifact> `
  --qdrant-path data\qdrant_code_local `
  --collection-name code_custom_r1_v1

& .\.venv\Scripts\python.exe scripts\verify_code_qdrant.py `
  <embedded-code-index-artifact> `
  --qdrant-path data\qdrant_code_local `
  --collection-name code_custom_r1_v1
```

Indexing refuses an existing collection and never modifies the FDD collection.
Verification requires exact collection count, every deterministic point ID,
every provenance payload field, and vector dimension. A failed new generation
may remain for investigation but cannot become active. The previous collection
and lexical artifact remain unchanged for rollback. Activation is deliberately
deferred until Steps 162-170 retrieval, citation, answer, and SME gates pass.
