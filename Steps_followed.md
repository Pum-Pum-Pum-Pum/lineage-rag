# Steps Followed

## Step 99 - Secure request correlation and privacy-safe API auditing

### Goal
Create one correlation path across HTTP responses, structured audit events, and
persisted answer traces without logging request content, credentials, concrete
conversation identifiers, or internal errors.

### Files touched
- `app/core/request_observability.py`
- `app/api/main.py`
- `app/api/routes/query.py`
- `app/services/answer_orchestration.py`
- `app/services/answer_trace.py`
- `tests/test_request_observability.py`
- `tests/test_query_api.py`
- `tests/test_answer_trace.py`
- `README.md`
- `docs/project_plan.md`
- `Steps_followed.md`
- `interview-questions.md`

### What changed
- Added an HTTP middleware that accepts only bounded, injection-safe request IDs
  and generates a UUID when the supplied value is absent or invalid.
- Added one structured JSON completion event with method, route template,
  status, and elapsed milliseconds.
- Added defensive no-store, MIME-sniffing, framing, referrer, and browser
  permissions response headers.
- Passed the API request ID to answer traces as `correlation_id`.
- Kept correlation IDs separate from unique trace IDs and filenames so a
  repeated or attacker-controlled request ID cannot overwrite trace artifacts.

### Python/code pattern used
```python
request_id = _resolve_request_id(request.headers.get("X-Request-ID"))
token = request_context.set(request_id)
response = await call_next(request)
response.headers["X-Request-ID"] = request_id
audit_logger.info(json.dumps(safe_request_metadata))
```

### Why it is implemented this way
Correlation is useful only when the same safe identifier crosses API, logs, and
traces. A client-supplied identifier is untrusted input, so it is validated
before becoming a header or log value. Audit events use route templates such as
`/conversations/{conversation_id}` rather than resource IDs and exclude bodies,
queries, titles, credentials, IP addresses, and exception details.

The trace ID remains server-generated and unique. Treating a caller's request
ID as the trace filename would allow collisions and overwrite prior artifacts.

### Production interpretation
An operator can join an API response to its safe request event and answer trace
without searching user content. Status and latency provide a minimal base for
error-rate and latency aggregation. Security headers reduce caching and common
browser misuse but do not establish user identity or access control.

### Validation
- Focused request/trace/query/conversation suite: 25 passed.
- Full regression suite: 304 passed.
- Existing non-failing upstream Starlette `TestClient`/HTTPX deprecation
  warning remains.

### Failure-mode thinking
Tests inject newline/log-forging content and a secret-like token into
`X-Request-ID`; the value is replaced before logging. They verify bodies,
private titles, concrete conversation IDs, and unsafe IDs do not enter audit
events. They also prove correlation and trace identifiers remain different.

This step does not claim authentication, authorization, rate limiting, TLS,
centralized retention, tamper evidence, or Oracle deployment packaging. Those
depend on the selected production identity and runtime boundary.

## Step 100 - Deterministic native deployment bundle and preflight

### Goal
Package the application for an Oracle-approved native runtime without Docker,
without embedding secrets or mutable business data, and without assuming an OS
service manager before the target environment is confirmed.

### Files touched
- `app/deployment/native_package.py`
- `app/deployment/preflight.py`
- `app/deployment/__init__.py`
- `deployment/native_runtime.json`
- `scripts/build_native_deployment.py`
- `scripts/check_deployment_preflight.py`
- `tests/test_native_deployment_package.py`
- `tests/test_deployment_preflight.py`
- `README.md`
- `docs/project_plan.md`
- `Steps_followed.md`
- `interview-questions.md`

### What changed
- Added a deterministic ZIP builder with sorted entries, fixed timestamps and
  permissions, and SHA-256 hashes for every included file.
- Allowlisted runtime source, scripts, lock/configuration templates, and the
  native process contract.
- Excluded secrets, certificates, virtual environments, raw documents,
  processed artifacts, Qdrant state, SQLite history, traces, logs, tests, and
  generated exports.
- Added an offline deployment preflight for Python 3.12, lockfiles,
  environment labeling, model configuration presence, mode-specific retrieval
  state, and writable conversation/trace paths.
- Added an explicit native runtime contract for locked installation, preflight,
  FastAPI, and Streamlit processes while leaving supervision target-specific.

### Python/code pattern used
```python
result = build_native_package(
    project_root=ROOT_DIR,
    output_path="data/exports/deployment/lineage-rag-native.zip",
)
report = run_deployment_preflight(
    settings,
    project_root=ROOT_DIR,
)
```

### Why it is implemented this way
Source and immutable dependency metadata have a different lifecycle from
indexes, documents, conversations, and traces. Bundling mutable state would
make deployments large, leak data, and blur rollback/backup ownership.

Deterministic archives let release engineering compare hashes and reproduce
the exact bundle from the same source. The caller must still verify provenance
and sign or attest the artifact in the approved release pipeline.

Preflight checks local prerequisites without spending tokens or depending on
network availability. Service-manager configuration is deferred because
systemd, Windows Service, and Oracle-managed supervisors have materially
different identity, restart, logging, and secret-injection contracts.

### Production interpretation
The bundle can be installed with `uv sync --locked --no-dev`, validated before
traffic, and launched using the recorded process commands. Mutable retrieval
and conversation state must be mounted/provisioned separately and protected by
environment-specific backup, permission, and recovery controls.

### Validation
- Focused native deployment suite: 11 passed.
- Full regression suite: 308 passed.
- Two consecutive real builds produced the identical SHA-256:
  `b032289e194d36259154b199ffe066bdd09f557ed274ac89be2ccb8b857684d6`.
- Final archive contains 97 files.
- Local offline preflight passed with `--allow-development`.
- The lockfile was not modified. A separate `uv lock --check` could not run in
  this shell because the `uv` executable/module is not installed on its PATH;
  CI already enforces the committed lock with pinned `uv`.
- Existing non-failing upstream Starlette `TestClient`/HTTPX warning remains.

### Failure-mode thinking
Tests prove that `.env` and data files do not enter the archive, builds are
byte-for-byte deterministic, and missing Python/lock/model/retrieval/writable
requirements fail with actionable messages that do not expose secret values.
The `--allow-development` result explicitly states that it is local validation,
not production safety.

The bundle is not a signed supply-chain artifact and does not configure TLS,
authentication, process restart, resource limits, centralized logging,
backups, or zero-downtime rollout. Those controls belong to the confirmed
deployment platform and release pipeline.

## Step 101 - Tamper-evident local audit journal and verification boundary

### Goal
Persist the existing privacy-safe API request events in a durable integrity
chain that can be verified locally and exported to a future approved central
audit platform, without claiming that local mutable storage is immutable.

### Files touched
- `app/core/audit_journal.py`
- `app/core/config.py`
- `app/core/request_observability.py`
- `app/api/main.py`
- `app/deployment/preflight.py`
- `scripts/verify_audit_journal.py`
- `tests/test_audit_journal.py`
- `tests/test_request_observability.py`
- `tests/test_deployment_preflight.py`
- `.env.example`
- `deployment/native_runtime.json`
- `README.md`
- `docs/project_plan.md`
- `Steps_followed.md`
- `interview-questions.md`

### Python/code pattern used
```python
unsigned = {
    "sequence": sequence,
    "event": asdict(safe_event),
    "previous_hmac": previous_hmac,
}
record_hmac = hmac.new(
    secret_key,
    canonical_json(unsigned),
    hashlib.sha256,
).hexdigest()
journal.write(canonical_json({**unsigned, "hmac_sha256": record_hmac}))
journal.flush()
os.fsync(journal.fileno())
```

### What the code does
- Persists only the Step 99 fixed safe event schema, never request bodies,
  prompts, titles, credentials, concrete resource IDs, IPs, or raw errors.
- Chains each canonical JSONL record to the prior HMAC using a secret containing
  at least 32 UTF-8 bytes.
- Verifies schema, contiguous sequence, chain links, and every record HMAC on
  startup or through an offline CLI.
- Accepts an externally trusted final HMAC and count so verification can detect
  deletion of an otherwise valid suffix.
- Flushes and `fsync`s each record for a durability-first audit policy.
- Rejects a file changed by another writer rather than silently forking the
  chain.
- Keeps API responses available on journal write failure and emits a minimal
  critical event that must be alerted.
- Requires enabled, strong, writable audit configuration in production
  preflight while allowing explicit local development validation without it.

### Why it is implemented this way
A plain SHA-256 chain can be recomputed by anyone who can edit the journal.
HMAC makes undetected rewriting depend on access to a separately held secret.
Canonical JSON makes signing and verification deterministic. Sequence and
previous-HMAC fields expose insertion, edits, and reordering.

`fsync` favors audit durability over peak throughput. Fail-open request behavior
avoids turning a log-volume failure into a total RAG outage, but it creates an
audit gap; therefore the critical failure event is an operational page, not a
harmless warning.

### Production interpretation
The final HMAC and record count are checkpoint material, not business evidence.
They should be shipped to a separately controlled platform along with the
journal. The HMAC key belongs in the approved secret store with access control
and rotation procedures. Current retrieval evidence and validated citations
remain the authority for functional answers; audit integrity does not make an
answer grounded.

The journal is a local integrity/export boundary, not centralized retention or
WORM storage. The present native API command uses one process. Multi-worker or
multi-host ordering should be delegated to the selected central audit service
rather than approximated with one shared local file.

### Failure-mode testing
- Modified event content invalidates the HMAC and blocks writer startup.
- Malformed UTF-8 returns a safe verification failure without echoing bytes.
- Valid suffix deletion passes internal chain validation but fails when checked
  against an externally trusted final count/HMAC.
- A stale second writer is rejected instead of creating a silent fork.
- An attacker-controlled unmatched URL path is recorded as `<unmatched>`.
- A weak HMAC key fails production preflight without printing the key.
- Simulated journal I/O failure leaves the API response available, emits only a
  safe critical event, and does not expose the internal exception.

### Validation
- Focused audit/observability/deployment/package/docs suite: 20 passed.
- Final core audit/observability/deployment suite after verifier hardening:
  14 passed.
- A first full run exposed 12 health/readiness failures because older settings
  doubles did not define the new optional flag. App creation now uses a
  backward-compatible disabled default; the affected suite then passed 25/25.
- Final full regression suite: 316 passed.
- One existing non-failing upstream Starlette `TestClient`/HTTPX deprecation
  warning remains.
- Two consecutive deployment builds were identical: 99 files, SHA-256
  `8891058ce848c1ed2c66e2a61064c39a2b317dec90c676311893f4f323f78439`.
- The bundle manifest includes `app/core/audit_journal.py` and
  `scripts/verify_audit_journal.py`.

## Step 102 - Measure local audit durability cost

### Goal
Measure the actual local cost of the Step 101 per-request HMAC, append, flush,
and `fsync` policy before deciding whether production should use synchronous
durability, grouped commits, or a central durable collector.

### Files touched
- `app/core/audit_benchmark.py`
- `scripts/benchmark_audit_journal.py`
- `tests/test_audit_benchmark.py`
- `README.md`
- `docs/project_plan.md`
- `Steps_followed.md`
- `interview-questions.md`

### Python/code pattern used
```python
started = perf_counter_ns()
journal.append(synthetic_event)
latencies_ms.append((perf_counter_ns() - started) / 1_000_000)

verification_started = perf_counter_ns()
verification = verify_audit_journal(journal_path, ephemeral_key)
verification_ms = (perf_counter_ns() - verification_started) / 1_000_000
```

### What the code does
- Generates only synthetic request IDs and the fixed `/benchmark` route.
- Creates a random ephemeral benchmark HMAC key in memory rather than reading or
  exposing the production key.
- Writes warm-up events, then measures individual durable append latency for a
  configurable number of events.
- Reports linearly interpolated p50, p95, p99, maximum latency, measured
  single-writer throughput, journal size, average bytes per record, and
  full-chain verification time.
- Verifies the generated HMAC chain before accepting the measurement.
- Removes the temporary journal and persists only an aggregate local JSON
  report under `data/exports/audit_benchmarks/`.

### Why it is implemented this way
Durability policy should be based on measured storage behavior and the
business's acceptable audit-loss window. Averages hide tail latency, so the
report emphasizes percentiles and maximum latency. Warm-up operations reduce
first-write noise. The temporary journal prevents synthetic events from
contaminating the real audit trail.

The benchmark does not disable `fsync` for comparison because the current
production candidate is the durability-first writer. Grouped commit would be a
separate implementation with a defined loss window and must not be simulated by
silently weakening the existing writer.

### Local result
For 200 measured events after 10 warm-up events on the current Windows
filesystem and Python 3.12.13:

- p50 append latency: `3.392300 ms`
- p95 append latency: `6.351130 ms`
- p99 append latency: `7.638621 ms`
- maximum append latency: `28.502300 ms`
- measured single-writer throughput: `256.203 events/second`
- average storage: `406.390 bytes/record`
- verification time for 210 records: `10.213200 ms`

Report:
`data/exports/audit_benchmarks/audit-journal-benchmark.json`

### Production interpretation
On this one local run, synchronous audit durability adds several milliseconds
to the request path and shows a materially higher maximum. That may be small
relative to LLM latency, but health checks, refusals, cached operations, and
future low-latency endpoints may feel the overhead more strongly.

The `256 events/second` figure is not API capacity. It is a single-writer local
journal result without concurrent requests, model calls, retrieval, central
shipping, antivirus variation, production disks, or repeated-trial confidence
intervals. Production selection still requires representative end-to-end load,
p95/p99 SLOs, event volume, storage-growth projections, and an explicit maximum
acceptable loss window.

### Failure-mode testing
- Empty or negative latency samples are rejected.
- Zero measured events and negative warm-up counts fail before creating work.
- The real benchmark verifies its chain before returning metrics.
- Temporary synthetic journals are removed after the run.
- The persisted report is checked not to contain HMAC key fields or query
  content.

### Validation
- Focused audit benchmark/journal suite: 13 passed.
- Full regression suite: 323 passed.
- Existing non-failing upstream Starlette `TestClient`/HTTPX warning remains.
- Two consecutive native bundle builds were identical: 101 files, SHA-256
  `5d8fd5c25585bb1db2a8e34b0e20e69b75e494b47bf4d90d5bd6a8931fbc31c8`.
- The bundle includes `app/core/audit_benchmark.py` and
  `scripts/benchmark_audit_journal.py`.

## Step 103 - Extract a storage-neutral audit sink boundary

### Goal
Decouple FastAPI request auditing from the local JSONL implementation so the
storage target can later become a mounted/network filesystem, database, central
collector, or grouped-commit writer without changing request middleware or
copying privacy logic.

### Files touched
- `app/core/audit_sink.py`
- `app/core/config.py`
- `app/core/request_observability.py`
- `app/api/main.py`
- `app/deployment/preflight.py`
- `tests/test_audit_sink.py`
- `tests/test_deployment_preflight.py`
- `.env.example`
- `README.md`
- `docs/project_plan.md`
- `Steps_followed.md`
- `interview-questions.md`

### Python/code pattern used
```python
class AuditSink(Protocol):
    backend: str
    durability: Literal["durable_on_return", "accepted_not_durable"]

    def append(self, event: ApiAuditEvent) -> AuditAppendResult: ...

audit_sink = build_audit_sink(settings)
install_request_observability(app, audit_sink)
```

### What the code does
- Introduces an `AuditSink` protocol containing only the privacy-safe event
  append contract and explicit durability semantics.
- Wraps the existing `AuditJournal` in `HmacJsonlAuditSink` without weakening
  its HMAC, flush, `fsync`, verification, or failure behavior.
- Declares the current adapter `durable_on_return` and returns a checkpoint in
  a storage-neutral result object.
- Moves backend construction into one factory, leaving FastAPI unaware of file
  paths, JSONL serialization, HMAC keys, or future database clients.
- Adds `AUDIT_SINK_BACKEND=hmac_jsonl`; disabled audit still builds no sink.
- Rejects unsupported production backends before service startup/preflight.

### Why it is implemented this way
Changing the existing path already supports another local directory, mounted
volume, or syntactically valid network path. That does not make a network share
equivalent to a local durable disk: remote caching, acknowledgements, mount
options, disconnects, locking, and server failure change what `fsync` means.

A database adapter should own its transaction, schema, integrity, retry, and
idempotency behavior. A grouped adapter should own its queue, batch commit,
backpressure, shutdown, and loss-window contract. Keeping those policies behind
one protocol prevents storage-specific code from entering request middleware.

### Production interpretation
The boundary improves maintainability; it does not prove that every future
adapter is safe. `durable_on_return` means the current call has completed local
flush/`fsync` under the filesystem's contract. `accepted_not_durable` is
reserved for a future buffered adapter whose acknowledged events can still be
lost before batch commit.

The next grouped-commit experiment must compare performance and failure loss
against the Step 102 synchronous baseline. It must not silently change the
production default or claim remote durability without deployment-specific
evidence.

### Failure-mode testing
- Disabled auditing produces no sink and performs no storage initialization.
- The HMAC JSONL adapter persists and verifies a valid chain through the new
  boundary.
- Unknown backend configuration fails without echoing the attacker-controlled
  backend value.
- Production preflight rejects unsupported backends with a safe message.
- Existing middleware, audit failure, health, readiness, and deployment tests
  confirm the refactor preserves current behavior.

### Validation
- Focused sink/journal/observability/preflight/health/readiness suite: 30 passed.
- Full regression suite: 327 passed.
- One existing non-failing upstream Starlette `TestClient`/HTTPX deprecation
  warning remains.
- Two consecutive native bundles were identical: 102 files, SHA-256
  `3ffa4df318a23c164570bc65c41f0bc03bc5a427cc343c7452754ea6cfc07059`.
- The bundle contains `app/core/audit_sink.py`.

## Step 104 – Portable master FDD ingestion and verified archival

### Objective

Provide one portable command that processes all reviewed FDD DOCX files in
`data/raw_specs/` through the existing ingestion, embedding, Qdrant indexing,
and verification stages, then archives only verified source files.

### Files added or changed

- `scripts/master_ingestion_embedding_docs.py`
- `scripts/run_embedding_smoke_test.py`
- `scripts/check_qdrant_index.py`
- `app/embeddings/client.py`
- `app/core/config.py`
- `docs/Steps_for_FDD_Ingestion.md`
- `.env.example`
- `tests/test_master_ingestion_embedding_docs.py`
- `tests/test_qdrant_index_verification.py`
- `tests/test_embedding_client.py`

### Python/code pattern used

```python
commands = build_pipeline_commands(
    documents=documents,
    cache_directory=settings.cache_dir / "embeddings",
    request_batch_size=args.request_batch_size,
)
for command in commands:
    subprocess.run(command, cwd=ROOT_DIR, check=True)

verify_embedding_artifacts(
    client=client,
    collection_name=collection_name,
    artifact_paths=artifact_paths,
)
archive_documents(documents, settings.embedded_docs_dir)
```

### What the code does

- Adds the portable master command:
  `uv run --locked python scripts/master_ingestion_embedding_docs.py`.
- Reuses existing child scripts rather than duplicating DOCX extraction,
  OpenAI embedding, Qdrant indexing, or Qdrant inspection logic.
- Adds `--all-units` to the existing per-document embedding script and splits
  uncached retrieval units into bounded OpenAI embedding requests (64 by
  default, configurable with `--request-batch-size`).
- Extends Qdrant inspection to verify every deterministic point ID and its
  identifying payload metadata from the specific embedding artifacts.
- Adds `EMBEDDED_DOCS_DIR`, defaulting to `data/docs_embedded/`.
- Supports `--dry-run`, which lists selected documents and exact child commands
  without API, Qdrant, or file-move actions.
- Archives only after every child command, including exact Qdrant verification,
  has succeeded. Archive-destination conflicts stop the batch before child
  stages begin.

### Why it is implemented this way

A `.bat` file would be Windows-only and difficult to test; the Python command
works with the pinned project interpreter on Windows, Linux, CI, and future
cloud environments. The master contains orchestration only. Existing scripts
continue to own their individual stage behavior.

A collection-level count is not proof that a new FDD was indexed. Exact
deterministic IDs plus `unit_id`, release, document-family, content-hash, and
cache-key payload checks prevent an unsupported archival claim.

### Production interpretation

The master processes documents sequentially but bounds one embedding API call
by retrieval units, not by documents. Smaller request batches reduce the blast
radius of a transient API failure but increase requests, latency, and possibly
cost. Repeated runs reuse unchanged cached vectors; altered text, chunking,
artifact version, or embedding model requires re-embedding.

The workflow is not an all-or-nothing transaction across OpenAI, Qdrant, and
filesystem archival. A failure before archival leaves source DOCX files in
`data/raw_specs/`; Qdrant/cache artifacts may be partially written but are
safe to reconcile through deterministic IDs and exact verification before any
archive move. A filesystem failure during a multi-file archive can leave an
already-verified batch partly archived and requires operator reconciliation.

### Failure-mode testing

- A missing `data/raw_specs/` batch fails before external actions.
- Dry run invokes no child process and moves no source document.
- A simulated Qdrant-stage subprocess failure leaves the DOCX in
  `data/raw_specs/`.
- An existing archive destination blocks all stages and avoids overwriting a
  prior source document.
- A missing Qdrant point fails exact verification.
- Non-positive embedding API batch size is rejected.

### Validation

- Focused embedding/Qdrant/master tests: 18 passed.
- Actual `--dry-run` with the current empty `data/raw_specs/` directory failed
  safely before API, Qdrant, or archive actions.
- Full regression suite: 336 passed; one existing non-failing Starlette/HTTPX
  deprecation warning remains.
- No live OpenAI embedding request, Qdrant write, or source-document move was
  performed for this implementation step.

## Step 105 – Duplicate-content embedding safety and explicit Qdrant rebuild

### Objective

Correct the runtime failure found during the first full R21 ingestion: identical
chunk content created one embedding cache key but persisted different vectors,
which also exposed that Qdrant point IDs were not citeable-unit unique.

### Files added or changed

- `app/embeddings/client.py`
- `app/vectorstore/qdrant_upsert.py`
- `scripts/run_embedding_smoke_test.py`
- `scripts/run_qdrant_indexing.py`
- `scripts/master_ingestion_embedding_docs.py`
- `docs/Steps_for_FDD_Ingestion.md`
- `tests/test_embedding_client.py`
- `tests/test_embedding_artifact_quarantine.py`
- `tests/test_qdrant_upsert.py`
- `tests/test_qdrant_script_client_cleanup.py`
- `tests/test_master_ingestion_embedding_docs.py`

### Python/code pattern used

```python
grouped_records = _group_uncached_records_by_cache_key(uncached_records)
for _, representative, matching_records in unique_request_records:
    vector = embed(representative.text)
    for index, record in matching_records:
        updated_records[index] = replace(record, vector=vector)

point_id = uuid5(
    NAMESPACE_URL,
    json.dumps({"cache_key": record.cache_key, "unit_id": record.unit_id}),
)
```

### What the code does

- Deduplicates identical uncached content before calling the embedding API and
  copies the one resulting vector to every matching retrieval unit.
- Retains a content-based cache key for cost-efficient reuse, but builds each
  Qdrant point ID from both cache identity and `unit_id` so duplicate text in
  separate chunks/releases remains separately citeable.
- Keeps persisted cache-conflict detection strict; it does not silently choose
  between conflicting stored vectors.
- Adds `--replace-existing-artifact`, which quarantines a specifically selected
  prior artifact outside the active cache glob rather than deleting it.
- Adds destructive Qdrant rebuild support only behind both `--rebuild` and
  `--confirm-rebuild`; the master exposes this only as `--rebuild-qdrant`.
- Documents the explicit, reviewable recovery command with dry-run first.

### Why it is implemented this way

Embedding-vector reuse and vector-store identity serve different purposes.
Identical content can share one vector to reduce cost, but each document/release
occurrence needs independent payload metadata, retrieval filtering, and
citations. A point ID based only on content can overwrite another occurrence
and create a false lineage claim.

The original R21 artifact contains three records with one cache key and three
different vector fingerprints. Selecting one silently would hide an integrity
incident. Explicit quarantine plus regeneration makes the recovery visible and
preserves the artifact for investigation.

### Production interpretation

Changing the deterministic point-ID scheme requires a Qdrant rebuild; otherwise
old point IDs remain and may produce duplicate or stale retrieval evidence. The
rebuild is deliberately opt-in because it deletes only the configured local
collection. It never deletes FDD source, processed artifacts, active embedding
cache, or quarantined conflicting artifacts.

Before executing recovery, back up or otherwise retain the local Qdrant state
if operational policy requires it, review the dry-run command, and confirm that
the selected raw FDD batch is the intended scope. The actual recovery will make
new OpenAI embedding requests and therefore has cost.

### Failure-mode testing

- Three duplicate-content units make one API request and receive the same
  resulting vector in deterministic tests.
- Duplicate-content units create distinct Qdrant point IDs and persist as two
  points rather than overwriting one payload.
- The artifact quarantine preserves the original file outside the active cache.
- `--rebuild` without `--confirm-rebuild` exits before collection access.
- The real R21 recovery dry run lists artifact replacement and Qdrant rebuild
  commands without API calls, Qdrant writes, or source moves.

### Validation

- Focused duplicate-content/cache/Qdrant/master suite: 35 passed.
- Real rebuild-guard and master recovery dry-run checks: passed without
  mutation.
- Full regression suite: 343 passed; one existing non-failing Starlette/HTTPX
  deprecation warning remains.
- The live R21 recovery command was not run and awaits explicit user approval.

## Step 106 – Preserve duplicate evidence units and version the Qdrant collection

### Objective

Correct the second R21 runtime failure: the Qdrant indexer reused the
content-cache dictionary and therefore discarded five repeated-content evidence
units before upsert. Correct the unsafe embedded-Qdrant rebuild assumption and
separate full document identity from lineage family/release metadata.

### Evidence from the failed live run

- The R21 artifact had 149 records, 149 unique `unit_id` values, and 149 unique
  point IDs, so filename/release parsing did not cause the missing IDs.
- R21 had 145 unique content cache keys. Exactly five units were absent from
  Qdrant because the indexer loaded one record per cache key.
- A disposable persistent-Qdrant probe inserted one old point, deleted and
  recreated the collection, then inserted one new point; it still counted two
  points. Embedded local Qdrant delete-and-recreate cannot support a truthful
  in-place rebuild guarantee.

### Files added or changed

- `app/embeddings/embedding_cache.py`
- `app/embeddings/embedding_contract.py`
- `app/ingestion/filename_parser.py`
- `app/ingestion/retrieval_ready_artifact.py`
- `app/vectorstore/qdrant_indexer.py`
- `app/vectorstore/qdrant_upsert.py`
- `app/llm/answer_contract.py`
- `scripts/check_qdrant_index.py`
- `scripts/run_qdrant_indexing.py`
- `scripts/master_ingestion_embedding_docs.py`
- `docs/Steps_for_FDD_Ingestion.md`
- parser, artifact, embedding-cache, Qdrant-indexer, Qdrant-script, and master
  regression tests.

### Python/code pattern used

```python
# Cache lookup: one canonical vector per exact content key.
cache = {record.cache_key: record for record in load_embedding_records(path)}

# Indexing: retain every citeable occurrence, including repeated text.
records = load_embedding_records(path)

document_id = full_filename_without_docx
document_family = cross_release_lineage_family
release_label = "R21"
```

### What the code does

- Adds `load_embedding_records`, which validates vector consistency but returns
  every usable record. Cache lookup remains deduplicated; Qdrant indexing no
  longer is.
- Adds `document_id` as the full filename without `.docx` to new ingestion,
  embedding, Qdrant payload, and citation metadata while preserving
  `document_family` and `release_label`.
- Makes Qdrant verification include the document identifier for newly generated
  artifacts.
- Disables `--rebuild` and `--rebuild-qdrant` with a safe error before any
  collection access.
- Documents a versioned-collection migration: set a new
  `QDRANT_COLLECTION_NAME`, build/verify it, then point the API/UI to it.

### Why it is implemented this way

Content-vector reuse and evidence occurrence indexing have different identity
requirements. A dictionary keyed by content is appropriate for cache lookup but
wrong for a citation-bearing vector index. Each occurrence must survive upsert.

Multiple FDDs can share one R21 release. Replacing `document_family` with the
full filename would prevent valid cross-release lineage grouping; a distinct
`document_id` retains full-source identity without collapsing the other axes.

### Production interpretation

The old `functional_specs` collection is now considered legacy/stale because it
contains mixed point-ID generations. Preserve it for investigation and rollback
but do not use it for new grounded answers. Create a new versioned collection,
such as `functional_specs_v2`, through the `.env` configuration and validate it
before switching application traffic.

Existing old embedding artifacts do not yet contain the new optional
`document_id`; their `unit_id` still retains the full filename. Reprocessing a
source document upgrades its artifact and payload without requiring semantic
guesses. Do not claim complete metadata migration until all intended source
documents have been reprocessed.

### Failure-mode testing

- Duplicate-content records remain distinct in the Qdrant indexing batch and
  become distinct points.
- Legacy rebuild CLI flags fail before collection access.
- A disposable persistent-Qdrant probe demonstrates old-point retention after
  delete-and-recreate, proving why versioned collections are required.
- Full filename identity is present separately from family and release in new
  retrieval-ready artifacts.

### Validation

- Focused identity/cache/indexing/master suite: 40 passed.
- Full regression suite: 345 passed; one existing non-failing Starlette/HTTPX
  deprecation warning remains.
- No additional live OpenAI call, Qdrant upsert, archive action, or `.env`
  collection-name change was performed after the failed recovery run.
