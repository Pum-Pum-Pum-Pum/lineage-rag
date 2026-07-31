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
