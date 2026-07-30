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
