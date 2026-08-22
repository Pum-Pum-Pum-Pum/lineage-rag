# Culling Blade Lineage — GenAI RAG System

Production-minded local RAG system for enterprise functional specification documents with release-lineage awareness, hybrid retrieval, grounded answers, citations, safe refusals, local traces, and a minimal FastAPI backend.

## Current backend API milestone

The project currently exposes a local FastAPI backend with:

- `GET /health` — lightweight liveness/config check
- `GET /ready` — dependency/artifact readiness check for the active retrieval mode
- `POST /query` — grounded answer query endpoint
- local answer trace artifacts under `data/exports/answer_runs/`
- config-driven retrieval mode: `dense`, `lexical`, or `hybrid`
- `POST /conversations` and `GET /conversations` for chat lifecycle
- `GET /conversations/{conversation_id}` for durable history
- `POST /conversations/{conversation_id}/messages` for grounded multi-turn
  submission
- `POST /conversations/{conversation_id}/archive` for read-only archival

The API calls the shared `run_grounded_answer_query(...)` orchestration service. The API layer should validate requests and format responses; it should not duplicate retrieval, sufficiency, generation, or trace-writing logic.

## Conversation-memory foundation

The project now has conversation-scoped domain records and a local durable
SQLite adapter under `app/conversation/`. The `ConversationStore` protocol keeps
persistence separate from the RAG pipeline so a future Oracle implementation can
replace SQLite without changing conversation orchestration.

The conversation-memory foundation supports:

- independent conversations and ordered user/assistant messages
- optional trace IDs on assistant messages
- versioned, forward-only summary checkpoints
- archived conversations that remain readable but cannot receive new messages
- configurable local storage through `CONVERSATION_DB_PATH`
- token-triggered rolling summaries that retain a recent verbatim suffix
- explicit context reserves for system instructions, retrieved evidence, and
  answer output
- configurable budgets through `CONVERSATION_MAX_CONTEXT_TOKENS`,
  `CONVERSATION_RESERVED_SYSTEM_TOKENS`,
  `CONVERSATION_RESERVED_EVIDENCE_TOKENS`,
  `CONVERSATION_RESERVED_ANSWER_TOKENS`, and
  `CONVERSATION_SUMMARY_TARGET_TOKENS`
- explicit failure when mandatory recent context or generated summaries exceed
  their allocation instead of silent truncation

`ApproximateTokenCounter` provides a deterministic UTF-8-aware preflight
estimate without adding a tokenizer dependency. The context builder accepts a
`TokenCounter` adapter so a model-specific tokenizer can replace it when exact
model accounting is required. Conversation summaries preserve chat intent and
constraints only; functional-spec claims must still be grounded in newly
retrieved evidence and citations.

The conversation API and Streamlit multi-turn interface now use these
contracts. Durable backend history remains the source of truth; Streamlit
session state only caches transient debug details for turns created in the
current UI session.

## Setup

Install the locked runtime and development dependencies:

```bash
uv sync --locked
```

`uv` creates and manages the local `.venv` automatically. The environment is
disposable and ignored by Git; `pyproject.toml` declares direct dependencies and
`uv.lock` stores the exact cross-platform resolution committed for reproducible
development, CI, and deployment.

## Run the FastAPI backend

Start the local API server:

```bash
uv run --locked uvicorn app.api.main:app --reload
```

Default local URL:

```text
http://127.0.0.1:8000
```

Interactive OpenAPI documentation:

```text
http://127.0.0.1:8000/docs
```

## Run the Streamlit UI

Keep the FastAPI backend running, then start the UI in a second terminal:

```bash
uv run --locked streamlit run app/ui/streamlit_app.py
```

The UI defaults to `http://127.0.0.1:8000`. Override it from the sidebar or set:

```bash
RAG_API_BASE_URL=http://127.0.0.1:8000
```

Use **New chat** to create a durable conversation, select active conversations
from the sidebar, and use **Archive active chat** to make history read-only.
Select **Functional documents**, **Visible custom code**, or **Documents +
custom code** independently from the dense/lexical/hybrid retrieval technique.
Code and combined requests also expose explanation or impact-analysis intent;
impact locations remain candidates rather than proven root causes. Every
submitted chat message performs a mode-specific readiness check before
`POST /conversations/{conversation_id}/messages`; a failed readiness check
blocks cost-bearing retrieval and generation.

The UI renders durable user/assistant history, grounded answers or safe
refusals, trace IDs, and optional evidence/debug details containing sufficiency,
global requested-claim support, per-section combined support, separate FDD/code
citations, token-budget state, model usage, and estimated cost. A user-only
partial turn is rendered as retryable instead of being hidden. The local trace
output path and raw backend error bodies are not exposed.

`CODE_MODES_ENABLED=false` remains the safe default. Selecting code or combined
mode before deliberate backend activation fails closed and directs the user back
to functional-document mode; it does not silently fall back or spend model cost.

## Smoke test the API

Start with the health-only smoke test:

```bash
uv run --locked python scripts/run_api_smoke_test.py --base-url http://127.0.0.1:8000
```

Expected interpretation:

- confirms the backend process is reachable
- prints active retrieval configuration from `GET /health`
- logs `No query supplied. Skipped POST /query.`
- does **not** run retrieval, embeddings, LLM generation, Qdrant collection checks, or trace writing

Use this first when you only want a cheap liveness/config check.

Health plus readiness smoke test:

```bash
uv run --locked python scripts/run_api_smoke_test.py --base-url http://127.0.0.1:8000 --check-ready
```

Expected interpretation:

- checks whether required local dependencies/artifacts are available for the active retrieval mode
- validates retrieval runtime configuration
- checks model configuration is present without calling model APIs
- checks `.retrieval_ready.json` artifacts when lexical or hybrid retrieval needs local lexical evidence
- checks Qdrant collection existence when dense or hybrid retrieval needs vector search
- returns `503 Service Unavailable` with structured readiness details when a required dependency is missing
- does **not** run retrieval, embeddings, LLM generation, answer trace writing, or a user query
- when `--check-ready` fails, the smoke-test client stops before any optional `POST /query`

Health plus readiness plus query smoke test:

```bash
uv run --locked python scripts/run_api_smoke_test.py --base-url http://127.0.0.1:8000 --check-ready --query "What changed in branch reports?" --limit 5
```

Expected interpretation:

- calls `GET /health` first
- calls `GET /ready` only when `--check-ready` is supplied
- then calls `POST /query` only because `--query` was explicitly supplied
- may trigger retrieval, embedding calls, LLM generation, local answer trace writing, latency, and API cost depending on active retrieval mode and model settings
- logs answer status, evidence sufficiency, refusal reason when present, trace ID, and citations

With filters:

```bash
uv run --locked python scripts/run_api_smoke_test.py ^
  --base-url http://127.0.0.1:8000 ^
  --check-ready ^
  --query "What changed in branch reports?" ^
  --limit 5 ^
  --release-label R24 ^
  --source-kind paragraph
```

PowerShell users can replace `^` line continuations with backticks.

### Smoke-test failure interpretation

- If health-only smoke testing fails, debug backend startup, port, routing, or configuration first.
- If `/query` returns `503` in dense or hybrid mode, the required Qdrant collection is unavailable; run indexing first with `uv run --locked python scripts/run_qdrant_indexing.py`.
- If `/query` returns an insufficient-evidence response, treat that as a safe refusal signal, not a backend crash.
- If the smoke-test client reports an HTTP failure, it intentionally avoids printing raw server response bodies because they may contain secrets, stack traces, local file paths, or internal configuration values.
- For answered queries, inspect `data/exports/answer_runs/` to reproduce which retrieved evidence, sufficiency decision, prompt version, citations, and usage metadata produced the response.

## API behavior

### `GET /health`

Lightweight endpoint for backend liveness and active retrieval configuration.

It intentionally does **not** run:

- retrieval
- embeddings
- LLM generation
- Qdrant collection checks

This keeps `/health` cheap and separates liveness from readiness.

### `GET /ready`

Readiness endpoint for checking whether the backend can serve the active retrieval mode.

It may check:

- retrieval runtime configuration
- required model configuration values
- local `.retrieval_ready.json` artifacts for lexical/hybrid modes
- Qdrant collection existence for dense/hybrid modes

It intentionally does **not** run:

- user retrieval
- embedding API calls
- LLM generation
- answer trace writing

If a required readiness check fails, `/ready` returns `503 Service Unavailable` with a structured payload such as `status=not_ready`, `is_ready=false`, and per-check details. This is dependency readiness, not answer sufficiency. Missing corpus evidence during a real query should produce an insufficient-evidence/refusal response, not a readiness failure.

### `POST /query`

Runs the shared grounded answer orchestration flow:

1. configured retrieval mode
2. evidence sufficiency
3. grounded answer generation or safe refusal
4. citation formatting
5. local answer trace writing

Minimal request example:

```json
{
  "query": "What changed in branch reports?",
  "limit": 5
}
```

Filtered request example:

```json
{
  "query": "What changed in branch reports?",
  "limit": 5,
  "release_label": "R24",
  "source_kind": "paragraph",
  "min_top_score": 0.25
}
```

### Conversation endpoints

Create a conversation:

```http
POST /conversations
Content-Type: application/json

{"title": "R24 branch-report investigation"}
```

List active conversations with `GET /conversations`; add
`?include_archived=true` to include archived history. Retrieve messages and the
current rolling-summary checkpoint with
`GET /conversations/{conversation_id}`.

Submit a grounded turn:

```http
POST /conversations/{conversation_id}/messages
Content-Type: application/json

{
  "content": "What changed in R24?",
  "limit": 5,
  "release_label": "R24"
}
```

The endpoint persists the user message, builds bounded conversation context,
runs the same grounded orchestration used by `POST /query`, and persists the
assistant answer with its trace ID. Conversation memory may help interpret
follow-up intent, but factual claims still require newly retrieved evidence and
citations.

If retrieval or generation fails after accepting the message, the durable user
message remains and no assistant message is invented. This visible partial turn
supports audit and retry handling. Token-budget overflow returns `413` before
the grounded query; archived conversations return `409`; unknown conversations
return `404`. Archive with
`POST /conversations/{conversation_id}/archive`; archived history remains
readable but cannot receive new messages.

## Important operational notes

- Dense and hybrid retrieval require a Qdrant collection.
- Hybrid retrieval combines dense and lexical ordinal ranks with weighted
  Reciprocal Rank Fusion (`0.40` dense, `0.60` lexical by default) rather than
  adding incompatible normalized score scales. Result payloads retain raw
  retriever scores, ranks, RRF contributions, and a bounded final hybrid score
  for debugging.
- Lexical-only retrieval uses local `.retrieval_ready.json` artifacts and does not require a Qdrant collection check.
- `/health` is a cheap liveness/config check; `/ready` is a dependency/artifact readiness check.
- If dense/hybrid mode is active and the Qdrant collection is missing, `POST /query` returns `503`.
- If `/ready` returns `503`, inspect the failed readiness check before running `POST /query`.
- `scripts/run_api_smoke_test.py --check-ready --query ...` stops before `/query` when `/ready` fails.
- Unexpected API errors return safe generic messages instead of raw exception details.
- Answer traces are written locally for debugging and reproducibility.
- Conversation summaries are context, not evidence, and never replace fresh
  retrieval or citation validation.
- This corpus treats indexed functional releases as production-deployed.
  Questions containing current/latest/now use the resulting state after the
  highest relevant retrieved release. An `Existing Functionality` section in
  that release is treated as its pre-change baseline.
- Current-state queries retrieve a wider candidate set once, resolve release
  labels numerically (`R24` after `R2`), remove older-release evidence, and then
  pass the requested top-k evidence to generation. Referential conversation
  queries such as “summarize it” may inherit an explicit release from bounded
  conversation memory, but all factual claims still require fresh evidence.
- LLM prompt evidence uses complete selected retrieval units; the 240-character
  citation previews returned to API/UI clients are display metadata only and
  are never used as the model's evidence.
- Prompt evidence is admitted as whole ranked units within the reserved
  evidence-token budget. If the highest-ranked unit cannot fit, generation
  stops with a safe refusal instead of silently truncating the evidence.
- A failed conversation turn may contain a persisted user message without an
  assistant message; clients should render this as retryable rather than assume
  strict user/assistant pairing.

### Retrieval-mode dependency matrix

| Active retrieval mode | `/ready` dependency checks | `/query` retrieval dependencies | Model/API calls during `/query` | Failure boundary |
| --- | --- | --- | --- | --- |
| `lexical` | validates retrieval config, model config presence, and local `.retrieval_ready.json` artifacts | local lexical artifacts only; no Qdrant client or collection check | no embedding call for retrieval; LLM call only when evidence is sufficient | missing lexical artifacts make `/ready` return `503`; insufficient evidence should produce a safe refusal |
| `dense` | validates retrieval config, model config presence, and Qdrant collection existence | Qdrant collection and embedding call for dense vector search | embedding call for retrieval; LLM call only when evidence is sufficient | missing Qdrant collection makes `/ready` or `POST /query` return `503` before answer generation |
| `hybrid` | validates retrieval config, model config presence, local `.retrieval_ready.json` artifacts, and Qdrant collection existence | both Qdrant dense search and local lexical artifacts | embedding call for dense side of retrieval; LLM call only when fused evidence is sufficient | missing lexical artifacts or Qdrant collection makes `/ready` return `503`; missing Qdrant collection makes `POST /query` return `503` |

This matrix is an operational contract. Lexical mode is the cheapest degraded retrieval path and should not instantiate Qdrant or call embedding APIs for retrieval. Dense and hybrid modes are higher-quality semantic retrieval paths, but they must fail fast when vector-store state is unavailable so the system avoids wasted model spend and misleading answers.

## Run tests

Targeted API tests:

```bash
uv run --locked pytest tests/test_health_api.py tests/test_readiness_api.py tests/test_query_api.py tests/test_api_smoke_script.py -q
```

Full regression suite:

```bash
uv run --locked pytest -q
```

Evaluate a persisted answer trace against the R24 current-state contract:

```bash
uv run --locked python scripts/evaluate_answer_trace.py \
  --trace data/exports/answer_runs/<trace-id>.json \
  --case-id r24_current_teller_and_branch_report_state
```

The same evaluator also has an explicit abstention contract:

```bash
uv run --locked python scripts/evaluate_answer_trace.py \
  --trace data/exports/answer_runs/<trace-id>.json \
  --case-id unsupported_mobile_login_abstention
```

Conversation reliability coverage uses deterministic dependency doubles while
exercising the real FastAPI, SQLite, context, grounding, and persistence
boundaries:

```bash
uv run --locked pytest \
  tests/test_conversation_rag_reliability_e2e.py \
  tests/test_conversation_reliability_evaluation.py -q
```

This suite covers follow-up context, conversation isolation, summary identifier
drift, safe context overflow behavior, abstention persistence, and invalid
citation suppression. It is not a substitute for load, browser, live-model,
semantic-entailment, or cross-browser evaluation.

## API request correlation and safe audit events

Every API response includes a validated `X-Request-ID`. Clients may supply an
ID containing only letters, digits, `.`, `_`, `:`, or `-` up to 128 characters;
invalid values are replaced with a generated UUID before logging. Query answer
traces store this value as `correlation_id`, separately from the unique trace
ID and filename.

The `api_audit` logger emits one compact JSON event per request containing only:

- event name
- request ID
- HTTP method
- route template rather than concrete resource IDs
- response status
- elapsed milliseconds

Request/response bodies, query text, conversation titles, credentials, client
IP addresses, exception details, and concrete conversation IDs are excluded.
Responses also set `Cache-Control: no-store`, `X-Content-Type-Options:
nosniff`, `X-Frame-Options: DENY`, `Referrer-Policy: no-referrer`, and a
restricted `Permissions-Policy`.

These controls improve correlation, log safety, and defensive API behavior.
They do not provide authentication, authorization, rate limiting, transport
TLS, centralized log retention, or tamper-evident auditing; those belong at the
application identity layer and production platform boundary.

## Tamper-evident local audit journal

The API can persist the same fixed-schema request events to a durable
HMAC-SHA256 chained JSONL journal:

```text
AUDIT_JOURNAL_ENABLED=true
AUDIT_SINK_BACKEND=hmac_jsonl
AUDIT_JOURNAL_PATH=/approved/mutable-state/api_audit.jsonl
AUDIT_HMAC_KEY=<secret containing at least 32 UTF-8 bytes>
```

The key must come from the approved secret store and must not be committed or
placed beside the journal. Each record contains a sequence number, UTC
timestamp, safe request event, previous-record HMAC, and record HMAC. The
writer flushes and calls `fsync` before reporting success. If persistence fails,
the request remains available and a content-free critical event is emitted;
operations must alert on that event because the audit trail has a gap.

Verify the chain without printing request events:

```bash
python scripts/verify_audit_journal.py
```

For suffix-deletion detection, provide a final HMAC and record count previously
stored in a separate trusted system:

```bash
python scripts/verify_audit_journal.py \
  --expected-record-count 1000 \
  --expected-final-hmac <trusted-checkpoint>
```

An HMAC chain detects edits, insertion, and reordering while its key remains
secret. It cannot detect deletion of a valid suffix without an external trusted
checkpoint, and an attacker who obtains both journal and key can forge a new
chain. The writer rejects a journal changed by another writer; the current
native contract is therefore one API process. Centralized append-only
retention, checkpoint custody, key rotation, multi-host ordering, access
review, and retention deletion remain platform controls.

FastAPI depends on an `AuditSink` protocol rather than directly on the JSONL
file writer. The currently supported `hmac_jsonl` adapter declares
`durable_on_return`, meaning its `append()` returns only after flush and
`fsync`. The configured path may be a local path, mounted volume, or reachable
network-filesystem path, but remote `fsync` durability and failure behavior must
be validated for the actual filesystem and mount options.

Future database, central collector, or grouped-commit adapters should be added
behind this boundary. A buffered adapter must declare
`accepted_not_durable` when returning before its batch is committed and must
define queue limits, flush triggers, shutdown draining, backpressure, loss
window, and health reporting. No such buffered adapter is enabled yet.

Measure the local per-record HMAC, append, flush, and `fsync` cost before
selecting a durability policy:

```bash
python scripts/benchmark_audit_journal.py \
  --events 200 \
  --warmup-events 10
```

The report is written to
`data/exports/audit_benchmarks/audit-journal-benchmark.json`. The benchmark
uses synthetic metadata, an ephemeral in-memory key, and a temporary journal
that is removed afterward. It reports append p50/p95/p99/max latency, measured
single-writer throughput, storage bytes per record, and full-chain verification
time. It does not call retrieval or model services and is not an end-to-end
capacity, concurrency, or production SLO test.

## Native deployment bundle

Docker is intentionally not required. Build a deterministic Python 3.12 runtime
bundle for an approved native process supervisor:

```bash
python scripts/build_native_deployment.py
```

The default artifact is
`data/exports/deployment/lineage-rag-native.zip`. It contains application and
operational Python code, `pyproject.toml`, `uv.lock`, `.env.example`, the
runtime process contract, and a SHA-256 manifest for every bundled file. ZIP
ordering, metadata, and timestamps are fixed so identical source produces an
identical archive hash.

The package deliberately excludes `.env`, credentials, certificates, raw
documents, processed retrieval artifacts, Qdrant state, conversations, traces,
logs, virtual environments, tests, and generated exports. Mutable state must be
provisioned and backed up separately.

After extracting the bundle and injecting configuration through the approved
secret mechanism:

```bash
uv sync --locked --no-dev
uv run --locked --no-dev python scripts/check_deployment_preflight.py
```

For local validation only, a development environment label can be allowed:

```bash
python scripts/check_deployment_preflight.py --allow-development
```

Preflight checks Python 3.12, locked project files, a non-development
environment label, presence—not values—of model configuration, retrieval state
required by the active mode, a strong enabled audit-journal configuration, and
writable conversation/trace locations. Development package validation may
explicitly leave the journal disabled. Preflight does not call embedding, chat,
or vector services and never prints secret values.

[`deployment/native_runtime.json`](deployment/native_runtime.json) records the
locked install, preflight, FastAPI, and Streamlit command contracts. An
Oracle-approved systemd, Windows Service, or other native supervisor must add
restart policy, service identity, resource limits, TLS/reverse proxy,
centralized logs, and secret injection after the actual target is selected.

## Continuous integration

GitHub Actions runs the full Python 3.12 regression suite for every push and
pull request. The workflow checks that `uv.lock` is current, installs the locked
development environment, and runs tests without requiring application secrets
or live external services:

```bash
uv lock --check
uv sync --locked --dev
uv run --locked pytest -q
```

The uv installer action and uv executable version are pinned in
`.github/workflows/ci.yml`. Dependency caching is keyed from `uv.lock`.

Check dependency reproducibility without modifying the lockfile:

```bash
uv lock --check
```

Install runtime dependencies only for deployment:

```bash
uv sync --locked --no-dev
```

If a deployment platform requires a pip-compatible file, generate it from the
committed lock rather than maintaining a second dependency source manually:

```bash
uv export --locked --no-dev --format requirements-txt --output-file requirements.txt
```

## UI integration status

The Streamlit interface is available in `app/ui/streamlit_app.py` and uses the
typed client in `app/ui/api_client.py` for health/readiness and the conversation
lifecycle/message endpoints. It provides new/select/archive controls,
`st.chat_message` history, `st.chat_input` submission, readiness gating, partial
turn warnings, explicit knowledge/analysis selectors, lane-specific citations,
per-section support states, and an optional evidence/debug panel.

Network, timeout, `404`, `409`, `413`, `503`, generic HTTP, malformed JSON, and
schema-validation failures map to safe presentation-layer errors without
exposing backend response bodies. Conversation history is reloaded from the
backend; only debug details returned during the current UI session are cached
in Streamlit state.
