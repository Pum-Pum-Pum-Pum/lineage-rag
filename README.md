# Culling Blade Lineage — GenAI RAG System

Production-minded local RAG system for enterprise functional specification documents with release-lineage awareness, hybrid retrieval, grounded answers, citations, safe refusals, local traces, and a minimal FastAPI backend.

## Current backend API milestone

The project currently exposes a local FastAPI backend with:

- `GET /health` — lightweight liveness/config check
- `GET /ready` — dependency/artifact readiness check for the active retrieval mode
- `POST /query` — grounded answer query endpoint
- local answer trace artifacts under `data/exports/answer_runs/`
- config-driven retrieval mode: `dense`, `lexical`, or `hybrid`

The API calls the shared `run_grounded_answer_query(...)` orchestration service. The API layer should validate requests and format responses; it should not duplicate retrieval, sufficiency, generation, or trace-writing logic.

## Setup

Install dependencies:

```bash
pip install -r requirements.txt
```

If you use a virtual environment, activate it before running the commands below.

## Run the FastAPI backend

Start the local API server:

```bash
python -m uvicorn app.api.main:app --reload
```

Default local URL:

```text
http://127.0.0.1:8000
```

Interactive OpenAPI documentation:

```text
http://127.0.0.1:8000/docs
```

## Smoke test the API

Start with the health-only smoke test:

```bash
python scripts/run_api_smoke_test.py --base-url http://127.0.0.1:8000
```

Expected interpretation:

- confirms the backend process is reachable
- prints active retrieval configuration from `GET /health`
- logs `No query supplied. Skipped POST /query.`
- does **not** run retrieval, embeddings, LLM generation, Qdrant collection checks, or trace writing

Use this first when you only want a cheap liveness/config check.

Health plus readiness smoke test:

```bash
python scripts/run_api_smoke_test.py --base-url http://127.0.0.1:8000 --check-ready
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
python scripts/run_api_smoke_test.py --base-url http://127.0.0.1:8000 --check-ready --query "What changed in branch reports?" --limit 5
```

Expected interpretation:

- calls `GET /health` first
- calls `GET /ready` only when `--check-ready` is supplied
- then calls `POST /query` only because `--query` was explicitly supplied
- may trigger retrieval, embedding calls, LLM generation, local answer trace writing, latency, and API cost depending on active retrieval mode and model settings
- logs answer status, evidence sufficiency, refusal reason when present, trace ID, and citations

With filters:

```bash
python scripts/run_api_smoke_test.py ^
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
- If `/query` returns `503` in dense or hybrid mode, the required Qdrant collection is unavailable; run indexing first with `python scripts/run_qdrant_indexing.py`.
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

## Important operational notes

- Dense and hybrid retrieval require a Qdrant collection.
- Lexical-only retrieval uses local `.retrieval_ready.json` artifacts and does not require a Qdrant collection check.
- `/health` is a cheap liveness/config check; `/ready` is a dependency/artifact readiness check.
- If dense/hybrid mode is active and the Qdrant collection is missing, `POST /query` returns `503`.
- If `/ready` returns `503`, inspect the failed readiness check before running `POST /query`.
- `scripts/run_api_smoke_test.py --check-ready --query ...` stops before `/query` when `/ready` fails.
- Unexpected API errors return safe generic messages instead of raw exception details.
- Answer traces are written locally for debugging and reproducibility.

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
python -m pytest tests/test_health_api.py tests/test_readiness_api.py tests/test_query_api.py tests/test_api_smoke_script.py -q
```

Full regression suite:

```bash
python -m pytest -q
```

## Current limitation

This project currently prioritizes a reliable FastAPI backend contract. Streamlit UI is deferred until the backend contract is stable and a UI adds clear demo value beyond FastAPI's built-in OpenAPI docs.
