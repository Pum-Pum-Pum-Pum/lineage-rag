# ChatGPT Secure MCP Tunnel — Phase 1 runbook

## Purpose and boundary

Phase 1 adds ChatGPT as a retrieval interface without creating a second RAG
implementation:

```text
FastAPI /query ─┐
FastAPI /search ├─> KnowledgeRetrievalService ─> approved FDD/code indexes
MCP search/fetch┘
```

FastAPI continues grounded answer generation. The MCP tools return bounded FDD
and visible custom-code evidence only; ChatGPT interprets it. The MCP process
does not call FastAPI over HTTP, does not expose a public listener, and exposes
only `search(query, mode)` and `fetch(id)`.

The active approved generations remain FDD v5, code v2, and the reviewed lineage
artifact. This runbook does not rebuild, migrate, or activate a knowledge
generation. A retrieval result is evidence, not proof that all source behavior is
known. Hidden kernel behavior, external schema targets, dynamic SQL, and
conditional branches remain qualified unknowns.

## Prerequisites and safety checks

1. Use the project virtual environment and locked dependencies:

   ```powershell
   uv sync --locked
   uv lock --check
   ```

2. Confirm the FDD and, when enabled, code generations are locally ready before
   exposing MCP. `GET /ready` is the runtime readiness check for FastAPI; the MCP
   child also performs a bounded local configuration preflight before registering
   tools.

3. Do not put the real `CONTROL_PLANE_API_KEY` in `.env`, a script, a trace, or a
   command history. It is injected only into the `tunnel-client` terminal through
   the approved secret mechanism. The tunnel launcher removes it before Python
   starts the MCP child.

4. Review the risk: enabling MCP disclosure allows returned internal FDD and
   custom PL/SQL evidence to leave the laptop through the approved ChatGPT tunnel.
   Use it only for an approved test session and only with authorized reviewers.

## Configuration

Keep these safe defaults in `.env` for normal FastAPI/Streamlit operation:

```dotenv
INTERFACE_MODE=fastapi
RETRIEVAL_INDEX_PATH=data/processed
MCP_EVIDENCE_DISCLOSURE_ENABLED=false
```

`RETRIEVAL_INDEX_PATH` and historic `PROCESSED_DIR` identify the same FDD lexical
artifact directory. If both are set they must resolve to the same path or startup
fails. Do not set `CONTROL_PLANE_API_KEY` in `.env`; the blank template entry is
only a reminder that tunnel-client, not application code, owns that credential.

For an approved ChatGPT retrieval session, an operator deliberately changes the
following in `.env` before starting processes:

```dotenv
INTERFACE_MODE=mcp
MCP_EVIDENCE_DISCLOSURE_ENABLED=true
```

For an approved parallel local API/UI plus ChatGPT session:

```dotenv
INTERFACE_MODE=both
MCP_EVIDENCE_DISCLOSURE_ENABLED=true
```

After a test, restore the safe state and restart affected processes:

```dotenv
INTERFACE_MODE=fastapi
MCP_EVIDENCE_DISCLOSURE_ENABLED=false
```

The disclosure flag is an MCP-only emergency egress kill switch. When false,
`search` and `fetch` return only `Evidence disclosure is disabled.` They do not
load the catalog, open Qdrant, perform lexical retrieval, embed a query, resolve
an opaque ID, or return evidence metadata. FastAPI and Streamlit behavior is not
disabled by this flag.

`OPENAI_API_KEY` remains the application key. Dense/hybrid retrieval can use it
to embed a query. `CONTROL_PLANE_API_KEY` authenticates tunnel-client to the
OpenAI control plane and must never be available to the MCP Python child.

## Build or verify local indexes

Do not rebuild active collections in place. A changed corpus requires an isolated,
complete generation, exact manifest-to-point verification, retrieval evaluation,
SME review, deliberate activation, and retained rollback generation.

For new FDD DOCX files, place only approved files in `data/raw_specs/`. Inspect
the planned non-mutating work first:

```powershell
uv run --locked python scripts/master_ingestion_embedding_docs.py --dry-run
```

The real FDD command may call OpenAI embeddings and archive a source only after
Qdrant verification succeeds. Obtain the required paid-use and internal-evidence
disclosure authority before running it:

```powershell
uv run --locked python scripts/master_ingestion_embedding_docs.py --request-batch-size 64
```

Verify the selected FDD collection and artifacts using the existing ingestion
runbook: [Steps_for_FDD_Ingestion.md](Steps_for_FDD_Ingestion.md).

For code, snapshot intake, parsing, dependency/lineage review, deterministic
artifact preparation, paid embeddings, isolated indexing, and exact verification
remain distinct gates. The relevant commands are intentionally separate:

```powershell
uv run --locked python scripts/prepare_code_index_artifacts.py --help
uv run --locked python scripts/embed_code_index_artifacts.py --help
uv run --locked python scripts/index_code_qdrant.py --help
uv run --locked python scripts/verify_code_qdrant.py --help
```

Do not use MCP as an ingestion, embedding, database, or arbitrary-file interface.

## Start the existing FastAPI/Streamlit interface

Use `INTERFACE_MODE=fastapi` or `both`. FastAPI and Streamlit refuse startup in
`mcp` mode.

```powershell
# Terminal 1
uv run --locked uvicorn app.api.main:app --host 127.0.0.1 --port 8000 --reload

# Terminal 2
uv run --locked streamlit run app/ui/streamlit_app.py --server.address 127.0.0.1 --server.port 8501
```

For a cheap readiness check that does not retrieve, embed, or generate an answer:

```powershell
uv run --locked python scripts/run_api_smoke_test.py --base-url http://127.0.0.1:8000 --check-ready
```

`POST /search` is retrieval-only and uses the same service as MCP. It accepts a
business query and knowledge mode, not a retrieval strategy:

```powershell
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/search `
  -ContentType application/json `
  -Body '{"query":"How is the AML batch processed?","mode":"combined"}'
```

Code and combined modes additionally require `CODE_MODES_ENABLED=true` under the
existing approved activation contract. MCP disclosure alone does not enable them.

## Direct MCP Inspector testing

Direct startup is for local MCP Inspector/protocol testing only; it is not the
tunnel operation. Set `INTERFACE_MODE=mcp` and
`MCP_EVIDENCE_DISCLOSURE_ENABLED=true` only for the approved test session.

With the MCP Inspector installed or available through `npx`, configure it to
launch this command from the repository root:

```text
.venv\Scripts\python.exe -m app.mcp.server
```

The child must receive the non-secret application environment:

```text
INTERFACE_MODE=mcp
MCP_EVIDENCE_DISCLOSURE_ENABLED=true
```

In Inspector, verify that only these read-only tools are advertised:

- `search(query, mode)` where `mode` is `fdd`, `code`, or `combined`
- `fetch(id)` where `id` matches `fdd_<64 hex>` or `code_<64 hex>`

`search` returns at most five items per single lane and at most five FDD plus five
code items for combined mode. Excerpts are capped at 240 characters. `fetch` can
resolve only an active approved catalog item; raw paths, SQL, database commands,
and arbitrary files are not accepted.

## Configure and run Secure MCP Tunnel

Use the current official [Secure MCP Tunnel guide](https://developers.openai.com/api/docs/guides/secure-mcp-tunnels)
and the organization-approved tunnel-client installation method. This project
does not create a tunnel automatically.

Initialize a named tunnel-client profile with the tunnel ID supplied by the
control plane. The MCP command must be the project launcher, not a FastAPI URL:

```powershell
tunnel-client init --sample sample_mcp_stdio_local `
  --profile culling-blade-local `
  --tunnel-id <OPENAI_TUNNEL_ID> `
  --mcp-command "powershell -NoProfile -ExecutionPolicy Bypass -File scripts/run_mcp_stdio.ps1"
```

Before connecting, validate the profile:

```powershell
tunnel-client doctor --profile culling-blade-local
```

In the dedicated tunnel terminal, inject the real control-plane key using the
approved secret-management method, then run:

```powershell
# Terminal 3 only: the key is injected by the approved secret mechanism.
tunnel-client run --profile culling-blade-local
```

The tunnel client launches and owns the MCP stdio child through `--mcp-command`.
There is no separately started MCP-server terminal for tunnel use. The child
startup preflight rejects a child that inherited `CONTROL_PLANE_API_KEY`.

## Connect in ChatGPT Developer Mode

After `tunnel-client doctor` succeeds and `tunnel-client run` is connected, use
the tunnel-client/OpenAI control-plane workflow to connect the configured tunnel
in ChatGPT Developer Mode. Confirm the connection advertises exactly `search` and
`fetch`, then begin with `search` before `fetch`.

Do not treat a ChatGPT response as automatically grounded. It must use returned
FDD/code evidence appropriately, distinguish documented functionality from visible
implementation, and retain explicit unknowns where the custom code corpus cannot
prove hidden kernel or runtime behavior.

## Terminal layout

### FastAPI only

```text
Terminal 1: FastAPI
Terminal 2: Streamlit
```

### MCP only

```text
Terminal 1: tunnel-client run
            └─ owns and launches the MCP stdio child
```

### Both interfaces

```text
Terminal 1: FastAPI
Terminal 2: Streamlit
Terminal 3: tunnel-client run
            └─ owns and launches the MCP stdio child
```

Do not add a fourth manually started MCP server terminal in tunnel mode.

## Troubleshooting

| Symptom | Safe interpretation and action |
| --- | --- |
| `Evidence disclosure is disabled.` | This is the expected kill-switch response. Confirm approved authorization, set `MCP_EVIDENCE_DISCLOSURE_ENABLED=true`, restart the MCP child through tunnel-client, and retest. |
| MCP refuses startup in `fastapi` mode | Set `INTERFACE_MODE=mcp` or `both`; do not bypass the mode guard. |
| FastAPI/Streamlit refuse startup in `mcp` mode | Use `fastapi` or `both` for those processes. |
| MCP child rejects `control_plane_key_isolation` | The key reached the child. Use the supplied launcher and verify tunnel-client passes the configured command unchanged. Never place the key in `.env`. |
| Missing lexical artifact / code artifact / lineage preflight | Restore the approved local artifacts and configuration. Do not point MCP at a draft stage or silently switch generations. |
| Missing Qdrant collection on dense/hybrid search | Verify the selected FDD/code collection and exact artifact-to-point verification. Do not fall back silently or rebuild an active collection in place. |
| Embedding/key error on dense or hybrid search | Query embedding may require `OPENAI_API_KEY` and may incur cost. Confirm approval, key availability, provider status, and whether lexical mode is an acceptable temporary diagnostic path. |
| Tunnel-client doctor/run failure | Stop before testing ChatGPT retrieval. Verify profile, tunnel ID, outbound connectivity, command working directory, and control-plane credential injection using the official guide. |
| Extra text, log, or warning corrupts MCP connection | Treat stdout contamination as a protocol failure. Remove `print()` and stdout handlers; diagnostics belong on stderr/file artifacts. Re-run subprocess protocol tests. |
| Tool metadata missing or unexpected tools appear | Stop exposure. Only read-only `search` and `fetch` are permitted; inspect the MCP server registration and rerun protocol tests. |

## Cost, retention, and rollback

Dense/hybrid MCP search can create a query embedding and therefore cost. Combined
dense/hybrid search creates one query embedding and reuses it across FDD and code
lanes. MCP does not call the answer-generation API itself, but ChatGPT retrieval
still discloses the returned internal evidence when the kill switch is enabled.

MCP operational logs and traces must not copy full source text. Citeable tool
results contain approved original evidence and provenance; operational records
should retain only safe request/tool/status/count/identity metadata under the
applicable retention and access policy.

Rollback is immediate: set `MCP_EVIDENCE_DISCLOSURE_ENABLED=false`, restart the
tunnel-owned child, and confirm a valid `search` returns only the generic disabled
message. For broader interface rollback, set `INTERFACE_MODE=fastapi` and stop the
tunnel-client process. This does not replace existing generation rollback,
evaluation, or production recovery controls.
