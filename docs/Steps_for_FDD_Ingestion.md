--FastAPI::
python -m uvicorn app.api.main:app --host 127.0.0.1 --port 8000 --reload

--Streamlit::
python -m streamlit run app/ui/streamlit_app.py --server.address 127.0.0.1 --server.port 8501

# Add FDD Documents to the RAG System

Use this workflow for reviewed, deployed FDDs. The currently active local
generation is `functional_specs_v4`, paired with
`data/indexes/functional_specs_v4/processed`. **Do not ingest new documents
directly into that active pair.** Build and evaluate a new versioned generation,
then activate its Qdrant collection and lexical directory together.

The commands below use `v5` as an example. If either target collection or stage
directory already exists, increment the generation name; never delete or reuse a
partially built generation.

## 1. Put new FDDs in the intake folder

Copy one or more reviewed DOCX files into `data/raw_specs/`:

```powershell
Copy-Item -LiteralPath 'C:\approved-fdds\FS_ASNB_R25_Teller_Change.docx' `
  -Destination 'data\raw_specs\FS_ASNB_R25_Teller_Change.docx'
```

Enabled FDD extensions are centralized in
`config/ingestion_sources.toml`. The current `.docx = "docx"` mapping enables
the implemented DOCX handler. Adding another extension for that same handler
is configuration-only when the underlying OPC document is compatible. Adding
a genuinely new format such as PDF still requires a new extractor and handler;
the policy rejects unimplemented handler names.

Requirements:

- The filename must include a numeric release label such as `R25`.
- Do not copy a filename already present in `data/docs_embedded/`.
- A release may contain multiple FDDs. The full filename without `.docx` is the
  citeable `document_id`; the release label alone is not a unique source ID.
- `docs/` contains documentation only. Source documents belong under `data/`.

## 2. Preview an isolated intake run

Prepare the locked environment once:

```powershell
uv sync --locked
```

Set process-local intake targets. These values apply only to the current
PowerShell window and prevent the master command from writing into live v4:

```powershell
$env:QDRANT_COLLECTION_NAME='functional_specs_v5_intake'
$env:INGESTION_OUTPUT_DIR='data/staging/functional_specs_v5_intake/processed'

uv run --locked python scripts/master_ingestion_embedding_docs.py --dry-run
```

The dry run must list only the intended files. It does not call OpenAI, write to
Qdrant, create ingestion artifacts, or move DOCX files.

Before continuing, confirm that `functional_specs_v5_intake` is a new,
disposable intake collection name. It is not a serving collection.

## 3. Run extraction, embedding, intake indexing, and verification

In the same PowerShell window:

```powershell
uv run --locked python scripts/master_ingestion_embedding_docs.py
```

The master command calls the existing Python stages in order:

1. Extract, normalize, and chunk every DOCX in `data/raw_specs/`.
2. Build paragraph and parent-linked table retrieval artifacts.
3. Embed all unique uncached retrieval units with the configured OpenAI
   embedding model, using at most 64 units per request.
4. Index the embedding artifacts into the isolated intake collection.
5. Verify each intended point, payload, and vector against Qdrant.
6. Move the new DOCX files to `data/docs_embedded/` only after all stages pass.

To reduce the failure blast radius per paid request:

```powershell
uv run --locked python scripts/master_ingestion_embedding_docs.py `
  --request-batch-size 32
```

The batch size counts unique content units, not documents. Lower values create
more requests and latency. Compatible cached vectors may be reused, but every
document/chunk occurrence still receives its own identity for filtering and
citations.

## 4. Build the complete next generation from the archive

The intake collection proves the new files can be processed, but it is not the
release candidate. Rebuild all archived FDDs into one isolated, complete
generation:

```powershell
uv run --locked python scripts/stage_archived_fdd_rebuild.py `
  --dry-run `
  --source-directory data/docs_embedded `
  --stage-directory data/staging/functional_specs_v5 `
  --collection-name functional_specs_v5 `
  --index-generation functional_specs_v5
```

Review the source count, filenames, SHA-256 hashes, target directory, target
collection, embedding model, and cache-reuse plan. Then explicitly authorize
the paid operation and rerun without `--dry-run`:

```powershell
uv run --locked python scripts/stage_archived_fdd_rebuild.py `
  --source-directory data/docs_embedded `
  --stage-directory data/staging/functional_specs_v5 `
  --collection-name functional_specs_v5 `
  --index-generation functional_specs_v5
```

The stage must finish with `status: verified` in
`data/staging/functional_specs_v5/stage_manifest.json`. Unchanged compatible
embedding inputs reuse cached vectors. New or changed retrieval text—including
new parent-linked table context—is embedded again.

## 5. Evaluate before activation

Keep `.env` pointing at v4 while evaluating v5 through explicit paired
overrides. At minimum:

```powershell
uv run --locked python scripts/run_fdd_retrieval_gate.py `
  --eval-file data/evaluations/fdd_grounded_eval_v2_reviewed.jsonl `
  --collection-name functional_specs_v5 `
  --lexical-artifact-directory data/staging/functional_specs_v5/processed
```

Run reviewed document-specific and lineage cases for the newly added FDDs as
well as the existing regression set. Paid grounded-answer evaluation requires
explicit authorization because questions and retrieved internal evidence are
sent to OpenAI.

Do not activate the generation unless all required checks pass:

- stage manifest and exact Qdrant verification;
- complete expected document/point coverage and vector dimension;
- reviewed retrieval, release-selection, table-linkage, citation, conflict,
  abstention, and answer-correctness gates;
- readiness check and a retained rollback generation.

## 6. Promote and activate the pair

After approval, copy the verified lexical artifacts to a stable runtime path,
verify file counts and SHA-256 hashes against the stage, and update both values
in `.env` together:

```text
QDRANT_COLLECTION_NAME=functional_specs_v5
PROCESSED_DIR=data/indexes/functional_specs_v5/processed
```

Restart FastAPI and Streamlit, verify their effective configuration and
readiness, run a known grounded query with citations, and retain v4 for rollback.
Changing only one of these settings creates a mixed vector/lexical generation
and is a release-blocking error.

## Artifact locations

| Purpose | Location |
| --- | --- |
| New source awaiting verified intake | `data/raw_specs/` |
| Verified source archive | `data/docs_embedded/` |
| Disposable isolated intake artifacts | `data/staging/functional_specs_v5_intake/` |
| Complete immutable release-candidate stage | `data/staging/functional_specs_v5/` |
| Embedding reuse cache | `data/cache/embeddings/` |
| Persistent local Qdrant state | `data/qdrant_local/` |
| Stable active lexical artifacts after promotion | `data/indexes/functional_specs_v5/processed/` |

These are mutable/generated data artifacts and must remain excluded from Git.
The source manifest, hashes, evaluation reports, SME decisions, and activation
decision provide the audit trail.

## Failure and recovery behavior

- If extraction, embedding, indexing, or exact verification fails, the affected
  DOCX remains in `data/raw_specs/`. Fix the cause and rerun.
- If one filename exists in both `data/raw_specs/` and `data/docs_embedded/`, the
  master command fails before child stages—even during `--dry-run`.
- If a stage directory or target collection exists, choose a new versioned name.
  Do not delete/recreate or append to it.
- If cached vectors conflict for the same cache key, preserve both conflicting
  artifacts for diagnosis. Do not silently choose one or use
  `--replace-existing-embedding-artifacts` as routine ingestion behavior.
- `--rebuild-qdrant` is intentionally unsupported for embedded local Qdrant.
- A failed release candidate must not change `.env`; the active v4 pair remains
  the rollback baseline.
- Successful point counts alone are insufficient: stale, duplicate, wrong-ID,
  wrong-payload, or wrong-schema points can still exist.

Conversation summaries and prior successful answers are never indexing proof.
Use current manifests, exact verification, retrieval traces, citations, and SME
evaluation evidence.
