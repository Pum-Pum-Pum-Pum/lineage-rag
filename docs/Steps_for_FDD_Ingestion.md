& .\.venv\Scripts\python.exe -m uvicorn app.api.main:app `
  --host 127.0.0.1 --port 8000 --reload

& .\.venv\Scripts\python.exe -m streamlit run app/ui/streamlit_app.py `
  --server.address 127.0.0.1 --server.port 8501

# Add FDD Documents to the RAG System

Use this workflow for a reviewed, deployed FDD. The master command processes
every `.docx` file currently in `data/raw_specs/` and archives only documents
whose exact vectors have been verified in Qdrant.

## 1. Put FDDs in the input folder

Copy one or more reviewed FDD DOCX files into `data/raw_specs/`.

Do not copy a filename that is already present in `data/docs_embedded/`. The
master command rejects that duplicate before it runs ingestion or embeddings;
rename only when it is genuinely a distinct, reviewed FDD with its own source
identity.

```powershell
Copy-Item -LiteralPath 'C:\approved-fdds\FS_ASNB_R25_Teller_Change.docx' `
  -Destination 'data\raw_specs\FS_ASNB_R25_Teller_Change.docx'
```

The file name must contain a numeric release label such as `R25`. `docs/`
contains runbooks; it must not be used as a source-document folder.

The system keeps three different identities: `document_family` groups related
FDDs across releases, `release_label` identifies the release (for example
`R21`), and `document_id` is the full filename without `.docx`. Multiple FDDs
can therefore belong to R21 without sharing a citeable document identity.

## 2. Preview the batch first

From the project root, prepare the locked environment once, then preview the
files and commands. This does not call OpenAI, change Qdrant, or move files.

```powershell
uv sync --locked
uv run --locked python scripts/master_ingestion_embedding_docs.py --dry-run
```

## 3. Run the complete ingestion batch

```powershell
uv run --locked python scripts/master_ingestion_embedding_docs.py
```

The command runs these existing stages in order:

1. DOCX extraction, normalization, chunking, and retrieval-artifact creation.
2. Full-document OpenAI embeddings for each FDD, sequentially. It sends at
   most 64 unique uncached content units in each API request. Identical chunk
   text reuses one vector, while keeping separate evidence records.
3. Qdrant indexing.
4. Exact Qdrant verification for every embedding artifact.
5. Move verified DOCX files from `data/raw_specs/` to `data/docs_embedded/`.

To use a smaller bounded OpenAI request size, for example during a cautious
first batch, run:

```powershell
uv run --locked python scripts/master_ingestion_embedding_docs.py --request-batch-size 32
```

The size is **unique content units per API request**, not documents. Lower
values reduce per-request failure blast radius but create more API requests and
can increase latency.

## Expected local artifacts

| Result | Location |
| --- | --- |
| Source FDD waiting for processing | `data/raw_specs/` |
| Inspectable extraction and chunk artifacts | `data/processed/` |
| Embeddings and metadata | `data/cache/embeddings/` |
| Persistent local vectors | `data/qdrant_local/` |
| Successfully verified source FDD archive | `data/docs_embedded/` |

## Failure behavior

If ingestion, OpenAI embedding, Qdrant indexing, or exact-Qdrant verification
fails, the master command stops and does **not** archive the affected source
DOCX files. Fix the reported problem and rerun; deterministic embedding cache
keys and Qdrant point IDs make a safe rerun possible.

If the same filename appears in both `data/raw_specs/` and
`data/docs_embedded/`, the command also stops before every child stage,
including `--dry-run`. This prevents accidental duplicate ingestion and repeat
embedding cost. Remove the accidental raw copy or investigate the archived
source before retrying; never overwrite the archived file.

### Explicit recovery for a duplicate-embedding cache conflict

If the command reports `Conflicting cached embeddings found for cache_key`, do
not delete files or archive the source DOCX. The embedded local Qdrant client
does not reliably clear old points after delete-and-recreate, so do **not** use
`--rebuild` or `--rebuild-qdrant`.

First, choose a new versioned collection name and set it in your local `.env`:

```text
QDRANT_COLLECTION_NAME=functional_specs_v2
```

Restart the API/UI after changing this setting. Then preview the recovery:

```powershell
uv run --locked python scripts/master_ingestion_embedding_docs.py `
  --dry-run --replace-existing-embedding-artifacts
```

After reviewing scope, run the same command without `--dry-run`.

This explicitly quarantines the selected prior embedding artifact outside the
active cache rather than deleting it, regenerates the selected document's
embeddings with duplicate-content deduplication, and indexes every active
embedding record into the new collection. The old collection remains untouched
for investigation and rollback. Do not point the API/UI to the new collection
until exact verification has passed.

Do not use a conversation summary as evidence that a document was indexed.
Confirm the command output, local artifacts, Qdrant verification, and grounded
answer citations.
