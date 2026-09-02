# FDD generation

Put each new, reviewed `.docx` FDD in:

```text
data/raw_specs/
```

For a full-replacement document stream such as REST API Services, retain prior
versions in the archive and add the exact stream key to
`config/fdd_document_lineage.toml` only after review. Direct embedded `.xlsx`
attachments are extracted as linked workbook/sheet/row evidence; do not place
the Excel file separately in `data/raw_specs/`.

Before OpenAI is called, the launcher path performs an exact local embedding
input preflight. All source units are bounded to 6,000 UTF-8 bytes; oversized
paragraphs, tables, and spreadsheet cells are deterministically split first.
If this preflight fails, fix and locally verify the chunking contract before
requesting authorization for another paid run.

Run these stages one at a time from the repository root. Use a new generation
name; never reuse an existing stage directory or Qdrant collection.

```powershell
.\scripts\run_fdd_generation.ps1 -Generation functional_specs_v7 -Stage prepare
.\scripts\run_fdd_generation.ps1 -Generation functional_specs_v7 -Stage embed-index
.\scripts\run_fdd_generation.ps1 -Generation functional_specs_v7 -Stage evaluate
.\scripts\run_fdd_generation.ps1 -Generation functional_specs_v7 -Stage activate
```

`embed-index` and `evaluate` ask the operator to type `APPROVE` before any
operation that might send internal evidence to OpenAI or incur embedding cost.
`activate` first performs a no-write preflight, then requires the exact typed
confirmation `ACTIVATE functional_specs_v7`. On confirmation it:

- validates the verified stage manifest and its exact Qdrant verification;
- copies the matching lexical artifacts to `data/indexes/functional_specs_v7/`;
- verifies their SHA-256 directory identity;
- atomically updates `QDRANT_COLLECTION_NAME`, `PROCESSED_DIR`, and
  `FDD_GENERATION` in `.env` (and `RETRIEVAL_INDEX_PATH` when present); and
- writes immutable activation evidence under `data/exports/activations/fdd/`.

It does **not** restart processes. Restart FastAPI and Streamlit if they are
running. In Codex Desktop, toggle the MCP server off and on so its
Desktop-owned child process loads the new `.env` values. The prior generation
remains retained for rollback.

After a successful ingestion, source documents are archived in
`data/docs_embedded/`. The candidate remains under
`data/staging/functional_specs_v7/` until the existing evaluation, SME,
readiness, promotion, and rollback gates are complete.

For the full artifact contract and manual promotion procedure, see
[Steps_for_FDD_Ingestion.md](Steps_for_FDD_Ingestion.md).
