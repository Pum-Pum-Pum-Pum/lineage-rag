# FDD generation

Put each new, reviewed `.docx` FDD in:

```text
data/raw_specs/
```

Run these stages one at a time from the repository root. Use a new generation
name; never reuse an existing stage directory or Qdrant collection.

```powershell
.\scripts\run_fdd_generation.ps1 -Generation functional_specs_v6 -Stage prepare
.\scripts\run_fdd_generation.ps1 -Generation functional_specs_v6 -Stage embed-index
.\scripts\run_fdd_generation.ps1 -Generation functional_specs_v6 -Stage evaluate
.\scripts\run_fdd_generation.ps1 -Generation functional_specs_v6 -Stage activate
```

`embed-index` and `evaluate` ask the operator to type `APPROVE` before any
operation that might send internal evidence to OpenAI or incur embedding cost.
`activate` is intentionally a checklist only: it never edits `.env`, restarts
services, or changes the active generation.

After a successful ingestion, source documents are archived in
`data/docs_embedded/`. The candidate remains under
`data/staging/functional_specs_v6/` until the existing evaluation, SME,
readiness, promotion, and rollback gates are complete.

For the full artifact contract and manual promotion procedure, see
[Steps_for_FDD_Ingestion.md](Steps_for_FDD_Ingestion.md).
