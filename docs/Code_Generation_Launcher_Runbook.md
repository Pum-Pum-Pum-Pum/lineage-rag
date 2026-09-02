# Custom-code generation

Put one complete custom-code snapshot here; do not supply only changed files:

```text
data/raw_code/fci-custom-r2/
|-- snapshot_request.json
`-- source/
```

Run the stages one at a time from the repository root:

```powershell
.\scripts\run_code_generation.ps1 -SnapshotRequest fci-custom-r2 -Stage intake-parse

.\scripts\run_code_generation.ps1 -SnapshotRequest fci-custom-r2 -Stage prepare-index `
  -DependencyReviewLedger data\exports\code_analysis\reviews\<snapshot-id>-dependency-review-ledger.json

.\scripts\run_code_generation.ps1 -SnapshotRequest fci-custom-r2 -Stage embed-index `
  -CollectionName code_custom_r2_v1

.\scripts\run_code_generation.ps1 -SnapshotRequest fci-custom-r2 -Stage evaluate

.\scripts\run_code_generation.ps1 -SnapshotRequest fci-custom-r2 -Stage activate
```

`embed-index` asks the operator to type `APPROVE` before it can send prepared
internal PL/SQL to OpenAI. `activate` is intentionally a checklist only: it
never edits `.env`, restarts services, or switches the active code collection.

The complete snapshot is immutably archived in `data/code_snapshots/`; derived
stages are written under `data/staging/`. New collections remain staged until
retrieval, citation, answer, SME, readiness, and rollback gates are approved.

For dense/hybrid code evaluation, supply the exact collection and reviewed
precomputed query vectors. Combined evaluation still requires its approved FDD
generation and reviewed FDD-to-code lineage artifact. See
[Steps_for_Code_Snapshot_Ingestion.md](Steps_for_Code_Snapshot_Ingestion.md)
for the full contract.
