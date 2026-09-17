# Coordinated Knowledge Update Runbook

This is the recurring release workflow for FDD retrieval, code retrieval, and
reviewed FDD-to-code lineage. It preserves the active release until the new
combination passes its gates. It does not connect to Oracle or implement
Text-to-SQL.

## Lifecycle

```text
FDD additive folder ─┐                         ┌─ reviewed lineage ─┐
                     ├─ prepare → build → SME ─┤                    ├─ promote → restart → verify
Complete code tree ──┘                         └─ final evaluations ┘
                         ^ paid-input approval       ^ final hash approval
```

There are three normal human checkpoints:

1. Approve the frozen uncached embedding inputs and a maximum USD amount.
2. Review the one or two evidence-bound Markdown packets shown by `status`.
3. Approve the exact promotion hash after all final gates pass.

The third checkpoint is a release decision, not an SME decision. A deferred
lineage item remains available as unreviewed candidate evidence; it never
becomes a confirmed implementation mapping.

## Source requirements

- `-Mode fdd` accepts an **additive folder** of new or revised `.docx` files.
  The coordinator merges it with the active archived corpus. Do not copy all
  existing FDDs merely to make a new generation.
- `-Mode code` accepts a **complete source tree**. Missing prior code is a
  deletion and stops for reconciliation.
- `-Mode both` uses both rules. The FDD stage is built first, so discovery can
  find a new FDD relationship to unchanged code.
- `-Mode review` is reserved for a review-only successor; use it only when no
  source or embedding contract changes. Unsupported review amendments stop
  rather than altering historical evidence.

The only selected code extensions remain `.sql`, `.spc`, `.prc`, and `.fnc`.
External source folders are read only. The workflow copies the selected input
to `data/knowledge_updates/<RunId>/intake/` and verifies hashes while copying.

## Commands

Choose an immutable, meaningful `RunId`, for example `Release_006`. It is a
release label, not an SVN revision.

```powershell
# FDD-only additive update
.\scripts\run_knowledge_update.ps1 -Action init -RunId Release_006 `
  -Mode fdd -FddGeneration functional_specs_v10 `
  -FddSourceDirectory 'C:\OFSS\FDD_Updates' `
  -Reviewer Pum -PricePerMillion 0.13 -PricingBasis 'OpenAI 2026-09-16'

# Code-only update (the source directory is a complete intended tree)
.\scripts\run_knowledge_update.ps1 -Action init -RunId Release_007 `
  -Mode code -CodeSourceDirectory 'C:\OFSS\Complete_Code' `
  -SvnRevision 73142 -ApplicationBuild 14.7 -Reviewer Pum `
  -PricePerMillion 0.13 -PricingBasis 'OpenAI 2026-09-16'

# Two-sided update
.\scripts\run_knowledge_update.ps1 -Action init -RunId Release_008 `
  -Mode both -FddGeneration functional_specs_v11 `
  -FddSourceDirectory 'C:\OFSS\FDD_Updates' `
  -CodeSourceDirectory 'C:\OFSS\Complete_Code' `
  -SvnRevision 73143 -ApplicationBuild 14.7 -Reviewer Pum `
  -PricePerMillion 0.13 -PricingBasis 'OpenAI 2026-09-16'

.\scripts\run_knowledge_update.ps1 -Action prepare -RunId Release_008
.\scripts\run_knowledge_update.ps1 -Action build -RunId Release_008 -MaxUsd 2.00
.\scripts\run_knowledge_update.ps1 -Action finalize -RunId Release_008
.\scripts\run_knowledge_update.ps1 -Action activate -RunId Release_008 -ServicesStopped
.\scripts\run_knowledge_update.ps1 -Action verify -RunId Release_008
.\scripts\run_knowledge_update.ps1 -Action status -RunId Release_008
```

`prepare` makes no OpenAI calls, Qdrant writes, activation, or changes to the
external source folders. It produces the exact cache-miss request. `build`
requests approval only if there are uncached inputs; a zero-miss release skips
that checkpoint. The recorded ceiling covers this workflow's ingestion calls,
not unrelated account activity or provider billing adjustments.

## Reviews and finalization

After `build`, run `status`. It prints every review packet that exists:

- `data/knowledge_updates/<RunId>/review.md` — changed/superseded FDD evidence
  and new-FDD-to-existing-code candidates.
- `data/code_updates/<RunId>-code/review.md` — complete-code dependency,
  affected caller/validation context, new-code lineage and regression items.

The adjacent HTML is read-only. Edit only `Decision`, `Rationale`, and
`Correction JSON` in the Markdown. Accepted carry-forward entries are already
bound to unchanged evidence. Changed support, a new conflicting FDD, deleted
implementation, changed caller, validation, or dependency invalidates a carry.

`finalize` imports decisions, builds the final lineage, runs the final FDD,
code, combined, boundary, cache-reuse, inventory and bounded local MCP gates
that apply to the mode, then writes `release_manifest.json` and a hash-bound
`promotion_request.json`. A failed required case is not bypassed by an average
pass rate. Read the report path printed by the error, correct evidence or
review decisions, and resume the same action.

## Recovery, no-op, replacements and rollback

Every run is resumable from `data/knowledge_updates/<RunId>/`. Closing
PowerShell loses no IDs, paths, approvals or successful batch receipts. Run:

```powershell
.\scripts\run_knowledge_update.ps1 -Action status -RunId Release_008
```

after an interruption; it prints the exact next action and review files.

- A duplicate source with identical content is a no-op and reuses its vector.
- A replacement or withdrawal must be explicit. Full-replacement policy keys
  supersede their older revision; supplementary FDDs remain additive.
  For a same-path document with changed bytes, pass a frozen replacement
  manifest at `init`; its form is:

  ```json
  {
    "schema_version": "knowledge_fdd_replacement_manifest_v1",
    "replacements": ["relative/path/to/the.docx"]
  }
  ```

  The manifest itself is hash-bound input and cannot be substituted on resume.
- A source move requires unambiguous matching content or a reviewed mapping.
  Splits, merges, case-insensitive collisions, missing attachments and parser
  coverage failures stop rather than guessing.
- An unknown paid request outcome is held for reconciliation. It is not resent.
- A no-op release with no review or technical change completes without a new
  collection or promotion.
- Rollback restores the recorded prior compatible FDD selection and disables
  code modes under the existing safety policy:

```powershell
.\scripts\run_knowledge_update.ps1 -Action rollback -RunId Release_008 -ServicesStopped
```

Restart and verify that rollback as well. Do not use legacy FDD-only activation
once code/combined serving is enabled; it cannot establish a compatible joint
lineage combination.

## Scenario checklist

The complete maintained [30-scenario matrix](Knowledge_Update_Scenario_Matrix.md)
covers the following release situations:

| IDs | Scenario family | Required outcome |
|---|---|---|
| 1–4 | no-op, FDD-only, code-only, two-sided | correct source policy and cache reuse |
| 5–9 | revised, replacement, withdrawal, duplicate, move | explicit identity/review decision |
| 10–14 | Excel-only, attachment failure, caller-only, overload, table behavior | complete evidence or actionable stop |
| 15–19 | new FDD/old code, new code/old FDD, conflicts, deferred, undocumented | no invented approved mapping |
| 20–24 | cache/model migration, budget, quota, unknown outcome, crash | no duplicate paid send or unsafe resume |
| 25–30 | stale review, corrupted artifact, store lock, concurrent promotion, restart and rollback | preserved evidence and no mixed serving state |

Unknown cases are unsupported until added as a new deterministic fixture and
recovery rule. Historical R3/R4/Release_005 documentation and low-level
launchers remain valid diagnostic evidence; this runbook is the preferred
recurring-update surface.
