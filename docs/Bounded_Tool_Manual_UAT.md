# Bounded-tool manual retrieval UAT

This local UAT checks which FDD, code, and reviewed-lineage evidence the bounded
tools select. It does not call OpenAI, generate an answer, or expose a new API/UI
route.

## Run one question

Use the project virtual-environment interpreter from the repository root:

```powershell
& .\.venv\Scripts\python.exe scripts\run_local_bounded_tool_uat.py `
  --mode combined `
  --question "How is batch transaction data sent to FlagRight according to the documents and the visible custom implementation?" `
  --limit 8 `
  --output-file data\exports\evaluations\manual-uat-001.json `
  --acknowledge-internal-evidence-output
```

Valid modes are `fdd`, `code`, and `combined`. Choose a new output filename for
every run; the script refuses overwrite.

The acknowledgement is mandatory because the JSON output contains retrieved
internal FDD and PL/SQL source text. Keep these reports in approved local storage
and do not commit or share them outside the authorized boundary.

## Review each result

Inspect the `execution.outputs` entries and record:

```text
Question:
Mode:
SME verdict: accepted | retrieval_gap | wrong_source | needs_more_context
Expected FDD document, if material:
Expected code path/symbol, if material:
Were reviewed lineage edges used correctly?: yes | no | not_applicable
Rationale:
Required follow-up:
```

## Run the formal ten-case draft batch

The versioned batch reuses previously reviewed business questions but keeps the
new bounded-tool UAT verdicts in draft state until the packet is reviewed:

```powershell
& .\.venv\Scripts\python.exe scripts\run_local_bounded_tool_uat_batch.py `
  --manifest data\evaluations\bounded_tool_manual_uat_v1_draft.jsonl `
  --output-directory data\exports\evaluations\bounded-tool-uat-v1-local `
  --batch-report data\exports\evaluations\bounded-tool-uat-v1-local-batch.json `
  --review-packet data\exports\evaluations\bounded-tool-uat-v1-local-sme-review.md `
  --acknowledge-internal-evidence-output
```

Choose unused paths because the batch preflights every target and refuses to
overwrite an earlier run. Individual case reports contain source text; the batch
index and SME packet retain identities and checks without copying that text.

For `combined` mode, confirm that documentation and code evidence remain distinct.
A reviewed file-level mapping does not prove that every routine in the file
implements the FDD. Kernel behavior, unresolved dependencies, dynamic SQL, and
external schemas must remain qualified unknowns.

## Suggested first manual round

Use 15–20 realistic business questions split across:

- documented functionality;
- exact routine/package explanation;
- combined FDD/code behavior;
- likely impact locations;
- missing or hidden kernel behavior;
- unrelated-module confusion;
- ambiguous wording and safe refinement;
- questions whose expected evidence should not be present.

Do not change ranking from one surprising answer. Record the case first, confirm
the expected evidence with an SME, reproduce it locally, and then decide whether
the issue belongs to the benchmark, retrieval, lineage, evidence packing, or the
later answer-generation layer.

## Current boundary

The runner uses local lexical retrieval with a fixed caller-selected mode and
bounded plan. Automatic routing is disabled. A successful retrieval UAT does not
prove generated-answer quality, citation entailment, provider stability,
concurrency, or production readiness. API/UI exposure requires a separately
approved activation change.
