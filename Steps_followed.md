## Step 134 — Preserve historical evidence for multi-part current-state questions

### Python/code

```python
permitted_releases = {effective_release, *plan.referenced_release_labels}
scoped = [
    result
    for result in results
    if _normalized_payload_release(result) in permitted_releases
]
```

For a current-state query, textual release mentions are now retained as
historical evidence references, not retrieval filters. After broad retrieval,
the planner scopes evidence to the latest retrieved release plus any explicit
historical releases named in the question. `effective_release_label` still
states the latest deployed state; `referenced_release_labels` makes the
additional historical scope visible in the trace.

### Production interpretation

This supports questions such as “R2 PDF format and current T-1 name” with R2
and R24 evidence in one grounded answer. It does not interpret every prose
release name as an API filter; callers that need a single release retain the
explicit request `release_label` authority.

### Failure-mode testing

The regression includes R1, R2, and R24 candidates. It proves R1 is excluded,
R2 baseline evidence is preserved, and R24 remains the effective current
release. This prevents repairing stale current answers by accidentally losing
the historical sub-question.

## Step 135 — Deterministic multi-scope and trace-contract verification

### Python/code

```powershell
& .\.venv\Scripts\python.exe -m pytest `
  tests/test_temporal_query.py `
  tests/test_retrieval_router.py `
  tests/test_answer_orchestration_service.py `
  --basetemp C:\tmp\fdd-v4-multiscope -p no:cacheprovider
```

### Result and production interpretation

Result: `18 passed`. Tests cover current-state historical references,
multi-part R2/R24 scope preservation, explicit request filters, dense/lexical
candidate lanes, and trace metadata. They prove the deterministic temporal and
observability contracts, not live retrieval relevance or LLM answer quality.

### Failure-mode testing

The test uses an unrelated R1 candidate to confirm that preserving named R2
evidence does not disable release bounds. A malformed or missing raw lane is
still represented as an empty lane summary rather than silently inferred.

## Step 136 — Reviewed two-case v4 replay manifest and no-cost preflight

### Python/code

```powershell
& .\.venv\Scripts\python.exe scripts\run_fdd_grounded_eval.py `
  --dry-run `
  --eval-file data\evaluations\fdd_v4_current_state_replay_20260805.jsonl `
  --collection-name functional_specs_v4 `
  --lexical-artifact-directory data\staging\table_context_v1_retry1\processed
```

### Result and production interpretation

The dedicated manifest contains only the two SME-approved current-state
regressions and preserves the 30-case source manifest as a draft baseline. The
preflight reported `cases=2`, `reviewed=2`, and `draft=False`. This makes the
next paid run eligible to produce release-gate evidence rather than a draft
result.

### Failure-mode testing

An earlier dry run of the source manifest reported `reviewed=0` and
`draft=True`; it was not used as a quality gate. The dedicated manifest fixes
review governance without silently changing the original draft cases. The dry
run made no OpenAI, Qdrant, trace, or configuration changes.

### Batch gate

Steps 134–136 are complete. Keep v2 live. The next action is a single paid v4
run of the reviewed two-case manifest, followed by trace inspection and SME
review. Do not activate v4 from a structural pass alone.

### Batch interview evaluation

Pass. The user correctly separated current R24 state from historical R2
evidence, justified excluding unrelated R1 evidence, understood lane-level
diagnostics and the limits of focused tests, preserved the reviewed replay as a
separate release-gate artifact, and required semantic SME inspection beyond
structural checks.

Precision correction: `referenced_release_labels` contains releases explicitly
mentioned in the current-state query. It does not describe releases discovered
in retrieved evidence. Retrieved candidates determine the effective latest
release, while an explicit API/request `release_label` is a hard eligibility
filter.

### Gate

Steps 134–136 are accepted. Keep v2 live. Await explicit approval for the paid
two-case v4 replay; the run will make query-embedding and answer-generation API
calls, read paired v4 Qdrant/lexical evidence, and write local traces/report.

## Step 137 — Paid reviewed two-case v4 current-state replay

### Python/code

```powershell
& .\.venv\Scripts\python.exe scripts\run_fdd_grounded_eval.py `
  --eval-file data\evaluations\fdd_v4_current_state_replay_20260805.jsonl `
  --collection-name functional_specs_v4 `
  --lexical-artifact-directory data\staging\table_context_v1_retry1\processed `
  --output-file data\exports\evaluations\fdd-grounded-v4-current-state-replay-20260806.json
```

### Result

The explicitly approved run made two embedding and two answer-generation API
calls. `confusion-release-004` passed structurally; `lineage-r24-006` answered
but failed because its citations contained R24 only while the reviewed manifest
expected R2 and R24 release labels. The report records `draft=False`, paired v4
targets, two cases, and one structural pass. Application cost is recorded as
zero only because pricing is unconfigured; it is not billing evidence.

### Production interpretation and failure modes

One structural pass out of two does not authorize activation. The run changed
no `.env`, collection, lexical artifact, or live v2 state. Both durable traces
must be semantically reviewed, and a structural failure must be classified
before retrieval or benchmark changes.

## Step 138 — Candidate-lane and citation-entailment diagnosis

### Python/code

```powershell
$trace = Get-Content $tracePath -Raw | ConvertFrom-Json
$trace.retrieval_metadata.candidate_lanes
$trace.retrieval_results
$trace.answer_response.citations
```

### Diagnosis

For `lineage-r24-006`, R2 was eligible and appeared at dense ranks 6–7, but no
R2 unit survived final fusion/evidence selection. R24 was correctly resolved as
effective and directly supported the answer. SME judgment must determine
whether the question requires a separately supported R2 baseline claim or only
the requested current R24 identification.

For `confusion-release-004`, the answer correctly used R2 for PDF and R24 for
the current T-1 name. However, the direct PDF citation had an empty
`document_id`; a nearby R2 citation allowed the aggregate structural check to
pass even though that nearby unit did not state PDF. The run and pending SME
fields are recorded in
`data/evaluations/fdd_v4_current_state_replay_review_20260806.json`.

### Production interpretation and failure modes

The first case is not automatically a retrieval failure or benchmark defect;
that depends on the material SME expectation. The second proves that
case-level citation identity checks can hide unit-level citation entailment and
metadata gaps. No global RRF change is justified.

## Step 139 — Preserve lexical document identity through hybrid fusion

### Python/code

```python
document_id=str(unit.get("document_id") or fallback_document_id)

if payload_key not in state["payload"] or (
    existing_value in (None, "") and payload_value not in (None, "")
):
    state["payload"][payload_key] = payload_value
```

The lexical loader now propagates `document_id` from each unit or its artifact
fallback. Hybrid fusion fills missing or blank metadata from the other lane but
does not overwrite an existing nonblank identity.

### Production interpretation

Future citations preserve the exact FDD occurrence even when the selected unit
is lexical-only or one lane has incomplete metadata. This changes metadata
provenance only; embeddings, vectors, ranks, evidence text, and release logic
remain unchanged.

### Failure-mode testing

```powershell
& .\.venv\Scripts\python.exe -m pytest `
  tests\test_lexical_search.py tests\test_retrieval_router.py `
  tests\test_answer_contract.py tests\test_answer_orchestration_service.py `
  --basetemp C:\tmp\fdd-v4-citation-metadata -p no:cacheprovider
```

Result: `22 passed`. Tests prove artifact-level fallback identity, lexical
payload propagation, blank dense-metadata backfill from lexical evidence, and
unchanged citation/orchestration contracts.

### Batch gate

Steps 137–139 are complete. Keep v2 live. Await the nine-question interview and
SME verdicts for both runtime answers. Do not run another paid evaluation or
activate v4 before those decisions are recorded.

### Batch interview and SME evaluation

Pass. All nine answers meet the production rubric. The user distinguished
structural checks from activation evidence, provider billing from local cost
configuration, process-local v4 evaluation from live v2 configuration, and
case-level checks from unit-level citation entailment.

Precision note: the `lineage-r24-006` trace proves R2 entered the dense lane at
ranks 6–7 and was lost during fusion/final candidate selection before temporal
evidence packing. The SME approved the R24 current-state answer and removed the
non-material R2 citation requirement. The SME accepted
`confusion-release-004` for continued development with its historical citation
gap explicitly retained; the deterministic metadata repair addresses future
runs.

### Gate

Steps 137–139 are accepted. The user authorized v4 as the local development
baseline while retaining v2 for rollback. This authorization does not assert
production readiness.

## Step 140 — Activate paired v4 local development configuration

### Python/code

```env
QDRANT_COLLECTION_NAME=functional_specs_v4
PROCESSED_DIR=data/staging/table_context_v1_retry1/processed
```

Both settings were changed together so hybrid retrieval cannot mix v4 dense
vectors with the previous lexical generation. The reviewed replay manifest now
requires only the material R24 claim for `lineage-r24-006`. The runtime review
ledger records both SME decisions and the accepted development risk.

### Verification

```text
effective_collection=functional_specs_v4
effective_processed_dir=data/staging/table_context_v1_retry1/processed
lexical_units=937
manifest_status=verified
manifest_qdrant_verified_records=937
functional_specs_v4_points=937
functional_specs_v2_points=579
full_regression=372 passed
```

### Failure-mode testing

The initial full run found one stale test fixture that supplied a headerless
model response while expecting citation-validation behavior. The fixture now
uses `DECISION: ANSWER` with an invalid citation, reaching the intended
fail-closed citation branch. The targeted test passed `2/2`; the complete suite
then passed `372/372`, with one existing non-failing upstream Starlette
TestClient/HTTPX deprecation warning.

### Production interpretation

v4 is now the baseline loaded by newly started local API/UI processes. v2
remains intact as the rollback collection. This is a development cutover, not
evidence for authentication, capacity, live-model stability, operational
supervision, or production deployment readiness. Any already-running FastAPI
or Streamlit process must be restarted to reload `.env`.

### Gate

Step 140 is complete. Continue feature development on v4. Preserve v2 and do
not delete earlier generations or claim production readiness.

## Step 141 — Separate mutable ingestion output from active lexical retrieval

### Python/code

```python
ingestion_output_dir: Path = Field(
    default=ROOT_DIR / "data" / "processed",
    alias="INGESTION_OUTPUT_DIR",
)
```

`scripts/run_ingestion_pipeline.py` now writes raw, chunked, and
retrieval-ready artifacts to `settings.ingestion_output_dir`. `PROCESSED_DIR`
remains the active lexical retrieval input used by the API, readiness checks,
deployment preflight, and query/evaluation scripts.

### Production interpretation

Future ingestion can build mutable work artifacts without writing directly
into the serving v4 lexical generation. Activation becomes an explicit
promotion operation instead of an accidental side effect of ingestion.

### Failure-mode testing

Added `tests/test_ingestion_pipeline_script.py`. It configures different active
and ingestion directories and proves all three ingestion writers use only
`INGESTION_OUTPUT_DIR`. This catches future regression that could partially
overwrite a serving lexical index after a failed ingestion.

## Step 142 — Promote v4 lexical artifacts to a stable runtime path

### Python/code

```env
QDRANT_COLLECTION_NAME=functional_specs_v4
PROCESSED_DIR=data/indexes/functional_specs_v4/processed
INGESTION_OUTPUT_DIR=data/processed
```

The 24 processed v4 artifacts were copied from the immutable stage to
`data/indexes/functional_specs_v4/processed`. Every relative filename, size,
and SHA-256 hash matched. The original
`data/staging/table_context_v1_retry1` directory and its historical manifest
remain unchanged.

### Production interpretation

The runtime path now expresses the promoted collection generation rather than
the operational retry that built it. Staging provenance and serving identity
remain separate and auditable. `data/indexes/*` is ignored by Git because it is
local generated state, like local Qdrant storage.

### Failure-mode testing

```text
promoted_files=24
sha256_verification=passed
active_lexical_units=937
collection=functional_specs_v4
active_lexical_dir=data/indexes/functional_specs_v4/processed
ingestion_output_dir=data/processed
focused_tests=28 passed
```

The focused suite covers ingestion isolation, readiness, deployment preflight,
query search, answer smoke tests, and staged evaluation targeting. The existing
full suite had already passed `372/372` before this path-boundary change.

### Batch gate

Steps 140–142 are complete. v4 remains the local development baseline using a
stable promoted runtime path; v2 and the original v4 staging generation remain
available. Restart already-running API/UI processes to load the new path.

### Interview evaluation for Steps 140–142

The user answered all nine questions satisfactorily. The answers correctly
covered paired vector/lexical generation switching, the distinction between
local activation and production readiness, v2 rollback preservation, ingestion
isolation, immutable staging provenance, SHA-256 promotion limits, and Git
exclusion of generated indexes. The remaining production gate is broader than
artifact integrity: security, capacity, live configuration, operational
supervision, and rollback evidence are still required.
