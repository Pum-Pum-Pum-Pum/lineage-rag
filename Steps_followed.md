# Active progress record

The complete Phase 2 implementation history for Steps 150-183 is archived in
`Steps_followed_phase2.md`.

## Current handoff after Step 183

- Explicit `fdd`, `code`, and `combined` knowledge-mode API contracts exist.
- Knowledge mode remains separate from dense, lexical, or hybrid retrieval mode.
- Combined mode reuses one query embedding while retrieving and ranking the FDD
  and code lanes independently with separate evidence and citations.
- Mode-aware readiness and deterministic feature-flag rollback are implemented.
- Activation-readiness report:
  `data/exports/evaluations/code-combined-activation-readiness-v3-20260822.json`
- Readiness identity:
  `654c3c7907f97755f8a08c6479f2c4470c1dd62a541d0f23fb37ebdb79488d93`
- Readiness result: **9/9; activation-ready; activation not performed**.
- `CODE_MODES_ENABLED` remains `false`; `.env` was not changed.
- Latest complete regression result: **556 passed**, with one existing
  non-failing Starlette/HTTPX deprecation warning.

The next batch selected for implementation was Steps 184-186: Streamlit
knowledge-mode selection, lane-specific evidence rendering, and fail-closed UX.

## Step 184 - Wire explicit knowledge modes through durable conversations

Extended `ConversationMessageRequest` with validated `knowledge_mode` and
`analysis_kind` fields, then forwarded both through the conversation endpoint
into the existing `QueryRequest` contract. The typed UI client now requests
readiness for the exact selected knowledge mode.

The code/combined runtime receives bounded conversation memory for reference
resolution. It enriches the single shared retrieval query and includes an
explicit prompt instruction that conversation memory is not source evidence and
must never support or receive citations.

```python
class ConversationMessageRequest(BaseModel):
    content: str
    knowledge_mode: Literal["fdd", "code", "combined"] = "fdd"
    analysis_kind: Literal["explanation", "impact_analysis"] = "explanation"

readiness = api.get_readiness(knowledge_mode=request.knowledge_mode)
if readiness.knowledge_mode != request.knowledge_mode:
    raise UiApiError(code="readiness_mismatch", ...)
```

Production interpretation: a Streamlit selection now reaches the authoritative
runtime lane instead of being silently dropped by the conversation API. A
mode-mismatched readiness response fails closed. Failure testing verified field
forwarding, mode-specific readiness, and bounded context propagation without an
external API call.

## Step 185 - Add lane-aware Streamlit controls and evidence rendering

Added explicit sidebar controls for Functional documents, Visible custom code,
and Documents + custom code. Explanation and impact-analysis intent are exposed
only where meaningful. FDD-only request filters are hidden for code/combined
modes because the current extended runtime does not honor those per-request
filters; displaying them would create a false control.

The evidence panel now reports the global requested-claim state, related-context
state, independent combined-section states, `[F#]` document citations, and
`[C#]` code citations with snapshot, path, symbol, and exact line ranges.

```python
if response.combined_sections:
    for name, section in response.combined_sections.items():
        marker = "SUPPORTED" if section.status == "answered" else "REFUSED"

for index, citation in enumerate(response.code_citations, start=1):
    render(f"C{index}: {citation.source_path}:{citation.start_line}-{citation.end_line}")
```

Production interpretation: users can distinguish documented intent from visible
implementation and can see when only one section is supported. Hidden kernel,
dynamic SQL, and external-schema limits remain explicit. Failure-mode UX directs
users to FDD mode when code modes are disabled instead of silently falling back.

## Step 186 - Verify disabled-mode UX and reversible rollback behavior

Added deterministic tests for explicit mode payloads, conversation forwarding,
mode-aware readiness, readiness-identity mismatch, lane-rendering contracts, and
a false -> true -> false feature-gate sequence. Only the enabled middle request
may reach message submission.

```python
with pytest.raises(UiApiError):
    run_code_turn()          # disabled
assert run_code_turn()      # enabled
with pytest.raises(UiApiError):
    run_code_turn()          # rolled back
assert api.submit_count == 1
```

Focused verification passed **62/62**. The complete regression suite passed
**559 tests** with the one existing non-failing Starlette/HTTPX deprecation
warning. Python Ruff was not installed in the project environment, so no
dependency was added; compilation and whitespace checks are recorded separately.

Production interpretation: the serving control is reversible in the tested
single-process contract. This does not prove multi-worker propagation, in-flight
request behavior, live answer quality, or disaster recovery. `CODE_MODES_ENABLED`
remains `false`; no `.env` change, service restart, paid API call, Qdrant write,
or activation occurred.

## Step 187 - Define an immutable activation and approval contract

Added frozen Pydantic contracts for an activation request, a separate approval,
preflight checks, and the config-switch result. The request binds the exact FDD
and code collections, processed/index paths, code artifact and lineage hashes,
the accepted readiness report, and a deterministic hash of the runtime query,
readiness, conversation, orchestration, and schema source files.

```python
request = build_activation_request(
    settings=settings,
    readiness_report_path=readiness_report,
    requested_by="AIAgentSmith",
)
assert request.status == "pending_approval"
assert request.target_configuration["CODE_MODES_ENABLED"] == "true"
```

Production interpretation: activation now identifies exactly what is proposed;
directory order and implicit latest files are not authority. Request and approval
hashes detect accidental or silent changes, but they are integrity bindings—not
user authentication or cryptographic signatures. A real approver remains an
organizational control.

Failure testing caught and fixed inconsistent datetime canonicalization that
initially made an unchanged request fail its own identity verification.

## Step 188 - Produce a fail-closed offline activation preflight

Implemented local preflight checks for request integrity, readiness-report byte
identity, runtime configuration drift, safe disabled start state, and a matching
approval. Reports contain no API key names or secret values.

```python
preflight = evaluate_activation_preflight(
    request=request,
    settings=settings,
    readiness_report_path=readiness_report,
    approval=None,
)
assert preflight.ready_to_apply is False
```

Generated pending artifacts:

- `data/exports/activation/code-modes-activation-request-20260822.json`
- `data/exports/activation/code-modes-preflight-pending-20260822.json`
- request identity:
  `a892169b2d50825912b80ee0572055928e701bf3a68732fe20411650e0d0c145`
  (initial request, superseded by the Step 191 regeneration)

The real preflight passes four of six checks and intentionally fails both the
explicit-key and approval checks, so `ready_to_apply=false`. `.env` currently
relies on the safe application default and does not contain a switchable
`CODE_MODES_ENABLED=false` entry. Production interpretation: evaluation success
cannot silently become serving authority, and the activation mechanism will not
invent or append a control key. Configuration or readiness drift also blocks the
switch and requires a new reviewed request.

## Step 189 - Add approval-bound atomic switch and rollback mechanics

Added scripts to record an explicit approval and to dry-run or apply one atomic
change to `CODE_MODES_ENABLED`. The switch requires a matching approved request,
the requested action, and a passing current preflight. It refuses missing or
duplicate `.env` keys, unexpected starting state, rejected/stale approvals, and
failed preflight.

```python
result = switch_code_modes(
    env_path=env_path,
    action="activate",
    request=request,
    approval=approval,
    preflight=preflight,
    apply=False,
)
assert result.applied is False
```

The apply path writes a same-directory temporary file, flushes and `fsync`s it,
then uses `os.replace`; it never prints the `.env` contents. Tests prove dry-run
non-mutation and a false -> true -> false file-level sequence while preserving an
unrelated secret line.

Production interpretation: this is an atomic configuration-file switch, not an
atomic multi-process deployment. It deliberately does not restart FastAPI or
Streamlit, drain in-flight requests, call readiness, or run a paid smoke test.
Those remain explicit activation operations. Focused activation tests passed
**5/5**. The exact final-state regression passed **564 tests** with the one
existing non-failing Starlette/HTTPX deprecation warning. Python compilation and
`git diff --check` also passed.

## Step 190 - Establish a durable explicit-disabled baseline

Added a dry-run-by-default initializer that appends only
`CODE_MODES_ENABLED=false` when the key is absent. It refuses an enabled value,
duplicates, and invalid values while preserving every unrelated `.env` line.

```python
result = initialize_disabled_baseline(env_path=Path(".env"), apply=False)
assert result.before == "missing"
assert result.applied is False
```

The apply path uses the same temporary-file, file-`fsync`, and atomic-replace
mechanism. It now attempts parent-directory `fsync` where the platform exposes
`O_DIRECTORY` and reports whether that durability step succeeded. On this
Windows filesystem it reported `parent_directory_fsynced=false`, making the
remaining durability boundary explicit rather than claiming it was guaranteed.

The initializer was run against the real `.env`: dry-run first, then apply.
`CODE_MODES_ENABLED=false` now appears exactly once. No `true` transition or
process reload occurred.

## Step 191 - Regenerate the approval-pending preflight

Added a reusable preflight command that evaluates an existing immutable request
without overwriting it and optionally consumes a separate approval. The final
request also binds the activation implementation source hash, so changes to the
switching mechanism invalidate the approval target.

The request and preflight were regenerated after the explicit disabled baseline:

- final request identity:
  `d1de15f2a9c83a33666bed88bd6ce7cac21d62fa464dec0796d6e6b5325be23d`
- preflight result: **5/6 passed**;
- sole failed check: `approval`;
- `ready_to_apply=false`.

Production interpretation: every technical prerequisite currently represented
by the offline preflight passes, but technical readiness still cannot authorize
activation.

## Step 192 - Define complete activation execution evidence

Added an immutable execution-evidence contract bound to the request and approval.
It records configuration application, process restart, effective runtime state,
separate code and combined readiness, smoke trace IDs, rollback owner, and
rollback rehearsal.

```python
evidence = build_execution_evidence(
    request=request,
    approval=approval,
    configuration_applied=True,
    service_restart_confirmed=True,
    effective_code_modes_enabled=True,
    code_readiness_passed=True,
    combined_readiness_passed=True,
    smoke_trace_ids=("trace-code", "trace-combined"),
    rollback_owner="operator",
    rollback_rehearsed=True,
)
```

`activation_complete` can become true only when every operational field passes
and the approval separately authorizes both paid smoke calls and internal-evidence
disclosure. Tests prove that activation-only approval remains incomplete even
when fabricated runtime fields are all true. Focused activation tests passed
**7/7** before the final regression run.

Production interpretation: the ledger is a validated evidence container, not a
source of truth for fabricated claims. Real process, readiness, trace, and
rollback evidence must be collected by the controlled activation operation.
The exact final-state regression passed **566 tests** with the one existing
non-failing Starlette/HTTPX deprecation warning. Python compilation and
`git diff --check` passed.

## Step 193 - Bind approval and pass the final activation preflight

Recorded the user's approval as a separate immutable artifact bound to request
`d1de15f2a9c83a33666bed88bd6ce7cac21d62fa464dec0796d6e6b5325be23d`.
The approval authorizes activation and rollback, two paid smoke requests, and
disclosure of their retrieved internal FDD/PLSQL excerpts.

```python
approval = build_activation_approval(
    request=request,
    approved_by="AIAgentSmith",
    paid_smoke_authorized=True,
    internal_evidence_disclosure_authorized=True,
)
```

The approval identity is
`f1515883c9903c6b50d66380ebdd4ad9ab83cd7668b97207620470dffd3f8694`.
The real preflight then passed **6/6**, and an approval-bound dry-run confirmed
the intended `false -> true` transition without mutation. Failure tests retain
the existing controls: stale/mismatched approval, configuration drift, an
unexpected start state, or an ambiguous `.env` still blocks activation.

Production interpretation: technical readiness and human authority were both
present for this exact request. Neither the approval nor the dry-run alone
claimed that the running processes had activated the feature.

## Step 194 - Apply activation, restart, and verify mode readiness

Applied the atomic `.env` transition, restarted local FastAPI and Streamlit, and
checked all requested modes before any paid smoke request.

```python
result = switch_code_modes(
    env_path=Path(".env"), action="activate",
    request=request, approval=approval,
    preflight=preflight, apply=True,
)
assert result.after == "true"
```

Observed runtime state before smoke:

- `/health`: healthy;
- FDD readiness: ready;
- code readiness: ready;
- combined readiness: ready;
- Streamlit: HTTP 200.

The switch reported `parent_directory_fsynced=false` on Windows, preserving the
previously documented crash-durability boundary. Production interpretation:
configuration and bounded dependency readiness passed, but semantic serving was
still conditional on both paid smoke cases. A stale process was deliberately
tested during rollback later; the port-binding conflict demonstrated why a file
change alone cannot prove effective runtime state.

## Step 195 - Run two paid smokes and fail closed with verified rollback

Added `scripts/run_code_modes_activation_smoke.py`, an approval-bound runner with
exactly two reviewed cases, no automatic retries, lane-specific support/citation
checks, and immutable report output. It refuses missing cost or disclosure
authority before any HTTP request. It now also records non-2xx response bodies
and status codes rather than raising before evidence persistence.

```python
report = run_smokes(
    client=client,
    base_url="http://127.0.0.1:8000",
    request=request,
    approval=approval,
)
assert report["automatic_retries"] == 0
```

Exactly two local smoke requests were attempted. The code-only request returned
HTTP 200 and produced grounded trace
`7055d845-2e79-433f-afc4-89c497004d42`: claim support was true, citations C2 and
C4 resolve to the exact `spSendBatchTxnEndData` specification/body ranges, the
query embedding used 24 tokens, and the answer call used 11,449 total tokens.
The combined request returned HTTP 400 and produced no durable answer trace.
Because the initial runner raised before preserving that response body, the
precise ValueError detail is not recoverable; the runner was corrected locally
for future attempts. No retry or third request was made.

The failure triggered the approved rollback. The first restart attempt exposed a
stale process still owning port 8000; exact port owners were then stopped and the
services restarted again. Final observed rollback state is:

- `.env`: `CODE_MODES_ENABLED=false`;
- FDD readiness: HTTP 200 and ready;
- code readiness: HTTP 503, explicitly disabled by configuration;
- Streamlit: HTTP 200;
- activation complete: false.

The local attempt record is
`data/exports/activation/code-modes-activation-attempt-20260822.json`; the
successful code trace remains under `data/exports/answer_runs/`. This proves the
rollback gate works and prevents a partial semantic success from becoming
serving authority. It does not establish the root cause of the combined 400;
diagnosis must occur without another paid request, followed by a new hash-bound
activation request because any runtime fix changes the approved contract.

Focused activation/smoke tests passed **11/11**. The full regression passed
**570 tests** with the one existing non-failing Starlette/HTTPX deprecation
warning. `git diff --check` passed; Ruff is not installed in the project virtual
environment.

## Step 196 - Diagnose the combined HTTP 400 without another paid call

Traced the local execution boundary from `generate_grounded_answer` through the
FastAPI query route. Combined answers required model output to parse as one exact
JSON/Pydantic contract. A parse or schema `ValueError` escaped generation, the
API converted it to HTTP 400, and orchestration wrote its durable answer trace
only after generation returned successfully.

```python
draft = CombinedAnswerDraft.model_validate(_parse_json_object(raw))
# Before this step, ValueError escaped and trace persistence was never reached.
```

The stored code-only trace and API access log prove the observed boundary: the
first request completed and persisted; the combined request returned 400 and no
combined trace exists. Because the original smoke runner discarded the HTTP
body, the exact malformed response and exact exception text cannot be recovered.
The diagnosis therefore establishes the failure class and trace-loss mechanism,
not the exact historical payload. No OpenAI call was made for diagnosis.

Production interpretation: structurally invalid model output is expected
nondeterminism at an LLM boundary and must be handled as an unsupported answer,
not as an untraceable client error.

## Step 197 - Convert invalid combined contracts into grounded safe refusals

Added `build_combined_contract_refusal` and changed paid combined-answer parsing
to catch only local answer-contract `ValueError`s after a successful model call.
Provider, transport, and empty-response failures remain exceptions; they are not
misrepresented as contract failures.

```python
try:
    answer = finalize_combined_answer(
        retrieval=retrieval,
        draft=CombinedAnswerDraft.model_validate(_parse_json_object(raw)),
    )
except ValueError as exc:
    answer = build_combined_contract_refusal(retrieval=retrieval)
    contract_valid = False
    contract_error = type(exc).__name__
```

The safe response sets `requested_claim_supported=false`, refuses documented,
implementation, and impact sections, returns no FDD/code citations, and explains
only that the generated contract could not be validated. It does not expose the
malformed output as functional evidence. Call metadata, prompts, evidence, raw
model output, `contract_valid=false`, and the error class can now reach the normal
local trace writer.

Production interpretation: users receive a semantically honest abstention with
HTTP 200, while operations retain enough evidence to diagnose provider/model
contract instability. This change affects runtime-bound source and therefore
invalidates the prior activation request; a new request will be required.

## Step 198 - Prove graceful failure and activation diagnostics offline

Added deterministic tests with a fake chat response containing malformed JSON.
They prove the response becomes a safe refusal, paid-call metadata remains
available for tracing, and the API returns HTTP 200 with an explicit unsupported
claim rather than HTTP 400.

```python
assert answer.requested_claim_supported is False
assert call["contract_valid"] is False
assert call["contract_error"] == "JSONDecodeError"
assert response.status_code == 200
```

The approval-bound smoke runner also records non-2xx status and response content
in future attempt reports and still performs no automatic retry. Tests cover
missing disclosure authority, exact two-request bounds, missing lane evidence,
and HTTP failure capture. Focused tests passed **26/26**. The full regression
passed **572 tests** with the one existing non-failing Starlette/HTTPX deprecation
warning. Python compilation and `git diff --check` passed. No OpenAI request,
embedding, Qdrant write, configuration switch, or service activation occurred.

The next activation path must regenerate readiness/request identities for the
changed runtime contract, receive a new approval, and repeat both controlled
smokes. Code/combined serving remains disabled meanwhile.

## Step 199 - Expand the hash-bound activation runtime surface

Replaced the private seven-file runtime tuple with the explicit
`ACTIVATION_RUNTIME_FILES` contract and added the files that governed the failed
live boundary:

- `app/fdd_code_lineage/paid_evaluation.py`;
- `app/fdd_code_lineage/combined_answer.py`;
- `scripts/run_code_modes_activation_smoke.py`.

```python
"runtime_contract_sha256": _canonical_sha256({
    name: _file_sha256(root_dir / name)
    for name in ACTIVATION_RUNTIME_FILES
})
```

Tests assert these files remain in the bound surface. A change to parsing,
refusal semantics, paid-call handling, or smoke validation now changes the
request identity. Production interpretation: approval covers the actual semantic
and operational gate, not merely the API entry point. Failure testing confirms
test request construction also fails if a required bound source is absent rather
than silently omitting it.

## Step 200 - Regenerate offline readiness after containment

Regenerated the activation-readiness artifact without an OpenAI call or runtime
switch. All **9/9** checks passed, including reviewed semantic evidence, corrected
evaluation contract, current/rollback FDD and code collections, explicit API
modes, mode readiness, orchestration, and rollback rehearsal.

```text
report_identity_sha256 =
cf563f5c2b958839465f3b2823b3c2425da9453f8f89220c161f20f45a3aae70
```

The artifact is
`data/exports/evaluations/code-combined-activation-readiness-v4-20260822.json`.
Production interpretation: existing reviewed quality evidence and retained
generations remain compatible with the containment change. This offline result
does not prove live-model stability or authorize activation.

## Step 201 - Create a new approval-pending request and reject stale authority

Created a new immutable activation request bound to readiness v4, the expanded
runtime contract, current configuration, collections, artifacts, and lineage.

```text
request_identity_sha256 =
36e3c1c244def6be37d8b1ff9ee02cbc3f072623a24e16b42f65af05761a0fd4
runtime_contract_sha256 =
cf9d3b045468bec994a0cf2e95a332c03b6ea9877d52415aab2f56689716fa32
```

Artifacts:

- `data/exports/activation/code-modes-activation-request-v2-20260822.json`;
- `data/exports/activation/code-modes-preflight-pending-v2-20260822.json`;
- `data/exports/activation/code-modes-preflight-stale-approval-v2-20260822.json`.

The pending preflight is **5/6**, failing only `approval`. Supplying the old
approval intentionally also remains blocked because it references request
`d1de15f...`, not the new `36e3c1c...` identity. This proves authorization cannot
carry across changed runtime semantics.

Focused tests passed **32/32**. The full regression passed **572 tests** with the
one existing non-failing Starlette/HTTPX deprecation warning. Code/combined mode
remains disabled; FDD readiness remains healthy. No paid call, embedding, Qdrant
write, service restart, configuration mutation, or activation occurred.

## Step 202 - Approve and preflight activation request v2

Recorded explicit approval for request
`36e3c1c244def6be37d8b1ff9ee02cbc3f072623a24e16b42f65af05761a0fd4`
under `AIAgentSmith`. The separate approval authorizes activation, rollback, two
paid smokes, internal FDD/PLSQL evidence disclosure, and the associated cost.

```python
approval = build_activation_approval(
    request=request,
    approved_by="AIAgentSmith",
    paid_smoke_authorized=True,
    internal_evidence_disclosure_authorized=True,
)
```

Approval identity:
`1870f7fba579009ecd934b117de120a31874f68c1a49f61deb60e37d0214cb86`.
The approval-bound preflight passed **6/6** and the switch dry-run proved the
intended `false -> true` transition without mutation. Production interpretation:
the previously missing authority was supplied for this exact expanded runtime
contract; it did not itself claim a live transition.

## Step 203 - Activate, restart exact services, and verify all modes

Applied the atomic feature-flag switch and replaced the exact Python processes
owning ports 8000 and 8501. The restart refused unexpected process types and
verified that the old listeners released both ports before launching replacements.

```python
result = switch_code_modes(
    env_path=Path(".env"), action="activate",
    request=request, approval=approval,
    preflight=preflight, apply=True,
)
assert result.after == "true"
```

Before the paid calls, `/health`, FDD readiness, code readiness, combined
readiness, and Streamlit all returned HTTP 200. Code and combined checks verified
the feature gate, exact code point count, reviewed lineage, FDD collection, and
retrieval artifacts. Production interpretation: the new process had loaded the
enabled state and its bounded dependencies were available, but answer-contract
behavior still required live smoke evidence.

## Step 204 - Preserve the second paid failure and roll back safely

Ran exactly two approval-bound HTTP smokes with zero automatic retries. The
immutable smoke report identity is
`3727857d6ef6f487f6ced3dc747e5cd0b3c130562c3315c03442795513ddf73b`.

The code smoke passed:

- trace: `c480b670-3fb1-4628-a4dd-c0faefda7086`;
- embedding tokens: 24;
- answer tokens: 11,355;
- contract valid: true;
- exact code citations: 2.

The combined smoke returned HTTP 200 but correctly failed the activation gate:

- trace: `93deeb26-824a-46f5-a2ce-2ad6fe34a1a5`;
- embedding tokens: 22;
- answer tokens: 22,738;
- retrieval: 8 FDD units, 8 code units, 3 reviewed lineage links;
- `requested_claim_supported=false`;
- all functional sections safely refused;
- citations returned: zero;
- contract diagnostic: `ValidationError`.

The raw response is now durably available in the restricted trace and reveals
the exact defect: the model returned `requested_claim_supported` as a section
object containing `status` and `text`, while `CombinedAnswerDraft` requires a
boolean. The system prompt is ambiguous because it lists the boolean field and
then says “Each value must be an object.” This is a prompt/schema alignment bug,
not a retrieval or lineage failure.

The failed smoke immediately triggered rollback. The exact active port owners
were stopped, disabled processes were restarted, and final checks prove:

- `.env`: `CODE_MODES_ENABLED=false`;
- health/FDD/UI: HTTP 200;
- code readiness: HTTP 503 with `disabled by configuration`;
- activation retained: false.

Production interpretation: the new containment path worked as designed—the
second paid failure became a traceable safe refusal instead of an HTTP 400—but a
safe refusal does not satisfy a positive combined-mode smoke. The stage is not
activated. No third call or retry was attempted. The next fix should remove the
prompt/schema contradiction and add deterministic exact-schema fixtures before
seeking another activation approval. No Python source changed during Steps
202-204, so the already completed **572-test** regression remains the code-state
verification; post-operation configuration, process ownership, readiness, trace
hashes, request count, and `git diff --check` were verified separately.

## Step 205 - Align the combined prompt with an enforced response schema

Removed the contradictory instruction that treated every top-level response
value as a section object. The combined prompt now states unambiguously that
`requested_claim_supported` is a JSON boolean and that only the four named
answer sections contain `status` and `text`.

```python
response_format = {
    "type": "json_schema",
    "json_schema": {
        "name": "combined_grounded_answer",
        "strict": True,
        "schema": CombinedAnswerDraft.model_json_schema(),
    },
}
```

The combined Chat Completions call now supplies this Pydantic-derived schema as
strict Structured Outputs and records the response-format kind plus canonical
schema SHA-256 in the answer trace. This follows OpenAI's documented contract
that `json_schema` Structured Outputs constrains model output to the supplied
schema. Code-only generation retains its existing text contract. Production
interpretation: the provider receives one machine-enforced contract instead of
having to infer types from prose, while local Pydantic validation remains a
defense-in-depth boundary. Failure mode: a provider/SDK that rejects the schema
fails the paid operation rather than silently falling back to unconstrained JSON.

## Step 206 - Add exact-schema and fail-closed regression coverage

Added deterministic checks for the strict schema, prompt alignment, trace schema
identity, malformed JSON, and the exact historical wrong-type shape where
`requested_claim_supported` is an object. Both malformed cases produce a safe
refusal with `requested_claim_supported=false` and no citations.

```python
assert schema["properties"]["requested_claim_supported"]["type"] == "boolean"
assert "Each value must be an object" not in COMBINED_SYSTEM_PROMPT
```

The readiness assessor now includes a `combined_structured_output_contract`
check, so future prompt/schema drift blocks readiness before paid execution.
Production interpretation: a local deterministic gate catches the previously
observed defect without disclosing evidence or consuming tokens. Failure tests
prove invalid JSON and schema-valid JSON with the wrong field type remain
distinguishable (`JSONDecodeError` versus `ValidationError`) while both fail
closed. Focused verification passed **16/16**.

## Step 207 - Rebuild readiness and activation authority offline

The full regression passed **574 tests** with the one existing non-failing
Starlette/HTTPX deprecation warning. Offline readiness then passed **10/10**,
including the new structured-output contract check:

```text
readiness identity = 8e598067ed06eadbf997a95986be6bac5c3a19dacf8eaf39d86016d916139775
```

Created a new immutable, approval-pending request:

```text
request identity = f927d16dde75bdf6ef3fc8d96ad07279ae22185b1ea6c3aea030e57b44692fff
runtime contract = 44218850a432d022dcb13af73600c00c28fb602b5c498f963c3e9865ed279c7d
```

Artifacts:

- `data/exports/evaluations/code-combined-activation-readiness-v5-20260822.json`;
- `data/exports/activation/code-modes-activation-request-v3-20260822.json`;
- `data/exports/activation/code-modes-preflight-pending-v3-20260822.json`;
- `data/exports/activation/code-modes-preflight-stale-approval-v3-20260822.json`.

The pending preflight is **5/6**, failing only approval. The prior v2 approval
was tested against v3 and rejected. Production interpretation: reviewed
knowledge artifacts remain usable, but changed prompt/schema behavior creates a
new serving contract that requires fresh authority. `CODE_MODES_ENABLED` remains
`false`; no OpenAI request, service restart, `.env` mutation, embedding, Qdrant
write, or activation occurred. Generated pytest runtime data is now explicitly
ignored under `data/test_runtime/`.

## Step 208 - Bind approval and complete activation preflight

Recorded `AIAgentSmith` approval for request
`f927d16dde75bdf6ef3fc8d96ad07279ae22185b1ea6c3aea030e57b44692fff`.
The approval separately authorizes activation, rollback, two paid smokes,
internal FDD/PLSQL evidence disclosure, and the associated OpenAI API cost.

```text
approval identity = 6a2d0d3c54639cbdf31400827028386c59f6a9b4f8c2c7d47bdb44512dd8a394
preflight = 6/6
```

The switch dry-run first proved the intended `false -> true` transition without
mutation. Production interpretation: the authorization is bound to the repaired
runtime contract and cannot be reused for another request. Failure testing still
rejects absent, stale, or hash-invalid approval before configuration changes.

## Step 209 - Activate and establish unambiguous runtime ownership

Applied the atomic `.env` transition to `CODE_MODES_ENABLED=true`. Initial
process startup discovered a stale FastAPI listener on port 8000 and duplicate
Streamlit ownership. The new FastAPI process could not bind, so readiness and
paid execution remained blocked.

The exact Python commands and parent/child process identities were inspected.
Only the verified FastAPI and Streamlit service trees were stopped. Both ports
were confirmed released before launching fresh virtual-environment services.
Final owners are:

```text
FastAPI  port 8000 -> PID 46412, parent PID 28968
Streamlit port 8501 -> PID 24324, parent PID 42728
```

Health, FDD readiness, code readiness, combined readiness, and Streamlit then
returned HTTP 200. Production interpretation: process ownership is part of the
activation evidence; merely starting a process does not prove it serves traffic.
Failure-mode handling prevented paid calls against a stale process that might
have loaded the prior disabled configuration.

## Step 210 - Pass both paid smokes and retain rollback capability

Ran exactly the two approval-bound cases with zero automatic retries. Both
passed:

```text
code trace     = 642a27b6-6408-48ab-b417-587fb4df43a7
code tokens    = 11,704
code citations = 5

combined trace     = 8726d6a7-2cfd-44a5-b751-fcfb06627268
combined tokens    = 22,342
FDD citations      = 5
code citations     = 6
```

The combined trace records `contract_valid=true`, `response_format=json_schema`,
and schema SHA-256
`3f31aa0e73dce41327b21dc8c4be916bc466ce5f2cc6f74cd5993cb4363e9e9d`.
Smoke report identity:
`4f15ed0ebbb55205cc73e94a0913545087f13c23cd723b4146b64c76aa8412ff`.

Activation execution evidence is complete with identity
`859d9499ed4d5e4d34b05a82cbc67eb2b5b892662d2e006f133a6f002300f306`.
Post-smoke health/readiness remains HTTP 200 and
`CODE_MODES_ENABLED=true`. A rollback preflight and dry-run passed without
changing the active state, proving the current request remains immediately
rollback-capable. The prior FDD/code generations are retained.

Production interpretation: the local code and combined capability is now
deliberately active and has passed the defined activation gate. This does not
prove production concurrency, long-duration availability, latency SLOs,
centralized identity/access controls, or disaster recovery. Failure policy
remains fail closed and apply the already authorized rollback if a required
runtime gate later fails. The previously completed **574-test** regression
remains the source-state verification; no Python source changed during this
activation batch.

## Step 211 - Add an explicit bounded FDD-search tool contract

Added the versioned `config/agentic_tools.toml` policy and typed explicit-plan
models under `app/agentic_tools`. The policy hash is:

```text
dcdd5b90790c77e913ddcfeaa619715311b8cdca9491d8b78a806472e5b789e2
```

The default contract allows at most three calls, eight results per call, and 16
total retrieved evidence units. `automatic_routing=false` is enforced as a
literal value. FDD mode exposes only `fdd_search`.

```python
plan = create_explicit_tool_plan(
    knowledge_mode="fdd",
    invocations=(("fdd_search", question, 8),),
)
policy.validate_plan(plan)
```

The FDD tool requires complete unit, document, family, release, source-kind, and
text identity before returning evidence. It consumes at most `limit + 1` items
even if a buggy retriever returns an endless iterator. Production
interpretation: the tool is a read-only adapter over the existing grounded FDD
retrieval lane; it does not grant an agent arbitrary search scope or make memory
authoritative. Failure tests reject automatic routing, mode/tool mismatch,
tampered plan identities, over-budget plans, and incomplete citation identity
before downstream claims are possible.

## Step 212 - Add a bounded custom-code search tool contract

Added a separate `code_search` tool that accepts an injected read-only code
retriever and returns the existing source-grounded `CodeEvidence` contract. Code
mode exposes only this tool.

```python
retrieval = search_runner(invocation.query, invocation.limit)
if retrieval.query != invocation.query:
    raise RuntimeError("Code retriever returned evidence for a different query")
```

The output preserves snapshot, path, symbol, exact line range, parser state,
conditional state, and source text. It caps returned units to the approved call
limit and does not perform embeddings, generation, writes, or patch creation.
Production interpretation: code evidence remains independently ranked and
citeable; the tool cannot silently substitute evidence retrieved for another
question. Failure testing verifies truncation and fails closed on query/result
identity mismatch rather than attaching unrelated code to the active plan.

## Step 213 - Add reviewed-lineage impact graph and explicit orchestration

Added a one-hop `impact_graph` tool. It creates FDD-to-code edges only from
`ReviewedLineageUse` records already produced by validated combined retrieval.
It also adds static dependency edges whose source lines overlap selected code
evidence, retaining resolution states such as `kernel_unavailable` instead of
claiming hidden behavior.

```python
execution = execute_explicit_tool_plan(
    plan=caller_authored_plan,
    policy=policy,
    handlers=handlers_by_invocation_id,
)
assert execution.trace.automatic_routing_used is False
```

Graph nodes and edges are capped during construction, not merely sliced after
building an unbounded graph. The executor runs only the caller-authored plan in
sequence, stops after the first blocked/failed call, and records policy, plan,
call, result-count, and evidence identities. Citeable outputs are kept separate
from the privacy-safer trace so source text need not be duplicated in operational
logs.

Production interpretation: this is bounded deterministic tool orchestration,
not autonomous agent routing. Missing reviewed lineage becomes an explicit
unknown, missing handlers become a blocked trace, and handler exceptions record
only their type before later calls are suppressed. Focused and related suites
passed **38/38**; the full regression passed **584 tests** with the existing
non-failing Starlette/HTTPX deprecation warning. No OpenAI call, Qdrant write,
API/UI route change, feature-flag change, or runtime activation occurred. The
existing local FDD/code/combined service remains active, but these new tools are
not yet exposed to users; deterministic tool-level evaluation is the next gate
before runtime integration.

## Step 214 - Define the bounded-tool evaluation contract

Created `app/agentic_tools/evaluation.py` and the six-case draft manifest
`data/evaluations/bounded_agentic_tools_v1_draft.jsonl`. The contract validates
unique case IDs, explicit review state, expected source identities, and the exact
tool sequence for each knowledge mode: FDD uses `fdd_search`, code uses
`code_search`, and combined uses `fdd_search`, `code_search`, then
`impact_graph`. Draft execution requires the explicit `--allow-draft` switch.

Production interpretation: previously reviewed answer questions are useful seeds,
but they do not automatically approve new tool-level expectations. Tool order is
part of the bounded execution contract because changing it can change the inputs
available to later tools. Failure tests cover duplicate IDs, invalid mode/tool
contracts, unreviewed cases, and missing draft authorization. The manifest
SHA-256 is `f38e19ec5942c8b24754fca304629b302a7c5378b5df2c68bf85e46949514aef`.

## Step 215 - Run local lexical evaluation against promoted artifacts

Added `scripts/run_bounded_tool_eval.py`. It loads the promoted FDD v5 lexical
artifacts, the prepared code artifact, and reviewed lineage locally, then runs the
fixed plans without OpenAI, query embeddings, or answer generation. The code path
uses the existing lexical retrieval implementation, for example:

```python
evidence = retrieve_code_evidence(query=case.question, mode="lexical", limit=limit)
```

The real-corpus draft result was **5/6 positive cases**. The FDD cases and both
other combined cases passed. `tool-combined-aml-batch-send-004` retrieved the
correct FDD, code path, and reviewed lineage, but the expected
`spSendBatchTxnEndData` symbol did not enter the combined top-eight code evidence;
the code-only case retrieved that exact symbol at rank eight.

Production interpretation: this isolates a combined evidence-selection or
ranking gap without spending money or disclosing source externally. It does not
measure dense/hybrid retrieval, generated answers, citation entailment, or user
usefulness. The failing reviewed expectation is preserved before any tuning so a
benchmark defect can be distinguished from a retrieval defect.

## Step 216 - Preserve the result, safety evidence, and SME review packet

The evaluator now writes no-overwrite JSON reports and source-minimized SME
review packets. The report identity is
`a7ee4d4bf0a04c7a8d8b5dff8db8849da2fca5088dbfbd2480fd953f9c45758a`.
Safety checks passed **5/5**: automatic routing stayed disabled, over-budget plans
and missing handlers failed before execution, operational traces omitted source
outputs, and external API calls remained zero. Because the cases are draft and
one positive expectation failed, `release_gate_passed=false`.

The packet is
`data/exports/evaluations/bounded-agentic-tools-v1-draft-sme-review-20260823.md`.
It records identities, observed results, and SME fields without copying source
text. Unit fixtures cover capped unresolved static dependencies; the real-corpus
run did not load the protected dependency-analysis directory, so it does not
claim real-corpus dependency precision/recall. Focused tests passed **15/15** and
the full regression passed **589 tests** with the existing non-failing
Starlette/HTTPX deprecation warning.

Production interpretation: the bounded tools remain offline and are not exposed
through the API or UI. No OpenAI call, Qdrant write, configuration activation, or
runtime restart occurred. SME review of the six cases, especially the combined
batch-send expectation, and real-corpus dependency evaluation remain gates.

## Step 217 - Import the bounded-tool SME decision safely

Added `app/agentic_tools/review.py` and
`scripts/import_bounded_tool_eval_review.py`. The importer parses each packet
section by stable case ID and accepts only unchanged `accepted` verdicts. It
requires the packet to contain the exact draft-manifest SHA-256 and evaluated
report identity. A correction or `needs_more_context` verdict fails closed and
requires a new manifest instead of silently rewriting an evaluated expectation.

Production interpretation: the user's chat confirmation is recorded as the
approval source and durable rationale, while the exact packet remains the review
artifact. This separates human acceptance from the deterministic result. Tests
cover CRLF packets, nonaccepted verdicts, corrected expectations, scope/hash
mismatches, and missing approval notes.

## Step 218 - Promote a separate reviewed manifest and ledger

Created `data/evaluations/bounded_agentic_tools_v1_reviewed.jsonl` without
modifying the evaluated draft manifest. All six cases now have
`review_status=reviewed` and `sme_reviewed=true`. Created the hash-bound ledger
`data/evaluations/bounded_agentic_tools_v1_review_20260823.json` under reviewer
`AIAgentSmith`.

The reviewed manifest SHA-256 is
`5f55f5ab9edd073370b835584b95479160bd11257e7cdcac435e3db7faba89a7` and the
ledger identity is
`454cc7a316f4c11ca6a4cbbcb373a9dfdb939bc8c2b689f322eacfc361a29f41`.
Both reviewed outputs refuse overwrite.

Production interpretation: review promotion changes approval state, not observed
retrieval behavior. In particular, SME acceptance of case 4 confirms that its
exact-symbol expectation is valid; it does not turn its structural failure into
a pass.

## Step 219 - Rerun the reviewed deterministic release gate

Reran the local lexical evaluator without `--allow-draft` and wrote
`data/exports/evaluations/bounded-agentic-tools-v1-reviewed-20260823.json`.
The reviewed result is **5/6 positive**, **5/5 safety**, all cases reviewed, and
`release_gate_eligible=false`. The reviewed report identity is
`a66c71b303c88439730e4b597d618f9313a1083b4a9df92b63d092a0290e8b1e`.

The evaluator's exit code 1 is the intended gate signal: the approved case
`tool-combined-aml-batch-send-004` still lacks `spSendBatchTxnEndData` in the
bounded combined code evidence. No external API calls occurred. Focused review
and tool tests passed **18/18**; the full regression passed **592 tests** with the
existing non-failing Starlette/HTTPX deprecation warning.

Production interpretation: bounded-tool API/UI exposure remains blocked. The
next change should target the reviewed combined evidence-selection gap and prove
the fix locally, while preserving code-only behavior, budgets, reviewed lineage,
trace privacy, and the five safety controls.

## Step 220 - Repair the reviewed combined evidence-selection gap

Candidate inspection proved that `spSendBatchTxnEndData` existed at lexical rank
10 for the natural combined question while the bounded output limit was eight.
Added a combined-only identifier-affinity reservation in
`app/agentic_tools/tools.py`. It tokenizes routine names and query wording,
normalizes bounded aliases such as `txn` to `transaction` and `sent` to `send`,
and may reserve one slot when an already retrieved candidate matches at least
three identifier terms more strongly than the weakest selected item.

```python
evidence = select_identifier_affinity_evidence(
    query=invocation.query,
    evidence=retrieval.evidence,
    limit=invocation.limit,
)
```

Production interpretation: the repair does not create evidence, increase the
eight-unit budget, alter embeddings, or change global dense/lexical/RRF weights.
Code-only behavior is unchanged. Failure tests cover weak/no affinity, bounded
replacement, stable leading order, invalid limits, and exact case-4 selection.

## Step 221 - Close the reviewed deterministic gate

Reran the unchanged reviewed manifest and wrote
`data/exports/evaluations/bounded-agentic-tools-v1-reviewed-selector-v2-20260823.json`.
The result is **6/6 positive**, **5/5 safety**, all cases reviewed, zero external
calls, and `release_gate_eligible=true`. The report identity is
`09e52e3a6ecfb2de8a9b8362a97eafc286e0549896d9cc535b767d74836059ec`.

Production interpretation: this closes the deterministic lexical-tool gate for
the six reviewed cases. It does not prove dense/hybrid behavior, generated
answers, broader-corpus recall, or production serving properties. The full suite
passed **595 tests** with the existing non-failing Starlette/HTTPX warning.

## Step 222 - Add a local manual retrieval-UAT boundary

Added `app/agentic_tools/uat.py`,
`scripts/run_local_bounded_tool_uat.py`, and
`docs/Bounded_Tool_Manual_UAT.md`. The runner accepts an explicit mode and
question, enforces the configured result budget, runs only the fixed local
lexical plan, and writes a no-overwrite identity-bound JSON report. It requires
`--acknowledge-internal-evidence-output` because citeable outputs contain internal
FDD/PLSQL source text.

The first case-4 UAT report is
`data/exports/evaluations/bounded-tool-manual-uat-case4-20260823.json`, identity
`689577f9bd8d7bf91fdf645a3f47c3b0eec61ef6ebc991c85e86aceaf97a5a02`.
It completed three fixed calls with 16 total evidence units and included
`spSendBatchTxnEndData` among exactly eight code units.

Production interpretation: manual retrieval UAT can now begin locally without
OpenAI cost or disclosure. No API/UI route, feature activation, service restart,
or automatic routing was added. Generated-answer UAT and any API/UI exposure
remain separate, approval-bound later steps.

## Step 223 - Define a formal ten-case manual retrieval-UAT contract

Added `ManualToolUatCase` and related batch models in
`app/agentic_tools/uat.py`, plus the draft manifest
`data/evaluations/bounded_tool_manual_uat_v1_draft.jsonl`. The ten cases reuse
questions whose business expectations were reviewed previously, while keeping
the new tool-level UAT state explicitly `draft`. The scope covers FDD-only, four
code cases, four positive combined cases, impact analysis, and an unavailable
hidden-kernel request.

Production interpretation: prior SME review is provenance for the source
question, not automatic approval of a new bounded-tool run. The contract records
expected source identities, reviewed-lineage requirements, and whether the tool
should provide evidence or a qualified unknown. Duplicate IDs, invalid limits,
and unexpected schema fields fail validation.

## Step 224 - Preserve a source-aware local UAT batch

Added `scripts/run_local_bounded_tool_uat_batch.py`. It preflights all target
paths, executes fixed local lexical plans, writes one no-overwrite source-bearing
report per case, then writes a source-minimized batch index and SME packet. The
explicit internal-evidence acknowledgement is mandatory. No query embedding,
answer generation, or external call occurs.

The first immutable batch produced **9/10** diagnostics. All positive cases
passed. The hidden-Java-kernel negative case retrieved useful nearby PL/SQL and
reviewed lineage but did not emit an explicit unavailable-boundary state. This
was preserved as a real contract failure rather than overwritten.

Production interpretation: retrieval of nearby visible code is not itself wrong,
but it cannot satisfy a request for an exact hidden Java method or defect line.
The tool graph must retain visible evidence and independently qualify the
unavailable requested boundary.

## Step 225 - Qualify unavailable kernel detail and prepare SME review

Added a bounded unavailable-boundary detector to the impact tool. It requires a
boundary term (`kernel`, `Java`, or `J2EE`), an unavailable-scope term (`hidden`,
`unavailable`, or `internal`), and an implementation-detail term before adding
the unknown. Ordinary visible-code questions remain unaffected.

The new immutable batch passed **10/10 diagnostics**, remained unreviewed, and
made **0 external calls**. Its identity is
`0ecd0b9856b920dc127d2d9c2d10eb878642e317298998646a0ececdd9482b61`.
The review packet is
`data/exports/evaluations/bounded-tool-uat-v1-kernel-qualified-20260824-sme-review.md`.
Focused tests passed **16/16** and the full regression passed **598 tests** with
the existing non-failing Starlette/HTTPX deprecation warning.

Production interpretation: the ten diagnostic results are not a release gate
until SME-reviewed. Paid-use permission has been acknowledged, but it does not
authorize disclosure of retrieved internal FDD/PLSQL excerpts. Paid grounded-
answer execution therefore remains blocked until the UAT packet is accepted and
internal-evidence disclosure is explicitly authorized.

## Step 226 - Bind UAT acceptance, disclosure, and paid-request limits

Added `app/agentic_tools/uat_review.py` and
`scripts/import_manual_bounded_tool_uat_review.py`. The review ledger binds the
draft manifest, successful local batch, source-minimized packet, reviewed
manifest, reviewer/chat approval, paid-use permission, and internal-evidence
disclosure permission. It limits execution to ten answer requests, zero query-
embedding requests, and zero automatic retries.

The first preflight exposed a Windows newline defect: the ledger hashed LF
content while the persisted reviewed manifest contained CRLF bytes. It failed
before any OpenAI call. The writer now uses explicit newline-preserving output;
the original mismatched artifacts remain preserved. The final reviewed manifest
is `data/evaluations/bounded_tool_manual_uat_v1_reviewed_v3.jsonl`; its ledger is
`data/evaluations/bounded_tool_manual_uat_v1_review_20260824_v3.json`, identity
`49d5911a4ff8370216fc9c42e59a4571ce6f8d4d54249e740e73f5be6a46138f`.

Production interpretation: semantic equality is insufficient for a byte-bound
authorization contract. A platform newline transformation must invalidate the
preflight rather than be silently accepted. Tests now compare the recorded hash
with the exact persisted manifest bytes.

## Step 227 - Build the no-retry paid bounded-tool evaluator

Added `app/agentic_tools/paid_uat.py` and
`scripts/run_paid_bounded_tool_uat.py`. The runner reconstructs prompt evidence
from the already preserved local UAT reports, so it makes no new query-embedding
requests. Code mode uses citeable code units; FDD and combined modes use the
strict combined JSON response contract with separate FDD/code sections and
unknown-boundary notes.

The first evidence preflight exposed the same CRLF mismatch in individual UAT
report hashes and stopped with zero paid calls. All UAT writers were corrected,
and the identical ten-case local batch was regenerated under a new immutable
namespace. The corrected batch passed **10/10** with identity
`b5137e83db33f20118389ac4a2c02810d63303a59c001663320796ae2f6e9575`.

Production interpretation: the runner refuses changed manifests, ledgers, batch
reports, or source-bearing case reports; preserves partial state on failure; and
does not retry automatically. Prompts, evidence, raw provider output, request
metadata, typed answers, and structural results are retained locally for the
authorized SME review.

## Step 228 - Execute the authorized paid grounded-answer evaluation

Executed the ten authorized OpenAI answer-generation requests using the preserved
local evidence. Query-embedding requests were **0**, answer requests were **10**,
and automatic retries were **0**. The run completed as
`completed_pending_sme_review` with **8/10 structural passes**.

Two immutable findings require SME review:

- `uat-code-aml-offline-impact-005` safely refused instead of returning the
  expected visible-code impact guidance;
- `uat-combined-aml-unitholder-008` answered but did not cite the expected R22 FDD
  document.

The review packet is
`data/exports/evaluations/bounded-tool-paid-answer-v2-20260824/sme-review.md`.
Focused paid-UAT and serialization tests passed **19/19**. The full regression
passed **601 tests** with the existing non-failing Starlette/HTTPX deprecation
warning.

Production interpretation: 10/10 completion proves the authorized paid scope ran;
8/10 structural pass does not establish semantic acceptance or activation. The
original answers must be SME-reviewed before deciding whether either finding is a
benchmark issue, an acceptable safe refusal/citation choice, or a targeted product
gap. No retry, API/UI exposure, automatic routing, or activation occurred.

## Step 229 - Import the paid SME decisions without rewriting machine results

Added `scripts/import_paid_bounded_tool_uat_review.py`. It binds the paid run
state, edited SME packet, and all ten exact trace hashes while preserving each
immutable structural result. The ledger records **9 semantic acceptances** and
**1 correction required**, with activation false and no additional paid requests
authorized.

The ledger is
`data/evaluations/bounded_tool_paid_answer_v1_review_20260824.json`, identity
`7f642f1061f4711c1275f388849fbee82faee6cf48514a547daad8a565d3c31f`.

Production interpretation: case 8 remains structurally failed because its expected
R22 citation was absent, but the SME explicitly accepted its answer. Human
acceptance is recorded alongside—not substituted for—the machine result. Case 5
remains a remediation item and prevents semantic gate completion.

## Step 230 - Localize the case-5 failure to citation formatting

Inspected only the two structurally failed traces. Case 5 already retrieved
`spOfflineParallelUserEnd` as `[C2]`, and the provider generated useful impact
guidance naming that routine. However, the raw response used bare references such
as `C2` and `Evidence: C2` instead of the required bracketed form `[C2]`.
`finalize_code_answer` therefore failed citation validation and safely returned
`invalid_or_missing_citation`.

Production interpretation: this is not a retrieval or reranking failure. Changing
RRF or evidence selection would treat the wrong layer and could regress unrelated
queries. The correct repair belongs to the generation/citation-format contract.
Case 8 is not changed because the SME accepted its useful R24-grounded answer;
its original missing-R22 structural observation remains available for future
benchmark refinement.

## Step 231 - Strengthen exact code-citation syntax locally

Updated `CODE_SYSTEM_PROMPT` to require exact square-bracket citations such as
`[C1]` and explicitly forbid bare forms such as `C1` or `Evidence: C2`. Existing
citation validation remains authoritative and still fails closed if the provider
does not follow the instruction.

```text
Every citation must use exact square-bracket syntax such as [C1] or [C2].
Never write a bare citation such as C1, "Evidence: C1", or "Evidence: C2".
```

Focused review/prompt tests passed **15/15** and the full regression passed
**602 tests** with the existing non-failing Starlette/HTTPX deprecation warning.

Production interpretation: deterministic tests prove the revised prompt contract
is present and the validator remains fail-closed; they do not prove a live model
will comply. The original ten-request authorization is exhausted. No retry was
made. A new activation-bound prompt identity plus explicit authorization for one
paid case-5 request and its internal code evidence are required before live replay.

## Step 232 - Bind the one-case replay authorization

Added `app/agentic_tools/replay.py` and
`scripts/prepare_paid_bounded_tool_case_replay.py`. The immutable authorization
binds the exact reviewed case, reviewed manifest, prior SME ledger, prior paid
trace, preserved local UAT evidence, current code-system-prompt hash, and answer
model. Its executable limits are exactly one answer request, zero query-embedding
requests, and zero automatic retries.

```text
maximum_answer_requests=1
maximum_query_embedding_requests=0
automatic_retries=0
authorization_identity_sha256=3ce3a5a8374a1de8439fd2bfdf594166c7925592d14c00fe1d16517b1e452eee
```

Production interpretation: prior disclosure does not create an open-ended retry
right. The new authorization is limited to the corrected prompt and the exact
previously retrieved evidence; any content, model, prompt, or request-limit drift
invalidates that authorization.

Failure-mode testing: deterministic tests reject authorization tampering, missing
approval notes, expanded request/embedding/retry limits, disabled permissions, and
attempted authorization-file overwrite.

## Step 233 - Add and exercise the fail-closed replay runner

Added `scripts/run_paid_bounded_tool_case_replay.py`. It validates all bound hashes,
the exact reviewed case, configured model, current prompt identity, disclosure
permission, and a new output namespace before constructing the preserved evidence.
It uses the no-retry OpenAI client and makes no query-embedding request.

The first execution intentionally omitted the execution-only disclosure flag. The
preflight reported one planned answer request, zero embeddings, and zero retries,
then stopped with `PermissionError` before creating the output directory or calling
OpenAI.

Production interpretation: authorization artifacts and execution confirmation are
separate gates. A valid stored authorization cannot be executed accidentally by a
partial command, and a failed preflight consumes neither cost nor disclosure budget.

Failure-mode testing: the runner refuses changed evidence or approval artifacts,
prompt/model drift, missing or duplicate cases, missing disclosure confirmation,
and an existing result directory. Paid failures are persisted as `failed_closed`
and are never retried automatically.

## Step 234 - Execute the authorized one-call replay

Executed exactly one paid answer request for
`uat-code-aml-offline-impact-005` using its previously preserved PL/SQL evidence.
The run used zero query embeddings and zero retries. It completed with a structural
pass: the answer is marked answered, provides visible-code impact candidates, uses
valid bracketed citations including `[C2]`, and retains explicit limitations for
missing routine bodies and unproven call paths.

Artifacts:

- authorization:
  `data/evaluations/bounded_tool_case5_replay_authorization_20260824.json`;
- trace and run state:
  `data/exports/evaluations/bounded-tool-case5-replay-20260824/`;
- SME packet:
  `data/exports/evaluations/bounded-tool-case5-replay-20260824/sme-review.md`.

The run state is `completed_pending_sme_review`, with
`answer_requests_completed=1`, `query_embedding_requests_completed=0`,
`automatic_openai_retries=0`, `structural_passed=true`, and
`activation_authorized=false`. Focused tests passed **21/21**. The full regression
passed **611 tests** with the existing non-failing Starlette/HTTPX deprecation
warning.

Production interpretation: this result demonstrates that the targeted live model
followed the repaired citation contract for this one case. It does not prove broad
model stability, semantic SME acceptance, or activation readiness. The preserved
SME packet must be reviewed; no additional paid call is authorized.

Failure-mode testing: citation validation remains independent of the prompt and
will still convert malformed or bare citations into a safe refusal. The immutable
prior failure remains preserved for before/after diagnosis, and the new runner
cannot overwrite this replay result.

## Step 235 - Parse the replay SME verdict without field bleed

Added `app/agentic_tools/replay_review.py` with a line-bounded review parser. The
reviewed packet records `accepted` for `uat-code-aml-offline-impact-005`, and its
displayed structural result matches the immutable replay result `pass`.

```text
SME verdict: accepted
Structural result: pass
```

The parser deliberately uses horizontal whitespace around field values rather
than `\s*`. This prevents a blank `SME rationale:` line from consuming the next
`Required follow-up:` label. Because the packet rationale is blank, the ledger
uses the explicit chat-confirmed acceptance note and identifies that source.

Production interpretation: human review fields must be parsed as separate records;
an apparently harmless multiline-regex choice can corrupt rationale provenance.
The parser does not infer acceptance from structural success or from an empty
follow-up field.

Failure-mode testing: focused tests reject duplicate/missing packet scope, blank or
unsupported verdicts, and mismatches between the displayed and stored structural
result. A blank rationale is verified to remain blank instead of absorbing the
following field label.

## Step 236 - Import a hash-bound remediation-closure ledger

Added `scripts/import_paid_bounded_tool_case_replay_review.py`. The new ledger
binds the exact replay run state, trace, edited SME packet, one-call authorization,
and original ten-case SME ledger. It verifies that the original case verdict was
`corrected`, the replay verdict is `accepted`, and the original suite had exactly
one unresolved semantic correction.

The ledger is
`data/evaluations/bounded_tool_case5_replay_review_20260824.json`, identity
`d3d1ab100a1c44e0dd1ecd49c5ad86082cd9629ca75bf970c59e7b81eec63e82`.
It records effective semantic acceptance as **10/10**, closes only the case-5
remediation item, authorizes zero additional paid calls, and keeps activation
false.

Production interpretation: the original 9/10 ledger and failed trace remain
immutable. The closure ledger forms a provenance chain rather than rewriting the
historical failure into a pass.

Failure-mode testing: the importer rejects authorization/hash drift, a nonterminal
replay, request/embedding/retry count drift, a mismatched case, an absent prior
correction, a non-accepted replay verdict, and attempted ledger overwrite.

## Step 237 - Close the bounded-tool semantic remediation gate

The effective reviewed result for the bounded-tool paid UAT set is now **10/10
semantically accepted after targeted replay**. Case 5 changed from an immutable
structural failure and SME correction to a separately recorded structural pass and
SME acceptance. No other case result or benchmark expectation was modified.

Focused replay/review tests passed **13/13**. The full regression passed **615
tests** with the existing non-failing Starlette/HTTPX deprecation warning.

Production interpretation: the targeted semantic remediation gate is closed. This
does not authorize another OpenAI request, enable automatic routing, or independently
prove broad bounded-agent runtime quality. Any subsequent API/UI exposure or new
agentic capability still requires its own readiness, privacy, serving, monitoring,
and rollback decision.

Failure-mode testing: `git diff --check` remains the documentation/code whitespace
gate, the ledger is no-overwrite, and activation remains explicitly false even
after complete SME acceptance.

### Phase 2 scoped completion decision

The Steps 235-237 learner gate is accepted **9/9**. Together with the existing
local code/combined activation evidence, reviewed lineage, deterministic retrieval,
source-line citations, safe unknown handling, manual UAT, paid answer evaluation,
and the accepted targeted replay, the approved initial **Phase 2 PL/SQL scope is
complete**.

This closure does not redefine later work as already delivered. JavaScript corpus
support, bounded-tool API/UI exposure, automatic agent routing, larger-corpus
evaluation, concurrency/load evidence, and production security/operations controls
remain explicit future scopes with independent gates.

## Step 238 - Define MCP-ready retrieval configuration and source contracts

Added safe configuration defaults for `INTERFACE_MODE=fastapi` and
`MCP_EVIDENCE_DISCLOSURE_ENABLED=false`. Added `RETRIEVAL_INDEX_PATH` as a
compatibility alias for the existing FDD lexical `PROCESSED_DIR`; configuration
now fails closed if both resolve to different directories.

```python
@property
def fdd_retrieval_artifact_dir(self) -> Path:
    return _resolve_project_path(
        self.retrieval_index_path or self.processed_dir,
        self.root_dir,
    )
```

Added framework-neutral search/fetch result models and SHA-256 opaque source IDs
in `app/retrieval/knowledge_service.py`. Public source references never expose
absolute paths or an FDD internal unit ID.

Production interpretation: configuration identifies one reproducible FDD lexical
generation while preserving explicit code and Qdrant locations. The new models
allow FastAPI and MCP to serialize the same evidence without sharing transport
logic.

Failure-mode testing: conflicting lexical paths fail validation; duplicate active
source identities fail catalog construction; malformed and unknown opaque IDs fail
fetch lookup without resolving a path.

## Step 239 - Add the shared KnowledgeRetrievalService

Added `KnowledgeRetrievalService`, the framework-independent boundary for FDD,
code, and combined retrieval. It owns configured lexical/dense/hybrid selection,
Qdrant lifecycle, active code/FDD artifact validation, reviewed lineage checks,
safe source catalog lookup, and bounded result formatting.

```python
result = service.retrieve(
    query="AML batch processing",
    mode="combined",
    limit=5,
)
```

Combined dense/hybrid retrieval creates one query vector and passes that vector to
both FDD planned retrieval and code retrieval. It does not merge FDD and code
scores; combined output keeps the existing five-per-lane contract.

Production interpretation: FastAPI and MCP can use exactly the same source,
retrieval, ranking, lineage, and Qdrant controls. A future adapter cannot bypass
the approved retrieval contract with a simplified search path.

Failure-mode testing: focused tests use missing/duplicate catalog identities and a
dense FDD path with a fake vector store. A combined hybrid test proves one
embedding call is reused across both lanes and both Qdrant clients close.

## Step 240 - Let answer orchestration consume prepared retrieval

`run_grounded_answer_query(...)` now accepts an optional `planned_retrieval`
argument and rejects it if it belongs to a different query. Code/combined answer
orchestration accepts a prepared `KnowledgeRetrievalExecution` and generates and
traces from that evidence without a second retrieval call.

```python
run_code_or_combined_query(
    mode="combined",
    query=query,
    analysis_kind="explanation",
    settings=settings,
    retrieval_config=config,
    limit=5,
    correlation_id=request_id,
    retrieval_execution=prepared_execution,
)
```

Production interpretation: retrieval is now a reusable, independently testable
phase. Existing answer contracts, citations, safe refusals, conversation context,
and trace format remain owned by their established answer layers.

Failure-mode testing: a prepared retrieval for another query or knowledge mode is
rejected. Existing orchestration, answer-service, and retrieval-config tests
remain green.

Focused verification: **16 passed** across shared-service, existing orchestration,
answer-service, and retrieval-config tests. `git diff --check` found no whitespace
errors; only pre-existing Windows line-ending advisories were reported.

Gate status: **awaiting learner answers for Steps 238-240.** No MCP SDK, tunnel,
disclosure, live OpenAI call, or data-egress operation has occurred.

### Learner evaluation — Steps 238-240

**Accepted, 9/9.** The learner correctly explained fail-closed lexical-generation
selection, stable occurrence/content identity, portable source references,
per-lane ranking, shared query-vector reuse, duplicate-catalog rejection,
query-bound prepared evidence, the answer-orchestration/MCP boundary, and the
remaining stdio protocol test gap. The answers were precise and production-aware;
no remediation was needed.

## Step 241 — Add retrieval-only FastAPI search through the shared service

Added the constrained `POST /search` contract and switched `/query` to prepare
retrieval with `KnowledgeRetrievalService` before passing the same result into its
established answer orchestration. `/search` accepts only a nonblank `query` and
knowledge lane `fdd`, `code`, or `combined`; it does not accept paths, point IDs,
SQL, arbitrary filters, or a retrieval-strategy override.

```python
@router.post("/search", response_model=KnowledgeSearchResponse)
def search_evidence(request: SearchRequest) -> KnowledgeSearchResponse:
    return build_knowledge_retrieval_service(
        settings=get_settings(),
        retrieval_config=build_retrieval_runtime_config(get_settings()),
    ).search(query=request.query, mode=request.mode, limit=5)
```

Production interpretation: API retrieval and answer generation now share one
source/ranking/lineage implementation. Existing answer contracts still own model
generation, citations, refusals, conversation context, and traces.

Failure-mode testing: code/combined remains blocked by `CODE_MODES_ENABLED`;
missing retrieval dependencies return safe 503 responses; a prepared retrieval
cannot be used with another query. FastAPI now refuses startup when
`INTERFACE_MODE=mcp`, preserving an explicit interface boundary.

## Step 242 — Add the guarded, read-only MCP adapter and structured encoding

Added `app/mcp/adapter.py` and `app/mcp/server.py`, using the maintained `mcp`
2.1.0 SDK and its `MCPServer` stdio transport. The adapter calls
`KnowledgeRetrievalService` directly; it contains no HTTP client and never calls
FastAPI.

```python
def search(self, *, query: str, mode: KnowledgeMode) -> KnowledgeSearchResponse:
    settings = self._disclosure_enabled_settings()
    return self._service_factory(settings).search(query=query, mode=mode, limit=5)
```

`MCP_EVIDENCE_DISCLOSURE_ENABLED=false` is checked before service construction.
Disabled `search` and `fetch` return only `Evidence disclosure is disabled.` with
no structured content. Therefore they cannot load a catalog, open Qdrant, embed a
query, resolve an opaque ID, or disclose source identifiers/metadata.

`encode_mcp_result` validates the Pydantic result, makes one canonical dictionary,
uses it as MCP `structuredContent`, and derives fallback text from that same
dictionary. The server exposes only read-only, idempotent, closed-world `search`
and `fetch`; `fetch` accepts only the existing SHA-256 opaque-ID shape.

Production interpretation: the kill switch is an MCP-only emergency egress
control, not a change to FastAPI or Streamlit RAG behavior. The ChatGPT side gets
bounded evidence for interpretation; it does not receive an alternate retrieval
implementation.

Failure-mode testing: disabled calls perform zero service work and emit no source
payload; malformed/unknown fetch IDs fail without resolving paths; non-activated
code lanes fail safely; MCP startup refuses `INTERFACE_MODE=fastapi`; Streamlit
also refuses `INTERFACE_MODE=mcp`.

## Step 243 — Verify adapter equivalence and prepare stdio-safe logging

Configured MCP startup logging explicitly to `sys.stderr` with `force=True`,
captured Python warnings, and reset known HTTP/Qdrant/OpenAI/MCP dependency logger
handlers to propagate to that stderr handler. The implementation deliberately does
not replace `sys.stdout`, because the MCP SDK owns stdout for JSON-RPC frames.

```python
logging.basicConfig(
    level=getattr(logging, level.upper(), logging.INFO),
    handlers=[logging.StreamHandler(sys.stderr)],
    force=True,
)
logging.captureWarnings(True)
```

Focused equivalence tests confirm the FastAPI `/search` route and direct MCP
adapter request the same shared result for `fdd`, `code`, and `combined` modes.
They also validate canonical structured/text encoding, disclosure-disabled output,
read-only tool metadata, and interface-mode startup refusal.

Production interpretation: this establishes one local retrieval implementation
behind two transports. Dense/hybrid remains internal retrieval configuration;
combined execution reuses one query vector across FDD and code lanes, so no new
public strategy parameter or duplicate embedding path exists.

Failure-mode testing: 46 focused tests passed. `uv lock --check` passed after
pinning the maintained MCP SDK. `git diff --check` found no whitespace errors;
only pre-existing Windows line-ending advisories were emitted. Actual subprocess
JSON-RPC framing and stderr/stdout isolation remain the explicit Steps 244-246
gate; no tunnel, live OpenAI call, or evidence disclosure occurred.

Gate status: **awaiting learner answers for Steps 241-243.**

### Learner evaluation — Steps 241-243

**Accepted, 9/9.** The learner accurately distinguished knowledge lanes from
retrieval strategy, explained deterministic prepared evidence, separated
retrieval and generation failure classes, described true pre-retrieval egress
control, opaque-ID safety, independent FDD/code feature gates, canonical output
serialization, stdout ownership, and the remaining subprocess wire test. No
answer needed strengthening.

## Step 244 — Test the actual MCP stdio JSON-RPC transport

Added a raw subprocess protocol harness in `tests/test_mcp_stdio_protocol.py`.
It starts `python -m app.mcp.server`, sends `initialize`,
`notifications/initialized`, `tools/list`, and a disclosure-disabled
`tools/call`, then parses every stdout line as JSON-RPC.

```python
message = json.loads(line)
assert message["jsonrpc"] == "2.0"
```

The test also enables a test-only diagnostic injection. An `httpx` logger warning
and a Python `RuntimeWarning` are emitted during startup and asserted on stderr.
The test proves they do not contaminate stdout protocol frames.

Production interpretation: stdio is a protocol boundary, not a console. A single
log prefix, warning, or `print()` on stdout could break a ChatGPT/tunnel session;
the test exercises the real child transport rather than assuming object-level JSON
returns prove wire correctness.

Failure-mode testing: the disabled call is an MCP tool error with exactly the
generic message and no `structuredContent`; tools publish only `search` and
`fetch` with read-only, non-destructive, closed-world annotations. No source
catalog, Qdrant, embedding, tunnel, or OpenAI call is used.

## Step 245 — Add child-process MCP startup preflight

Added `app/mcp/preflight.py`. Before tools register, the MCP child validates the
effective interface mode, retrieval strategy, FDD lexical artifact directory, and
when code modes are enabled, the code artifact, analysis directory, and reviewed
lineage artifact. These checks are local-only and deliberately do not load a
catalog, open Qdrant, or call OpenAI.

```python
report = run_mcp_startup_preflight(settings)
if not report.passed:
    raise RuntimeError("MCP startup preflight failed: ...")
```

The preflight checks only whether `CONTROL_PLANE_API_KEY` is present in the child
environment; it never reads, logs, traces, or returns the value. Presence fails
closed, proving the tunnel credential was improperly inherited.

Production interpretation: a descriptive runtime manifest cannot inject or prove
child environment. The child must validate its actual configuration before it
advertises tools. This prevents an MCP process from starting with a mismatched
interface mode, stale lexical directory, invalid retrieval policy, or leaked
tunnel-control credential.

Failure-mode testing: tests reject `INTERFACE_MODE=fastapi`, unsupported retrieval
strategy, absent artifact directory, and inherited control-plane credential; the
safe error contains no secret value.

## Step 246 — Record tunnel-client process ownership

Added `scripts/run_mcp_stdio.ps1` and updated
`deployment/native_runtime.json`. The tunnel client is the process owner; it
launches the PowerShell wrapper through `--mcp-command`. The wrapper removes the
parent-only control-plane key before starting the project virtual-environment
Python MCP child.

```powershell
Remove-Item -LiteralPath "Env:CONTROL_PLANE_API_KEY" -ErrorAction SilentlyContinue
& $pythonPath -m app.mcp.server
```

The runtime manifest records the MCP command, working directory, allowed interface
modes, non-secret application environment names, parent-only control-key rule, and
the fact that it describes—not injects—environment.

Production interpretation: in `both` mode there will be exactly three terminals:
FastAPI, Streamlit, and `tunnel-client run`. The latter owns the stdio child; no
separate manually started MCP server is used for tunnel operation.

Failure-mode testing: tests assert the native metadata omits
`CONTROL_PLANE_API_KEY` from application environment requirements and that the
launcher removes it before Python starts. The focused protocol/preflight/runtime
suite passed **17/17**; `uv lock --check` and `git diff --check` passed (only
pre-existing Windows line-ending advisories were emitted).

Gate status: **awaiting learner answers for Steps 244-246.** No tunnel was
created, no external API request occurred, and no internal evidence was disclosed.

### Learner evaluation — Steps 244-246

**Accepted, 9/9.** The learner clearly separated in-process logic testing from
wire-level protocol proof, explained whole-stream validation and independent
diagnostic paths, justified bounded local preflight, described credential
presence-only isolation, fail-closed tool registration, tunnel lifecycle
ownership, least-privilege key removal, and the limits of local tests versus live
ChatGPT/tunnel evidence. No remediation was needed.

## Step 247 — Publish the Phase 1 operator runbook

Created `docs/ChatGPT_Secure_MCP_Tunnel_Phase1.md` and added a concise command
section to `README.md`. The runbook covers generation verification, FastAPI/UI
startup, direct MCP Inspector testing, explicit disclosure enablement,
tunnel-client profile initialization/doctor/run, ChatGPT Developer Mode
connection, three-terminal `both` operation, costs, safe rollback, and concrete
troubleshooting.

```python
# The documented child process is the actual application entry point.
def main() -> None:
    settings = get_settings()
    configure_mcp_stdio_logging(settings.log_level)
    create_mcp_server(settings=settings).run(transport="stdio")
```

Production interpretation: documentation makes operational authority visible.
The approved operator must set `MCP_EVIDENCE_DISCLOSURE_ENABLED=true` before MCP
retrieval testing; this is deliberate data egress, not a hidden convenience
setting. Dense/hybrid queries can also incur query-embedding cost.

Failure-mode testing: the troubleshooting table provides fail-closed actions for
disclosure-disabled, missing environment/artifacts/Qdrant, embedding, tunnel,
stdout-corruption, and unexpected tool-metadata failures. It explicitly rejects a
fourth manually started MCP terminal in tunnel mode.

## Step 248 — Document process ownership, interface selection, and key boundary

The runbook and README now make the selected runtime topology executable:

```text
Terminal 1: FastAPI          (only fastapi/both)
Terminal 2: Streamlit        (only fastapi/both)
Terminal 3: tunnel-client run → owns MCP stdio child (mcp/both)
```

`scripts/run_mcp_stdio.ps1` is the documented tunnel command. It removes the
parent-only `CONTROL_PLANE_API_KEY` before virtual-environment Python starts;
the application validates only its absence, never its value. The real key is
injected only into Terminal 3 by the approved secret mechanism and is not stored
in `.env`, emitted in documentation examples, logs, traces, or tool output.

Production interpretation: interface mode and disclosure state are separate.
`INTERFACE_MODE` controls which local processes may start; disclosure controls
whether the already-authorized MCP transport may return internal evidence.
Neither flag enables code/combined retrieval without the established
`CODE_MODES_ENABLED` activation control.

Failure-mode testing: metadata/launcher tests verify parent-only key handling and
tunnel-client ownership. The startup preflight blocks a child with incompatible
configuration before it can advertise tools.

## Step 249 — Final offline verification

Added/updated MCP adapter, protocol, preflight, runtime-metadata, API, and
documentation coverage. A first full suite revealed two regressions: legacy API
test doubles did not define the new `interface_mode` field, and test-time MCP
factory construction globally reset logging before unrelated audit assertions.
Both were corrected: an absent legacy field defaults safely to `fastapi`, and
stderr logging is configured only in the actual MCP `main()` entry point.

```python
if getattr(settings, "interface_mode", "fastapi") == "mcp":
    raise RuntimeError("FastAPI is disabled when INTERFACE_MODE=mcp.")
```

Production interpretation: Phase 1 preserves existing FastAPI/Streamlit behavior
while adding a private, read-only MCP transport. The full suite validates code
compatibility and fail-closed behavior; it does not create a tunnel, disclose
source evidence, call OpenAI, or prove an external ChatGPT session.

Failure-mode testing and final checks:

```powershell
uv lock --check
uv run --locked pytest
git diff --check
```

The final clean suite passed **637 tests in 127.46 seconds**. `uv lock --check`
passed and `git diff --check` reported no whitespace errors (only pre-existing
Windows line-ending advisories). The temporary test-output files used to capture
the long suite were removed from `data/tmp` after verification.

Gate status: **offline implementation accepted.** No automated test made a live
OpenAI call, created a tunnel, or disclosed internal evidence. Live Secure MCP
Tunnel and ChatGPT validation remains a separate, operator-authorized exercise.

### Learner answers and mentor evaluation

**Accepted, 9/9.** The learner correctly distinguished the three independent
controls, immutable generation activation, Inspector versus tunnel-client
ownership, the parent-only control-plane key, transport health versus retrieval
quality, safe default interface behavior, logging scope, and the limits of the
637-test offline result. Phase 1 implementation is complete; the next work is
the documented manual tunnel setup and evidence-grounding validation.

## Step 250 — Controlled FastAPI dense retrieval probe

After the local Codex experiment showed that a shell-enabled client can launch a
separate child with caller-supplied environment variables, dense retrieval was
tested through the established FastAPI path rather than through Codex.

```http
POST /search
Content-Type: application/json

{"query":"What is the AML batch processing behavior?","mode":"fdd"}
```

The temporary localhost FastAPI process used `RETRIEVAL_MODE=dense`, FDD mode,
and code mode disabled. The operator explicitly authorized one paid query
embedding. The result was `HTTP 200`, `retrieval_mode=dense`, five FDD results,
and no answer-generation call. Only aggregate outcome metadata was recorded;
no source excerpt was copied into this progress log. The temporary process was
stopped after the request.

Production interpretation: dense/hybrid retrieval should be exercised through a
single controlled retrieval runtime. A local embedded Qdrant store is not a
multi-process service: competing MCP/FastAPI children can conflict on its lock.
An environment-variable disclosure switch is a useful gate for the intended MCP
child, but is not an enforceable egress boundary against a client that has local
shell and filesystem access to the repository.

Failure-mode testing: two stale MCP children were identified by command line and
terminated only after operator approval. The dense probe then completed without
a Qdrant lock. No hybrid probe was run or authorized.

## Steps 251-253 — Local MCP/Qdrant runtime hardening

### Step 251 — Safe embedded-Qdrant lock errors

`app/vectorstore/qdrant_schema.py` now recognizes the local embedded-Qdrant
storage-lock failure and raises a safe application error instead of passing a
filesystem path through the retrieval stack:

```python
try:
    return QdrantClient(path=str(storage_path))
except RuntimeError as exc:
    if _looks_like_local_storage_lock(exc):
        raise LocalQdrantLockError(
            "Local Qdrant storage is in use by another process."
        ) from exc
    raise
```

Explanation: the underlying client can report the local storage location in a
lock exception. That detail is useful locally but should not be returned to an
MCP/Chat client.

Production interpretation: this is failure containment, not multi-process
support. Embedded Qdrant remains an exclusive local-store runtime.

Failure-mode test: a mocked Qdrant lock with an internal path raises only the
generic `LocalQdrantLockError`; `tests/test_qdrant_schema.py` proves the path is
not exposed.

### Step 252 — Single MCP-child launcher guard

`scripts/run_mcp_stdio.ps1` now acquires a Windows named mutex before launching
the stdio server:

```powershell
$mutex = New-Object System.Threading.Mutex($false, "Local\CullingBladeLineageMcpStdio")
if (-not $mutex.WaitOne(0, $false)) {
    $mutex.Dispose()
    throw "Culling Blade MCP server is already running. Stop the existing MCP child before starting another."
}
```

Explanation: this blocks a duplicate local MCP child before both processes try
to open the same embedded Qdrant files. The existing parent-only control-plane
key removal remains in the launcher.

Production interpretation: the mutex protects duplicate **MCP** children only.
It does not permit simultaneous FastAPI and MCP dense/hybrid access to embedded
Qdrant. That topology requires a separately selected shared Qdrant server.

Failure-mode test: `tests/test_mcp_stdio_launcher.py` verifies key stripping,
the mutex, non-blocking acquisition, the safe duplicate message, and the
expected MCP module command. PowerShell parsing was also checked without
starting a server.

### Step 253 — Explicit local-runtime boundary

`docs/ChatGPT_Secure_MCP_Tunnel_Phase1.md` now documents the actual laptop
contract:

```text
lexical MCP search: no application embedding API call
dense/hybrid MCP search: application query embedding; API usage/cost possible
embedded Qdrant: one process at a time for a given local store
```

Explanation: ChatGPT can generate the final answer from MCP evidence, but that
does not remove the application embedding cost when dense or hybrid retrieval
is configured. The local direct-stdio test is valid for one client process; it
is not the server topology for concurrent interfaces.

Production interpretation: for the current personal laptop testing path, run
one local MCP child and select lexical retrieval when zero application embedding
cost is required. Before concurrent FastAPI + MCP dense/hybrid use, choose and
validate a shared Qdrant server deployment, credentials, health checks, and
rollback plan.

Verification: focused hardening tests passed **9/9**. `uv lock --check` passed.
The complete suite was executed with the new tests collected (640 total); no
test failure was reported by the runner. `git diff --check` reported no
whitespace failure. No OpenAI call, tunnel, or evidence disclosure occurred in
these hardening steps.
