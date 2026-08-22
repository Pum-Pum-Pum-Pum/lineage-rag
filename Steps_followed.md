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
