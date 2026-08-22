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
