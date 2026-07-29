# Interview Questions

## Step 94 - Streamlit multi-turn chat UI

### Questions asked
1. Why must the backend conversation store remain the source of truth while
   Streamlit session state is only a UI cache?
2. Why should the UI call `/ready` before submitting a cost-bearing conversation
   message, and what race condition still exists after a successful readiness
   response?
3. Why should a user-only partial turn be displayed explicitly instead of
   hidden or automatically paired with an error-shaped assistant message?
4. What value does the optional evidence/debug panel provide, and what
   information should still be hidden from users?
5. What did the unit tests and live browser QA prove separately, and which
   production UI guarantees remain unproven?

### Correct answer key
1. Durable backend storage survives Streamlit reruns, browser refreshes, and
   process restarts and can be shared by API consumers. Session state is tied to
   one UI session and may disappear or become stale. It may cache display-only
   details, but conversation selection and history must be reconciled with the
   backend on each rerun.
2. Readiness prevents known dependency or artifact failures from starting
   retrieval, embedding, or generation cost. It is still a point-in-time check:
   Qdrant, artifacts, networking, or model services may fail between readiness
   and the subsequent message request, so the submission path must retain its
   own safe failure handling.
3. The persisted user message is an honest record of accepted input and enables
   audit and retry. Hiding it loses history; inventing an assistant error message
   misrepresents a completed model outcome. An explicit partial state tells the
   user what happened and supports later idempotent retry design.
4. The panel exposes citations, release labels, scores, sufficiency, trace IDs,
   context-budget state, summary checkpoints, usage, and cost for trust,
   evaluation, and debugging. It should not expose raw backend bodies, stack
   traces, secrets, local filesystem paths, credentials, or unrestricted
   internal logs.
5. Unit tests prove typed request/response handling, state-selection logic,
   filter validation, readiness ordering, partial-turn detection, safe error
   mapping, and presence of required UI contracts. Browser QA proves the real
   processes start, Streamlit renders, controls are wired, readiness appears,
   lifecycle reruns work, and no browser errors occur in the tested path. It does
   not prove concurrent users, accessibility completeness, responsive layouts,
   long-history performance, streaming quality, idempotent retries, live model
   answers, or cross-browser behavior.

### Gate
Step 94 implementation is complete. Await the user's answers before proceeding
to Step 95 evaluation coverage.
