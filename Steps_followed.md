# Steps Followed

## Step 94 - Streamlit multi-turn chat UI

### Goal
Replace the single-query Streamlit form with a durable, conversation-scoped chat
experience that preserves the grounded RAG boundary and exposes safe operational
debugging information.

### Files touched
- `app/ui/api_client.py`
- `app/ui/streamlit_app.py`
- `tests/test_ui_api_client.py`
- `tests/test_streamlit_ui.py`
- `README.md`
- `docs/project_plan.md`
- `Steps_followed.md`
- `interview-questions.md`

### What changed
- Extended the typed UI client with create, list, detail, archive, and
  conversation-message methods.
- Added safe UI error codes for missing conversations, archived writes, context
  overflow, unavailable dependencies, and generic HTTP failures.
- Rebuilt Streamlit around `st.chat_message` and `st.chat_input`.
- Added New Chat, active-conversation selection, and Archive Active Chat
  controls.
- Reloaded conversation messages from the backend so durable storage remains the
  history source of truth.
- Added readiness gating before cost-bearing message submission.
- Added optional evidence/debug details for citations, sufficiency, trace ID,
  context budget, summary checkpoint, usage, and estimated cost.
- Added explicit user-only partial-turn warnings for failed prior attempts.
- Kept per-turn debug responses in Streamlit session state only as a transient
  cache because the current history endpoint stores messages and trace IDs but
  not the full citation/debug response.

### Python/code pattern used
```python
prompt = st.chat_input(
    "Ask a grounded question about the functional specifications",
    disabled=detail.conversation.is_archived,
)
if prompt:
    request = _build_message_request(content=prompt, ...)
    turn = _run_ready_turn(api, active_id, request)
    _cache_turn(st.session_state, turn)
    st.rerun()
```

### Why it is implemented this way
The UI is a presentation layer, not a second source of conversation truth.
Every rerun reloads conversations and messages from FastAPI, while session state
stores only replaceable display details. Readiness is checked before message
submission to avoid preventable retrieval/model cost. Typed response validation
and safe error mapping keep backend bodies, paths, and exception details out of
the page.

Partial user-only turns are rendered rather than hidden because Step 93
deliberately persists accepted user input before grounded processing. This makes
failures auditable and gives the future retry/idempotency flow a visible state.

### Production interpretation
The page is now a usable local multi-turn product surface with durable chat
lifecycle and debugging visibility. It remains intentionally local and
single-user: authentication, authorization, pagination, idempotent retries,
streaming responses, and persistent citation/debug history remain future
production concerns.

Debug visibility improves trust and incident diagnosis, but it is optional to
avoid overwhelming ordinary users. Conversation memory still cannot prove
functional-spec claims; the displayed citations and evidence sufficiency remain
the factual grounding signals.

### Validation
- Focused UI/client/conversation/documentation suite: 35 passed.
- Full regression suite: 279 passed.
- Existing upstream Starlette `TestClient`/HTTPX deprecation warning remains
  non-failing.
- `git diff --check` passed with only Windows line-ending notices.
- Live FastAPI `/health`: `ok`, hybrid retrieval.
- Live Streamlit HTTP check: 200.
- Browser QA verified initial render, New Chat creation, active selection,
  enabled chat input, readiness details, archive removal from the active list,
  and an empty browser warning/error console.
- No model query was submitted during QA.
- Temporary FastAPI and Streamlit listeners were stopped after verification.

### Failure-mode thinking
Tests cover malformed responses, safe mappings for `404`/`409`/`413`/`503`/500,
readiness blocking, partial-turn detection, blank input, filter normalization,
and stale active-conversation fallback. Browser QA also exposed that a fixed
startup delay can race a real server; service verification must poll an
authoritative endpoint rather than infer readiness from process launch.
