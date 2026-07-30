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

## Step 95 - Complete, token-bounded prompt evidence

### Goal
Repair an answer-quality defect where retrieval found the correct BOR evidence
but answer generation received only each citation's 240-character UI preview.

### Files touched
- `app/llm/prompt_template.py`
- `app/services/answer_generation.py`
- `tests/test_prompt_template.py`
- `tests/test_answer_generation_service.py`
- `README.md`
- `docs/project_plan.md`
- `Steps_followed.md`
- `interview-questions.md`
- `Culling Blade Lineage- Gen AI RAG System_Prompt_for_GPT.txt`

### What changed
- Introduced a separate `PromptEvidence` model containing the complete text of
  each selected retrieval unit.
- Kept `Citation.text_preview` as a 240-character API/UI display field only.
- Selected evidence in retrieval-rank order using the configured reserved
  evidence-token budget.
- Admitted only whole evidence units; no unit is silently cut in the middle.
- Returned a deliberate safe refusal without calling the LLM when the
  highest-ranked complete unit cannot fit.
- Returned only citations corresponding to evidence actually sent to the LLM.
- Added BOR and B-04 regressions with decisive facts placed beyond character
  240, plus budget and no-LLM-on-overflow tests.

### Python/code pattern used
```python
prompt_evidence = select_prompt_evidence(
    request.retrieved_results,
    max_evidence_units=max_citations,
    max_evidence_tokens=max_evidence_tokens,
)
evidence_block = build_evidence_block(prompt_evidence)
citations = build_citations_from_results(
    request.retrieved_results[: len(prompt_evidence)],
    max_citations=len(prompt_evidence),
)
```

### Why it is implemented this way
Citation previews and model evidence have different jobs. A short preview keeps
API responses and UI panels compact, but using it as evidence can remove a
decisive row late in a table. Complete ranked units preserve semantic integrity.
The lightweight UTF-8 token estimate is replaceable with exact model
tokenization. Whole-unit admission makes the boundary observable and avoids
answers based on incomplete table fragments.

### Production interpretation
The first BOR failure is now addressable because the acronym row can reach the
model even when it occurs after character 240. This step does not yet guarantee
the second answer, “17 reduced to 4,” because hybrid top-5 retrieval omitted the
decisive R24 realignment tables for that natural query. Retrieval fusion,
alias-aware query handling, table-aware chunking, and answer-level evaluation
remain the next quality layer.

### Validation
- Focused prompt/answer/API/conversation suite: 36 passed.
- Full regression suite: 283 passed.
- Existing upstream Starlette `TestClient`/HTTPX warning remains non-failing.

### Failure-mode thinking
Tests prove that late BOR/B-04 facts survive prompt construction, citation
previews remain bounded, lower-ranked evidence is omitted only at whole-unit
boundaries, oversized top evidence produces a safe refusal, and the LLM is not
called on that refusal path. They do not prove that hybrid retrieval ranks every
decisive R24 table inside top-k or that the model always derives the correct
temporal count.

## Step 96 - Weighted Reciprocal Rank Fusion

### Goal
Repair the demonstrated hybrid-ranking defect where excellent lexical table
results were capped below mediocre dense-only results because incompatible
score scales were normalized and added.

### Files touched
- `app/retrieval/hybrid_search.py`
- `app/retrieval/evaluation.py`
- `app/core/config.py`
- `app/services/answer_orchestration.py`
- `.env.example`
- `data/eval/retrieval_eval.json`
- `tests/test_hybrid_search.py`
- `tests/test_retrieval_evaluation.py`
- `tests/test_retrieval_config.py`
- `README.md`
- `docs/project_plan.md`
- `Steps_followed.md`
- `interview-questions.md`

### What changed
- Replaced max-normalized score addition with weighted Reciprocal Rank Fusion.
- Selected lexical-heavy defaults of `0.40` dense and `0.60` lexical for this
  identifier- and table-heavy functional-spec corpus.
- Used an RRF rank constant of `1.0` for the short ten-result candidate lists.
- Normalized only the final RRF score to `[0,1]`, preserving the existing
  sufficiency-threshold scale without comparing raw dense and lexical scores.
- Added raw scores, ranks, per-retriever RRF contributions, fusion method, and
  rank constant to result/trace diagnostics.
- Extended retrieval evaluation with `expected_text_contains_all` so a test can
  require baseline, teller, and branch evidence together.
- Added an R24 regression reproducing the observed dense and lexical ranking
  pattern and requiring both realignment tables inside final top-5.

### Python/code pattern used
```python
rrf_contribution = weight / (rrf_rank_constant + rank)
raw_rrf_score += rrf_contribution
hybrid_score = raw_rrf_score / maximum_possible_rrf_score
```

### Why it is implemented this way
Cosine similarity and lexical relevance are different measurement scales. Rank
fusion uses only their ordering, so one retriever's arbitrary score magnitude
cannot dominate another. A lexical-heavy weight is appropriate here because
report identifiers such as `B-17`, action terms, and table headers carry exact
business meaning. The final normalization is solely a stable API/sufficiency
scale; it does not reintroduce input-score comparison.

### Production interpretation
For the user's original wording, real hybrid retrieval moved the R24 branch
realignment table to rank 2 and teller table to rank 4. The previous irrelevant
traceability result was removed from final top-5. No reindex was required.

RRF did not solve the complete answer. A controlled live variant retrieved the
detailed R24 business-requirements table but `gpt-5-mini` still answered 6/17,
treating the document's pre-change “Existing Functionality” as current. This
isolates the next defect to production-deployed latest-release temporal
semantics, query/release scoping, and answer-level validation.

### Validation
- The new RRF regression failed under the old normalized-score algorithm.
- Focused retrieval/evaluation suite: 42 passed.
- Full regression suite: 285 passed.
- Real retrieval check returned the branch and teller tables inside top-5 for
  the original natural-language query.
- Live answer trace:
  `c49b8880-be04-4216-80c3-20e86d9797c5`.
- The live check made one embedding call and one answer-generation call.
- Existing upstream Starlette `TestClient`/HTTPX warning remains non-failing.

### Failure-mode thinking
The implementation rejects invalid weights, limits, and RRF constants. Tests
cover overlap, rank-only evidence, final limits, bounded score output, and
multi-fact coverage failure. The live answer intentionally demonstrates that
retrieval recall is necessary but not sufficient: correct evidence can still be
synthesized under the wrong temporal policy.

## Step 97 - Current-state temporal synthesis and release scoping

### Goal
Make “current” mean the resulting production state after the latest relevant
deployed release, resolve conversational release references safely, and prevent
the R24 pre-change 6/17 baseline from being returned as the current answer.

### Files touched
- `app/retrieval/temporal_query.py`
- `app/services/answer_orchestration.py`
- `app/llm/answer_contract.py`
- `app/llm/prompt_template.py`
- `app/services/answer_generation.py`
- `app/llm/answer_evaluation.py`
- `data/eval/answer_eval.json`
- `scripts/evaluate_answer_trace.py`
- `tests/test_temporal_query.py`
- `tests/test_prompt_template.py`
- `tests/test_answer_orchestration_service.py`
- `tests/test_answer_evaluation.py`
- `README.md`
- `docs/project_plan.md`
- `Steps_followed.md`
- `interview-questions.md`

### What changed
- Added deterministic current-state intent detection and numeric release
  parsing so `R24` sorts after `R2`.
- Added a guard so domain phrases such as “current application date” do not
  incorrectly trigger latest-release behavior.
- Allowed referential queries such as “summarize it” to inherit an explicit
  release label from bounded conversation memory.
- Expanded current-state retrieval to the configured hybrid candidate depth in
  one search, then scoped the final results to the highest relevant deployed
  release before answer generation.
- Added retrieval expansion terms for retained, consolidated, renamed, and
  removed items while explicitly marking Existing Functionality as baseline.
- Added prompt rules stating that all indexed releases are deployed and that
  current/latest answers must apply the effective release changes.
- Added a reusable answer-trace evaluator requiring the 6/17 baseline, current
  2/4 result, T/B report identifiers, R24-only citations, and the baseline and
  teller/branch table evidence units.

### Python/code pattern used
```python
plan = build_temporal_query_plan(
    query_text,
    requested_release_label=release_label,
    conversation_context=conversation_context,
)
routed = retrieve_query_evidence(
    query_text=plan.retrieval_query,
    limit=max(limit, hybrid_candidate_limit),
    ...
)
results, plan = scope_results_to_temporal_plan(
    routed.results,
    plan,
    limit=limit,
)
```

### Why it is implemented this way
Conversation memory may resolve what “it” refers to, but it is not proof. The
resolved release is therefore used only as a retrieval filter; the answer still
requires fresh indexed R24 evidence. Candidate expansion and in-memory release
scoping avoid a second embedding/API call while giving change tables room to
enter the final evidence set.

Release numbers are parsed numerically rather than sorted as strings. The prompt
separates the latest release's baseline section from its resulting state so the
model must count retained/renamed outputs and exclude removed reports.

### Production interpretation
The exact reported question now returns:
- Pre-R24 baseline: 6 teller and 17 branch reports.
- Current deployed R24 state: 2 teller reports (`T-1`, `T-2`) and 4 branch
  reports (`B-01` through `B-04`).

All five live citations were R24. The older R2 result was removed, and the
evidence contained the baseline paragraph plus teller, branch, and detailed
requirements tables. The retrieval trace records original and expanded queries,
current-state intent, candidate depth, effective release, and how release scope
was resolved.

### Validation
- Focused temporal/answer/API/conversation suite: 41 passed.
- Full regression suite: 296 passed.
- Exact live query answered 2 teller and 4 branch reports.
- Live trace: `b286e5c6-c4ed-4338-af10-9b95927650a2`.
- Persisted live trace passed `r24_current_teller_and_branch_report_state`.
- Live validation made one embedding request and one `gpt-5-mini` generation
  request; current-state candidate expansion did not make a second embedding
  request.
- Existing upstream Starlette `TestClient`/HTTPX warning remains non-failing.

### Failure-mode thinking
Tests cover numeric release ordering, explicit-filter precedence,
conversation-derived release scope, older-release removal, missing or malformed
release metadata, non-temporal “current application date,” incomplete answers,
baseline-as-current answers, R2 citation leakage, and missing table citations.

The current resolver is deterministic and deliberately narrow. If future
conversations compare multiple releases or the corpus includes draft/unreleased
documents, explicit lifecycle metadata and a richer query planner will be
required instead of assuming every indexed release is deployed.
