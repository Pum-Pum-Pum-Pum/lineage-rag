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

### User answer evaluation - Step 94 - 2026-07-29

#### Overall verdict
Pass. All five answers correctly identify the UI state, reliability,
auditability, observability, and validation boundaries.

#### What was strong
- Q1 clearly separates durable backend history from transient UI state and
  correctly requires reconciliation after each Streamlit rerun.
- Q2 correctly treats readiness as a cost-saving point-in-time signal rather
  than a guarantee that dependencies will remain available during submission.
- Q3 correctly connects accepted user input to audit and retry while rejecting
  hidden failures or fabricated assistant outcomes.
- Q4 gives a strong observability boundary: expose grounding, trace, budget,
  usage, and cost signals while withholding secrets and unsafe internal details.
- Q5 accurately separates deterministic contract/logic tests from live
  rendering and wiring evidence, then names the major production guarantees that
  remain unproven.

#### Precision improvements
- Backend reconciliation should also clear or replace a session-selected
  conversation when that conversation has been archived or removed elsewhere.
- A successful `/ready` result does not cover every cost-bearing dependency:
  model APIs are intentionally not called by readiness, and even checked
  dependencies can fail immediately afterward.
- Persistent debug details across sessions would require a backend API or trace
  lookup contract; current Streamlit state only retains details returned during
  the active UI session.

#### Stronger interview-quality answer
Durable backend storage is authoritative because it survives Streamlit reruns,
browser refreshes, and process restarts. Session state is appropriate only for
replaceable display caches and must be reconciled with the backend, including
clearing stale selections.

Readiness blocks work when known required configuration, artifacts, or Qdrant
state is unavailable, reducing wasted cost. It is a point-in-time check and does
not call model APIs, so submission still needs independent safe handling for
dependencies that fail or change after the check.

An accepted user message is an auditable retryable event. If downstream work
fails, the UI should display the resulting partial turn explicitly rather than
erase history or represent an assistant response that never existed.

The debug panel should expose citations, release lineage, retrieval scores,
sufficiency, trace IDs, context budgeting, token usage, and estimated cost.
Secrets, credentials, raw error bodies, stack traces, local paths, and
unrestricted logs must remain hidden.

Unit tests establish typed contracts, validation, state-selection logic,
readiness ordering, partial-turn detection, and safe error mapping. Browser QA
establishes real Streamlit startup, rendering, control wiring, lifecycle reruns,
and browser-console health for the tested path. Neither proves high concurrency,
full accessibility, responsive behavior, long-history performance, streaming,
idempotent retry, real model quality, or cross-browser compatibility.

#### Gate
Step 94 interview gate accepted. Proceed to Step 95.

## Step 95 - Complete, token-bounded prompt evidence

### Questions asked
1. Why must the evidence sent to the LLM be separate from the short citation
   preview returned to the API and Streamlit UI?
2. Why does the evidence budget admit complete ranked units instead of
   truncating text exactly at the remaining character or token limit?
3. Why should an oversized highest-ranked evidence unit cause a safe refusal
   before the LLM call?
4. Which part of the reported BOR problem does Step 95 fix, and why can the
   “17 reduced to 4” answer still fail?
5. What do the new regressions prove, and what answer-level evaluation is still
   required before claiming the BOR scenario is solved end to end?

### Correct answer key
1. The model needs complete factual evidence, while clients need compact,
   bounded display metadata. Reusing a 240-character preview as prompt evidence
   can remove a decisive acronym or table row and force an unnecessary refusal.
2. Mid-unit truncation can separate labels from values or omit later rows,
   changing a table's meaning. Whole-unit admission preserves integrity and
   makes budget behavior deterministic; exact model tokenization can later
   replace the lightweight estimate.
3. Calling the model with a knowingly incomplete highest-ranked unit risks an
   unsupported or misleading answer and wastes cost. A deliberate application
   refusal is observable, testable, and safer than accidental truncation.
4. It fixes the prompt-construction loss: BOR and B-04 facts beyond character
   240 can now reach the model. It does not fix hybrid ranking; the trace showed
   that the decisive R24 realignment tables were outside hybrid top-5 for the
   second natural query.
5. The tests prove full selected evidence reaches the prompt, previews stay
   short, whole-unit budgeting works, and oversized evidence avoids an LLM call.
   We still need a natural-language answer-level case asserting the supported
   facts “17 currently” and “4 after R24,” plus retrieval recall/rank checks for
   all required supporting units.

### Gate
Step 95 implementation is complete. Await the user's answers before proceeding
to the retrieval-ranking and current-state evaluation step.

## Step 96 - Weighted Reciprocal Rank Fusion

### Questions asked
1. Why is adding normalized dense cosine scores to lexical relevance scores
   unsafe even when each list is divided by its own maximum?
2. How does weighted RRF combine retrievers, and why is the rank constant
   important for a short candidate list?
3. Why was the final RRF score normalized to `[0,1]`, and how is that different
   from normalizing the dense and lexical input scores?
4. Why is a lexical-heavy weight defensible for this corpus, and what evaluation
   risk would make that choice unsafe to generalize?
5. What did the failed live answer prove even after retrieval improved?

### Correct answer key
1. The raw scores have different definitions and distributions. Dividing each
   by its query maximum does not calibrate them; it merely makes each winner
   equal to one. With fixed weights, a mediocre dense-only result could still
   outrank lexical rank 1 regardless of its absolute lexical relevance.
2. Each result contributes `weight / (k + rank)` for every retriever that found
   it. Smaller `k` preserves more separation between early ranks; large `k`
   makes ranks within a short list nearly equal and can over-reward any overlap.
3. Final normalization preserves ordering while producing a bounded score
   compatible with logging and the existing sufficiency threshold. It does not
   compare or combine the retrievers' raw score magnitudes; only RRF rank
   contributions have already been combined.
4. Exact report IDs, abbreviations, action words, and table headers are
   unusually important in functional specifications, and lexical retrieval
   ranked the decisive tables first. The weight must still be evaluated across
   semantic paraphrases and unanswerable cases; optimizing only the BOR example
   would overfit and could damage dense-retrieval recall.
5. It proved that retrieval recall is necessary but insufficient. Even with
   detailed R24 change evidence in the prompt, the model applied the wrong
   meaning of “current,” so explicit latest-deployed-release semantics,
   release-aware query scoping, and answer-level fact assertions are required.

### Gate
Step 96 is complete. Await the user's answers before implementing current-state
temporal synthesis and conversation-aware release scoping.

### User answer evaluation - Step 96 - 2026-07-30

#### Overall verdict
Pass. All five answers correctly explain score calibration, RRF mechanics,
bounded final scoring, corpus-specific weighting, and the separation between
retrieval recall and temporal answer synthesis.

#### What was strong
- Q1 precisely identifies that per-query maximum division does not calibrate
  dense and lexical scores or make their magnitudes comparable.
- Q2 gives the correct weighted RRF equation and accurately explains how the
  rank constant changes top-rank separation.
- Q3 correctly distinguishes bounded output-score normalization from combining
  incompatible retriever input scores.
- Q4 justifies lexical emphasis using exact identifiers and table language while
  naming paraphrase and hard-negative validation as the protection against
  overfitting.
- Q5 correctly interprets the failed live answer as a temporal-policy defect
  after retrieval recall improved.

#### Precision improvements
- Final RRF normalization also preserves ranking because every fused result is
  divided by the same query-level maximum possible RRF score.
- RRF weights and the rank constant should be selected against a representative
  evaluation set containing semantic paraphrases, exact identifiers,
  multi-table questions, and unsupported queries rather than one BOR case.
- Release-aware retrieval alone is insufficient: answer generation must treat
  the latest relevant deployed release as the effective current state and
  distinguish its pre-change baseline from its resulting state.

#### Stronger interview-quality answer
Dense cosine similarity and lexical relevance have different statistical
meanings and distributions. Dividing each list by its maximum only makes both
winners equal to one; it does not calibrate the remaining values. Weighted RRF
avoids that comparison by summing `weight / (k + rank)` contributions. For
short candidate lists, a smaller `k` retains meaningful separation among early
ranks, while a large `k` compresses them and disproportionately rewards overlap.

The combined RRF score can then be divided by the maximum possible RRF score to
produce a bounded diagnostic and sufficiency value without changing result
ordering or comparing raw retriever magnitudes. Lexical-heavy weighting is
defensible for identifier- and table-heavy functional specifications, but it
must be validated across semantic paraphrases, hard negatives, and unsupported
questions to avoid optimizing only the BOR case.

The failed live answer shows that evidence recall is necessary but not
sufficient. The model received relevant R24 change evidence but interpreted the
pre-change “Existing Functionality” section as current. The next layer must
resolve release scope and explicitly define current state as the result after
applying the latest relevant deployed release.

#### Gate
Step 96 interview gate accepted. Proceed to Step 97 current-state temporal
synthesis and conversation-aware release scoping after resolving the reported
Codex diff-viewer issue.

## Step 97 - Current-state temporal synthesis and release scoping

### Questions asked
1. Why may conversation memory resolve the release referenced by “it” while
   still being forbidden as factual evidence for the answer?
2. Why does a current-state query retrieve a wider candidate set before
   selecting the latest release, instead of immediately filtering to the
   globally highest release?
3. Why must release labels be ordered numerically, and what failure would
   lexicographic ordering cause?
4. Why is `Existing Functionality` in R24 treated as the pre-change baseline
   rather than the current deployed result in this project?
5. What does the answer-level evaluation prove beyond retrieval recall, and
   what future corpus change would invalidate the “all releases are deployed”
   assumption?

### Correct answer key
1. Memory is appropriate for intent resolution because it records the user's
   conversational reference, but it may be stale, summarized, or unsupported.
   It can select R24 for retrieval; the model may make functional claims only
   from newly retrieved R24 evidence and citations.
2. The globally highest release may belong to an unrelated topic. Broad
   retrieval first establishes a relevance-bounded candidate set; the highest
   release among those candidates is then selected. Expanding once also allows
   baseline and change tables to survive before the final top-k is applied,
   without paying for a second query embedding.
3. String ordering can place `R9` after `R10` because it compares characters.
   Parsing the numeric suffix produces the actual lineage order:
   `R2 < R9 < R10 < R24`.
4. The corpus represents deployed release changes. R24's Existing Functionality
   describes the state entering that change, while its realignment tables define
   the result after deployment. Therefore 6/17 is historical baseline context
   and 2/4 is the current state.
5. Retrieval recall proves that evidence was available; answer evaluation also
   verifies temporal wording, derived counts, report IDs, release isolation,
   and citation coverage. If drafts, proposals, rejected changes, or partially
   deployed releases enter the corpus, release number alone can no longer prove
   effective current state; lifecycle/deployment metadata becomes mandatory.

### Gate
Step 97 implementation is complete. Await the user's answers before proceeding
to the next step.

### User answer evaluation - Step 97 - 2026-07-30

#### Overall verdict
Pass. All five answers correctly explain the boundary between conversational
intent and evidence, relevance-bounded release selection, numeric lineage,
baseline-versus-resulting-state interpretation, and answer-level evaluation.

#### What was strong
- Q1 correctly allows memory to resolve references while requiring fresh R24
  evidence and citations for every functional claim.
- Q2 accurately explains both reasons for wider candidate retrieval: selecting
  the latest relevant release and retaining complementary baseline/change
  evidence without a second embedding request.
- Q3 gives the correct numeric ordering and clearly rejects lexicographic
  comparison.
- Q4 applies the project's deployed-corpus rule correctly: 6/17 is the R24
  incoming baseline and 2/4 is the resulting current state.
- Q5 distinguishes evidence availability from answer correctness and correctly
  identifies when deployment lifecycle metadata becomes mandatory.

#### Precision improvements
- The selected release is the highest release among relevance-bounded retrieved
  candidates, not the highest release anywhere in the corpus.
- Lifecycle metadata would be required not only for non-final releases, but also
  for drafts, rejected changes, partial deployments, rollbacks, and releases
  deployed only to selected environments or customers.
- Answer evaluation should verify that cited evidence entails the associated
  claims, in addition to checking that required citation units are present.

#### Stronger interview-quality answer
Conversation memory may resolve a reference such as “it” because it captures
the user's prior intent, but it may be stale, summarized, or unsupported. It may
therefore scope retrieval to R24 but cannot prove any functional claim; those
claims require newly retrieved R24 evidence and citations.

Current-state retrieval first gathers a wider relevance-bounded candidate set,
then selects the highest numeric release represented in that set. This avoids a
globally newer but unrelated document, preserves baseline and change tables,
and avoids a second query-embedding request. Numeric parsing is essential
because string ordering can misorder `R9` and `R10`.

For this deployed corpus, R24's Existing Functionality section describes the
state entering the release, while its realignment tables define the resulting
production state. Thus 6/17 is historical baseline context and 2/4 is current.
Answer-level evaluation must verify temporal framing, counts, identifiers,
release isolation, citation coverage, and ultimately citation entailment—not
just retrieval recall. If drafts, rejected changes, partial deployments,
rollbacks, or environment-specific releases enter the corpus, explicit
lifecycle and deployment metadata becomes mandatory.

#### Gate
Step 97 interview gate accepted.
