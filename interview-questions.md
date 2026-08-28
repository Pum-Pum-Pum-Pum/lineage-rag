# Active interview record

The complete Phase 2 interview and evaluation history for Steps 150-183 is
archived in `interview-questions_phase2.md`.

## Current gate

- Steps 181-183 learner gate: **accepted, 9/9**.
- Code and combined capabilities are activation-ready but remain inactive.
- Activation and paid/internal-evidence smoke testing remain separate explicit
  approval boundaries.
- No interview questions are currently pending.

Questions and evaluated answers for the next completed three-step batch will be
appended here.

## Steps 184-186 - Knowledge-mode Streamlit integration

### Step 184 - Durable mode and conversation-context wiring

1. Why must `knowledge_mode` and `analysis_kind` pass through the conversation
   request instead of existing only on the direct `/query` endpoint?
2. Why may conversation memory help resolve a code follow-up while remaining
   forbidden as evidence, and what retrieval risk comes from adding memory to
   the query representation?
3. Why must the UI verify that the readiness response names the same knowledge
   mode requested before it submits a potentially paid turn?

### Step 185 - Lane-aware controls and evidence rendering

4. Why must the UI keep the knowledge-lane selector separate from the
   dense/lexical/hybrid retrieval technique?
5. Why does combined mode need both `requested_claim_supported` and independent
   section statuses, even when one section contains strong evidence?
6. Why must document and code citations use distinct labels and preserve code
   snapshot/path/symbol/line identity rather than showing only a text preview?

### Step 186 - Failure and rollback UX

7. What does the deterministic disabled -> enabled -> disabled UI test prove,
   and which multi-process or production behaviors remain unproven?
8. Why is hiding an unsupported FDD filter in code/combined mode safer than
   displaying it and allowing the backend to ignore it?
9. After these UI tests pass, what configuration, readiness, live-smoke,
   provenance, privacy, and rollback evidence is still required before code or
   combined mode may be deliberately activated?

Gate status: **awaiting learner answers.** Code and combined modes remain
activation-ready but inactive; no paid/internal-evidence operation was run.

### Evaluation

Overall result: **9/9 accepted.**

1. **Accepted with precision.** The fields belong to each durable turn's
   execution/evidence contract and must survive the conversation boundary. A
   conversation may intentionally change modes between turns, so mode is not a
   permanent property of the whole conversation unless a future schema makes
   that policy explicit.
2. **Accepted.** Memory can resolve references but is generated contextual state,
   not authoritative evidence. Query enrichment can introduce stale-topic bias,
   identifier drift, prior-message prompt injection, extra token cost, and
   privacy exposure; material claims still require freshly retrieved source
   units and valid citations.
3. **Accepted.** Exact mode readiness prevents silent lane fallback and stops
   embedding/generation cost before it begins when the requested evidence lane
   or activation state is unavailable.
4. **Accepted.** Knowledge authority answers *where evidence may come from*;
   retrieval mode answers *how eligible evidence is searched*. Combining them
   would couple a product contract to one retrieval implementation.
5. **Accepted.** Global requested-claim support prevents a strong subsection
   from overstating the whole answer, while section states preserve useful
   partially grounded results and explicit unknowns.
6. **Accepted.** Separate `[F#]` and `[C#]` namespaces preserve claim authority.
   Snapshot, path, symbol, and line identity make a code claim reproducible
   against one immutable source occurrence rather than merely a preview.
7. **Accepted with precision.** The test proves fail-closed submission and
   reversible routing in the deterministic UI/API-boundary contract. It does not
   restart real processes or prove configuration propagation, in-flight request
   behavior, multi-worker consistency, semantic quality, or disaster recovery.
8. **Accepted.** A visible ignored filter is a false security/scoping control.
   Hiding it avoids presenting an unconstrained answer as though the requested
   restriction were enforced.
9. **Accepted.** The answer covers effective configuration, mode readiness,
   authorized live smoke calls, privacy, provenance, evaluation/SME evidence,
   safe refusal, retained generations, and rollback. Activation must also record
   the approver, exact identities/configuration diff, effective post-restart
   state, smoke trace IDs/results, rollback trigger and owner, and the final
   activation outcome.

Gate status: **Steps 184-186 learner gate accepted, 9/9.** Code and combined
modes remain activation-ready but inactive. Deliberate activation and any paid
internal-evidence smoke call remain separate explicit approval boundaries.

## Steps 187-189 - Approval-bound activation preparation

### Step 187 - Activation and approval contracts

1. Why must an activation request bind runtime source hashes as well as artifact,
   collection, lineage, and configuration identities?
2. Why must activation approval be a separate hash-bound artifact rather than a
   Boolean field edited into the original request?
3. What do SHA-256 identities protect here, and why do they not authenticate the
   human approver or make the artifacts tamper-proof?

### Step 188 - Offline activation preflight

4. Why is `ready_to_apply=false` the correct result when four technical checks
   pass but the approval check is missing?
5. Why must configuration drift invalidate the approved target instead of being
   silently absorbed into the activation operation?
6. Why must activation reports exclude secrets even though the settings object
   necessarily loads credentials for the application?

### Step 189 - Atomic config switch and rollback mechanics

7. What does `fsync` followed by same-filesystem `os.replace` protect during the
   `.env` update, and which crash/durability guarantees can still depend on the
   operating system and filesystem?
8. Why must dry-run be the default and an unexpected current flag value fail,
   rather than making activate/rollback idempotently force the requested value?
9. Why does an atomic `.env` edit not constitute a complete activation, and what
   process restart, effective-config, readiness, live-smoke, trace, and rollback
   evidence remains required?

Gate status: **awaiting learner answers, an explicit disabled `.env` key, and
explicit activation approval.** The pending real preflight is intentionally
blocked; `.env` remains unchanged and no service restart, paid call,
internal-evidence disclosure, or activation occurred.

### Evaluation

Overall result: **9/9 accepted.**

1. **Accepted.** Activation authority applies to one exact runtime and evidence
   state. Binding source, configuration, collection, artifact, and lineage
   identities prevents a reviewed request from being reused for different code
   or data merely because generation labels look familiar.
2. **Accepted.** The request and approval are separate immutable events. Editing
   the request in place would erase the submitted state and weaken evidence of
   what the approver actually reviewed.
3. **Accepted.** SHA-256 supports change detection only when compared with a
   separately trusted identity. It neither authenticates a person nor stops an
   actor with write access from replacing content and recomputing hashes;
   identity, authorization, ACLs, and ideally signed/tamper-evident retention are
   separate controls.
4. **Accepted.** Technical eligibility cannot grant operational authority.
   Missing one required authorization gate must keep the conjunction false even
   when every infrastructure check passes.
5. **Accepted.** Approval is scoped to exact inputs. Absorbing drift would turn a
   reviewed activation into an unreviewed one and destroy reproducibility.
6. **Accepted.** Reports require proof of configuration presence and identity,
   not credential values. Persisting secrets would expand disclosure, retention,
   backup, and incident-response risk without strengthening activation evidence.
7. **Accepted with an implementation-specific refinement.** File `fsync` makes
   the temporary file's bytes more durable and same-filesystem `os.replace`
   prevents partial-file visibility. The current code does not `fsync` the parent
   directory after replacement, so rename-metadata durability after sudden power
   loss remains filesystem/OS dependent. Neither operation reloads processes or
   proves application state.
8. **Accepted.** Dry-run requires deliberate escalation to mutation. Refusing an
   unexpected start value prevents a stale command, repeated action, or operator
   misunderstanding from silently changing service state.
9. **Accepted.** Complete activation still needs controlled process restart,
   effective-config verification, exact mode readiness, separately authorized
   live smoke calls and internal-evidence disclosure, trace/citation inspection,
   multi-worker consistency evidence, and an owned immediate rollback decision.

Gate status: **Steps 187-189 learner gate accepted, 9/9.** The activation request
remains pending. The real preflight is still blocked by the missing explicit
disabled `.env` key and missing approval; no activation authority is inferred
from these interview answers.

## Steps 190-192 - Disabled baseline and execution-evidence contract

### Step 190 - Explicit disabled baseline and durability

1. Why is an explicit `CODE_MODES_ENABLED=false` entry operationally stronger
   than relying only on the application default, even though both currently
   disable the feature?
2. Why must the initializer refuse an existing `true`, duplicate keys, or an
   invalid value instead of normalizing them automatically?
3. What does a reported `parent_directory_fsynced=false` mean for atomicity and
   crash durability, and why should the system expose rather than hide it?

### Step 191 - Final pending request and preflight

4. Why must changing the activation mechanism itself invalidate the bound
   request even when the serving query code and collection identities are
   unchanged?
5. Why is a 5/6 preflight with only approval missing still correctly blocked,
   rather than treated as proportionally ready?
6. When configuration changes legitimately after a request is created, why is a
   new request/preflight safer than updating hashes inside the existing request?

### Step 192 - Activation execution evidence

7. Why can an activation evidence model validate structure and consistency but
   not prove that an operator truthfully observed a restart or readiness result?
8. Why must paid-smoke authorization and internal-evidence disclosure
   authorization both be present before smoke trace IDs can complete the gate?
9. Which evidence fields distinguish configuration written, process activated,
   capability ready, semantically smoke-tested, and safely rollback-capable
   states, and why must they remain separate?

Gate status: **awaiting learner answers and explicit activation approval.** The
real preflight is 5/6 with only approval missing. The explicit flag remains
`false`; no restart, paid call, disclosure, or activation occurred.

### Evaluation

Overall result: **9/9 accepted.**

1. **Accepted.** An explicit disabled value is persisted, observable, and
   reviewable independently of source-code defaults. It prevents a future
   default change or different runtime build from silently changing the intended
   deployment state.
2. **Accepted.** Existing `true` may represent an authorized active state,
   duplicates create parser/precedence ambiguity, and invalid values require
   human correction. Normalizing any of them would turn a baseline initializer
   into an uncontrolled state-changing repair tool.
3. **Accepted.** Atomic replacement prevents readers from observing partial
   content. Without a successful parent-directory `fsync`, persistence of the
   renamed directory entry across abrupt power/storage failure remains dependent
   on Windows and the underlying filesystem.
4. **Accepted.** Approval covers both the target and the mechanism used to reach
   it. Changing the switching implementation changes the reviewed operational
   contract even if every serving collection and artifact remains identical.
5. **Accepted.** Required activation checks form a conjunction, not a percentage
   score. Authorization cannot be inferred from technical readiness, so one
   missing approval keeps the transition fully blocked.
6. **Accepted.** A new request preserves the original proposal and its decision
   history. Replacing hashes in place would retroactively expand any prior review
   or approval to inputs the approver never saw.
7. **Accepted.** Schema and hash validation establish format, internal
   consistency, and exact referenced bytes. Truthful operational claims require
   independently collected observations, authenticated actors, and preferably
   system-produced evidence rather than operator-entered booleans alone.
8. **Accepted.** Cost authority and disclosure authority govern independent
   risks. For the live smoke path this covers paid query embedding/answer
   generation and transmission of retrieved internal FDD/PLSQL excerpts; neither
   permission implies the other.
9. **Accepted.** The answer correctly separates persisted configuration,
   effective process state, bounded readiness, controlled functional evidence,
   and recoverability. In the implemented ledger those map to configuration and
   restart flags, effective enabled state, code/combined readiness, smoke trace
   IDs, rollback owner, and rollback rehearsal.

Gate status: **Steps 190-192 learner gate accepted, 9/9.** The real activation
preflight remains 5/6 and blocked only on approval. Interview acceptance is
learning evidence, not activation authorization; `CODE_MODES_ENABLED=false`
remains unchanged.

## Steps 193-195 - Approved activation attempt and fail-closed rollback

### Step 193 - Approval-bound authority

1. Why must the approval identity bind the exact request identity even when the
   approver and requested action are unchanged from a prior attempt?
2. Why do paid-smoke authorization and internal-evidence disclosure remain two
   independent booleans rather than one general approval?
3. What does a 6/6 offline preflight prove, and what live behavior does it still
   leave unproven?

### Step 194 - Effective activation and readiness

4. Why must process restart and observed mode-specific readiness follow the
   atomic `.env` transition before sending a paid smoke request?
5. Why did the stale process retaining port 8000 mean the first rollback restart
   was not yet a valid rollback, even though `.env` already said `false`?
6. What do code/combined readiness checks establish, and why can they still pass
   before a live model returns an invalid answer contract?

### Step 195 - Paid smoke failure and rollback

7. Why does one successful code smoke plus one HTTP 400 combined smoke require
   rollback instead of accepting code mode alone under this activation request?
8. Why is preserving the HTTP error body and partial successful trace important,
   and why would automatically retrying the combined request be unsafe here?
9. What evidence proves the rollback is effective, and what investigation is
   required before preparing a new activation request?

Gate status: **awaiting learner answers.** The activation attempt is closed as
failed and rolled back. FDD mode remains ready; code/combined serving is disabled.

### Evaluation

Overall result: **9/9 accepted.**

1. **Accepted.** Approval is authority over one exact proposed state, evidence
   set, and transition mechanism. Any changed hash or generation requires a new
   request rather than silently expanding old authority.
2. **Accepted.** Cost and disclosure are independent risk decisions. Neither
   financial authority nor technical access implies permission to transmit
   internal evidence.
3. **Accepted.** Offline preflight proves only the defined static prerequisites
   and authorization. It cannot prove process reload, live dependency access,
   model-contract stability, successful serving, or operational rollback.
4. **Accepted.** Persisted configuration is inert until the serving process loads
   it. Readiness after restart establishes the effective process and bounded
   dependencies before paid traffic is attempted.
5. **Accepted.** The answer correctly identifies the process/port identity gap.
   A new process failing to bind means the old process still serves observations,
   so a configuration-file rollback cannot yet be called an operational rollback.
6. **Accepted.** Readiness is intentionally bounded and does not invoke every
   retrieval/generation path. Provider output, evidence-specific behavior,
   contract parsing, timeouts, and resource failures remain live-path risks.
7. **Accepted.** The approved request covered both code and combined modes as one
   activation gate. Partial semantic success cannot narrow that scope after the
   fact or overrule the defined fail-closed policy.
8. **Accepted.** Partial traces bound disclosure, cost, evidence, and the exact
   failure boundary. Automatic retries would add cost/disclosure, obscure the
   original incident, and violate the explicit two-request authorization.
9. **Accepted.** Effective rollback requires the disabled configuration, correct
   serving process, reconciled port ownership, mode-specific readiness, and a
   healthy retained FDD path. Reactivation requires local diagnosis, a targeted
   deterministic regression, and a new hash-bound request after any runtime fix.

Gate status: **Steps 193-195 learner gate accepted, 9/9.** The failed activation
attempt remains closed and rolled back; these answers do not authorize another
paid request or activation attempt.

## Steps 196-198 - Combined-contract failure containment

### Step 196 - Evidence-bounded diagnosis

1. Why does the code path prove the likely failure class but not the exact
   historical malformed model response from the failed smoke?
2. Why was returning HTTP 400 for generated contract failure a misleading API
   boundary even though the client request itself was valid?
3. Why must diagnosis use the successful trace, access log, and absence of a
   combined trace together rather than treating any one artifact as sufficient?

### Step 197 - Grounded graceful refusal

4. Why should malformed combined output set `requested_claim_supported=false`
   and remove citations instead of returning partially parseable sections?
5. Why catch contract-validation `ValueError` but continue propagating provider,
   transport, or empty-response failures through their distinct failure paths?
6. What is the difference between storing malformed raw output in a restricted
   trace and presenting it to the user as grounded answer content?

### Step 198 - Offline regression and reactivation boundary

7. What does the fake malformed-response test prove, and what live-model
   stability property does it still leave unproven?
8. Why does preserving an HTTP error body improve diagnosis without authorizing
   an automatic retry or proving that a paid call should be repeated?
9. Why must this runtime fix produce a new activation request even though the
   collections, artifacts, lineage mappings, and `.env` target are unchanged?

Gate status: **awaiting learner answers.** The deterministic containment fix
passes the full regression, but code/combined serving remains disabled and no
new paid operation or activation is authorized.

### Evaluation

Overall result: **9/9 accepted.**

1. **Accepted.** The surviving artifacts bound the path and failure category,
   but the original body was not persisted. Reconstructing its precise syntax or
   validation defect would exceed the evidence.
2. **Accepted.** The user supplied a valid request; the server failed while
   validating generated output. Treating that as a client 400 incorrectly assigns
   responsibility and encourages the wrong remediation.
3. **Accepted.** The successful trace establishes normal persistence, the access
   log establishes the 400 boundary, and the missing trace locates the loss before
   normal trace completion. No one artifact proves all three facts.
4. **Accepted.** An invalid envelope makes section identity, support state, and
   citation association untrustworthy. Failing closed prevents plausible fragments
   from acquiring unsupported authority.
5. **Accepted.** Contract failure means a response exists but is unusable under
   the grounded schema. Provider/transport/empty responses have different retry,
   availability, cost, and incident semantics and must remain distinguishable.
6. **Accepted.** Restricted trace data is diagnostic input under access and
   retention controls; user-facing evidence must first satisfy grounding and
   citation contracts. Retention never grants claim authority.
7. **Accepted.** The fixture proves deterministic containment of the modeled
   malformed-output class. It does not reproduce the lost historical payload or
   prove frequency, stability, or all provider failure variants.
8. **Accepted.** Capturing an error improves observability only. Retry changes
   cost and disclosure and needs a separately bounded policy or authorization.
9. **Accepted.** Runtime source and failure semantics are part of the approved
   serving contract. Unchanged collections do not allow an old approval to cover
   changed execution behavior.

Gate status: **Steps 196-198 learner gate accepted, 9/9.** The next operation is
limited to offline readiness/request regeneration; activation and paid calls
remain unauthorized.

## Steps 199-201 - Rebuild the activation authority boundary

### Step 199 - Runtime hash coverage

1. Why must the activation hash bind answer parsing and refusal code as well as
   API routes and orchestration?
2. Why should the exact smoke runner be approval-bound rather than treated as an
   interchangeable operator utility?
3. What security or reliability failure could occur if a runtime file affects
   serving behavior but is omitted from `ACTIVATION_RUNTIME_FILES`?

### Step 200 - Offline readiness regeneration

4. Why can the prior SME answer ledger remain valid after changing malformed
   response containment, while the activation request cannot?
5. What do the 9/9 readiness result and retained v4/v5 plus code v1/v2
   collections establish—and what do they not establish?
6. Why is it useful to regenerate readiness without enabling code mode or
   invoking OpenAI?

### Step 201 - New request and stale-approval rejection

7. Why must an approval for request `d1de15f...` fail against request
   `36e3c1c...` even when the same person would approve both?
8. Why is 5/6 with approval missing a completely blocked state rather than a
   nearly activated state?
9. What exact additional authority is required before the new request may alter
   `.env`, restart services, or issue paid smoke requests?

Gate status: **awaiting learner answers and new explicit approval.** Request
`36e3c1c...` is pending, the preflight is 5/6, and code/combined modes remain
disabled.

### Evaluation

Overall result: **9/9 accepted.**

1. **Accepted.** Parsing, support-state derivation, citation handling, and
   refusal behavior are user-visible runtime semantics. Binding only routes would
   leave a material behavioral gap in the approved contract.
2. **Accepted.** The runner determines case scope, request count, retries,
   validation, and evidence persistence. Approval of a smoke operation must bind
   the exact implementation that controls cost and disclosure.
3. **Accepted.** An omitted behavior-changing file creates an unhashed path by
   which execution can change while the request identity remains stable. That is
   both a provenance defect and an authorization bypass.
4. **Accepted.** The SME ledger judges already captured semantic evidence; the
   activation request authorizes a specific future runtime transition. A runtime
   containment change does not rewrite the reviewed answers, but it does change
   what would be served.
5. **Accepted.** Readiness proves all defined bounded prerequisites, not live
   model stability, semantic smoke success, authorization, concurrency, or
   production behavior.
6. **Accepted.** Offline regeneration validates identities and local dependencies
   without exposing internal evidence, incurring cost, or changing serving state.
   This keeps diagnosis and authorization boundaries separate.
7. **Accepted.** Request identity is the object of approval. A changed runtime
   contract is materially different even when collections and the approver remain
   unchanged.
8. **Accepted.** Required checks are conjunctive controls, not a readiness score.
   Missing approval cannot be compensated for by additional technical successes.
9. **Accepted.** The answer identifies all distinct authorities: the exact new
   request, runtime/configuration transition, internal-evidence disclosure, and
   paid smoke usage. It also preserves the correct operational order and trace/
   rollback requirements.

Gate status: **Steps 199-201 learner gate accepted, 9/9.** Request
`36e3c1c244def6be37d8b1ff9ee02cbc3f072623a24e16b42f65af05761a0fd4`
remains pending approval. `CODE_MODES_ENABLED=false`; no restart, paid request,
or activation is authorized by these interview answers.

## Steps 202-204 - Second activation attempt and traceable safe rollback

### Step 202 - Exact approval and preflight

1. Why did the new approval make the preflight 6/6 while leaving runtime state
   unchanged until the separate apply operation?
2. Why must the approval separately enumerate rollback, paid smoke, and internal
   evidence disclosure instead of inferring them from activation permission?
3. What does the approval identity protect, and which approver-authentication or
   tamper-prevention properties does a SHA-256 identity still not provide?

### Step 203 - Runtime transition and bounded readiness

4. Why was verifying exact port release before launching replacement services
   stronger than merely starting another FastAPI/Streamlit process?
5. Why did HTTP 200 readiness for combined mode not justify keeping the feature
   enabled before the paid smoke completed?
6. Which combined-readiness checks help rule out missing collections/artifacts,
   and why do they not test prompt/schema alignment?

### Step 204 - Safe refusal and rollback evidence

7. Why is the combined result a prompt/schema failure rather than a retrieval or
   lineage failure, given the persisted trace evidence?
8. Why did HTTP 200 plus a safe refusal still fail a positive activation smoke,
   and why is that the correct semantic gate behavior?
9. What deterministic fix and tests are required before requesting another paid
   activation attempt, and why is the current approval no longer reusable after
   that fix?

Gate status: **awaiting learner answers.** The second activation attempt is
closed and rolled back. Code/combined serving remains disabled. Both authorized
paid smokes were consumed; no further paid request or retry is authorized.

### Evaluation

Overall result: **9/9 accepted.**

1. **Accepted.** Approval completed the authorization gate for one exact request;
   it did not apply configuration or alter a running process.
2. **Accepted.** Activation, rollback, paid use, and internal-evidence disclosure
   have different consequences and therefore require explicit independent scope.
3. **Accepted.** The hash binds content identity and detects drift. It neither
   authenticates `AIAgentSmith` nor prevents an authorized writer from replacing
   artifacts and computing new hashes.
4. **Accepted.** Port ownership makes process identity observable. Without
   release verification, readiness could accidentally test a stale listener.
5. **Accepted.** HTTP readiness is a bounded availability/dependency result; it
   cannot substitute for a positive semantic smoke.
6. **Accepted.** Collection and artifact checks are local infrastructure
   properties. Prompt/schema alignment is exercised at the later generation
   boundary unless a separate deterministic contract check is added.
7. **Accepted.** The persisted 8 FDD units, 8 code units, and 3 reviewed mappings
   establish successful evidence construction before validation rejected the
   generated response shape.
8. **Accepted.** HTTP 200 proves handled transport completion. The positive gate
   still requires a supported, contract-valid grounded answer; safe refusal is
   correct containment but not positive-case success.
9. **Accepted.** The required repair is deterministic prompt/schema alignment,
   strict type enforcement, and malformed-output regression coverage. Because
   this changes serving behavior, the prior hash-bound approval cannot be reused.

Gate status: **Steps 202-204 learner gate accepted, 9/9.** Offline repair and
verification may proceed. Code/combined serving remains disabled, and no further
paid call or activation is authorized.

## Steps 205-207 - Structured-output repair and new pending authority

### Step 205 - Prompt and schema alignment

1. Why is a strict JSON Schema stronger than asking the model in prose to return
   a boolean for `requested_claim_supported`?
2. Why should local Pydantic validation remain after enabling provider-side
   Structured Outputs?
3. Why is recording the response-schema hash in each answer trace useful for
   diagnosing future contract failures?

### Step 206 - Deterministic failure-mode coverage

4. Why must the historical object-shaped support field be tested separately
   from syntactically invalid JSON?
5. Why must malformed combined output clear citations even when the raw response
   contains plausible cited prose?
6. What does 16/16 focused success prove, and which live-provider property does
   it still not prove?

### Step 207 - Readiness and activation boundary

7. Why did readiness increase from 9 to 10 checks, and what defect can the new
   check block before a paid operation?
8. Why must approval v2 fail against request v3 even though the FDD, code, and
   lineage artifacts did not change?
9. What exact new authority is required before request
   `f927d16dde75bdf6ef3fc8d96ad07279ae22185b1ea6c3aea030e57b44692fff`
   may change `.env`, restart services, or run paid smokes?

Gate status: **awaiting learner answers and new explicit approval.** Readiness is
10/10, but preflight remains 5/6 because approval is absent.
`CODE_MODES_ENABLED=false`; no paid request or activation is authorized.

### Evaluation

Overall result: **9/9 accepted.**

1. **Accepted.** Strict JSON Schema turns an ambiguous natural-language request
   into an enforceable type contract and rejects an object where a boolean is
   required.
2. **Accepted.** Provider enforcement and local Pydantic validation are separate
   safeguards. The application must independently validate the response before
   trusting support state, sections, or citations.
3. **Accepted.** The schema hash binds each trace to the exact output contract
   used for that call and prevents later schema versions from being assumed
   retroactively.
4. **Accepted.** Invalid JSON fails at decoding, while valid JSON with an invalid
   field type fails schema/model validation. Both must remain observable and
   fail closed.
5. **Accepted.** Plausible prose cannot restore trust after the response envelope
   fails. Clearing citations prevents malformed content from appearing grounded.
6. **Accepted.** The focused suite proves the deterministic local contract and
   containment cases. It does not prove live-provider compliance, availability,
   latency, cost, or end-to-end smoke success.
7. **Accepted.** The tenth readiness check directly covers the prompt/schema
   alignment gap that collection and infrastructure readiness could not detect.
8. **Accepted.** Approval binds runtime behavior, not only knowledge artifacts.
   The prompt/schema repair changed that behavior and therefore invalidated the
   earlier approval.
9. **Accepted with authority clarification.** The response correctly enumerates
   the required new approval scope: exact request/runtime contract, activation
   and rollback, paid usage, and internal-evidence disclosure. Describing that
   requirement is not itself an explicit grant of authority.

Gate status: **Steps 205-207 learner gate accepted, 9/9.** Request
`f927d16dde75bdf6ef3fc8d96ad07279ae22185b1ea6c3aea030e57b44692fff`
remains pending approval. `CODE_MODES_ENABLED=false`; no `.env` change, service
restart, paid smoke, disclosure, or activation is authorized by these interview
answers.

## Steps 208-210 - Approved activation and successful live smokes

### Step 208 - Approval-bound preflight

1. Why does a 6/6 preflight authorize the configuration transition but still not
   prove that the newly started service will pass a live model request?
2. Why are the paid-smoke count and evidence-disclosure permission recorded in
   the approval rather than inferred from general activation authority?
3. What would cause this approval to become stale even though the approver and
   desired `CODE_MODES_ENABLED=true` value stayed the same?

### Step 209 - Process ownership and readiness

4. Why did the stale port-8000 listener block paid execution even though another
   FastAPI process had been started successfully at the operating-system level?
5. Why was it necessary to verify both the virtual-environment parent process
   and the actual child process owning each listening port?
6. What do HTTP 200 results for health, FDD, code, combined, and Streamlit prove,
   and what did they still not prove before Step 210?

### Step 210 - Paid smokes and active state

7. Which trace fields prove that the combined prompt/schema repair reached the
   live provider path and returned a usable grounded response?
8. Why does a rollback dry-run add useful activation evidence without weakening
   the successfully enabled runtime state?
9. What is now legitimately active, and which production properties remain
   outside the evidence established by this local activation?

Gate status: **awaiting learner answers.** Both approved paid smokes passed,
activation execution evidence is complete, and local FDD/code/combined serving
is active with `CODE_MODES_ENABLED=true`. Rollback remains authorized and ready.

### Evaluation

Overall result: **9/9 accepted after answer 9 completion.**

1. **Accepted.** Preflight validates bounded local prerequisites and authority,
   while provider availability, transport, generation, and live schema behavior
   exist beyond that boundary.
2. **Accepted.** Request count bounds cost and repeated disclosure; disclosure
   permission governs whether internal evidence may leave the local boundary.
   Neither control implies the other.
3. **Accepted.** The answer correctly covers behavioral source, prompt/schema,
   runner, routes, configuration, evidence generations, lineage, and activation
   mechanism. A change to any bound identity requires a new request.
4. **Accepted.** An unidentified stale listener breaks provenance because the
   paid request may exercise old code or configuration rather than the approved
   runtime.
5. **Accepted.** Launching a parent does not establish serving identity. The
   actual listener may be a child, reloader, or surviving duplicate and therefore
   must be reconciled explicitly.
6. **Accepted.** Readiness proved reachability and its declared bounded local
   dependencies. Live generation, citations, Structured Outputs, and semantic
   success required the later paid smokes.
7. **Accepted.** The answer names the essential evidence: provider request ID,
   schema identity, valid typed contract, support state, citations, and usage.
   The combined trace provides these as `response_format=json_schema`, the schema
   hash, `contract_valid=true`, boolean support, and validated citations.
8. **Accepted.** A dry-run verifies current-state assumptions and the authorized
   reversal path without disrupting a healthy active service.
9. **Accepted after completion.** The final answer correctly distinguishes the
   active local Phase 2 capability from production evidence. It explicitly keeps
   concurrency/load reliability, long-term provider availability, broad semantic
   quality, monitoring, disaster recovery, and future FDD/code generations
   outside the claims established by two successful local smokes. Authentication,
   authorization, rate controls, TLS, supervision, accessibility, and formal
   production SLOs likewise remain unproven boundaries from the recorded project
   architecture.

Gate status: **Steps 208-210 learner gate accepted, 9/9.** The local Phase 2
FDD/code/combined capability remains active and rollback-capable. This closes the
local Phase 2 activation gate without asserting broader production readiness.

## Steps 211-213 - Bounded Phase 2B analysis tools

### Step 211 - Explicit FDD-search plan and policy

1. Why must `automatic_routing=false` be enforced by the typed policy rather
   than treated as a prompt instruction?
2. Why does the FDD tool consume at most `limit + 1` results instead of
   materializing everything returned by the underlying retriever?
3. Why must plan and invocation identities be hash-validated before executing
   even a read-only retrieval tool?

### Step 212 - Code-search evidence boundary

4. Why must code-search output retain snapshot, path, symbol, and line identity
   even when the source text itself looks sufficient?
5. Why should the tool reject a `CodeRetrievalResult` whose query differs from
   the invocation query rather than simply returning its highly ranked evidence?
6. What do bounded code-tool tests prove, and what retrieval-quality property
   remains unproven until reviewed tool-level evaluation?

### Step 213 - Impact graph and explicit orchestration

7. Why may the impact graph create an FDD-to-code edge only from reviewed
   lineage while still showing unresolved static dependency edges?
8. Why are citeable tool outputs separated from the operational execution trace,
   and which information should the trace retain?
9. Why is this mechanism “bounded deterministic tool orchestration” rather than
   an autonomous agent, and what evaluation must pass before API/UI exposure?

Gate status: **awaiting learner answers.** Steps 211-213 are implemented and the
full regression passes 584 tests. The tools remain offline library capabilities:
automatic routing is disabled, no new paid operation occurred, and runtime/API
integration is deferred until reviewed deterministic tool evaluation passes.

### Evaluation

Overall result: **9/9 accepted after answer 8 correction.**

1. **Accepted.** A typed literal and validated configuration create an
   enforceable, hashable boundary; prompt wording alone cannot prevent a model
   from selecting an unapproved tool.
2. **Accepted.** Reading one extra item distinguishes an exactly full result set
   from overflow while keeping consumption bounded to `limit + 1`.
3. **Accepted.** Hash validation binds tool name, query, limit, position, mode,
   and ordering, preventing altered or stale plans from retaining a trusted
   identity.
4. **Accepted.** Snapshot, path, symbol, and exact lines identify the immutable
   source occurrence needed for reproducible retrieval, impact analysis, and
   citations.
5. **Accepted.** A high score cannot repair provenance mismatch. Returning
   evidence produced for another query would contaminate grounding and make the
   plan trace misleading.
6. **Accepted.** Bounded mechanism tests establish resource and contract
   behavior, not relevance, recall, ranking, evidence completeness, or usefulness
   on representative questions.
7. **Accepted.** Reviewed mappings authorize the cross-lane implementation edge.
   Static unresolved dependencies remain legitimate observed edges only when
   their unresolved state is preserved; they cannot become resolved runtime
   claims.
8. **Accepted after correction.** Citeable outputs retain original FDD/code text
   and provenance for grounded answer construction. Operational traces retain
   plan/policy hashes, invocation IDs, tool names, statuses, counts, evidence
   identities, and safe error types while omitting full source text. Query
   retention remains a separate privacy and retention-policy decision.
9. **Accepted.** Execution is fixed by a caller-authored plan and bounded policy,
   with no model-selected routing or open-ended loop. The proposed evaluation
   areas are correct; the release gate should make them concrete through reviewed
   retrieval/citation cases, budget and failure tests, lineage/impact checks, and
   privacy review before API/UI exposure.

Gate status: **Steps 211-213 learner gate accepted, 9/9.** The bounded tools
remain offline pending reviewed deterministic tool-level evaluation; automatic
routing remains disabled.

## Steps 214-216 - Bounded-tool deterministic evaluation

1. Why do previously reviewed grounded-answer questions not automatically make
   newly derived tool-level expectations SME-reviewed?
2. Why must the evaluator enforce the order of tools in addition to checking the
   set of allowed tools?
3. Why does a free local draft run still require an explicit `--allow-draft`
   switch?
4. What does the exact-symbol success in the code-only case, followed by failure
   in the combined case, tell us about the likely failure boundary?
5. Why should the failing expected symbol remain recorded until SME review rather
   than immediately changing retrieval to force a pass?
6. What does this local lexical evaluation measure, and which dense, hybrid,
   citation, and answer-quality properties remain unmeasured?
7. Why can **5/5 safety checks** not compensate for a **5/6 positive result** and
   an unreviewed manifest?
8. Why should the SME packet contain identities, checks, and observed evidence
   references but omit copied FDD and PL/SQL source text?
9. What must be reviewed or proven before these bounded tools can be exposed in
   the API/UI, including the current real-corpus dependency-analysis boundary?

Gate status: **awaiting learner answers and SME review.** Draft deterministic
result: **5/6 positive**, **5/5 safety**, **0 external calls**. Automatic routing
and API/UI tool exposure remain disabled.

### Learner answers and mentor evaluation

1. **Accepted.** A previously reviewed answer benchmark does not approve newly
   derived tool plans, evidence identities, limits, or success criteria. Those
   constitute a new evaluation contract and require their own SME decision.
2. **Accepted.** Tool order is behavior: later tools may depend on bounded outputs
   and validated identities from earlier tools. Merely allowing the right set of
   tools would not preserve that evidence flow or its safeguards.
3. **Accepted.** The explicit override prevents convenient diagnostic execution
   from being confused with an approved release gate and keeps draft status
   visible in reports.
4. **Accepted.** Since the exact symbol is retrievable in code-only mode, the
   evidence points toward combined selection, ranking, diversity, or final-budget
   pressure rather than absence from the code index.
5. **Accepted.** Preserving the failure prevents tuning against an unverified
   assumption and lets the SME distinguish a benchmark defect from a retrieval
   defect.
6. **Accepted with refinement.** The run measures more than term matching: it
   also checks expected document/path/symbol identities, reviewed lineage use,
   fixed plan execution, limits, and local lexical evidence selection. It still
   does not establish dense/hybrid semantic relevance, completeness beyond the
   cases, citation entailment, generated-answer correctness, or user usefulness.
7. **Accepted.** Safety and positive retrieval are independent gates. Safe
   execution cannot repair missing evidence, and an unreviewed benchmark cannot
   provide release authority.
8. **Accepted.** Identities and observed checks support reproducibility and SME
   review without duplicating sensitive FDD/PLSQL text into another retained
   artifact. Authorized reviewers can inspect the immutable source separately.
9. **Accepted.** The stated gates are correct. Real-corpus dependency precision
   and recall must be evaluated before dependency-derived lineage is trusted;
   until then, combined analysis may rely only on reviewed lineage edges and must
   preserve other dependencies as qualified unknowns.

Gate status: **Steps 214-216 learner gate accepted, 9/9.** SME approval of the
six-case manifest and the case-4 expectation remains pending. The deterministic
result remains **5/6 positive**, **5/5 safety**, with automatic routing and API/UI
tool exposure disabled.

## Steps 217-219 - Reviewed tool gate promotion and rerun

1. Why should a chat-confirmed SME decision be recorded in a durable ledger with
   its approval source rather than left only in conversation history?
2. Why must review promotion bind the draft manifest, review packet, and original
   evaluated report identity together?
3. Why does accepting case 4 unchanged validate its expectation without changing
   its structural failure into a pass?
4. Why create a separate reviewed manifest instead of editing the paid or
   evaluated draft manifest in place?
5. Why must reviewed manifests and approval ledgers refuse overwrite?
6. What do the ledger and manifest SHA-256 identities prove, and which approver
   authentication or filesystem protections do they not provide?
7. Why is exit code 1 from the reviewed evaluator an intentional gate result
   rather than a runtime crash?
8. Why can **5/5 safety** and **all_cases_reviewed=true** still not authorize tool
   exposure when the positive result is **5/6**?
9. What evidence should a targeted case-4 fix provide before API/UI exposure,
   including protection against regressions in code-only retrieval and budgets?

Gate status: **awaiting learner answers.** The SME-reviewed contract is durable,
but the deterministic release gate remains blocked at **5/6 positive**. No paid
operation occurred and bounded-tool API/UI exposure remains disabled.

### Learner answers and mentor evaluation

1. **Accepted.** The ledger makes the human decision durable and binds reviewer,
   time, rationale, and exact artifacts rather than relying on chat context.
2. **Accepted.** Binding the draft, packet, and evaluated report prevents an
   approval from being reused with substituted expectations or results.
3. **Accepted.** SME acceptance validates the benchmark expectation. The observed
   failure remains a confirmed product gap until the required evidence is
   retrieved.
4. **Accepted.** Separate immutable draft and reviewed manifests preserve the
   preapproval proposal and the approved gate as distinct historical states.
5. **Accepted.** No-overwrite behavior prevents silent mutation; revisions need a
   new generation and approval trail.
6. **Accepted.** SHA-256 establishes byte identity/integrity comparison. It does
   not establish semantic correctness, reviewer authentication, authorization,
   filesystem protection, or production representativeness.
7. **Accepted.** Exit code 1 is a machine-actionable quality-gate outcome with a
   valid diagnostic report, not an unhandled runtime failure.
8. **Accepted.** Safety, SME validity, and positive retrieval are independent
   required gates; none substitutes for another.
9. **Accepted.** The proposed evidence is correct: recover the required case-4
   symbol without broad ranking distortion, preserve budgets and all other cases,
   then rerun the reviewed gate, safety checks, and regression suite.

Gate status: **Steps 217-219 learner gate accepted, 9/9.** The bounded-tool
architecture and reviewed benchmark may be finalized as offline foundations.
Serving exposure remains blocked until the confirmed case-4 gap is repaired and
the reviewed deterministic gate reaches 6/6.

## Steps 220-222 - Selection repair, gate closure, and manual retrieval UAT

1. Why is reserving one slot from already retrieved candidates safer than
   increasing the evidence limit or globally changing RRF weights?
2. Why require a minimum identifier-token affinity before replacing a selected
   evidence unit?
3. What failure modes could arise if identifier aliases such as `txn` and
   `transaction` were expanded without a bounded, reviewed policy?
4. What does **6/6 positive and 5/5 safety** prove for the reviewed lexical tool
   manifest, and what does it not prove?
5. Why must the reviewed manifest remain unchanged while testing the selector
   repair?
6. Why should code-only retrieval be regression-tested even though the selector
   applies only to combined mode?
7. Why must the manual UAT runner require acknowledgement that its output contains
   internal source text?
8. Why is a no-overwrite local UAT report preferable to repeatedly writing one
   `latest.json` file?
9. What should the SME inspect during manual combined UAT before concluding that
   a retrieved package or routine implements the documented FDD behavior?

Gate status: **awaiting learner answers and manual SME retrieval UAT.** The
reviewed deterministic lexical-tool gate passes **6/6 positive** and **5/5
safety**. API/UI exposure, automatic routing, and paid generated-answer UAT remain
disabled or unperformed.

### Learner answers and mentor evaluation

1. **Accepted.** The reservation improves bounded evidence coverage without
   increasing prompt pressure or changing global dense/lexical fusion behavior.
2. **Accepted.** A minimum affinity threshold prevents weak identifier overlap
   from displacing stronger ranked evidence merely to satisfy a diversity rule.
3. **Accepted.** Uncontrolled aliases can create broad false matches and promote
   the wrong package or symbol. Alias policy must remain explicit, bounded, and
   regression-tested.
4. **Accepted with terminology correction.** The **6/6** result proves the six
   reviewed positive retrieval cases met their defined identity/lineage checks.
   The **5/5** result covers orchestration safety controls—automatic-routing,
   budget, missing-handler, trace-privacy, and zero-external-call checks—not five
   abstention cases. Neither result proves corpus-wide semantics, citations,
   generated-answer correctness, load behavior, or production reliability.
5. **Accepted.** Keeping the reviewed target immutable prevents benchmark
   weakening from being mistaken for a retrieval improvement.
6. **Accepted.** Scope intent does not replace regression evidence; shared models,
   retrievers, and tool boundaries can still be affected unintentionally.
7. **Accepted.** The acknowledgement makes the sensitivity of locally retained
   proprietary evidence explicit. It supplements rather than replaces access,
   storage, retention, and sharing controls.
8. **Accepted.** No-overwrite reports preserve independent run identities and
   historical evidence instead of silently replacing the diagnostic record.
9. **Accepted.** The SME must verify the FDD, snapshot, path, symbol/overload, and
   semantic relationship. Name similarity and even a broad file-level mapping do
   not prove that a particular routine implements the documented behavior.

Gate status: **Steps 220-222 learner gate accepted, 9/9.** The reviewed local
lexical-tool gate is closed at **6/6 positive** and **5/5 orchestration safety**.
The next gate is a broader manual SME retrieval-UAT round; API/UI exposure and
paid generated-answer UAT remain separate and unapproved.

## Steps 223-225 - Formal manual UAT and unavailable-boundary qualification

1. Why do previously reviewed source questions not automatically make their new
   bounded-tool UAT cases reviewed?
2. Why should a formal UAT case distinguish `evidence` from
   `qualified_unknown` outcomes?
3. Why must the batch preflight every output path before executing any case?
4. Why was the initial **9/10** result preserved instead of regenerating into the
   same output namespace after the fix?
5. Why can nearby visible PL/SQL be useful while still failing a request for an
   exact hidden Java kernel method and defect line?
6. Why should an unavailable-boundary qualifier require multiple independent
   query signals rather than trigger on the word `kernel` alone?
7. What does the corrected **10/10 diagnostic** batch prove, and why is
   `all_cases_reviewed=false` still material?
8. Why do individual UAT reports retain source text while the batch index and SME
   packet omit it?
9. Why does approval to incur OpenAI cost not by itself authorize disclosure of
   retrieved internal FDD and PL/SQL evidence?

Gate status: **awaiting learner answers and SME packet review.** Local diagnostics
pass **10/10**, with **0 external calls**, but the UAT manifest remains draft.
Paid grounded-answer evaluation is additionally blocked pending explicit internal-
evidence disclosure authorization.

## Steps 226-228 - Authorized paid bounded-tool grounded-answer evaluation

1. Why should one ledger bind SME acceptance, paid-use permission, disclosure
   permission, and exact request limits separately?
2. Why could this paid evaluation reuse local evidence and make zero query-
   embedding requests?
3. Why must an LF-versus-CRLF byte difference invalidate a SHA-256-bound approval
   even when parsed JSON content is equivalent?
4. Why preserve both failed preflights instead of deleting them after correcting
   deterministic serialization?
5. Why are zero automatic retries important for paid calls that disclose internal
   evidence?
6. What does **10/10 completed** prove, and why is it different from **8/10
   structural passes**?
7. Why can a safe refusal still be a structural failure for a positive impact-
   analysis case?
8. Why can an otherwise useful combined answer fail when it omits the expected
   reviewed FDD citation?
9. What must happen before these bounded grounded answers can justify API/UI
   exposure or activation?

Gate status: **awaiting learner answers and SME semantic review of the paid
packet.** The authorized run completed **10/10 requests** with **8/10 structural
passes**, zero query embeddings, and zero retries. Activation remains unauthorized.

### Learner answers and mentor evaluation

1. **Accepted.** SME quality approval, paid-use authority, disclosure authority,
   and request ceilings govern different risks and must not imply one another.
2. **Accepted.** The paid runner reused the exact local citeable evidence already
   selected and reviewed, so generating another query vector would add cost and
   disclosure without changing the authorized evidence set.
3. **Accepted.** A byte-bound hash applies to exact serialized bytes. Treating
   semantically equivalent CRLF/LF files as identical would weaken the approved
   integrity contract.
4. **Accepted.** The failed preflights demonstrate fail-closed enforcement and
   zero paid calls under mismatched state; deleting them would remove safety and
   diagnostic evidence.
5. **Accepted.** Each retry is another paid disclosure event. With retries
   disabled, any further request must be deliberate, within a newly established
   authorization budget, and tied to preserved evidence.
6. **Accepted.** Completion measures execution coverage; structural passing
   measures compliance with the expected machine-checkable answer contract.
7. **Accepted.** A refusal can preserve grounding safety while failing the user-
   assistance objective of a reviewed positive case.
8. **Accepted.** Usefulness and plausibility do not replace the required authority
   or lane-specific citation contract.
9. **Accepted.** The listed semantic, citation, authorization, readiness, smoke,
   activation, and rollback gates remain independent prerequisites.

Gate status: **Steps 226-228 learner gate accepted, 9/9.** SME review records nine
accepted cases and one corrected case (`uat-code-aml-offline-impact-005`). Case 8
is human-accepted despite its immutable structural citation failure. Activation
remains blocked pending durable review import and targeted case-5 remediation.

## Steps 229-231 - SME ledger and targeted citation-contract remediation

1. Why must the semantic review ledger preserve the original structural result
   even when the SME accepts the answer?
2. Why should case 8 be recorded as SME-accepted rather than changing its stored
   structural failure to pass?
3. Why does one corrected verdict keep the semantic gate pending despite nine
   accepted answers?
4. What evidence proves case 5 was not a retrieval or reranking failure?
5. Why did a useful raw answer still become a safe refusal?
6. Why would changing RRF weights be an inappropriate fix for bare citation
   syntax?
7. Why should the prompt forbid bare `C2` while the validator still independently
   enforces `[C2]`?
8. What do **602 passing tests** prove about the correction, and what live-model
   property remains unproven?
9. Why does replaying only case 5 require new cost/disclosure authorization even
   though its evidence was disclosed in the earlier run?

Gate status: **awaiting learner answers and one-case replay authorization.** Nine
paid answers are SME-accepted; case 5 has a locally tested citation-contract fix.
No paid retry or activation has occurred.

## Steps 232-234 - One-call case-5 replay

1. Why must the replay authorization bind the current prompt hash and the exact
   previously retrieved evidence hashes?
2. Why did this replay permit one answer request but zero query-embedding requests?
3. Why did the evidence's earlier authorized disclosure not automatically permit
   this later replay?
4. Why is a no-call preflight useful even after a hash-valid authorization exists?
5. Why must the original failed trace remain immutable after the corrected replay
   passes?
6. What does the replay's structural pass prove, and what semantic property still
   requires SME judgment?
7. Why must citation validation remain fail-closed even after the prompt explicitly
   requires `[C#]` syntax?
8. If this one authorized call had failed transiently, why would an automatic retry
   still have been prohibited?
9. What must happen before this replay can close the case-5 remediation item or
   support any broader activation decision?

Gate status: **awaiting learner answers and SME review of the one-case replay.**
The replay completed with one answer request, zero query embeddings, zero retries,
and a structural pass. Activation remains false and no further paid call is
authorized.

## Steps 235-237 - SME replay ledger and remediation closure

1. Why can using `\s*` after a blank Markdown field accidentally consume the next
   field's label?
2. Why should a blank packet rationale use an explicitly identified acceptance note
   rather than inventing a rationale?
3. Why must the packet's displayed structural result match the immutable run state?
4. Why does the closure ledger bind both the old ten-case ledger and the new replay
   trace?
5. Why should the original failed trace and corrected verdict remain unchanged after
   the replay is accepted?
6. What does effective **10/10 semantic acceptance after targeted replay** mean?
7. Why does semantic remediation closure still authorize zero additional paid calls?
8. What failure modes are caught by refusing ledger overwrite and hash drift?
9. Which controls remain before exposing a new bounded-agent capability through the
   API/UI, despite this remediation gate being closed?

Gate status: **awaiting learner answers.** The case-5 remediation and bounded-tool
semantic review gate are closed at effective **10/10 acceptance**. Activation and
additional paid requests remain unauthorized by this ledger.

### Learner answers and mentor evaluation

1. **Accepted.** `\s` includes newlines, so `\s*` can cross a blank field boundary
   and associate the following label with the wrong field. Horizontal whitespace
   keeps each Markdown field line-bounded.
2. **Accepted.** A blank rationale is not permission to invent one. The separate,
   attributable chat confirmation supplies the human decision and its provenance.
3. **Accepted.** Review is valid only when the human-visible structural state and
   the immutable machine result describe the same artifact.
4. **Accepted.** The original ledger records the prior decision; the replay trace
   records the corrected execution. Binding both creates the complete remediation
   chain.
5. **Accepted.** Historical failure evidence must remain immutable. The accepted
   replay supersedes the remediation status without rewriting the earlier event.
6. **Accepted.** Effective 10/10 means the nine original acceptances plus the one
   separately accepted replay cover all ten unique cases; it does not transform the
   original failure or imply ten new executions.
7. **Accepted.** The one-call authorization was consumed. Review closure neither
   spends nor grants additional cost/disclosure authority.
8. **Accepted.** Hash checks detect substituted or changed inputs; no-overwrite
   behavior preserves history and prevents stale-generation or result replacement.
9. **Accepted with scope clarified.** The approved initial Phase 2 PL/SQL and local
   code/combined scope can close. New bounded-agent API/UI exposure, automatic
   routing, JavaScript expansion, and production-scale controls are separately
   gated future capabilities, not prerequisites for this scoped Phase 2 closure.

Gate status: **Steps 235-237 learner gate accepted, 9/9.** The case-5 remediation
is closed, the bounded-tool semantic set is effectively **10/10 accepted**, and the
approved initial PL/SQL Phase 2 scope is complete. No additional paid request or
new bounded-tool serving exposure is authorized by this decision.

## Steps 238-240 - Shared retrieval foundation

### Step 238 - Configuration and source identity

1. Why must `RETRIEVAL_INDEX_PATH` fail closed when it conflicts with
   `PROCESSED_DIR`, rather than silently prefer one variable?
2. Why should a public fetch ID include both stable source identity and a content
   hash instead of only a document filename or Qdrant point ID?
3. Why should a logical source reference avoid absolute local paths and internal
   FDD unit IDs, even when the source is approved for retrieval?

### Step 239 - Shared retrieval service

4. Why is keeping FDD and code rankings separate in combined mode safer than
   flattening their scores into one global top-five list?
5. In combined hybrid retrieval, why must exactly one query vector be reused by
   both lanes, and what would two independent embeddings make harder to audit?
6. Why does a source catalog need to reject duplicate active identities instead
   of selecting the first matching source during `fetch`?

### Step 240 - Answer orchestration boundary

7. Why must answer orchestration reject a prepared retrieval whose original query
   does not match the current request?
8. What behavior remains owned by answer orchestration after retrieval moves into
   `KnowledgeRetrievalService`, and why should it not move into the MCP adapter?
9. Why do the 16 focused tests not yet prove that the MCP interface is safe or
   correctly encoded on the stdio JSON-RPC wire?

Gate status: **awaiting learner answers for Steps 238-240.** The new service has
not opened an MCP transport, disclosed evidence, or made a live OpenAI call.

### Learner answers and mentor evaluation

**Accepted, 9/9.** The learner correctly identified stale-index ambiguity,
content-bound opaque identity, reference privacy/portability, independent lane
scores, one-vector reuse, exact catalog resolution, query-bound prepared evidence,
answer-orchestration ownership, and the remaining stdio wire-validation gap.

## Steps 241-243 — FastAPI and MCP adapters

### Step 241 — Shared FastAPI retrieval

1. Why must `/search` keep `mode` as a knowledge-lane selector while keeping
   lexical/dense/hybrid selection internal configuration?
2. Why is passing a prepared retrieval to `/query` safer than letting answer
   orchestration retrieve a second time after FastAPI has already retrieved once?
3. Why should API error handling distinguish a retrieval dependency failure from
   a later answer-orchestration failure?

### Step 242 — MCP evidence-disclosure boundary

4. Why must the disclosure switch be checked before constructing the retrieval
   service, rather than merely hiding fields in a response after retrieval?
5. Why is an MCP `fetch` tool limited to a validated opaque ID instead of taking
   a relative file path from ChatGPT?
6. Why must a code/combined MCP call still honor `CODE_MODES_ENABLED`, even when
   MCP evidence disclosure is explicitly enabled?

### Step 243 — Structured transport and operational safety

7. Why must fallback text be generated from the same canonical dictionary as
   `structuredContent`, instead of being formatted separately?
8. Why must MCP logging use stderr without globally redirecting stdout?
9. What do the 46 focused tests prove, and what exact protocol-level property is
   still unproven until the subprocess stdio tests in Steps 244-246?

Gate status: **awaiting learner answers for Steps 241-243.** No tunnel was
created, no internal evidence was disclosed, and no live OpenAI call occurred.

### Learner answers and mentor evaluation

**Accepted, 9/9.** The learner correctly explained internal strategy control,
prepared-evidence determinism, error classification, early disclosure gating,
opaque fetch IDs, independent code authority, canonical MCP serialization,
stderr/stdout separation, and the need for actual child-process frame validation.

## Steps 244-246 — Protocol, process, and readiness

### Step 244 — Actual stdio protocol verification

1. Why is a subprocess JSON-RPC test stronger than directly calling an MCP tool
   function in Python?
2. Why must the test validate every stdout line, rather than only the expected
   response frame for `tools/call`?
3. Why is injecting both a third-party logger warning and a Python warning useful
   for validating stdio hygiene?

### Step 245 — Child configuration preflight

4. Why should MCP startup preflight validate configuration and local artifacts
   without opening Qdrant, loading the source catalog, or calling OpenAI?
5. Why does checking only the *presence* of `CONTROL_PLANE_API_KEY` in the child
   environment satisfy isolation without authorizing application code to use it?
6. Why must an invalid child preflight prevent tool registration rather than
   allowing tools to register and fail later on their first call?

### Step 246 — Tunnel ownership and environment boundary

7. Why should `tunnel-client` own the MCP stdio child instead of an independently
   started server terminal?
8. Why does the PowerShell launcher remove the control-plane key before starting
   Python even though the tunnel client needs that key itself?
9. What does the 17-test protocol/preflight/runtime suite prove, and what does it
   still not prove about an actual Secure MCP Tunnel or ChatGPT connection?

Gate status: **awaiting learner answers for Steps 244-246.** No tunnel was
created, no external API request occurred, and no internal evidence was disclosed.

### Learner answers and mentor evaluation

**Accepted, 9/9.** The learner demonstrated a strong operational understanding of
wire-level testing, stdout hygiene, bounded configuration checks, process
ownership, key isolation, and the difference between local proof and an actual
ChatGPT/tunnel exercise.

## Steps 247-249 — Runbook and final verification

### Step 247 — Operator runbook

1. Why must the runbook require an explicit operator change to
   `MCP_EVIDENCE_DISCLOSURE_ENABLED=true` before ChatGPT retrieval testing,
   instead of enabling it automatically when `INTERFACE_MODE=mcp`?
2. Why should an index rebuild create a complete new generation rather than add
   points directly to the active FDD or code collection before MCP testing?
3. Why is direct MCP Inspector startup useful for local protocol diagnosis but
   not the operating model for a Secure MCP Tunnel session?

### Step 248 — Operational boundaries

4. Why are `INTERFACE_MODE`, `MCP_EVIDENCE_DISCLOSURE_ENABLED`, and
   `CODE_MODES_ENABLED` three independent controls rather than one “enable MCP”
   flag?
5. Why must a tunnel-client command launch the supplied key-stripping wrapper,
   rather than directly running `python -m app.mcp.server` in Terminal 3?
6. Why does a `tunnel-client doctor` success remain insufficient evidence that
   MCP answers are grounded, useful, or safe for the intended business question?

### Step 249 — Release-quality evidence

7. Why was defaulting an absent legacy test-double `interface_mode` to `fastapi`
   safer than changing all old test doubles to an implicitly enabled MCP mode?
8. Why did configuring logging in `create_mcp_server()` break unrelated audit
   tests, and why is `main()` the correct scope for the MCP stdio logging setup?
9. What does the final 637-test offline pass prove, and what manual evidence is
   still required before relying on a live ChatGPT Secure MCP Tunnel session?

Gate status: **awaiting learner answers for Steps 247-249.** No automated test
made a live OpenAI call, created a tunnel, or disclosed internal evidence.

### Learner answers and mentor evaluation

**Accepted, 9/9.** The learner correctly distinguished the three independent
controls, immutable generation activation, Inspector versus tunnel-client
ownership, the parent-only control-plane key, transport health versus retrieval
quality, safe default interface behavior, logging scope, and the limits of the
637-test offline result. Phase 1 implementation is complete; the next work is
the documented manual tunnel setup and evidence-grounding validation.

## Step 250 — Controlled dense retrieval probe

1. Why was `POST /search` safer than `POST /query` for the first paid dense
   probe?
2. What did the `HTTP 200`, dense mode, and five-result outcome prove, and what
   did they still not prove about answer quality?
3. Why can two local processes using embedded Qdrant conflict even when they use
   the same collection and files?
4. Why does a caller-controlled environment variable not provide an enforceable
   disclosure boundary when that caller can launch local processes and read the
   repository?

Gate status: **dense retrieval mechanically verified once.** Hybrid retrieval,
semantic evaluation, and any future evidence disclosure require separate,
explicit authorization and controls.

## Steps 251-253 — Local MCP/Qdrant runtime hardening

### Step 251 — Safe lock handling

1. Why is it safer to return a generic local-Qdrant-lock error than the original
   storage exception to an MCP client?
2. What does safe lock-error handling improve, and what does it *not* change
   about embedded Qdrant concurrency?
3. Why must the lock-error test include an internal path in the original mocked
   exception?

### Step 252 — Launcher mutex

4. What specific race does the Windows mutex prevent?
5. Why does an MCP-only mutex not make FastAPI plus MCP dense/hybrid safe on the
   same embedded store?
6. Why must the mutex failure happen before the Python MCP server starts?

### Step 253 — Operator topology and cost

7. Why is lexical MCP retrieval normally free of application embedding API cost,
   while dense/hybrid is not?
8. What must be selected and validated before intentionally serving concurrent
   FastAPI and MCP dense/hybrid traffic?
9. Why should a personal direct-stdio Chat client test be described as a local
   interface test rather than a Secure MCP Tunnel or production-security proof?

Gate status: **local runtime hardening complete; learner review pending.**

### Learner answers and mentor evaluation

**Accepted, 9/9.** The learner demonstrated sound understanding of safe error
containment, embedded-store exclusivity, launcher-level race prevention,
retrieval-cost boundaries, and the difference between local functional testing
and an enforceable production security boundary.
