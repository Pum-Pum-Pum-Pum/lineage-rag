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
