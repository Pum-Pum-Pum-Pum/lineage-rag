# Interview Questions

## Step 99 - Secure request correlation and privacy-safe API auditing

### Questions asked
1. Why must a client-provided request ID be treated as untrusted input before
   it is copied into response headers, logs, or traces?
2. Why are the request correlation ID and unique answer-trace ID separate, even
   though using one identifier everywhere appears simpler?
3. Why does the audit event record a route template instead of the concrete
   request path, and which useful fields are deliberately excluded?
4. What do the defensive response headers mitigate, and why do they not replace
   authentication, authorization, rate limiting, or TLS?
5. How would you turn these local JSON events into production observability and
   audit evidence without storing sensitive prompts or conversation content?

### Correct answer key
1. Headers are attacker-controlled. Unbounded or control-character values can
   forge log lines, poison downstream systems, create oversized records, or
   produce invalid response headers. Apply a strict character/length allowlist
   or generate a server UUID.
2. Correlation IDs may be retried, duplicated, or caller-controlled. Trace IDs
   identify unique artifacts and must remain server-generated to prevent
   collisions and overwrites. Store the correlation ID as a separate join key.
3. A template such as `/conversations/{conversation_id}` supports endpoint
   aggregation without disclosing resource identifiers or creating
   high-cardinality metrics. Bodies, query text, titles, credentials, IPs, raw
   errors, and concrete IDs are excluded to reduce privacy and secret leakage.
4. `no-store`, `nosniff`, frame denial, referrer restrictions, and permissions
   policy reduce caching, content-type confusion, framing, referrer leakage,
   and unnecessary browser capabilities. They do not identify callers,
   determine permissions, throttle abuse, or encrypt network traffic.
5. Ship structured events to an access-controlled centralized platform; define
   schemas, retention, rotation, clock synchronization, integrity controls,
   dashboards, latency/error SLOs, and alerts. Join to traces by correlation ID,
   apply least privilege and redaction, and keep content logging disabled unless
   a separately governed diagnostic workflow explicitly requires it.

### Gate
Step 99 implementation is complete. Await the user's answers before proceeding
to the next production-hardening step.

### User answer evaluation - Step 99 - 2026-07-30

#### Overall verdict
Pass. The answers correctly cover untrusted header handling, identifier
separation, low-cardinality privacy-safe logging, the limited role of browser
security headers, and the controls required for centralized observability.

#### What was strong
- Q1 names concrete attack and reliability outcomes rather than merely saying
  input should be validated.
- Q2 correctly separates a caller-controlled join key from a server-controlled
  unique artifact identity.
- Q3 balances endpoint aggregation with privacy, secret protection, and metric
  cardinality.
- Q4 correctly maps each header family to browser risk and clearly states the
  controls it cannot provide.
- Q5 includes schemas, retention, rotation, time synchronization, integrity,
  SLOs, alerting, redaction, and least privilege.

#### Precision improvements
- A strict allowlist also needs a length bound; character validation alone does
  not prevent oversized records.
- Route templates reduce identifier exposure and metric cardinality but are not
  anonymous by themselves; method, timing, request IDs, and joined systems can
  still become sensitive metadata.
- Operational logs become defensible audit evidence only with controlled
  access, documented retention, integrity or append-only controls, reliable
  clocks, export/review procedures, and evidence that the logging pipeline
  itself is monitored.
- Production events should preserve the correlation ID for joining API events
  to answer traces without copying prompt content into the log platform.

#### Stronger interview-quality answer
Client headers are untrusted and must pass both a character allowlist and a
strict length bound before entering response headers, logs, or traces;
otherwise generate a server UUID. This prevents control-character log forging,
invalid headers, oversized events, and downstream parser poisoning.

Correlation IDs are join keys and may be supplied, duplicated, or retried by a
caller. Trace IDs identify unique server artifacts and must be generated
independently to prevent collisions and overwrites. Audit events should use
low-cardinality route templates and exclude bodies, prompts, titles,
credentials, raw errors, IPs, and resource IDs. Even the retained metadata must
be access-controlled because correlation and timing data may still be
sensitive.

Defensive headers reduce caching, content-type confusion, framing, referrer
leakage, and unnecessary browser capabilities, but do not authenticate,
authorize, throttle, or encrypt. In production, structured events should flow
to a controlled centralized platform with a versioned schema, correlation
joins, synchronized clocks, retention and rotation, least-privilege access,
redaction, append-only or integrity controls, pipeline monitoring, dashboards,
SLOs, alerts, and documented audit export/review procedures.

#### Gate
Step 99 interview gate accepted.

## Step 100 - Deterministic native deployment bundle and preflight

### Questions asked
1. Why should application source and locked dependencies be packaged
   separately from documents, indexes, conversations, and answer traces?
2. What does a deterministic ZIP hash prove, and what important supply-chain
   properties does it not prove?
3. Why should deployment preflight check configuration presence and local
   state without calling the embedding, chat, or vector services?
4. Why is the native process command recorded now while systemd, Windows
   Service, or another supervisor configuration is deferred?
5. What additional controls are required before this bundle can be called
   production-ready in an Oracle environment?

### Correct answer key
1. Code/lockfiles are immutable release inputs; documents, vector indexes,
   SQLite history, traces, and logs are mutable or sensitive runtime state.
   Separating them enables small releases, independent backup/restore, safer
   rollback, least-privilege permissions, and avoids leaking business data.
2. Determinism proves identical selected bytes produce the same archive and
   helps detect accidental build drift. It does not prove source provenance,
   reviewer approval, dependency safety, builder integrity, or publisher
   identity. Add CI provenance, vulnerability/SBOM checks, signing or
   attestation, and verified promotion controls.
3. Preflight should fail fast on known local prerequisites without creating
   network dependency, latency, cost, or side effects. Live service reachability
   belongs in readiness and a controlled post-deployment smoke test.
4. The application command is portable, but supervisors differ in service
   identity, restart semantics, environment/secret injection, logging, limits,
   and shutdown behavior. Selecting one before the Oracle runtime is confirmed
   would create a misleading deployment artifact.
5. Confirm the OS and supervisor; add a non-privileged service identity,
   approved secret store, TLS/reverse proxy, authentication/authorization,
   rate controls, file/database permissions, backups and restore tests,
   centralized logs/metrics, resource limits, health-based restart, signed
   artifact promotion, rollback, and load/failure validation.

### Gate
Step 100 implementation is complete. Await the user's answers before selecting
the next step.

### User answer evaluation - Step 100 - 2026-07-30

#### Overall verdict
Pass. All five answers correctly distinguish immutable releases from mutable
state, reproducibility from supply-chain trust, offline preflight from live
readiness, portable commands from supervisor-specific policy, and packaging
from complete production operations.

#### What was strong
- Q1 ties separation to release size, least privilege, rollback, and restore
  rather than treating exclusions as simple archive cleanup.
- Q2 correctly refuses to equate a deterministic hash with provenance or
  safety and names SBOM, signing, and controlled promotion.
- Q3 correctly assigns local deterministic checks to preflight and live
  dependencies to readiness/post-deployment smoke testing.
- Q4 identifies the main supervisor-specific contracts: identity, restart,
  secrets, logs, resources, and shutdown.
- Q5 covers the major platform, identity, network, data, release, recovery, and
  performance controls required before production.

#### Precision improvements
- A deterministic build claim is strongest when an independent trusted builder
  reproduces the same artifact from an identified source revision and locked
  toolchain. Two builds on one machine detect drift but do not independently
  verify the builder.
- An SBOM inventories components; it does not prove that dependencies are safe.
  Pair it with vulnerability/license policy, provenance, signing/attestation,
  and verified deployment admission.
- Backups are not proven until restore and recovery-time/recovery-point
  objectives are tested.
- Health checks should drive supervisor and traffic-management behavior, with
  graceful shutdown and rollback verified under failure.

#### Stronger interview-quality answer
Immutable application source, lockfiles, and runtime contracts should be
released separately from mutable or sensitive documents, vector indexes,
conversation data, traces, and logs. This keeps artifacts small, permissions
least-privileged, rollback predictable, and state backup/restore independent.

Deterministic packaging demonstrates byte-for-byte reproducibility for the
selected inputs and builder behavior, but does not establish source provenance,
review approval, dependency safety, or publisher identity. Strong assurance
requires independent reproduction from an identified revision and pinned
toolchain, an SBOM plus vulnerability/license policy, signed provenance or
attestation, and verified promotion/admission controls.

Offline preflight should validate local prerequisites without network cost or
side effects; readiness and controlled smoke tests validate live services.
Portable process commands can be recorded before the supervisor is selected,
but service identity, restart/backoff, secret injection, logs, limits, graceful
shutdown, and health integration depend on the confirmed Oracle runtime.

Production readiness additionally requires the confirmed OS/supervisor,
non-privileged identity, approved secrets, TLS, authentication/authorization,
rate controls, permissions, centralized observability, tested backup/restore
with RPO/RTO, health-based restart and traffic removal, signed promotion,
rollback testing, and load/failure validation.

#### Gate
Step 100 interview gate accepted.

## Step 101 - Tamper-evident local audit journal and verification boundary

### Questions asked
1. Why is a keyed HMAC chain materially stronger than a plain SHA-256 chain for
   a mutable local audit file, and what attacker can still forge it?
2. Why can a completely valid internal chain fail to reveal deletion of its
   last records, and what exact external checkpoint closes that gap?
3. The API deliberately fails open when `fsync` or journal writing fails. What
   availability benefit and compliance/reliability risk does that create, and
   what operational response is required?
4. Why does the current local journal reject a stale second writer, and how
   should multi-worker or multi-host audit ordering be solved in production?
5. Does a valid audit HMAC prove that a RAG answer is factually correct and
   grounded? Explain the separate evidence controls still required.

### Correct answer key
1. A plain hash chain can be recalculated after modification by anyone with
   write access. HMAC verification requires a separately protected secret.
   An attacker who obtains both journal write access and the HMAC key can forge
   a replacement chain; key custody, access controls, rotation, and external
   checkpoints remain necessary.
2. Removing a valid suffix leaves the earlier sequence and HMAC links internally
   consistent. Compare both the final HMAC and record count with a checkpoint
   previously stored in a separate trusted, append-only system.
3. Fail-open preserves query availability and prevents an audit disk outage
   from becoming a total service outage. It permits an audit gap, so a safe
   critical event must page operations; traffic may need to be drained or the
   service placed in a governed degraded mode according to policy. Capacity,
   permissions, recovery, and reconciliation must be tested.
4. Independent writers can start from the same previous HMAC and fork or
   interleave the chain. The local boundary therefore detects an external
   change and is limited to one writer. Production multi-process/host events
   should be shipped independently to an approved centralized append-only
   service that supplies ordering, durable ingestion, retention, and integrity
   controls.
5. No. HMAC proves integrity/authenticity of recorded metadata under the key
   assumptions, not truth or semantic entailment. Functional claims still need
   fresh retrieval, current-release planning, sufficient complete evidence,
   validated citations, safe abstention, and answer/evidence evaluation.

### Gate
Step 101 implementation is complete. Await the user's answers before selecting
the next step.

### User answer evaluation - Step 101 - 2026-07-30

#### Overall verdict
Partial pass; gate not yet accepted. The user correctly located audit creation
around the FastAPI request boundary, distinguished the main local artifacts,
identified the privacy-minimized event fields, and explained `flush` versus
`fsync`. The operational conclusion about disabling audit because the tool is
internal is not sufficiently risk-based, and memory/context overflow was
incorrectly mixed with audit-storage failure.

#### What was strong
- Q1 correctly says the audit layer records the API operation rather than RAG
  message and evidence contents.
- Q2 broadly separates conversation continuity, answer debugging/evaluation,
  and request-level operational records.
- Q3 correctly identifies request metadata and deliberate content omission.
- Q4 accurately separates Python-buffer flushing from OS durability and names
  the per-request latency tradeoff.

#### Required corrections
- The middleware surrounds every FastAPI request; audit creation is not merely
  "at the call to FastAPI."
- Conversation summaries supply conversational context only. They are not
  functional-spec retrieval evidence, and cannot replace fresh retrieval and
  validated citations.
- The audit identifier is a validated correlation/request ID, not a business
  primary key. The event records the route template, not the concrete URL.
- Conversation token-memory overflow is unrelated to audit persistence.
  Relevant audit failures include full/unavailable disk, permissions, corrupt
  journal, concurrent writer, missing/weak key, and failed central shipping.
- Internal users can still make unauthorized, mistaken, or disputed actions.
  Audit requirements follow data sensitivity, action impact, regulation, and
  incident-response needs, not simply whether users are external.
- Disabling the journal removes durable local integrity evidence; the ordinary
  structured logger event remains, but it is not equivalent to a verified
  audit chain.
- `fsync` policy should be selected from measured latency/throughput and loss
  tolerance. Alternatives such as grouped commits or a durable collector reduce
  request-path cost but introduce a defined loss window or another dependency.

#### Stronger interview-quality answer
The FastAPI middleware surrounds each HTTP request and records a fixed,
privacy-safe completion event after endpoint processing. It stays outside the
retrieval and LLM layers so request auditing does not duplicate prompts,
evidence, answers, or sensitive conversation content.

Conversation SQLite stores durable chat turns and summaries for conversational
context; summaries never replace fresh functional-spec retrieval. Answer traces
store retrieval, sufficiency, answer, citation, and correlation details for RAG
debugging and evaluation. The audit journal stores a request/correlation ID,
method, route template, status, duration, timestamp, sequence, and HMAC chain.

`flush` transfers Python-buffered bytes to the OS, while `fsync` requests
durable filesystem persistence. Per-request `fsync` minimizes the acknowledged
record-loss window but adds disk latency and limits throughput. The policy must
be chosen from measured SLO impact and the maximum acceptable audit-loss
window, not from an unsupported assumption that internal users need no audit.

Audit storage can fail through disk exhaustion, permissions, corruption,
concurrent writers, bad key configuration, or failed central shipping.
Fail-open preserves RAG availability but creates an audit gap that must alert
operations. Disabling the journal is acceptable only as an explicitly accepted
environment risk; it leaves ordinary logs but no local tamper-evident chain.

#### Follow-up gate
Answer the focused follow-up questions in chat before Step 101 is accepted.

### Follow-up answer evaluation - Step 101 - 2026-07-30

#### Overall verdict
Pass. The user now clearly separates generated conversation memory from
source-of-truth retrieval evidence, justifies internal audit controls with
realistic authorization and abuse scenarios, selects durability policy from
measured SLO and loss-window requirements, and gives a governed response to an
audit gap.

#### What was strong
- Q1 precisely identifies omission, misunderstanding, and staleness risks in
  generated summaries and preserves the fresh-retrieval/citation boundary.
- Q2 gives both an authorization incident and an availability/abuse incident;
  neither requires an external attacker.
- Q3 names workload, storage, latency, and crash-loss measurements and maps
  them correctly to per-request durability, grouped commits, and development
  disablement.
- Q4 treats fail-open as an operationally visible degraded state rather than a
  harmless logging warning. It includes restriction of sensitive operations,
  evidence preservation, repair, trusted-checkpoint verification, and gap
  reconciliation.

#### Precision improvements
- Audit-event rate normally follows all HTTP traffic, including health and
  invalid-route requests, not just successful RAG request volume.
- Large or malformed-query abuse should also be controlled through request-size
  limits, timeouts, concurrency/rate controls, and capacity isolation; auditing
  records and supports investigation but does not prevent abuse.
- A missing checkpoint and a journal write failure are different signals:
  write failure is detected on the request path, while checkpoint mismatch is
  detected during verification or central reconciliation.
- Key restoration must follow an explicit rotation/chain-boundary procedure;
  silently replacing the key mid-chain would invalidate verification.

#### Stronger interview-quality answer
Conversation summaries are generated, lossy context artifacts that may omit,
distort, or retain stale details. They can help resolve conversational
references but cannot substantiate functional claims; those require fresh
retrieval from approved documents, sufficient complete evidence, and validated
citations.

Internal audit scenarios include unauthorized access attempts to restricted
release/domain information and repeated oversized or malformed requests that
consume capacity. Audit evidence supports detection and investigation, while
authorization, request limits, timeouts, and rate/concurrency controls provide
prevention.

Choose durability from measured total HTTP event rate, disk and `fsync`
latency, p95/p99 response SLOs, throughput, storage growth, and the maximum
acceptable record-loss window. Use per-request `fsync` for near-zero
acknowledged-event loss, grouped commits for a documented bounded loss window,
and disable the integrity journal only in explicitly accepted non-production
or no-audit-risk environments.

On fail-open, alert on the safe write-failure event and detect checkpoint/gap
problems through verification and reconciliation. Apply policy-based traffic
restriction, preserve evidence, repair capacity/permissions, restore the
correct key through a governed rotation or chain-boundary process, verify from
the last trusted checkpoint, and document the unrecoverable gap.

#### Gate
Step 101 interview gate accepted.

## Step 102 - Measure local audit durability cost

### Questions asked
1. Why are p95, p99, and maximum append latency more useful than only average
   latency when evaluating synchronous `fsync` on an API request path?
2. Why does the measured `256.203 events/second` not prove that the FastAPI RAG
   service can handle 256 concurrent or end-to-end requests per second?
3. At approximately `406.390 bytes/record`, what additional inputs are needed
   to estimate daily storage and retention cost?
4. Why does an LLM endpoint's high model latency not automatically make a
   several-millisecond synchronous audit cost irrelevant?
5. What evidence would justify replacing per-request `fsync` with grouped
   commits, and what new failure guarantee must be documented?

### Correct answer key
1. Tail percentiles expose slow storage operations that affect user-visible
   SLOs and can accumulate under queueing; an average can hide intermittent
   flush, filesystem, antivirus, or device stalls. Maximum is diagnostic but
   needs enough repeated samples before it is treated as stable.
2. The benchmark serially measures only `AuditJournal.append` on one local
   filesystem. It excludes HTTP handling, concurrency and lock contention,
   retrieval, model latency, conversation SQLite, central shipping, CPU/memory
   saturation, and production infrastructure.
3. Estimate total HTTP events per day across success, error, health, readiness,
   and unmatched routes; schema/identifier size distribution; retention days;
   indexes/metadata and replication overhead; rotation/compression; backup or
   WORM copies; growth margin; and central-platform ingestion/storage pricing.
4. Audit cost applies to every endpoint and adds directly to latency. It matters
   more for health, cached, refused, or otherwise fast requests; synchronous
   storage can also serialize writers and create queueing or an outage coupling
   under disk degradation.
5. Use representative repeated load tests showing synchronous audit causes an
   unacceptable SLO/capacity impact, together with business approval for a
   bounded loss window. Grouped commits must document maximum events/time that
   can be lost on process or host failure, flush triggers, backpressure,
   shutdown behavior, monitoring, and recovery/reconciliation.

### Gate
Step 102 implementation is complete. Await the user's answers before selecting
the next step.

### User answer evaluation - Step 102 - 2026-07-31

#### Overall verdict
Pass. All five answers are concise, technically correct, and appropriately
bounded. The user distinguishes tail behavior from averages, microbenchmark
throughput from service capacity, raw record size from retained-platform cost,
LLM latency from cross-endpoint audit overhead, and performance evidence from
the business decision to accept a bounded audit-loss window.

#### What was strong
- Q1 explains why tail latency exposes intermittent storage stalls and avoids
  treating a small-sample maximum as a stable capacity statistic.
- Q2 lists the major excluded service layers and production pressures rather
  than extrapolating `AuditJournal.append` throughput to FastAPI throughput.
- Q3 covers workload, representation, retention, operational copies, growth,
  and platform pricing needed for defensible storage planning.
- Q4 recognizes that audit cost applies to fast endpoints as well as slow LLM
  requests and identifies serialization and disk-failure coupling.
- Q5 requires both repeated representative testing and business approval, then
  names the essential grouped-commit operating contract.

#### Precision improvements
- Repeated tests should report run-to-run dispersion or confidence intervals,
  not merely increase the sample count within one run.
- End-to-end load should include the expected mix of query, conversation,
  health/readiness, refusal, invalid, and error responses because all create
  audit events.
- Storage projections should distinguish logical JSONL bytes from filesystem,
  central-index, replication, backup, and retention-tier billable bytes.
- Grouped-commit guarantees should state both a maximum time window and maximum
  event count at risk, including process crash, host crash, and forced shutdown.

#### Stronger interview-quality answer
Tail percentiles show the storage stalls that affect user-visible SLOs and
queueing while an average can hide them; maximum latency is diagnostic only
after repeated representative trials establish its variability. The measured
throughput applies solely to serial local `AuditJournal.append` operations and
cannot represent HTTP, concurrency, retrieval, model, SQLite, shipping, or
production resource behavior.

Storage planning requires the complete HTTP event mix and daily volume,
identifier/schema size distribution, retention, rotation/compression,
filesystem and index overhead, replication, backups/WORM copies, growth margin,
and platform ingestion/storage pricing. Audit latency applies to every endpoint
and can dominate otherwise fast paths or introduce queueing and disk-failure
coupling.

Replacing per-request `fsync` requires repeated representative end-to-end load
evidence with run-to-run variability, an SLO or capacity problem attributable
to synchronous durability, and business approval of a bounded loss window. The
new contract must specify maximum time and event count at risk, flush triggers,
backpressure, graceful and forced shutdown behavior, monitoring, recovery, and
reconciliation.

#### Gate
Step 102 interview gate accepted.

## Step 103 - Extract a storage-neutral audit sink boundary

### Questions asked
1. Why is changing `AUDIT_JOURNAL_PATH` sufficient for another filesystem path
   but insufficient for replacing JSONL with a database?
2. What does `durable_on_return` mean for the current adapter, and why must a
   grouped writer normally declare `accepted_not_durable`?
3. Why does successful `fsync()` on a network-mounted path not automatically
   prove the same failure durability as a local disk?
4. Which responsibilities belong inside a future database audit adapter rather
   than FastAPI middleware?
5. What lifecycle and failure controls must a grouped-commit adapter implement
   before it can safely replace the synchronous adapter?

### Correct answer key
1. A path change preserves the file API, JSONL schema, HMAC chain, and filesystem
   semantics. A database needs a client/connection lifecycle, schema,
   transactions, uniqueness/idempotency, integrity model, retries, and health
   behavior, so it requires another adapter behind the common event boundary.
2. `durable_on_return` means the adapter reports success only after its defined
   durable commit operation completes. A grouped writer commonly returns after
   enqueueing; until the batch commits, process or host failure can lose the
   event, so acceptance must not be mislabeled as durability.
3. Network filesystems vary in client caching, server acknowledgement, stable
   storage guarantees, mount options, locking, failover, and partition behavior.
   Validate the actual protocol, configuration, server, and failure scenarios.
4. Connection pooling, schema/migrations, transactions, ordering, idempotency,
   HMAC or platform integrity controls, retry classification, timeouts,
   backpressure, health, credentials, and safe errors belong in the adapter.
   Middleware should create the safe event and invoke the sink only.
5. Define bounded queue capacity, batch size/time triggers, one ordering model,
   maximum time/event loss window, backpressure or fail policy, commit retry and
   idempotency, health/metrics, graceful drain, forced-shutdown behavior, and
   recovery/reconciliation. Then prove the tradeoff under representative load
   and injected failures.

### Gate
Step 103 implementation is complete. Await the user's answers before starting
the grouped-commit experiment.

### User answer evaluation - Step 103 - 2026-07-31

#### Overall verdict
Pass. All five answers preserve the storage abstraction, distinguish acceptance
from durability, avoid assuming local-disk semantics for a network filesystem,
assign database concerns to the adapter, and define the controls required for a
grouped-commit lifecycle.

#### What was strong
- Q1 correctly separates a file-location change from a storage-technology
  change with different lifecycle and transactional semantics.
- Q2 defines durability at the adapter-return boundary and does not mislabel
  enqueue success as committed persistence.
- Q3 identifies the network protocol, caching, acknowledgement, locking,
  failover, and partition variables that invalidate generic `fsync` claims.
- Q4 keeps privacy-safe event creation in middleware while assigning storage
  lifecycle, correctness, credentials, and failure handling to the adapter.
- Q5 covers bounded resources, commit triggers, loss guarantees, pressure,
  retries, observability, shutdown, recovery, and proof under load/failure.

#### Precision improvements
- A database migration must preserve audit-verification continuity or document
  a governed chain boundary and trusted checkpoint; moving rows alone does not
  preserve the old integrity claim.
- Retry idempotency needs a stable event identity or uniqueness rule so an
  uncertain commit cannot silently duplicate records.
- A grouped adapter needs both readiness status and an explicit policy for a
  full queue: block, reject, spill durably, or fail open with a visible gap.
- Network-filesystem testing should include client and server crashes plus a
  partition after acknowledgement, not only a clean disconnect.

#### Stronger interview-quality answer
Changing a path preserves the file API, JSONL schema, and HMAC-chain behavior;
changing to a database introduces connection lifecycle, migrations,
transactions, ordering, stable event identity, idempotency, integrity
continuity, retry classification, credentials, and health semantics owned by a
separate adapter. Middleware should only construct the safe event and invoke
the selected sink.

`durable_on_return` means the adapter's defined durable commit completed before
success. A grouped writer that returns after enqueue must declare
`accepted_not_durable` and specify maximum time and event count at risk. It also
needs bounded queues, batch triggers, full-queue policy, backpressure, commit
retry/idempotency, readiness and metrics, graceful drain, forced-shutdown
behavior, recovery, and reconciliation. Network and database guarantees must be
validated under partitions, uncertain acknowledgements, and client/server
crashes before production claims are made.

#### Gate
Step 103 interview gate accepted. The project is paused at the storage-neutral
sink boundary; the grouped-commit experiment has not started.

## Step 104 – Portable master FDD ingestion and verified archival

### Interview questions

1. Why is a Qdrant collection point count insufficient evidence to archive a
   specific newly ingested FDD?
2. Why does `--request-batch-size=64` limit retrieval units per OpenAI request,
   rather than the number of FDD documents in the batch?
3. If the master command fails after embedding some documents but before the
   Qdrant verification command succeeds, what state may exist and what must the
   operator do before archiving anything?
4. Why is `--dry-run` valuable before a real ingestion batch, even though it
   cannot prove that OpenAI or Qdrant will succeed?
5. Why must an existing file in `data/docs_embedded/` stop the master command
   rather than be overwritten?

### Correct-answer rubric

1. A point count can include old, unrelated, partial, or stale points. The
   archive decision needs the expected deterministic point IDs and identifying
   payload metadata for the exact embedding artifact.
2. One document can generate many retrieval units of variable token size. API
   risk and limits apply to the request payload, so the bound must be on units
   (and later token-aware batching), while documents are processed sequentially.
3. The source files remain in `data/raw_specs/`, while cache artifacts and
   possibly partial Qdrant points can exist. Diagnose the failing stage, retain
   evidence, rerun idempotently, and archive only after exact verification
   passes for the intended batch.
4. Dry run proves document discovery, ordering, archive-destination safety, and
   command construction without cost or mutation. It cannot validate external
   credentials, API availability, model behavior, storage capacity, or Qdrant
   availability.
5. Overwriting destroys the prior source-of-truth archive and hides whether the
   same release was reprocessed, changed, or duplicated. Stop, compare hashes
   and release metadata, then make an explicit reconciliation decision.

### User answer evaluation

1. Partly correct. Qdrant point IDs enforce uniqueness, but the key point is
   that duplicate text must not collapse distinct citeable evidence. Reusing the
   vector saves embedding cost; distinct point IDs preserve each chunk's source,
   release, path, and citation metadata.
2. Needed explanation. A persisted vector conflict means the system cannot
   prove which candidate represents the cache key. Arbitrary selection makes
   retrieval and citations non-reproducible, can hide model/API or artifact
   corruption, and could attach a result to the wrong release/chunk payload.
3. Needed explanation. Lineage answers must cite the exact release and chunk
   that supports a statement. Identical text can appear in different releases
   with different business meaning, applicability, or current-state effect;
   point identity must retain that evidence boundary.
4. Correct direction. Rebuild the Qdrant collection after validating scope and
   backing up if required, rather than deleting it automatically. A destructive
   operation with the wrong configuration or incomplete cache could erase useful
   retrieval evidence.

### Stronger interview-quality answer

An embedding cache key answers, “Can this exact content reuse a vector?” A
Qdrant point ID answers, “Which specific evidence unit may be retrieved and
cited?” Identical text may reuse one vector, but it requires separate point IDs
because its document, release, chunk, and citation metadata differ. If stored
vectors conflict for one cache key, silently choosing one is unsafe because the
system cannot prove deterministic retrieval or grounded lineage attribution.
After a point-ID schema change, rebuild Qdrant deliberately from validated
artifacts; do not automatically delete a collection without scope checks,
backup/recovery policy, and explicit approval.

## Step 105 – Duplicate-content embedding safety and explicit Qdrant rebuild

### Interview questions

1. Why is it correct to reuse one vector for identical content but incorrect to
   reuse one Qdrant point for every occurrence of that content?
2. What business and RAG risks would remain if we only deduplicated API calls
   but kept point IDs based solely on `cache_key`?
3. Why is artifact quarantine preferable to deleting the failed R21 embedding
   artifact before investigating it?
4. What exact assets does the explicit Qdrant rebuild delete, and which local
   artifacts does it deliberately preserve?
5. Why must the recovery be dry-run reviewed and explicitly approved even after
   deterministic tests pass?

### Correct-answer rubric

1. One vector represents content similarity and can be reused for cost and
   consistency. One Qdrant point carries occurrence-specific payload/citation
   metadata, so each unit needs its own stable identity.
2. Distinct chunks/releases with identical text would overwrite each other,
   losing lineage metadata, creating incomplete retrieval coverage, and risking
   citations to the wrong evidence occurrence.
3. Quarantine preserves the failed artifact, its vector fingerprints, and its
   diagnostics for audit and debugging while removing it from the active cache;
   deletion destroys that evidence.
4. It deletes and recreates only the configured local Qdrant collection. It
   preserves source DOCX files, processed artifacts, active embeddings, and
   quarantined artifacts.
5. Tests prove designed cases, not live credentials, model availability, API
   cost, local storage capacity, real cache state, or operator intent. Dry run
   proves scope/command construction; explicit approval authorizes the paid and
   destructive live action.

## Step 106 – Preserve duplicate evidence units and version the Qdrant collection

### Correction to Step 105

The Step 105 in-place rebuild design is superseded. A disposable embedded local
Qdrant probe proved that delete-and-recreate retained old points. `--rebuild`
and `--rebuild-qdrant` now fail safely; use a new versioned collection through
`QDRANT_COLLECTION_NAME` instead.

### Interview questions

1. Why must `load_embedding_cache` return one canonical vector per content key,
   while `load_embedding_records` returns every occurrence for Qdrant indexing?
2. Why would replacing `document_family` with the full filename break valid
   cross-release lineage analysis?
3. What is the role of `document_id`, and why can multiple R21 FDDs still have
   different `document_id` values?
4. Why does a successful delete API response not prove that a local embedded
   vector store is safe to rebuild in place?
5. What must be validated before changing `QDRANT_COLLECTION_NAME` from
   `functional_specs` to `functional_specs_v2` for the API/UI?

### Correct-answer rubric

1. Cache lookup avoids repeat API work for identical content, so one canonical
   vector is correct. Qdrant must preserve every document/chunk occurrence for
   metadata filters, retrieval coverage, and citations, even when vectors match.
2. A family represents the logical FDD stream across R2, R21, R24, and later
   releases. A full filename is one document occurrence and would split related
   releases into unrelated groups.
3. `document_id` is the complete filename stem and identifies one exact FDD
   source. Release is not globally unique; several distinct R21 FDDs can each
   have their own document ID while sharing a family/release context where
   appropriate.
4. The probe showed stale points survive recreation. API acknowledgement covers
   the method call, not a verified physical-state guarantee. Reusing the name
   risks mixed schema/data and ungrounded retrieval.
5. Confirm the intended new name and config source, artifact/index coverage,
   point count and exact per-document verification, vector dimension, grounded
   retrieval/citation evaluation, API readiness, and rollback plan before
   directing the UI/API to the new collection.

### User answer evaluation

Pass. All five answers are production-ready.

- Q1 correctly separates cost-efficient content-vector reuse from preserving
  every citeable document/chunk occurrence in Qdrant.
- Q2 correctly keeps the family as the cross-release lineage stream and the
  full filename as one source occurrence.
- Q3 correctly identifies the full-source document ID as distinct from a
  release label that can be shared by multiple FDDs.
- Q4 correctly relies on the observed persistent-Qdrant probe rather than a
  delete API acknowledgement when deciding against in-place rebuild.
- Q5 covers configuration source, coverage, exact validation, vector contract,
  grounded evaluation, readiness, and rollback before routing API/UI traffic.

### Stronger interview-quality answer

The embedding cache de-duplicates exact content to control API cost, whereas
Qdrant must preserve each evidence occurrence because citation, filtering, and
lineage semantics depend on document and chunk identity. `document_family`
groups one logical stream across releases, `release_label` identifies the
release, and `document_id` identifies one exact FDD source; multiple R21 FDDs
therefore remain distinct without breaking cross-release grouping. Because the
embedded-Qdrant probe retained old points after recreation, migration must use
a new versioned collection, validated for artifact coverage, exact points,
dimensions, grounded retrieval/citations, readiness, and rollback before the
API/UI switches configuration.

### Gate

Step 106 interview gate accepted. Await the user's local `.env` update to a
new versioned Qdrant collection name and the master-command dry-run result
before any live collection build or API/UI switch.

### Live recovery confirmation

The user confirmed that the reviewed
`master_ingestion_embedding_docs.py` command completed successfully against
the new versioned collection. Step 106 is complete.

### Post-recovery interview evaluation

1. Partly correct. Preserve `functional_specs` as a recoverable prior index
   generation, not as an additional source to query beside v2. Its mixed old
   point-ID generations make it unsafe for current grounded answers; it may be
   useful for investigation or rollback while retained under its own name.
2. Needs correction. Release labels and current-state facts do not prove index
   coverage. Compare the active embedding-artifact occurrence count with the
   new collection's exact verified point count, then validate each deterministic
   point ID and payload (`document_id`, family, release, unit ID, vector
   dimension) using `check_qdrant_index.py`. Run retrieval/citation evaluation
   separately after structural coverage passes.
3. Correct. A UI/API still configured to the legacy collection can return stale
   or incomplete evidence, causing citations and current-state answers to be
   misleading even though v2 was built successfully.
4. Correct. Ingestion/indexing proves data movement and structure, whereas
   grounded answer quality also depends on retrieval, ranking, synthesis,
   citation validity, abstention, and reviewed correctness.
5. Needs explanation. Rollback means changing the API/UI configuration back to
   the previously validated, preserved collection name, restarting the
   processes, and recording the reason and affected time window. Do this only
   if that prior collection is known safe for the intended scope; a stale or
   mixed legacy collection is an investigation fallback, not an automatic
   production rollback target.

### Gate

Step 106 is accepted with remediation: the user understands the core safety
model and must retain the coverage-proof and rollback distinction for the next
collection migration.

## Step 107 — Four-FDD batch preflight and safe rejection

### Interview questions

1. Why is it valid for all four files to share `R1` and a document family, but
   unsafe for them to share one `document_id`?
2. Which operations in the master workflow can incur OpenAI cost or mutate
   local state, and why does `--dry-run` prove neither occurred?
3. If one FDD embeds successfully but exact Qdrant verification fails, what
   must happen to all four raw DOCX files, and why?
4. Why does a filename parser rejection improve grounded-RAG safety rather than
   merely input hygiene?
5. After this batch is indexed, why must the evaluation set include questions
   that distinguish the four R1 document IDs rather than only generic R1
   questions?

### Correct-answer rubric

1. Family/release support lineage grouping and release filtering; document ID
   identifies the exact source occurrence. Sharing it would collapse citation,
   payload, and audit identity across separate FDDs.
2. Embedding cache misses call OpenAI. Ingestion, artifact creation, Qdrant
   upsert, verification, and archival can write local state. Dry run prints the
   constructed commands but does not execute child processes, so it cannot call
   the provider, write Qdrant, or move source files.
3. The failing source must remain in `data/raw_specs`; the master must not
   archive an unverified document. Other sources should be handled only under
   the documented per-document success policy, never falsely reported as a
   fully verified batch.
4. An unparseable release/family cannot be reliably filtered, selected as
   current state, or cited in lineage answers. Rejecting it avoids evidence with
   ambiguous temporal metadata entering the index.
5. Generic release questions can pass while the system retrieves the wrong R1
   module. Document-specific questions test metadata filters, ranking,
   citations, cross-document confusion, and safe abstention when evidence is
   absent.

### User answer evaluation

Pass. All five answers match the production rubric.

1. Correctly separates lineage grouping from exact source identity and the
   citation/audit collision risk.
2. Correctly distinguishes paid embedding calls from local mutation and states
   why a master dry run cannot perform either.
3. Correctly retains an unverified source and avoids falsely reporting partial
   success as a verified batch.
4. Correctly connects parsing failure to unreliable temporal filtering,
   current-state selection, and citation rather than treating it as formatting.
5. Correctly identifies cross-module retrieval confusion that generic
   release-level questions would hide.

### Gate

Step 107 interview gate accepted. The four-FDD live master run is authorized.

## Step 108 — Clean versioned-collection reconstruction from cached embeddings

### Interview questions

1. Why was re-indexing from active embedding artifacts cheaper and safer than
   rerunning the master ingestion workflow after the wrong collection target
   was discovered?
2. Why is `579` collection points alone not sufficient evidence that the new
   collection is complete and correctly configured?
3. The legacy collection has 591 points while v2 has 579. Why must we not use
   the larger count as evidence that the legacy collection is better?
4. What does the nonexistent-collection negative test prove, and what does it
   not prove about retrieval quality?
5. Why must API/UI processes be restarted after changing `.env`, and what
   verification should occur before users rely on answers from v2?

### Correct-answer rubric

1. Existing artifacts already contain validated vectors and source metadata, so
   re-indexing avoids OpenAI cost and avoids reparsing/moving source DOCX files.
   It changes only the chosen vector-store generation.
2. A count can include stale, duplicate, wrong-schema, or wrong-payload points.
   Exact verification must compare all intended artifact records to their
   deterministic IDs, document/release payload, and vector contract.
3. The legacy collection is known to contain mixed point-ID generations;
   additional points can be stale duplicates rather than valid evidence.
   Correctness comes from the selected artifact manifest and exact validation,
   not a larger number.
4. It proves the verifier fails closed if its configured collection does not
   exist and will not silently create/accept it. It does not prove query
   relevance, ranking, citation entailment, abstention, or answer correctness.
5. Settings are loaded into process memory at startup. Restarting applies the
   new name; then verify effective configuration, readiness, a known retrieval
   query, citations, and the reviewed evaluation set before user traffic.

### User answer evaluation

Pass. All five answers meet the Step 108 production rubric. The user correctly
distinguished artifact reuse from re-embedding, structural coverage from a
point count, stale legacy state from a validated manifest, fail-closed
verification from retrieval quality, and startup configuration from user-ready
validation.

## Step 109 — Duplicate raw-versus-archive source guard

No interview questions were requested. This was a narrowly scoped validation
and maintainability improvement; its regression test demonstrates that a
case-insensitive filename collision is rejected before every child stage.

## Step 110 — Versioned FDD grounded-evaluation runner and draft gate

### Interview questions

1. Why is a benchmark with `sme_reviewed=false` useful for a draft baseline but
   invalid as a release-quality gate, even if every automated check passes?
2. Why must expected `document_id` citations be checked separately from release
   labels, particularly when multiple FDDs share R1?
3. Why does structural answer/citation evaluation not prove the expected claims
   are entailed by the answer and evidence?
4. What cost and state changes occur during a non-dry evaluation run, and which
   local artifacts make a failure diagnosable later?
5. If an abstention case returns citations but `is_answered=false` with a clear
   refusal reason, why can that still be safe behavior?

### Correct-answer rubric

1. Draft cases can expose retrieval/citation regressions, but their expected
   claims have not been accepted by a domain authority. Passing them cannot
   justify a 90% SME-correctness claim or release decision.
2. A release label groups multiple FDD occurrences. Document ID identifies the
   exact cited source and catches cross-module confusion that release-only
   checks would miss.
3. Structural checks establish state and source identity, not whether wording
   is complete, correctly qualified, non-contradictory, or actually supported
   by the cited text. An SME must review claim entailment.
4. Each case can make embedding and LLM calls, write answer traces/reports, and
   consume local Qdrant reads. The run report, trace directory, request IDs,
   retrieval metadata, citations, model usage, and cost estimates support later
   diagnosis.
5. A refusal may retain retrieved evidence to explain why the threshold was not
   met. It is safe if it does not make a functional claim, has
   `is_answered=false`, carries a machine-readable reason, and the UI presents
   it as an abstention rather than an answer.
