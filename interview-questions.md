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
