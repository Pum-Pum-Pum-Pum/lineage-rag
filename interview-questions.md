# Phase 2 interview questions

## Steps 150-152 - Interview evaluation

### Step 150 - Snapshot contracts

1. Why must `expected_changed_packages` remain an assertion rather than the
   authority for which files changed?
2. Why does freezing a Pydantic model not by itself make a source snapshot on
   disk immutable?
3. Why must snapshot paths reject absolute paths, traversal, and
   case-insensitive duplicates before reading source content?

### Step 151 - Streaming intake validation

4. Why should a file above 5 MiB be warned about but not automatically rejected,
   and which downstream controls still become necessary?
5. Why retain both exact-byte SHA-256 and newline-normalized text SHA-256?
6. Why must secret scanning report categories without logging the matched
   source value, and what operational problem can false positives create?

### Step 152 - Immutable snapshots and deterministic diffs

7. How does validate-copy-verify-atomic-promote protect against partial writes
   and an intake changing during publication?
8. Why are exact renames reported only for an unambiguous one-deleted to
   one-added hash match?
9. What does a verified code snapshot prove, and which parsing, retrieval,
   grounding, citation, and runtime properties remain completely unproven?

### Evaluation

Overall result: **9/9 accepted after the corrected Answer 9.**

1. **Accepted.** The deterministic comparison is the authority;
   `expected_changed_packages` is a reviewer assertion used to expose missing
   or unexpected changes.
2. **Accepted with precision.** Frozen Pydantic models prevent ordinary
   in-process field assignment. Snapshot immutability is enforced separately
   through content-addressed identity, no-overwrite publication, exact source
   verification, and tamper detection. It is not an operating-system write
   prohibition.
3. **Accepted.** Unsafe paths can escape the intake boundary, while
   case-insensitive collisions make identity nondeterministic across Windows
   and case-sensitive deployment filesystems.
4. **Accepted.** File size alone is not evidence of invalid source. Streaming
   checks and isolated parsing with explicit resource boundaries prevent one
   large package from exhausting the ingestion process.
5. **Accepted with precision.** The exact hash proves byte identity and enables
   exact rename detection. The normalized hash identifies formatting-only
   newline/BOM differences, but the current diff still reports changed bytes as
   `modified`; normalization does not silently erase the change.
6. **Accepted.** Diagnostics must identify only the secret category and source
   path. False positives can block legitimate intake and therefore require a
   controlled review/remediation process, never an unrecorded bypass.
7. **Accepted with precision.** Validation establishes the intended manifest;
   copy happens outside the final namespace; verification detects a source
   mutation or partial copy; atomic promotion makes only the verified directory
   visible. Later tampering is detected by integrity verification.
8. **Accepted.** Multiple same-hash deleted or added files do not prove which
   old path became which new path, so the system reports ambiguity rather than
   inventing lineage.
9. **Accepted after revision.** A verified Step 152 snapshot proves only that
   the reviewed request was valid, allowlisted source bytes were copied
   completely, hashes and manifest
   identity agree, the archive was published without overwrite, and the
   file-level diff is reproducible. It proves nothing yet about ANTLR parsing,
   conditional compilation, symbols, dependencies, embeddings, Qdrant points,
   retrieval relevance, citations, grounded answers, SME answer correctness,
   concurrency, performance, deployed behavior, or rollback of an active code
   index.

Gate status: **accepted. Steps 150-152 are complete.**

## Steps 153-155 - Interview gate

### Step 153 - Grammar and conditional compilation

1. Why must conditional preprocessing use a separate line-preserving parse
   view while immutable original source remains the only citation authority?
2. How should answer behavior differ when compiler context proves a branch is
   active versus when `$IF` conditions remain `conditional_unknown`?
3. Why do pinned grammar/runtime versions, source hashes, and real directive
   fixtures provide stronger reproducibility than merely observing directive
   tokens in the lexer grammar?

### Step 154 - Isolated and degraded parsing

4. Why is a token-aware structural splitter safer than ordinary regular
   expressions for procedures containing nested blocks, comments, and strings?
5. What does the successful >5 MiB timeout test prove, and what parser
   performance or semantic-correctness property does it still not prove?
6. How should operations treat `complete_with_degradation` differently from
   `complete` and `failed` before any later indexing or activation?

### Step 155 - Context-enriched retrieval units

7. Why must `retrieval_text` keep its derived context explicitly separate from
   exact citeable `text` and `source_map`?
8. Why link only referenced declaration units instead of prepending the full
   package header to every procedure, and what might selective linking miss?
9. What do Steps 153-155 now prove, and which overload identity, dependency,
   DDL, embedding, retrieval, citation, grounded-answer, and production
   properties remain unproven?

### Evaluation

Overall result: **9/9 accepted.**

1. **Accepted.** The parse view may normalize directive syntax only as a
   parser aid; preserved offsets and newlines keep every parsed range aligned
   with immutable original source, which remains the citation authority.
2. **Accepted.** A branch proven active by recorded compiler context may be
   described as selected for that context. `conditional_unknown` requires an
   explicit qualification that the deployed branch cannot be confirmed.
3. **Accepted.** Lexer tokens prove recognition only. Pinned tool/runtime and
   grammar hashes reproduce the implementation, while real fixtures prove the
   combined lexer, parser, parse-view, and state-handling behavior.
4. **Accepted.** Token-aware splitting distinguishes structural syntax from
   comments and literals and tracks nested constructs that ordinary regex
   boundaries cannot interpret safely.
5. **Accepted.** The test proves that a file above 5 MiB is not rejected or
   silently lost when a resource boundary forces declared fallback. It does
   not establish representative large-file latency, memory capacity, parser
   coverage, or semantic equivalence with a full parse.
6. **Accepted with precision.** `complete` permits progression to the next
   controlled gate; it does not authorize automatic indexing or activation.
   `complete_with_degradation` requires diagnostic review and an explicit
   acceptance policy. `failed` must stop downstream publication.
7. **Accepted.** Derived retrieval context can support candidate discovery but
   cannot support a source-code claim. Exact source text and its source map
   preserve auditable file-and-line citations.
8. **Accepted.** Selective links reduce duplicated tokens, embedding cost, and
   provenance noise. Static reference extraction can miss indirect, dynamic,
   aliased, conditional, or parser-degraded dependencies; later dependency
   extraction and explicit unknowns must handle that boundary.
9. **Accepted.** The answer correctly limits the completed evidence to the
   parsing and retrieval-unit foundation and identifies overload identity,
   dependency/DDL extraction, indexing, retrieval, grounded answering,
   evaluation, activation, and rollback as later gates.

Gate status: **accepted. Steps 153-155 are complete.** The next approved batch
is Steps 156-158: overload-safe Oracle symbol identity, dependency and boundary
extraction, followed by DDL and synonym modeling.

## Steps 156-158 - Interview gate

### Step 156 - Oracle symbol identity

1. Why must parameter modes and function return type affect the full
   declaration hash but not create a distinct overload discriminator?
2. Why may a package specification contain a default expression that its body
   omits without proving an incompatible implementation, and which differences
   should still fail the declaration/implementation gate?
3. Why are both a logical `symbol_key` and a source-specific `occurrence_id`
   required for overloaded retrieval, diffs, and exact citations?

### Step 157 - Dependencies and unknown boundaries

4. If a call has multiple overload candidates with the same argument count,
   why must the artifact retain every candidate, and how should a later answer
   qualify its impact analysis?
5. Why are kernel-package prefixes empty by default, and what evidence and
   review are required before configuring them for the real corpus?
6. What do static table edges and `dynamic_unknown` prove, and which aliases,
   runtime SQL construction, polymorphism, or execution behavior can they not
   establish?

### Step 158 - DDL and synonym modeling

7. Why must synonyms be resolved across the complete approved snapshot rather
   than independently inside each source file?
8. How should `database_link`, `external_schema`, `ambiguous`, and `cyclic`
   synonym states affect retrieval and generated answers?
9. What does static DDL extraction prove about tables and constraints, and
   which live Oracle metadata, privilege, edition, deployment, retrieval, and
   grounded-answer properties remain unproven?

### Evaluation

Overall result: **9/9 accepted, with a required precision correction to
Answer 5.**

1. **Accepted.** Modes, defaults, `NOCOPY`, return type, and deterministic
   declaration metadata belong in the semantic declaration hash. Oracle
   overload identity remains limited to permitted parameter characteristics;
   mode-only or return-only changes cannot manufacture a distinct overload.
2. **Accepted.** A caller-facing default may be declared in the specification
   without being repeated in the implementation. Compatibility must still
   enforce ordered parameter identity/types, quoted-name state, modes,
   `NOCOPY` where relevant, and function return type. Default metadata remains
   separately diffable rather than becoming a false body incompatibility.
3. **Accepted.** `symbol_key` supports logical overload identity across source
   occurrences. `occurrence_id` preserves the exact declaration or
   implementation location required for citations, diffs, and non-destructive
   storage of duplicate content.
4. **Accepted.** Argument count alone cannot resolve same-arity overloads when
   static argument types are unavailable. All candidates must remain visible,
   and later impact analysis must state that the target is candidate-based,
   not proven.
5. **Accepted after precision correction.** Empty defaults prevent the system
   from inventing a customer-specific kernel boundary. The required evidence
   is an SME-reviewed namespace or explicit package inventory identifying
   which unavailable packages are kernel-owned, ideally tied to application
   build/SVN scope and negative examples that must remain custom or external.
   Manifest, allowlist, secret-scan, parser-review, and embedding-approval gates
   remain important but do not prove that a package prefix means kernel code.
6. **Accepted.** Static edges prove source-visible references only.
   `dynamic_unknown` records detection of dynamic behavior while withholding an
   unsupported target. Alias interpretation, constructed names, runtime branch
   selection, polymorphic dispatch, and executed paths remain unproven.
7. **Accepted.** Snapshot-wide resolution is required for cross-file targets,
   chains, duplicate identities, and cycles. Per-file resolution would create
   false external or missing-target conclusions.
8. **Accepted.** External-schema and database-link states are qualified
   external dependencies, never live-target proof. Ambiguous and cyclic states
   must prevent definitive target claims and remain visible in retrieval and
   answer qualifications.
9. **Accepted.** Static DDL establishes only source-declared structure in the
   approved snapshot. Live ownership, validity, privileges, editions,
   deployment state, metadata drift, retrieval relevance, citation entailment,
   grounded answers, and production behavior remain later gates.

Gate status: **accepted. Steps 156-158 are complete.** Before paid indexing,
the next operational gate is a curated real snapshot run and parser/static-
analysis coverage review. Steps 159-161 remain blocked until that evidence is
accepted.

## Pre-Step 159 real-corpus readiness gate - Interview

### Snapshot intake

1. Why must this first snapshot contain the complete selected custom module set
   rather than only the files a reviewer believes changed?
2. What does adding `.spc` to the versioned allowlist prove, and what parsing or
   security property does the extension mapping itself not prove?
3. Why is `base_snapshot_id` absent for this request, and what comparison will
   become possible when a later complete snapshot references this snapshot?

### Parser resource recovery

4. Why is a separate bounded segmented attempt safer than going directly from
   a full-file timeout to anonymous line chunks?
5. Why must the 1,000-character segment boundary remain configurable and be
   recorded in the immutable generation contract?
6. What does `token_structural` prove about an oversized routine, and which
   declaration, dependency, control-flow, or runtime facts remain unproven?

### Readiness decision

7. Why do zero fallback files and 19/19 retained routine identities still not
   make the current artifacts safe to embed immediately?
8. Why is silently deleting all 1,795 unresolved dependencies unsafe, and what
   distinction must the next classifier preserve while removing SQL/table
   syntax false positives?
9. What exact evidence should be required before unblocking Steps 159-161, and
   why do 484 passing regressions not replace that real-corpus evidence?
