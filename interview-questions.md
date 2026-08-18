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

### Evaluation

Overall result: **6/9 accepted; Answers 7-9 require correction before the
pre-index gate is accepted.**

1. **Accepted.** A complete selected module set makes absence meaningful and
   supports deterministic added, modified, deleted, unchanged, and exact-rename
   comparison. A changed-files-only upload cannot distinguish deletion from an
   omitted file and loses unchanged dependency context.
2. **Accepted.** The `.spc` mapping authorizes intake and selects the PL/SQL
   handler. Content validation, secret scanning, grammar compatibility, symbol
   extraction, and dependency correctness remain separate controls.
3. **Accepted.** No earlier approved snapshot exists, so there is no valid base
   identity. A later complete snapshot can reference this immutable snapshot
   and produce reproducible file- and symbol-level changes.
4. **Accepted.** Token-aware segmentation preserves routine boundaries while
   respecting comments, strings, parentheses, and nested blocks. Anonymous
   line chunks preserve bytes but not trustworthy routine identity.
5. **Accepted.** The threshold is a measured implementation boundary, not a
   universal PL/SQL constant. Recording it makes parser outcomes reproducible;
   changing it requires a new generation and performance/coverage comparison.
6. **Accepted with precision.** `token_structural` proves only lexer-derived
   routine kind/name, conservative boundary, exact source range, and retained
   original text. It does not prove the declaration signature, nested symbols,
   dependency resolution, control flow, compile validity, or runtime behavior.
7. **Incomplete.** Generic ingestion gates are correct but miss the observed
   blocker: one retained routine is about 93.9 KB. Embedding it as one unit can
   exceed model/token limits, dilute retrieval, inflate cost, and overflow
   evidence packing. Exact bounded child chunks with parent identity and line
   maps must be proven first.
8. **Incomplete.** Genuine unknown edges must remain, but the answer must also
   distinguish them from classifier false positives. The real artifact treats
   SQL identifiers and keywords such as table names, `IN`, `TRUNC`, `AND`, and
   `EXISTS` as possible routine calls. The next classifier must suppress
   grammar-context false positives while retaining actual unresolved calls,
   dynamic SQL, kernel boundaries, and external targets, with recall tests.
9. **Incomplete.** The listed indexing gates occur later. Before Steps 159-161
   can begin, the renewed real-corpus gate must prove bounded child-unit sizes,
   complete and non-overlapping/declared-overlap source coverage, exact parent
   and line provenance, deterministic IDs, no routine loss, materially reduced
   false-positive calls, retained known unknowns, and reviewed dependency
   precision/recall fixtures. The 484 regressions cover existing synthetic and
   integration contracts, not these measured corpus-specific properties.

Gate status: **not yet accepted.** No indexing or paid embedding work is
authorized by this result.

### Revised Answers 7-9 - Evaluation

7. **Accepted.** The revised answer identifies the measured 93.9 KB unit and
   connects it to embedding limits, retrieval dilution, cost, and prompt
   packing. It correctly requires deterministic bounded child chunks with
   parent identity and exact source-line provenance.
8. **Accepted.** The revised answer preserves genuine unknown dependencies
   while separating them from observed classifier noise. It names the required
   categories: genuine unresolved calls, dynamic SQL, kernel/external
   boundaries, and SQL/table syntax that is not a routine call.
9. **Accepted.** The revised answer defines the correct evidence boundary:
   bounded units, complete controlled-overlap coverage, exact provenance,
   deterministic IDs, no lost routines, reduced false positives, retained
   genuine unknowns, and reviewed precision/recall.

Interview result: **9/9 accepted.** The learner-understanding gate is complete.
The operational pre-index gate remains pending because these requirements are
not yet implemented or measured on the real corpus. No paid embedding or code
indexing operation is authorized by the interview result alone.

## Steps 159A-159C - Interview gate

### Step 159A - Bounded child units

1. Why are the 500-character ANTLR segment bound and 6,000-character retrieval
   unit bound separate controls, and what failure does each prevent?
2. Why must a child unit record both its own exact source map and the parent
   unit ID/source map instead of using only sequential chunk numbers?
3. What do deterministic, bounded, gap-free child chunks prove, and what
   retrieval, embedding-model, or answer-quality properties remain unproven?

### Step 159B - Dependency classification

4. Why is token/context classification safer than adding `TBLBUNDLERPT`, `IN`,
   or `EXISTS` to a corpus-specific global ignore list?
5. How do separate routine-call, table, dynamic-SQL, kernel, and external edges
   improve grounded impact analysis and graceful failure?
6. Why is precision/recall `1.0` on the current fixture insufficient as a
   production claim, and how should an SME-reviewed evaluation set be expanded?

### Step 159C - Real-corpus gate

7. Why did the gate rebuild artifacts from original bytes instead of trusting
   the published JSON counts and point-like identities alone?
8. What did reproducing the same snapshot ID in a verification root establish,
   and what operational problem does the inaccessible original archive still
   reveal?
9. Which evidence now permits moving toward Steps 159-161, and which paid-call,
   embedding, Qdrant identity, retrieval, citation, and activation gates still
   remain separate?

### Evaluation

Overall result: **7/9 accepted. Answers 8-9 require precision corrections.**

1. **Accepted.** The ANTLR threshold bounds grammar-runtime risk, while the
   retrieval threshold bounds embedding and evidence-unit size. One threshold
   cannot represent both parser complexity and retrieval usefulness.
2. **Accepted.** Child provenance supports exact citations and bounded search;
   parent provenance preserves routine identity, reconstruction, grouping, and
   later deduplication across overlapping children.
3. **Accepted.** The answer correctly limits the proof to reproducible source
   coverage, size bounds, and provenance. Embedding behavior, retrieval recall,
   packing, citation entailment, and answer correctness remain unproven.
4. **Accepted.** Syntax/context rules generalize beyond observed names and
   avoid globally suppressing a name that could be callable elsewhere. A
   growing corpus-specific denylist would hide classifier defects.
5. **Accepted.** Typed edges preserve distinct evidence strengths and allow
   answers to qualify dynamic, hidden-kernel, and external behavior instead of
   converting uncertainty into a false implementation claim.
6. **Accepted.** The draft fixture is too small and narrow for a production
   quality claim. It must be expanded with representative positive, negative,
   ambiguous, overloaded, SQL-heavy, and boundary cases and then SME-reviewed.
7. **Accepted.** Rebuilding from immutable bytes proves the artifact can be
   reproduced independently of mutable staging JSON and detects source-map,
   newline, configuration, or implementation drift.
8. **Incomplete.** Reproducing the exact snapshot ID, content hash, file
   hashes, and policy hash establishes an equivalent content contract. The
   observed ACL incident was not evidence that unauthorized mutation was
   possible; it denied the legitimate parser read access. The operational risk
   is therefore archive availability and recoverability as well as permission
   correctness: an immutable archive that cannot be read cannot support
   verification, rebuild, rollback, or incident recovery.
9. **Incomplete.** The bounded-child and real-corpus mechanism gate has already
   passed under `analysis_v6`; it is not still pending. What remains pending is
   SME review/expansion of the draft dependency-classifier labels. That review
   should precede treating precision/recall as a quality gate. After that,
   Steps 159-161 may begin, but paid embedding still requires explicit approval
   and indexing must separately prove cache identity, isolated lexical/Qdrant
   generations, exact artifact-to-point verification, dimensions, and rollback.
   Steps 162-170 retain retrieval, citation, answer, SME, and activation gates.

Gate status: **learner gate pending revised Answers 8-9.** No paid embedding or
Qdrant operation is authorized.

### Revised Answers 8-9 - Evaluation

8. **Accepted.** Reproducing the same snapshot ID, content hash, file hashes,
   and policy hash proves an equivalent content contract. The answer correctly
   identifies the observed ACL failure as a legitimate-read availability and
   recoverability problem affecting verification, rebuild, rollback, and
   incident recovery.
9. **Accepted.** The answer correctly distinguishes the passed `analysis_v6`
   mechanism gate from the still-draft semantic-quality evidence. SME review
   and representative expansion of dependency labels remain necessary before
   precision/recall becomes a quality gate. Paid embedding, isolated indexing,
   exact point verification, retrieval, citation, grounded-answer, SME, and
   activation gates remain independent.

Final interview result: **9/9 accepted.** The Steps 159A-159C learner gate and
real-corpus mechanism gate are complete. Dependency-label SME review remains
the next unpaid quality task. No paid embedding or Qdrant operation is
authorized by this acceptance.

## Steps 159-161 - Interview gate

### Step 159 - Deterministic code index contract

1. Why must code evidence use snapshot/path/symbol/line metadata rather than
   reusing the FDD document-family and release contract unchanged?
2. Why may duplicate embedding text share a cache vector while each code source
   occurrence still requires a distinct deterministic Qdrant point ID?
3. What does artifact identity `922a253d...e75c` prove, and what embedding,
   retrieval, citation, or semantic property does it not prove?

### Step 160 - Isolated lexical and vector generations

4. Why must the code lexical artifact and `code_custom_*` Qdrant collection be
   isolated from `functional_specs_v4`, even when combined mode is planned?
5. Why does refusing to write an existing Qdrant collection improve rollback
   and diagnosis, and what operational cleanup burden can it create?
6. What did the local lexical query prove about `spPNBRPT006`, and what dense,
   hybrid, context-packing, or answer behavior remains untested?

### Step 161 - Exact verification and approval boundaries

7. Why are exact point count, deterministic IDs, payload equality, and vector
   dimensions all required rather than trusting Qdrant upsert acknowledgement?
8. Why are SME dependency-label review and explicit OpenAI code-disclosure/cost
   authorization separate gates, and why is either one alone insufficient?
9. What parts of Steps 159-161 are complete now, what real operations remain
   unexecuted, and which Steps 162-170 gates still prevent activation?

### Evaluation

Overall result: **9/9 accepted.**

1. **Accepted.** Code provenance is snapshot-, path-, symbol-, occurrence-, and
   line-based. It cannot be represented truthfully as an FDD release/document
   occurrence, and FDD-to-code linkage remains a separate reviewed mapping.
2. **Accepted.** Cache identity represents reusable semantic input, whereas
   point identity represents a citeable source occurrence. Reusing a vector
   must never collapse distinct files, symbols, snapshots, or line ranges.
3. **Accepted.** The answer correctly limits artifact identity to deterministic
   source/artifact correspondence. It does not establish embedding semantics,
   retrieval relevance, citation entailment, or answer correctness.
4. **Accepted.** Separate artifacts, payloads, thresholds, indexes, and Qdrant
   namespaces prevent code evidence from contaminating the active FDD lane and
   allow the two lanes to be evaluated independently before combined mode.
5. **Accepted.** New-generation-only writes prevent partial mutation and mixed
   evidence contracts. The trade-off is retained failed/stale generations that
   require explicit inventory, retention, and cleanup procedures.
6. **Accepted.** The real lexical query proves identifier-level candidate
   discovery in the isolated prepared artifact only. Dense/hybrid retrieval,
   representative recall, Qdrant, citations, and answers remain untested.
7. **Accepted.** Count, deterministic identity, provenance payload, and vector
   schema detect distinct failure classes and therefore must all pass.
8. **Accepted.** Semantic label quality and permission to disclose proprietary
   code externally are different authorities. Neither gate implies the other,
   and cost authorization remains part of the disclosure decision.
9. **Accepted.** The answer accurately separates completed parsing/indexing
   mechanisms from unexecuted SME review, external embedding, staged Qdrant
   indexing, exact real-point verification, retrieval/citation/answer
   evaluation, SME answer review, and deliberate activation.

Gate status: **Steps 159-161 learner gate accepted.** The real prepared contract
remains `dependency_review_status="draft"`. No OpenAI embedding call or Qdrant
write is authorized by this interview acceptance.

## Steps 161A-161C - Custom-code boundary policy interview gate

### Step 161A - Versioned suffix configuration

1. Why should `_CUSTOM` be stored in a versioned policy rather than embedded
   as a string literal in the dependency analyzer?
2. Why must changing this configuration change the policy hash and invalidate
   promotion of older derived artifacts?
3. Why does a filename ending `_custom.sql` provide useful intake evidence but
   not by itself prove every dependency called inside the file is custom?

### Step 161B - Package and table semantics

4. For `APP.PKG_REPORT_CUSTOM.BUILD_REPORT`, which component owns the routine,
   and why would checking only the first component classify it incorrectly?
5. Why must `APP.AUDIT_CUSTOM` remain a table edge rather than be classified as
   a custom package merely because its name ends `_CUSTOM`?
6. Why are unqualified unresolved calls not automatically labeled kernel even
   though known custom packages/functions use the `_CUSTOM` convention?

### Step 161C - Quality and generation gates

7. What does the updated draft result of precision/recall `1.0` prove, and why
   does a two-case synthetic fixture still not constitute SME approval?
8. Why must `analysis_v6` remain immutable after the boundary policy changes,
   rather than replacing its stored policy hash or analysis records?
9. What must be rebuilt and rechecked before any code excerpts may be sent for
   paid embeddings under the new policy?

### Evaluation

Overall result: **9/9 accepted.**

1. **Accepted.** The answer correctly treats `_CUSTOM` as an explicit,
   auditable classification policy rather than an intrinsic identifier rule.
   Configuration supports controlled reuse across module sets without hiding
   business conventions inside analyzer code.
2. **Accepted.** The answer correctly connects policy-dependent derivation to
   the policy hash. Reusing an old identity after changing classification rules
   would make provenance false and could promote stale results.
3. **Accepted.** File identity and dependency identity are separate. A custom
   source file can call kernel/external routines and access any table, so its
   suffix cannot be inherited by every extracted edge.
4. **Accepted.** `APP` is the schema namespace and `PKG_REPORT_CUSTOM` is the
   package owner. The answer correctly avoids schema-wide custom
   classification and identifies the relevant qualified-name component.
5. **Accepted.** Syntax and parse context establish that `APP.AUDIT_CUSTOM` is
   a table reference. A suffix must not override the independently extracted
   dependency kind.
6. **Accepted.** The answer preserves the important distinction between an
   unresolved target and a proven unavailable kernel boundary. Converting
   uncertainty into a kernel assertion would weaken grounded analysis.
7. **Accepted.** Perfect metrics describe only the current small draft fixture.
   The answer correctly withholds corpus-level and production claims pending a
   representative, reviewed label set.
8. **Accepted.** Immutability preserves the exact source-policy-result
   relationship and enables clean comparison and rollback. The new policy
   requires a new generation rather than mutation of `analysis_v6`.
9. **Accepted.** The answer names the necessary rebuild and verification
   controls and retains explicit paid/disclosure authorization as a separate
   boundary. A stronger operational phrasing is that the rebuilt generation
   must also produce a new prepared index contract whose stored policy and
   source identities match the reviewed analysis exactly.

Gate status: **Steps 161A-161C learner gate accepted.** The `_CUSTOM` policy
contract is understood. Existing `analysis_v6` and its prepared index remain
historical, non-promotable artifacts under the old policy. No OpenAI code
embedding or Qdrant write is authorized by this interview acceptance.

## Steps 161D-161F - Policy-v2 regeneration interview gate

### Step 161D - Immutable analysis generation

1. Why must policy v2 publish `analysis_v7` instead of rewriting
   `analysis_v6`, even though the source snapshot bytes are unchanged?
2. What does `complete_with_degradation` with one segmented file prove, and
   what semantic or performance property does it not prove?
3. Why did the short outer command timeout leave an unpublished temporary
   directory, and which validations made its removal safe?

### Step 161E - Focused dependency review packet

4. Why group 90 occurrences into 39 target/classification cases instead of
   asking an SME to review every occurrence independently?
5. Why are unresolved table identities excluded from this classifier packet
   while all 488 table edges remain in the analysis graph?
6. What do 19/19 routine retention, exact source mapping, and zero known false
   calls prove, and why do they not replace SME review of the 39 cases?

### Step 161F - Prepared-contract verification

7. Why must the verifier rebuild the complete artifact rather than check only
   its stored artifact identity string?
8. Why are 96 unique point IDs required even if a future corpus contains fewer
   than 96 unique embedding inputs?
9. What gates remain between the verified draft contract and a usable code RAG
   answer, and which next action remains unpaid?

The Steps 161D-161F questions above are retained as history but their proposed
blanket non-suffix kernel classification was superseded by the corrected
package-ownership policy before learner evaluation. Do not use that gate for
promotion.

## Steps 161G-161I - Corrected ownership and scale interview gate

### Step 161G - Declared program-unit validation

1. Why is the declared PL/SQL program-unit name authoritative while the
   filename remains a required assertion?
2. Why do routines inside `PKGTRANSACTIONBLL_P_CUSTOM` inherit custom-source
   availability even when their individual names lack `_CUSTOM`?
3. Why are `.ddl` tables exempt from program-unit suffix validation, and what
   risk would suffix-filtering tables create?

### Step 161H - Grounded dependency boundaries

4. Why is `custom_source_missing` materially different from
   `kernel_unavailable` for retrieval, remediation, and user answers?
5. Why did targets such as `ALC.TRANSACTIONNUMBER` prove blanket non-suffix
   kernel inference unsafe?
6. Why is a larger unresolved count preferable to a smaller count produced by
   unsupported kernel classifications?

### Step 161I - Scale and immutable regeneration

7. Which inputs must match before parse/retrieval artifacts may be reused, and
   why must dependency analysis still be rebuilt after a policy change?
8. How does canonical-name symbol indexing improve behavior for a 4,000-file
   corpus without changing grounded resolution semantics?
9. What does the verified v4 prepared contract prove, and what unpaid review,
   paid embedding, retrieval, citation, and answer gates remain?

## Steps 161J-161L - Parser-coverage remediation interview gate

### Step 161J - Independent declaration inventory

1. Why was comparing retrieval units only with already-detected routine
   segments a circular coverage check?
2. How does an independent declaration-start inventory detect a top-level
   routine omitted by the segmenter without pretending to fully parse it?
3. Why may a nested local declaration be covered by its retained parent while
   an uncovered top-level declaration must fail the stage?

### Step 161K - CASE-aware segmentation

4. Why did SQL `CASE ... END AS alias` cause `spPNBRPT023` to disappear under
   the old begin/end-depth algorithm?
5. Why is tracking CASE depth structurally safer than adding `END AS` to a
   corpus-specific exception list?
6. What does recovering exact lines 32499-33237 prove, and what retrieval or
   answer-quality property does it still not prove?

### Step 161L - Immutable v10 and fail-closed gates

7. Why must v10 reject reuse of v9 parse artifacts even though source hashes,
   ANTLR versions, and chunk limits are unchanged?
8. Why must declaration coverage, extracted-node coverage, retrieval-unit
   coverage, and symbol coverage be checked separately?
9. Why does the passing v10 pre-index gate remove `SPPNBRPT023` from ambiguity
   review but still require SME review of the other 40 cases before paid code
   embeddings?

### Evaluation

Overall result: **9/9 accepted.**

1. **Accepted.** The answer identifies the circular oracle correctly: a set of
   already-detected segments cannot reveal a declaration the segmenter omitted.
   The independent lexer inventory establishes a separate expected set.
2. **Accepted with precision refinement.** The implemented inventory detects
   declaration starts, names, kinds, and source positions; it does not determine
   complete routine boundaries. Comparing those starts with independently built
   segments is sufficient to expose an omitted top-level declaration.
3. **Accepted with precision refinement.** Parent containment proves that the
   nested source remains available in a retained citeable parent range. It does
   not create an independent nested symbol or prove local dependency resolution.
   An uncovered top-level declaration has no retained routine parent and must
   fail closed.
4. **Accepted.** The SQL CASE `END` reduced the old PL/SQL begin depth and the
   following `AS` prevented a valid routine terminator match, so the segment scan
   abandoned `spPNBRPT023`.
5. **Accepted.** CASE-depth state generalizes across aliases, whitespace, nested
   CASE expressions, and formatting, whereas an `END AS` exception would encode
   one corpus observation rather than the syntax relationship.
6. **Accepted.** Exact lines prove recovered source coverage and provenance for
   this procedure only. The later node, retrieval, symbol, and dependency gates
   separately prove those transformations; retrieval relevance and answer
   quality remain untested.
7. **Accepted.** Parse identity includes the derivation contract, not merely
   source bytes and ANTLR versions. Reusing v9 would preserve the defect under a
   misleading v10 generation label.
8. **Accepted.** Each boundary can lose or corrupt information independently,
   so declaration, node, retrieval-unit, and symbol checks provide distinct
   failure localization and must not be collapsed into one count.
9. **Accepted.** `SPPNBRPT023` now has one direct in-snapshot candidate and is no
   longer ambiguous. The remaining packet cases require SME judgment because
   static evidence has not established a definitive target or has exposed a
   likely classifier false positive.

Gate status: **Steps 161J-161L learner gate accepted.** The strengthened v10
pre-index mechanism is approved. The v10 dependency packet remains `draft`;
SME review of its 40 cases is still required. No OpenAI code disclosure, paid
embedding, Qdrant indexing, or activation is authorized by this acceptance.

## Steps 161M-161O - Canonical SME packet regeneration interview gate

### Step 161M - Preserve the submitted review

1. Why must a generation-mismatched SME submission be preserved rather than
   silently overwritten with a canonical packet?
2. Why is the packet header insufficient to prove that the body belongs to the
   stated parser generation?
3. Why should verdict migration use stable review IDs instead of ordinal case
   numbers or target names alone?

### Step 161N - Deterministic no-overwrite regeneration

4. Why regenerate from v10 analysis artifacts rather than repair the edited
   Markdown headings manually?
5. Why is a separate canonical output namespace safer than overwriting the
   SME-edited file?
6. What external cost or disclosure occurred during local regeneration, and
   why?

### Step 161O - Identity and scope verification

7. What does byte-identical JSON prove, and what SME-quality property does it
   not prove?
8. Why must both required-target presence and stale-target absence be checked?
9. Why do 40/40 placeholders represent a structurally correct packet but not a
   completed review gate?

### Evaluation

Overall result: **9/9 accepted.**

1. **Accepted.** The original submission is human-review evidence, including
   its generation mismatch. Overwriting it would erase diagnostic history and
   could falsely imply that its verdicts were made against canonical v10 cases.
2. **Accepted.** Header metadata is not a content-integrity control. Case IDs,
   target set, order, packet identity, and source artifact bindings must agree
   with the declared generation.
3. **Accepted with precision refinement.** Stable review IDs prevent ordinal
   drift, but a verdict must also be bound to the packet identity because the
   same-looking target can have different evidence or proposed states in a
   later packet.
4. **Accepted.** Deterministic regeneration reconstructs case identity,
   evidence, and provenance from v10 artifacts. Manual heading repair changes
   presentation without repairing the evidence contract.
5. **Accepted.** The separate namespace preserves both the SME submission and
   canonical output. In this incident it primarily separates canonical v10
   regeneration from a mixed human-edited submission, not merely v9 from v10.
6. **Accepted.** The operation read local verified artifacts and rendered local
   JSON/Markdown. It made no OpenAI request, embedding call, or Qdrant write and
   therefore incurred no provider cost or new source disclosure.
7. **Accepted with precision refinement.** Byte equality proves exact
   equivalence to the compared v10 JSON. By itself it does not establish that
   either file is authoritative; deterministic reconstruction and verified
   snapshot/analysis identities provide that provenance.
8. **Accepted.** Presence and absence detect complementary failure modes: an
   incomplete new packet and a mixed packet containing stale evidence.
9. **Accepted.** Placeholder completeness proves render/schema scope only. It
   explicitly proves that none of the 40 dependency judgments has yet been
   supplied in the canonical packet.

Gate status: **Steps 161M-161O learner gate accepted.** Canonical v10 packet
regeneration and identity verification are approved. The canonical packet
still requires SME verdicts for all 40 cases; paid code embeddings, Qdrant
indexing, and activation remain unauthorized.

## Steps 161P-161R - Dependency-noise remediation interview gate

### Step 161P - Approved infrastructure utilities

1. Why must debug and initialization procedures remain dependency edges even
   when they are excluded from business-dependency SME review?
2. Why is an exact versioned utility-call list safer than classifying every
   non-custom package as infrastructure or kernel?
3. What must happen to the policy hash and derived analysis when an approved
   utility name is added or removed?

### Step 161Q - Structural SQL and cursor classification

4. Why is `ALC.TRANSACTIONNUMBER(+)` not a function call despite containing
   parentheses?
5. Why should `FOR rec IN CUR_FINTXN(...)` create a `cursor_reference` instead
   of disappearing from the graph as a false positive?
6. Why was full-file cursor inventory necessary when the large package used
   token-structural segmented parsing?

### Step 161R - Immutable v12 quality gate

7. Why was v11 retained rather than overwritten after its 19-case cursor gap
   was discovered?
8. Why was parse/retrieval reuse safe for v12 while static dependency analysis
   still had to be rebuilt?
9. What does reducing the SME packet from 40 cases/121 occurrences to one
   case/nine occurrences prove, and what embedding/retrieval/answer properties
   remain unproven?

### Evaluation

Overall result: **9/9 accepted.**

1. **Accepted.** Debug and initialization calls remain real static graph edges,
   while the business-review filter prevents infrastructure scaffolding from
   consuming SME attention.
2. **Accepted.** Exact configured identities avoid the unsupported inference
   that every non-custom package is infrastructure or kernel.
3. **Accepted.** The utility set is part of the classification contract, so a
   change requires a new policy hash and a new immutable derived analysis.
4. **Accepted.** `(+)` is Oracle outer-join syntax on a column reference in this
   context, not a routine argument list.
5. **Accepted.** A distinct `cursor_reference` retains meaningful control/data
   flow without falsely representing cursor use as a procedure/function call.
6. **Accepted.** Package-level cursor declarations may sit outside a segmented
   routine, so a source-mapped full-file inventory is required to reconcile
   invocation identity across segment boundaries.
7. **Accepted.** Preserving v11 retains reproducible evidence of the failed
   cursor-quality gate and prevents history from being rewritten.
8. **Accepted.** Source, parser, and retrieval derivation inputs were unchanged,
   while the dependency policy changed; only the affected derived layer could
   be reused safely.
9. **Accepted.** The reduction demonstrates substantially better static
   classification and a smaller human-review burden on this corpus. It does not
   prove embeddings, retrieval relevance, citations, grounded answers, or
   production readiness.

Gate status: **Steps 161P-161R learner gate accepted.** The single canonical
v12 SME case was subsequently accepted and imported in Step 161S. This learner
gate alone did not authorize OpenAI code disclosure or paid embedding.

## Steps 161S-161U - Reviewed code-index contract interview gate

### Step 161S - Hash-bound SME decision ledger

1. Why must the ledger bind both the canonical packet JSON and the reviewed
   Markdown instead of storing only the final verdict?
2. Why does accepting `custom_source_missing` preserve an unknown boundary
   rather than resolve the missing package implementation?
3. Why should an attempted edit or overwrite of the ledger fail instead of
   silently producing a newer review file at the same path?

### Step 161T - Reviewed v12 prepared contract

4. Why must a `reviewed` code-index artifact contain both packet and ledger
   hashes, while a `draft` artifact must contain neither?
5. What do 111 records, point IDs, and cache keys prove separately, and what
   semantic retrieval property do they not prove?
6. Why is `prepared` deliberately different from `embedded`, `indexed`, and
   `active`?

### Step 161U - Paid-operation boundary

7. What information does the dry run provide for disclosure and cost review,
   and why can it not prove provider billing or embedding quality?
8. Why must the real embedding command fail closed when the exact authorization
   token is absent even though SME dependency review is complete?
9. After a paid embedding run is explicitly authorized, which exact-index,
   retrieval, citation, grounded-answer, rollback, and activation gates still
   remain?

### Evaluation

Overall result: **9/9 accepted.**

1. **Accepted.** The answer correctly binds the human decision to both the
   machine-readable canonical case contract and the exact presentation reviewed
   by the SME. A verdict without those inputs would not be reproducible.
2. **Accepted.** `custom_source_missing` remains a qualified absence state. SME
   acceptance validates that classification; it does not provide or infer the
   unavailable implementation.
3. **Accepted.** A no-overwrite ledger preserves the approved decision and its
   audit history. Any later correction must be represented by a separately
   identifiable review artifact rather than an in-place rewrite.
4. **Accepted with precision refinement.** The packet hash identifies the
   reviewed cases and proposed evidence, while the ledger hash identifies the
   resulting human decisions. A draft omits both review bindings because no
   accepted SME decision is being claimed.
5. **Accepted.** Record count checks scope, point IDs preserve distinct source
   occurrences, and cache keys identify reusable semantic inputs. None measures
   relevance, ranking, recall, or answer grounding.
6. **Accepted.** The four states correctly separate local contract preparation,
   external vector creation, isolated vector-store publication, and deliberate
   serving activation.
7. **Accepted with precision refinement.** The current dry run reports the
   exact record/input count and disclosure intent, but it does not calculate a
   reliable currency estimate. Actual tokens, provider billing, vectors,
   indexing, and retrieval behavior remain unproven.
8. **Accepted.** Dependency-label quality and permission to disclose internal
   code to an external paid service are independent approvals.
9. **Accepted.** The answer identifies the remaining generation-isolation,
   exactness, vector-schema, retrieval, citation, grounding, unknown-handling,
   SME, rollback, and activation gates in the correct order.

Gate status: **Steps 161S-161U learner gate accepted.** The reviewed v12
prepared contract may advance only after separate explicit authorization to
send its 111 code embedding inputs to OpenAI and incur the associated cost.
No such authorization is inferred from these interview answers.
