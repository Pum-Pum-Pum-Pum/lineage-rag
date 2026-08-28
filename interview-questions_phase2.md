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

## Steps 161V-161X - Code embedding and isolated indexing interview gate

### Step 161V - Authorized code embeddings

1. Why does four successful embedding requests not by itself prove all 111
   source occurrences were preserved correctly?
2. Why can the artifact content identity remain stable from prepared to
   embedded while vector completeness and dimension require separate checks?
3. Why is provider billing the authority for actual cost even though the local
   run records request and input counts?

### Step 161W - Isolated Qdrant generation

4. Why must the code collection use a separate path and namespace from
   `functional_specs_v4`?
5. Why should a second indexing attempt fail instead of upserting the same 111
   points into `code_custom_r1_v1`?
6. Why does successfully indexing `code_custom_r1_v1` not make it active for
   API, UI, code, or combined retrieval?

### Step 161X - Exact verification and remaining gates

7. What distinct failures are detected by point count, deterministic ID,
   payload, and vector-dimension verification?
8. Why was it important to verify the collection again after the deliberately
   rejected duplicate-index attempt?
9. Which Steps 162-170 capabilities and evaluations must pass before this code
   generation can be deliberately activated?

## Steps 161Y-161AA - Expanded five-file code generation interview gate

### Step 161Y - Immutable successor snapshot

1. Why must three additional files create a successor snapshot instead of
   being appended directly to the already indexed r1 snapshot?
2. Why are two unchanged files retained in the complete successor snapshot even
   though only three files were newly supplied?
3. What does the three-added/two-unchanged diff prove, and what parser or
   retrieval property does it not prove?

### Step 161Z - Parser resource behavior

4. Why did lowering the per-attempt timeout to 30 seconds correctly block
   publication rather than provide a faster acceptable generation?
5. Why can a 112 KB PL/SQL package require more parser time than a 1.66 MB
   package, and what does that imply for capacity planning?
6. Why are content-hash reuse and bounded concurrency better scale controls for
   4,000 files than globally increasing every parser timeout?

### Step 161AA - Pre-index and disclosure gates

7. What does zero uncovered declarations and exact source mapping prove, and
   what dependency/retrieval/answer quality remains unproven?
8. Why can the old accepted SME ledger not approve the new 33-case packet even
   when two source files are unchanged?
9. Why does the earlier authorization for 111 embedding inputs not authorize
   embedding this successor generation, and what must happen next?

## Steps 161AB-161AD - Dependency policy v5 and successor preparation interview gate

### Step 161AB - Dependency semantics

1. Why must an unsuffixed unresolved package call remain `routine_call /
   unresolved` instead of being inferred as kernel merely because its source is
   absent?
2. Why is `GET_STRING` better represented as an `object_method_call` than
   deleted as `not_routine_call`?
3. Why should `ipTxnData.Desc_Fields(...)(1)(...)` remain a
   `collection_reference` edge rather than disappear entirely?

### Step 161AC - Immutable v13 and review migration

4. Why could v13 reuse all parse/retrieval artifacts but not reuse v12 static
   dependency artifacts?
5. Why was matching target and verdict insufficient by itself when migrating
   the two accepted dynamic-SQL reviews?
6. What does reducing the SME packet from 33 cases to two prove, and what
   dependency accuracy does it still not prove?

### Step 161AD - Cached successor embedding plan

7. Why do the two unchanged source files need new Qdrant point identities even
   though 111 vectors can be reused?
8. Why does 111 cache hits plus 168 misses require verifying model, input text,
   cache key, and vector consistency rather than matching filenames?
9. What exactly would a new authorization permit, and which indexing,
   retrieval, citation, answer, rollback, and activation gates would remain?

## Steps 161AE-161AG - Cache-aware successor indexing interview gate

### Step 161AE - Authorized cache-aware embeddings

1. Why does `cached_records=111` prove more than simply finding 111 matching
   filenames, and which compatibility fields protect reuse correctness?
2. Why must cached and newly returned vectors still pass one common dimension
   check before the embedded artifact can be published?
3. What do six successful OpenAI requests prove, and what provider-cost or
   semantic-quality property remains unproven?

### Step 161AF - Immutable v2 collection

4. Why is v2 a complete 279-point generation rather than a 168-point delta
   collection containing only newly embedded inputs?
5. Why must a duplicate v2 indexing attempt fail instead of idempotently
   upserting points with the same IDs?
6. Why does creating `code_custom_r1_v2` not change API/UI behavior?

### Step 161AG - Dual-generation verification

7. Why verify v1 again after writing v2 when the collections have different
   names?
8. Why re-verify v2 after the rejected duplicate-index attempt?
9. What retrieval, citation, grounded-answer, impact-analysis, unknown-handling,
   rollback, and activation evidence is still required before v2 can serve
   users?

### Evaluation

Overall result: **9/9 accepted.**

1. **Accepted.** A cache hit proves exact embedding-input/cache identity, not a
   filename match. Source-occurrence identity remains separate.
2. **Accepted.** Cached and newly returned vectors must conform to the same
   collection dimension; reuse identity alone does not prove schema compatibility.
3. **Accepted.** Six successful requests prove only that those requests returned
   usable indexed responses. Coverage, semantic quality, retrieval, citations,
   answers, and authoritative provider cost remain separate checks.
4. **Accepted.** A complete 279-point v2 generation is independently verifiable
   and reversible; a 168-point delta would create a runtime dependency on v1.
5. **Accepted.** Rejecting duplicate construction preserves the immutable meaning
   of a verified collection name and avoids silent partial mutation.
6. **Accepted.** Index creation is not routing or activation, so API/UI behavior
   remains unchanged until a deliberate serving decision.
7. **Accepted.** Re-verifying v1 proves that constructing v2 did not disturb the
   retained rollback generation.
8. **Accepted.** Re-verifying v2 after the rejected write proves the failure path
   caused no partial mutation.
9. **Accepted.** The answer correctly retains retrieval, exact citations,
   grounding, bounded impact guidance, unknown handling, rollback, SME review,
   and deliberate activation as independent gates.

Gate status: **Steps 161AE-161AG learner gate accepted.**

## Steps 162-164 - Code retrieval, citation, and grounded-answer interview gate

### Step 162 - Explicit isolated code retrieval

1. Why must dense results be validated against the exact embedded artifact even
   when Qdrant already filters by `knowledge_lane` and `snapshot_id`?
2. Why does the retrieval service require an explicit mode and an externally
   supplied query vector instead of automatically routing or embedding a query?
3. What does the real lexical `spPNBRPT006` smoke test prove, and what dense or
   semantic retrieval property does it not prove?

### Step 163 - Exact source-line citation contract

4. Why must code citations use `citation_text`, source path, symbol, and exact
   line range rather than the derived `embedding_text`?
5. Why are candidate-lane summaries retained without repeating full source text?
6. Why must an answered response with no citation, or with an unknown citation
   ID such as `[C9]`, fail closed rather than be returned with a warning?

### Step 164 - Grounded explanation and graceful failure

7. Why can an impact-analysis answer identify candidate change locations but
   not claim that any location is a proven root cause?
8. How should `fallback_parse`, `conditional_unknown`, and unavailable kernel
   behavior affect a code answer even when some visible custom evidence exists?
9. What retrieval, paid answer evaluation, SME, API/UI routing, rollback, and
   activation work still remains after these deterministic contracts pass?

Gate status: **deferred by the user before evaluation** to prioritize the three
matching FDD documents and v5 activation. No learner score is claimed for this
gate; its technical coverage is partly revisited in Steps 165-167.

## Steps 164D-164F - FDD v5 ingestion and activation interview gate

### Step 164D - Controlled three-document intake

1. Why did the master intake use `functional_specs_v5_intake` rather than write
   the three documents directly into active v4 or final v5?
2. What do 188 exactly verified source occurrences prove, and what retrieval or
   answer property remains unproven?
3. Why may duplicate semantic text reuse an embedding while each FDD occurrence
   still requires a distinct point and citation identity?

### Step 164E - Complete immutable v5 rebuild

4. Why must v5 contain all 11 archived FDDs rather than only the 188 new units?
5. Why did 458 new embeddings reveal a cache-planning issue even though the
   rebuild completed successfully, and what should be checked before v6?
6. What separate evidence is provided by the source manifest, vector dimension,
   point count, and exact artifact-to-point verification?

### Step 164F - Paired activation with deferred evaluation

7. Why must `QDRANT_COLLECTION_NAME` and `PROCESSED_DIR` be activated together?
8. Why is the R22 document ranking second behind a related R24 document a real
   risk even though all three documents are indexed and searchable?
9. What can be claimed about local v5 activation, and what cannot be claimed
   while semantic retrieval and grounded-answer evaluation remain deferred?

### Evaluation

Overall result: **9/9 accepted.**

1. **Accepted.** Isolation protects both the active and final namespaces from
   partial ingestion and preserves a clean failure boundary.
2. **Accepted.** Exact occurrence verification establishes identity, payload,
   and storage completeness, not semantic retrieval or generated-answer quality.
3. **Accepted.** Vector identity belongs to semantic input; point and citation
   identity belong to each distinct source occurrence.
4. **Accepted.** A complete generation prevents activation from silently
   dropping the eight previously indexed FDDs and remains independently
   reproducible and reversible.
5. **Accepted with precision refinement.** The additional 270 older inputs were
   absent under compatible identities in the configured central seed cache.
   Before v6, inspect all compatible prior-stage cache locations and estimate
   misses from canonical input keys rather than document counts.
6. **Accepted.** The answer correctly separates source-byte identity, vector
   schema, namespace cardinality, and exact occurrence correspondence.
7. **Accepted.** Paired activation exposes one coherent vector/lexical evidence
   contract and prevents mode-dependent generation drift.
8. **Accepted with temporal refinement.** For an explicitly R22-scoped request,
   R24 ranking first is a clear failure. For a current-state request, R24 may be
   correct; the later evaluation must distinguish explicit historical scope from
   current-state release selection.
9. **Accepted with wording refinement.** V5 is already deliberately activated
   locally. What remains unproven is semantic and production-quality readiness,
   including ranking, lineage, citations, grounded answers, operational load,
   and SME acceptance.

Gate status: **Steps 164D-164F learner gate accepted.**

## Steps 165-167 - FDD/code lineage and combined-analysis interview gate

### Step 165 - Reviewed lineage mapping contract

1. Why must a user-supplied FDD/package association remain `candidate` until an
   SME reviews its scope, even when filenames and source comments align?
2. Why does an exact overload selector require the canonical qualified name,
   symbol kind, and overload discriminator hash together?
3. What risk remains when an SME accepts a broad file-level mapping instead of
   selecting individual symbols?

### Step 166 - Independent combined retrieval

4. Why must FDD and code retrieval keep separate scores, thresholds, evidence,
   and candidate traces rather than fuse both lanes into one RRF ranking?
5. Why should direct code retrieval still run when no reviewed lineage mapping
   exists, and how must the result be qualified?
6. What does the local Neo AML combined smoke test prove when it finds both FDD
   and AML package evidence but refuses to claim a reviewed link?

### Step 167 - Sectioned combined answer contract

7. Why may the documented-functionality section cite only `[F#]` evidence while
   implementation and impact sections require `[C#]` evidence?
8. Why should one invalid section fail independently rather than force every
   otherwise grounded section into the same answer/refusal state?
9. What SME mapping review, retrieval evaluation, paid answer evaluation, and
   activation work remains before combined mode can serve users?

### Evaluation

Overall result: **9/9 accepted.**

1. **Accepted.** A supplied association or similar name is candidate evidence,
   not proof of implementation lineage. Only explicit SME approval may promote
   it to reviewed authority.
2. **Accepted.** Qualified ownership, symbol kind, and overload discriminator
   jointly identify the intended callable. Return type alone cannot distinguish
   a PL/SQL overload.
3. **Accepted.** File-level review is deliberately broad and can over-attribute
   unrelated routines. Combined and impact answers must preserve that scope
   instead of describing every contained symbol as an exact implementation.
4. **Accepted.** The lanes have different evidence meanings, score
   distributions, and thresholds. RRF is appropriate within a lane, but a
   cross-lane ranking would blur documented requirements and visible code.
5. **Accepted.** Direct retrieval can surface useful candidate implementation
   evidence, but without a reviewed link it cannot establish that the code
   implements the retrieved FDD.
6. **Accepted.** The smoke test proves one local mechanism path and expected
   evidence reachability. It does not establish corpus-wide retrieval, mapping,
   citation, answer, or operational quality.
7. **Accepted.** Lane-specific citation prefixes preserve claim authority:
   `[F#]` supports documented behavior and `[C#]` supports visible custom-code
   and impact claims.
8. **Accepted.** Independent section failure preserves valid partial knowledge
   while preventing one unsupported section from contaminating another.
9. **Accepted.** The answer correctly retains mapping review, code/combined
   evaluation, citation validation, separately authorized paid generation, SME
   answer review, rollback, and deliberate activation as independent gates.

Gate status: **Steps 165-167 learner gate accepted.** Steps 168-170 remain
blocked on the three-item FDD-to-code SME mapping review; candidate mappings
cannot be silently promoted by the implementation.

### Mapping-review resolution

The SME subsequently marked all three mappings `reviewed` with rationale
`Correct mapping`. The decisions were imported into a separate hash-bound
artifact rather than modifying the candidate generation:

```text
review packet SHA-256: 47c7a8f0bb18167a4810be873f8d37a75cdeda643baac21fd2e0cb4a44ec20e1
reviewed artifact identity: 85f1623e298b73858abbd68596c66aab36c4409739eb48d9ab07f29998f9d738
```

The mapping-review prerequisite is now satisfied. Steps 168-170 may begin with
draft code-only and combined evaluation manifests; paid evaluation remains a
separate explicit authorization boundary.

## Steps 168-170 - Code and combined evaluation gate

### Step 168 - Evaluation manifest contracts

1. Why should normal evaluation questions omit release labels while retaining
   exact FDD document IDs as hidden expected metadata?
2. Why must a code-only case be forbidden from declaring FDD evidence or a
   reviewed FDD-to-code mapping?
3. What must the SME validate before the ten draft cases can become a reviewed
   release-gate manifest, beyond confirming that the wording looks reasonable?

### Step 169 - Deterministic retrieval gate

4. Why does retrieving the expected code file not prove that the evidence is
   sufficient when the expected routine is absent from final bounded evidence?
5. Why must the merged direct and mapping-bounded code evidence be capped again
   after both searches, even though each individual search already has a limit?
6. What does the `6/8` positive result prove, and why must the two abstention
   cases remain diagnostics rather than improve the positive pass rate?

### Step 170 - Paid answer-evaluation boundary

7. Why should paid answer generation be blocked when the deterministic
   retrieval threshold fails instead of letting the LLM try to recover?
8. Why must the future authorization explicitly cover both evaluation questions
   and retrieved internal FDD/PLSQL excerpts, rather than mentioning API cost
   alone?
9. After retrieval passes and the paid answers are generated, what evidence and
   reviews are still required before combined mode can be deliberately activated?

Gate status: **awaiting learner answers and SME review of the draft manifests.**
No paid OpenAI call was made in Steps 168-170.

### Evaluation

Overall result: **9/9 accepted.**

1. **Accepted.** Natural business wording tests retrieval and temporal reasoning
   without leaking release metadata. Hidden document IDs retain an objective,
   auditable expectation.
2. **Accepted.** Code-only mode must not borrow FDD authority. Keeping evidence
   lanes separate prevents documentation that was never retrieved from being
   used to validate a code claim.
3. **Accepted.** The answer correctly includes question realism, exact source and
   symbol expectations, mapping/overload scope, evidence sufficiency, and
   answered-versus-abstention behavior. SME approval converts developer
   assumptions into a reviewed benchmark contract.
4. **Accepted.** Package-level relevance is not symbol-level support. A claim
   about one routine requires that routine or a provenance-preserving bounded
   child in final evidence.
5. **Accepted.** Independent limits do not bound their union. Final merge
   deduplication and capping are necessary for deterministic prompt size, cost,
   and evidence balance.
6. **Accepted.** The answer correctly separates positive retrieval recall from
   safe-abstention behavior. Mixing them into one numerator would conceal the
   two positive retrieval failures.
7. **Accepted.** Generation cannot reliably reconstruct missing evidence. The
   deterministic evidence gate should fail before cost and model variability are
   introduced.
8. **Accepted.** Cost authority and disclosure authority are separate controls.
   Approval must cover the exact internal questions and FDD/PLSQL excerpts sent
   to the external service.
9. **Accepted.** Citation provenance, lane-specific entailment, unknown/refusal
   behavior, paid evaluation, SME answer review, rollback, and deliberate
   activation remain independent gates.

Gate status: **Steps 168-170 learner gate accepted.** The draft manifest still
requires SME review, and the two combined exact-symbol retrieval gaps still
block paid answer evaluation.

### SME-review resolution

The SME accepted all ten cases. Separate reviewed manifests and a hash-bound
ledger were published without modifying the drafts:

```text
review packet SHA-256: 7f06d2d275989e218d339f796ea3eb99511b2688fa7f33374ca2548219b1d913
review ledger identity: 76a09f9781ba0a34cf7febf19c3845b1ecb4632a71c2065ece5883f813a39592
```

The reviewed deterministic replay remained `6/8` positive (`0.75`). Benchmark
review is complete, but the two combined exact-symbol retrieval failures still
block paid answer evaluation.

### Source-of-truth clarification

Code is authoritative for what the **available custom-code snapshot visibly
implements**. The FDD is authoritative for **documented business intent and
functional requirements**. Neither is a universal source of truth:

- visible PL/SQL cannot prove hidden Java/kernel behavior, external services,
  live configuration, runtime data, privileges, conditional branches, or
  unresolved dynamic SQL;
- an FDD can omit implementation details or differ from deployed code;
- an archived code snapshot can differ from the version actually deployed.

Combined answers must therefore report documented functionality and visible
custom implementation separately, identify agreement or conflict, and label
runtime/deployment facts as unknown unless later confirmed by approved runtime
metadata or deployment evidence.

## Steps 171-173 - Parent-diverse code retrieval gate

### Step 171 - Candidate-lane diagnosis

1. What does finding the expected symbols at lexical ranks 12 and 13 prove, and
   which ingestion, mapping, retrieval, and generation failure causes does it
   rule out?
2. Why must candidate traces retain parent identity, symbol, path, rank, and line
   range while excluding full source text?
3. Why would increasing the final evidence limit be a weaker first correction
   than identifying repeated children from the same parent?

### Step 172 - Parent-first diversity

4. Why did a greedy maximum of two children per parent still fail even though it
   technically enforced the configured cap?
5. How does parent-first round-robin differ from deduplicating every symbol to a
   single chunk, and what trade-off does `max_units_per_parent=2` preserve?
6. Why is this correction safer than changing global lexical weight, embeddings,
   or the reviewed FDD-to-code mappings?

### Step 173 - Reviewed replay and paid boundary

7. What does the reviewed `8/8` deterministic result prove, and what semantic or
   operational properties remain unproven?
8. Why do the two kernel abstention cases remain outside the positive retrieval
   pass-rate numerator even after the release gate becomes eligible?
9. What exact disclosure, cost, trace, citation, SME, rollback, and activation
   evidence is still required before combined mode can serve users?

Gate status: **awaiting learner answers.** Deterministic retrieval is eligible
for the next gate; the paid OpenAI operation remains unexecuted and requires
explicit authorization.

### Evaluation

Overall result: **9/9 accepted.**

1. **Accepted with scope refinement.** Candidate ranks 12 and 13 prove the
   symbols exist in the approved artifact and were reached by lexical candidate
   retrieval. Their presence in the mapping-bounded trace also proves they were
   eligible under the reviewed broad mapping. It does not prove that the mapping
   is semantically precise, that ranking is acceptable, or that the symbols
   survive evidence packing.
2. **Accepted.** Compact identity/rank metadata makes the loss auditable while
   avoiding source duplication. Original source units remain the only citation
   authority.
3. **Accepted.** Raising the limit increases prompt size, latency, and cost while
   preserving the parent-dominance mechanism. Diversity corrects coverage at the
   existing budget.
4. **Accepted.** The greedy cap limited total children but still admitted a
   second child from an early parent before the first child of a later parent.
   Ordering, not merely the cap value, caused the remaining loss.
5. **Accepted with terminology refinement.** The important comparison is one
   chunk per **parent routine**, not one per symbol name. A strict one-parent-one-
   chunk rule improves breadth but can omit necessary later lines of a large
   routine. Parent-first round-robin gives every parent initial representation,
   then permits a second child when budget remains.
6. **Accepted.** The correction changes deterministic selection only; it leaves
   source bytes, vectors, points, scores, RRF weights, and reviewed lineage
   authority unchanged.
7. **Accepted.** `8/8` proves the reviewed deterministic positive expectations
   under this corpus, configuration, and budget. It does not prove semantic
   entailment, generated-answer quality, broader generalization, live-model
   stability, or production capacity.
8. **Accepted.** Safe refusal and positive retrieval measure different system
   properties and must remain separately reported.
9. **Accepted with completeness refinement.** In addition to authorization,
   provenance, lane-specific citations, safe refusal, SME review, rollback, and
   deliberate activation, the paid run must retain exact prompts/evidence,
   provider request and usage identifiers, generated answer contracts, failure
   reasons, and immutable report bindings. Production operational properties
   remain a later gate.

Gate status: **Steps 171-173 learner gate accepted.** The next action is the
separately authorized paid code/combined answer evaluation; no authorization is
inferred from these interview answers.

## Steps 174-176 - Paid code/combined grounded-answer evaluation

### Step 174 - Paid-call and evidence boundary

1. Why should one query embedding be reused across the FDD and code retrieval
   lanes instead of embedding the same question independently for each lane?
2. Why must the paid runner preserve exact prompts/evidence, provider request
   IDs, token usage, and finalized answer contracts without storing API secrets?
3. Why does disabling automatic retries matter for cost control and reproducible
   evaluation, and what operational trade-off does it create?

### Step 175 - Fail-closed execution and physical-store readiness

4. Why did verifying `functional_specs_v5` not prove that
   `code_custom_r1_v2` was available, even though both are Qdrant collections?
5. Why must collection existence, generation identity, vector dimension, and
   artifact-to-point verification happen before the first paid query embedding?
6. Why is preserving a partial failed run and explicitly resuming safer than
   restarting all ten cases from the beginning?

### Step 176 - Structural results and SME gate

7. Why does missing the expected `spOfflineParallelUserEnd` citation cause a
   structural failure even when the answer cites several plausible alternative
   offline-processing routines?
8. For the hidden-kernel negative case, when is it acceptable to refuse the exact
   request but still provide clearly separated, cited visible-code context, and
   why must the global answer state still indicate that the requested fact was
   unsupported?
9. What do `8/10` structural passes and `10/10` completed paid answers prove, and
   what must the SME verify before activation can be considered?

Gate status: **awaiting learner answers and SME review of the ten paid answers.**
No FDD/code generation or combined mode was activated.

### Evaluation

Overall learner result: **9/9 accepted.**

1. **Accepted.** One semantic query vector can be reused across independently
   searched evidence stores when the query text and embedding contract are the
   same. Lane-specific scores, filters, thresholds, and citations remain separate.
2. **Accepted.** Audit evidence must bind inputs, outputs, usage, and provider
   identifiers without persisting credentials. Secrets are operational authority,
   not evaluation evidence.
3. **Accepted.** Disabling implicit retries bounds cost and disclosure attempts.
   The trade-off is explicit recovery work and the need for durable resumable state.
4. **Accepted.** Collection readiness is scoped to one physical store and
   generation; FDD readiness cannot establish code-lane readiness.
5. **Accepted with one addition.** All listed gates are required. The paid-run
   preflight must also verify the exact collection names, vector dimensions,
   artifact-to-point identities, and that no process currently holds an
   incompatible local-Qdrant lock.
6. **Accepted.** Resume avoids duplicate cost and preserves the original failed
   state. Resume inputs must be hash-bound to the same manifests, plan, and
   completed case traces.
7. **Accepted.** A benchmark that names an exact symbol measures symbol-level
   citation coverage, not merely plausible package-level relevance.
8. **Accepted with a contract refinement.** Helpful visible-code context may be
   returned in separately labelled sections, but the response also needs a global
   machine-readable state saying the user's exact hidden-kernel claim was not
   supported. Per-section states alone are insufficient for this mixed response.
9. **Accepted.** Completion proves execution coverage; structural passes prove
   only the encoded contract. SME entailment, completeness, qualification,
   benchmark validity, rollback, and deliberate activation remain separate gates.

Gate status: **Steps 174-176 learner gate accepted.** The submitted SME packet
still requires normalization against immutable machine results before its verdicts
can be imported into a release-gate ledger.

## Steps 178-180 - Semantic ledger, corrected contract, and activation readiness

### Step 178 - Separate machine and SME decisions

1. Why must an SME semantic acceptance be stored separately from the immutable
   structural result instead of changing the structural `fail` to `pass`?
2. What do the run-state hash, packet hash, and per-trace hashes each protect in
   the answer-review ledger?
3. Why must a chat-confirmed verdict override record its source rather than
   silently replacing the verdict written in the Markdown packet?

### Step 179 - Correct future evaluation semantics

4. When should an expected code symbol be `all`, `any`, or `advisory`, and why
   is `advisory` appropriate for this open-ended impact-analysis case?
5. Why are `requested_claim_supported` and
   `related_grounded_context_provided` independent states?
6. Why is changing the v2 evaluation contract safer than modifying retrieval
   weights or regenerating the already reviewed paid answer?

### Step 180 - Activation readiness

7. Why do retained FDD/code current and rollback collections still not make
   combined mode activation-ready?
8. Which API, configuration, readiness, orchestration, and rollback controls are
   missing before code and combined modes can serve users?
9. What is the difference between an evaluated capability, an activation-ready
   capability, and an active production capability?

Gate status: **awaiting learner answers.** Semantic review is accepted, but code
and combined runtime activation remains blocked at `4/9` readiness checks.

### Evaluation

Overall result: **8/9 accepted; question 8 requires revision.**

1. **Accepted.** Structural execution evidence and SME semantic judgment answer
   different questions and must remain independently reproducible.
2. **Accepted.** Run-state, packet, and per-trace hashes protect execution scope,
   review presentation, and individual evidence/answer identity respectively.
3. **Accepted.** An override is a separate human decision event and must retain
   its source rather than masquerading as the original packet verdict.
4. **Accepted.** `all`, `any`, and `advisory` correctly distinguish mandatory
   conjunction, acceptable alternatives, and non-gating diagnostics.
5. **Accepted.** An unsupported requested claim may coexist with useful, clearly
   separated grounded context; one state cannot represent both facts safely.
6. **Accepted.** Repairing a reviewed benchmark defect preserves retrieval
   integrity; tuning the system to satisfy a wrong expectation is benchmark gaming.
7. **Accepted.** Collection retention proves availability, not correct routing,
   readiness, rollback execution, or user-serving behavior.
8. **Incomplete.** The answer mentions API code/combined retrieval but omits the
   required explicit mode request/response contracts, active code artifact and
   store configuration, mode-aware readiness validation, shared query embedding
   plus lane-specific orchestration, trace/citation contract, and rehearsed atomic
   rollback routing.
9. **Accepted.** Evaluated, activation-ready, and active are correctly separated
   as evidence, operational eligibility, and actual serving state.

Gate status: **not accepted until question 8 is completed.** Runtime integration
work remains intentionally paused.

### Question 8 correction

**Accepted.** The revised answer covers readiness, approved configuration and
policy, generation identity, explicit routing, orchestration, health/provenance,
and tested rollback. Implementation refinement: those categories must become
concrete controls—an explicit API knowledge mode and response schema, configured
artifact/store/lineage identities, mode-aware readiness checks, one shared query
embedding with lane-specific retrieval, persisted citations/traces, and an atomic
feature-gate rollback.

Gate status: **Steps 178-180 learner gate accepted, 9/9.**

## Steps 181-183 - Feature-gated runtime modes and activation readiness

### Step 181 - Explicit API and configuration contracts

1. Why should `knowledge_mode` be separate from dense/lexical/hybrid
   `retrieval_mode`, and what breaks if one field represents both concepts?
2. Why must code collection, artifact, analysis, snapshot/lineage, and FDD
   generation identities be configuration rather than inferred from directory
   contents at request time?
3. Why is `CODE_MODES_ENABLED=false` the correct default even after the offline
   evaluation and SME gate passed?

### Step 182 - Runtime code/combined orchestration

4. Why must combined mode embed the query once but keep FDD and code ranking,
   evidence thresholds, and citations separate?
5. Why must a combined response expose both a global requested-claim state and
   independent section states?
6. What privacy, storage, and debugging trade-offs arise from persisting exact
   code/FDD evidence and model prompts in local answer traces?

### Step 183 - Mode readiness and rollback

7. Why should ordinary readiness use bounded checks while exact artifact-to-point
   verification remains an activation/preflight gate rather than retrieving every
   vector on every health probe?
8. What does the deterministic false → true → false feature-flag test prove, and
   what live-process or multi-worker behavior does it not prove?
9. What exact actions and evidence are required at the separate deliberate
   activation boundary now that the readiness assessment is 9/9?

Gate status: **awaiting learner answers and explicit activation decision.** Code
and combined API paths exist but remain disabled because `.env` was not changed.

### Evaluation

Overall result: **9/9 accepted.**

1. **Accepted.** Knowledge authority and retrieval algorithm are orthogonal
   controls and must remain independently configurable and observable.
2. **Accepted.** Explicit identities prevent cross-generation evidence mixing and
   make traces, readiness, rollback, and incident diagnosis reproducible.
3. **Accepted.** Evaluation establishes capability evidence; activation is a
   separate operational authority and risk decision.
4. **Accepted.** One semantic query representation controls duplicate cost while
   independent ranking/threshold/citation contracts preserve lane authority.
5. **Accepted.** Global requested-claim support prevents a strong subsection from
   falsely turning a partially supported response into an overall answer.
6. **Accepted with one addition.** Access control, retention, and encryption are
   necessary; trace minimization, deletion procedures, audit access logging, and
   incident handling are also required because exact evidence may contain
   sensitive internal implementation details.
7. **Accepted.** Runtime probes must be cheap and bounded; exhaustive identity
   verification belongs to controlled preflight and integrity jobs.
8. **Accepted.** The deterministic flag test proves routing reversibility in the
   tested process contract, not multi-worker propagation, in-flight request
   behavior, semantic quality, or disaster recovery.
9. **Accepted with activation-boundary refinement.** In addition to the listed
   evidence, activation needs explicit configuration diff review, process restart,
   effective-config verification, mode-specific readiness, authorized live smoke
   calls, trace inspection, and an immediate rollback trigger/owner.

Gate status: **Steps 181-183 learner gate accepted, 9/9.** The capability is
activation-ready but inactive. Activation and paid/internal-evidence smoke testing
remain separate explicit approval boundaries.

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
