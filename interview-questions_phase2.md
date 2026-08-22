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
