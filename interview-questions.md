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

## Steps 153-155 - Interview gate (pending)

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
