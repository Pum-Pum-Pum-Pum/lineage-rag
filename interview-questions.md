## Steps 134–136 — Multi-scope temporal retrieval and reviewed replay preflight

1. Why must `effective_release_label` remain R24 while R2 evidence is retained
   for a multi-part question?
2. Why should the planner exclude unrelated R1 evidence even when it is highly
   ranked?
3. What is the difference between `referenced_release_labels` and an explicit
   API/request `release_label` filter?

4. Why does the R2/R24/R1 deterministic regression need all three releases?
5. Why does an empty dense or lexical candidate lane remain useful diagnostic
   evidence instead of being silently omitted?
6. Why do the 18 focused tests still not justify v4 activation?

7. Why create a separate reviewed two-case manifest rather than changing the
   review flags of the full 30-case draft benchmark?
8. Why is `draft=False` important before treating the paid replay as a
   release-gate signal?
9. What must the SME inspect in the two new traces beyond structural pass and
   citation release labels?

### Correct-answer rubric

1. R24 remains the effective current state; R2 is retained only for the
   historical sub-question and baseline evidence.
2. High rank does not make unrelated R1 evidence relevant; retaining it can
   introduce cross-module claims and citation confusion.
3. `referenced_release_labels` records releases explicitly named in the query.
   An explicit API/request `release_label` restricts candidate eligibility
   before ranking.
4. R2 and R24 prove required historical/current coverage, while R1 proves
   unrelated release exclusion.
5. An empty lane distinguishes candidate-generation absence from later fusion,
   filtering, packing, or generation failures.
6. Focused tests prove deterministic contracts only, not live retrieval,
   semantic correctness, citation entailment, or operational readiness.
7. A separate manifest preserves the draft benchmark and creates an auditable,
   stable reviewed target.
8. `draft=False` confirms the expectations are reviewed and eligible for use as
   quality evidence rather than exploratory results.
9. SME review must confirm complete source-entailment, correct temporal meaning,
   exact citations, absence of unsupported claims, and useful completeness.

### User answer evaluation

Pass. All nine answers meet the production rubric. Precision correction for
answer 3: `referenced_release_labels` comes from release labels explicitly
named in the query, not from releases present in retrieved evidence. Retrieved
candidates determine the effective latest release; an explicit request filter
remains authoritative.

### Gate

Steps 134–136 are accepted. Await explicit approval for the paid reviewed
two-case v4 replay before making external model calls.

## Steps 137–139 — Paid replay, trace diagnosis, and citation metadata repair

1. Why does one structural pass out of two not justify v4 activation?
2. Why is `estimated_llm_cost=0.0` not evidence that the paid replay was free?
3. What proves this replay evaluated v4 without changing the live v2 service?

4. Why is `lineage-r24-006` not automatically an incorrect answer merely
   because its R2 citation requirement failed?
5. What do the new lane summaries prove about R2 eligibility and where its
   evidence was lost in `lineage-r24-006`?
6. Why can `confusion-release-004` pass its aggregate structural checks while
   still containing a unit-level citation-contract weakness?

7. Why must lexical retrieval preserve `document_id` separately from
   `unit_id`, release, and document family?
8. Why may hybrid fusion fill a blank identity from another lane but not
   overwrite an existing nonblank identity?
9. Which two SME decisions are required before deciding whether to change the
   benchmark, evidence coverage, or run another paid replay?

### Correct-answer rubric

1. Activation requires consistent reviewed regression, grounding, readiness,
   rollback, and SME evidence, not one passing case.
2. Zero reflects unconfigured local prices; provider calls and billing still
   occurred and require provider usage records.
3. The report names paired v4 targets supplied through CLI overrides, while
   `.env` remained v2 during the replay.
4. R24 directly answers the material current-release question; missing R2 is a
   failure only if the historical baseline is material to the accepted case.
5. R2 appeared in dense ranks 6–7 and was lost during fusion/final candidate
   selection before temporal packing, not during ingestion.
6. Aggregate release/document presence can be satisfied by a nearby citation
   even when the direct claim citation lacks identity or entailment.
7. `document_id` identifies the exact FDD occurrence required for citation and
   audit; family/release/unit identifiers serve different scopes.
8. Blank backfill completes metadata; overwriting nonblank identity can corrupt
   provenance by replacing an already asserted source.
9. SME must decide whether R2 is material to the current-release answer and
   whether the R2/R24 distinction is semantically acceptable despite the
   historical citation gap.

### User answer and SME evaluation

Pass. The SME approved `lineage-r24-006`, making R24 the material claim and
removing the R2 citation requirement. The SME accepted
`confusion-release-004` for continued v4 development while recording that its
historical trace needs stronger unit-level citation identity. The lexical and
fusion metadata repair remains required regression coverage.

## Step 140 — Paired v4 local development activation

### Interview status

Step 140 will be included with the next two development steps in the next
nine-question batch. No additional interview is required at this point.

## Steps 140–142 — v4 activation, ingestion isolation, and runtime promotion

1. Why must the Qdrant collection and active lexical directory be switched as
   one coherent pair?
2. Why does making v4 the local development baseline not establish production
   readiness?
3. Why retain the v2 collection after v4 integrity and regression checks pass?

4. Why is it unsafe for ordinary ingestion and active lexical retrieval to
   share the same output directory?
5. What operational boundary does `INGESTION_OUTPUT_DIR` create relative to
   `PROCESSED_DIR`?
6. What failure could occur if a multi-document ingestion wrote directly into
   the active v4 runtime artifacts and stopped halfway?

7. Why was copying verified artifacts to a stable runtime path safer than
   renaming the immutable `table_context_v1_retry1` staging directory?
8. What does matching all 24 filenames, sizes, and SHA-256 hashes prove, and
   what answer-quality property does it not prove?
9. Why should `data/indexes/*` be excluded from Git while its promotion source
   and integrity decision remain documented?

### User answers and mentor evaluation

1. **Satisfactory.** You correctly identified that Qdrant and lexical artifacts
   must represent one coherent evidence generation. Mixing them would weaken
   provenance, citations, and reproducibility.
2. **Satisfactory.** Local v4 activation is a development cutover only. You
   correctly distinguished it from production readiness, which also requires
   operational, security, capacity, rollback, and live-configuration evidence.
3. **Satisfactory.** v2 remains the known-good rollback and comparison
   generation; retaining it preserves continuity while v4 is evaluated.
4. **Satisfactory.** Separating mutable ingestion output from serving artifacts
   prevents partial or unverified data from entering user-facing retrieval.
5. **Satisfactory.** `INGESTION_OUTPUT_DIR` establishes the staging/write side
   of the handoff, while `PROCESSED_DIR` remains the active lexical read side.
6. **Satisfactory.** You correctly described mixed-generation evidence,
   citation breakage, and difficult rollback as consequences of partial
   overwrites.
7. **Satisfactory.** Preserving the immutable retry stage maintains an auditable
   build checkpoint and prevents provenance from being blurred by reuse.
8. **Satisfactory.** SHA-256 proves byte-for-byte promotion integrity only; it
   does not establish semantic correctness, retrieval quality, or readiness.
9. **Satisfactory.** Generated indexes are mutable derived state, so excluding
   them from Git avoids accidental state commits while the promotion manifest
   and decision remain the durable record.

**Batch assessment:** 9/9 satisfactory. The key production distinction is
clear: artifact integrity and local activation are necessary gates, but neither
is sufficient evidence for production readiness.
