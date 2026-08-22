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
