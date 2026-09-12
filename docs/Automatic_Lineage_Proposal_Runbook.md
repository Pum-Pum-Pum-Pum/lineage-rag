# Automatic FDD/code lineage proposals

Use this optional discovery step after embedding/index verification and before
creating the candidate lineage definition. It replaces manual corpus searching
with a ranked shortlist, **not** with automatic approval.

## Run locally

From the repository root, run this command. Use a fresh output directory each time.

```powershell
.\.venv\Scripts\python.exe scripts\propose_fdd_code_lineage.py `
  --fdd-stage data\staging\functional_specs_v9 `
  --snapshot-directory data\code_snapshots\fci-custom-r2-ffd9732906d4 `
  --analysis-directory data\staging\code\fci-custom-r2-ffd9732906d4\plsql_antlr_4_13_2_analysis_v15 `
  --code-artifact data\staging\code_embeddings\fci-custom-r2-ffd9732906d4\code_index_text_embedding_3_large_v1\code_index_artifact.json `
  --enhancement-registry data\evaluations\enhancement_fdd_registry_v1.json `
  --output-directory data\exports\lineage_proposals\fci-custom-r2-fdd-v9-v1
```

The first run above has already been generated. Open its `review.html` locally;
do not rerun into that existing directory. For another generation, replace the
FDD stage, immutable snapshot, analysis generation and embedded artifact paths
together, and choose a new output name. Do not supply a prepared-only artifact.

No API key is required. The command reads existing embedding vectors directly
from files, computes cosine similarities locally, and does not open Qdrant,
call OpenAI, create embeddings, modify `.env`, or activate a generation.
The report contains internal evidence excerpts: keep it in approved local storage.

## How matching works

1. Validate selected snapshot, analysis and embedding identities, source hashes,
   exact source offsets, FDD unit text/cache bindings, model and vector dimensions.
2. Extract actual PL/SQL comments, excluding SQL string literals. Pair matching
   Start/End markers, preserving nested enhancement identities. Unpaired or
   crossed markers are diagnostics, not guessed code boundaries.
3. Compare FDD chunk vectors with code-unit vectors; group results by exact
   implementation routine and overload. Declaration/spec companions are retained
   separately. A declaration alone does not prove implementation.
4. Rank exact registered comment-region candidates first, then similarity-only
   candidates. Keep up to five candidates per document by default.
5. Write immutable `proposals.json` and searchable `review.html` with candidate
   status, snippets, source ranges, identities and input SHA-256 bindings.

The registry contains the operator's two explicit comment/FDD correspondences:

| Code marker | FDD identity |
| --- | --- |
| R66 — Neo Day2 | R24_Day2_Neo_AML_FlagRight_FCIS_Integration_v1.1 |
| R24 (REQ07) — NEO Day2 Part2 Enhancement | R24_Day2_Part2_Neo_AML_FlagRight_FCIS_Integration_v1.3 |

The R66/R24 difference is intentional. Never equate documents on release number
or a short title alone. Part2, SCR and Sonar suffixes remain distinct. To add a
confirmed correspondence, create a new registry version with its exact FDD ID,
code release, optional requirement, full title and attributable `basis`. A
registry entry is a discovery hint, not SME approval of every routine it touches.
Other FDDs still receive semantic candidate discovery without registry entries.

## Review the result

Open `review.html`, filter by document or package, then expand a document.
Read its FDD excerpt, code excerpt and comment ranges together. Full exact
selectors, declaration companions, input hashes, unmatched marker inventory and
diagnostics are available in `proposals.json`.

- A cosine score is similarity, **not a probability or confidence percentage**.
- Defaults `--minimum-similarity 0.45` and `--ambiguity-margin 0.03` are exploratory,
  not calibrated release gates. `--top-k` accepts 1–20; omitted candidates are
  explicitly counted. A low threshold can return suggestions for nearly every FDD.
- Similarity-only matches may reflect shared terminology rather than an
  implementation relationship. Do not bulk-accept them.
- Close scores can indicate several valid routines, not necessarily a conflict.
- Missing matches can mean unavailable code, incomplete vectors, no registered
  marker correspondence, or insufficient retrieval quality—not “no implementation”.
- Missing embedding coverage and unmatched markers are reported. Invalid or
  mismatched artifacts fail the run; no fallback paid embedding is attempted.
- Broad code units that cannot be assigned to one implementation are counted as
  unassigned; they are not arbitrarily attached to a routine.

The initial local run covers **126 FDD documents, six source files, 439 code
units and 134 comparable routines**. It produced 627 suggestions, not 627 verified
mappings. Expand the immutable code snapshot before claiming corpus-wide coverage.

## Continue through the existing approval workflow

For selected candidates, use `target.path`, `target.qualified_name`,
`target.symbol_kind`, `target.overload_discriminator_hash` and the FDD document
identity to populate the existing candidate definition. Supply a reviewed
relationship rationale; do not label similarity as proof. An assistant can draft
this definition from the proposal IDs you select, but must not invent acceptance.

Continue with steps 7–11 of [the code generation runbook](Code_Generation_Launcher_Runbook.md):
build the candidate artifact, review the packet, import the review, evaluate the
combined path and pass the separate runtime-promotion controls. These proposal
files are deliberately **not** reviewed lineage artifacts and cannot activate
themselves. Existing approved lineage and historical results remain unchanged.

Before wider use, measure suggestion precision and missed mappings on SME-labelled
positive, negative and similar-title cases. Local deterministic tests validate
the mechanism and safety boundaries, not the semantic correctness of its output.
