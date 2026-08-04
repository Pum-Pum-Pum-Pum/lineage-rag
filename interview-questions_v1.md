# Interview Questions

## Stage 1 — Project setup and architecture foundation

### Question 1
Why should a RAG system separate indexing flow from query flow?

**Strong answer:**
Because indexing and query workloads have different responsibilities, runtime patterns, failure modes, and scaling concerns. Indexing is batch-oriented and handles ingestion, cleaning, chunking, embedding, and vector upsert. Query flow is latency-sensitive and focuses on retrieval, evidence assembly, and grounded response generation. Separating them improves maintainability, testing, observability, and production reliability.

### Question 2
Why use centralized typed settings instead of reading environment variables directly throughout the codebase?

**Strong answer:**
Centralized typed settings reduce duplication, prevent inconsistent defaults, improve validation, and make the system easier to test. It also helps avoid hidden configuration bugs and supports cleaner dependency boundaries as the project grows.

### Question 3
Why define artifact and index versioning in Stage 1 before ingestion is implemented?

**Strong answer:**
Versioning must be defined early because processed outputs, embeddings, and indexes need reproducibility from the start. Without clear versioning, debugging retrieval issues later becomes difficult because you cannot confidently trace which artifact set or index produced a result.

### Question 4
What is the risk of skipping centralized logging in an LLM/RAG system?

**Strong answer:**
Without centralized logging, later failures in ingestion, retrieval, or generation become difficult to trace. RAG systems are multi-stage pipelines, so debugging requires consistent logs for settings, paths, retrieval behavior, latency, and failures across components.

### Question 5
Why is modular structure especially important in production-minded GenAI systems?

**Strong answer:**
Because GenAI systems combine multiple independently evolving concerns such as ingestion, chunking, retrieval, prompting, vector storage, APIs, and UI. A modular structure keeps responsibilities isolated, makes components swappable, simplifies testing, and supports future evolution without destabilizing the whole system.

---

## Stage 1 — Evaluation of user answers

### User answer review 1 — Centralized BaseSettings
**What was correct:**
- You recognized that a single master configuration point makes future changes easier.

**What was weak or incomplete:**
- "Best practice" alone is too vague.
- You did not explain validation, consistency, testability, or hidden configuration bugs.
- "the model will update" is the wrong framing. This is not about the model updating; it is about the application reading consistent settings.

**Stronger interview-quality answer:**
Centralized `BaseSettings` is better because it gives the project one typed, validated source of truth for configuration. That reduces duplicated environment reads, prevents inconsistent defaults across ingestion, retrieval, and API modules, and makes testing easier because settings can be controlled in one place. In a production RAG system, this avoids subtle bugs such as one module reading the wrong embedding model, another using the wrong data path, or different services silently using inconsistent configuration.

### User answer review 2 — Artifact and index versioning
**What was correct:**
- You understood that versioning helps track future changes.

**What was weak or incomplete:**
- You were too generic.
- You did not explain reproducibility.
- You did not explain the actual debugging failure mode: not knowing which processed artifacts or index generated a bad answer.

**Stronger interview-quality answer:**
We introduced `artifact_version` and `index_version` early so that reproducibility is built into the system from the start. Later, if retrieval quality changes or a generated answer becomes worse, we need to know exactly which processed corpus, chunking logic, or vector index version produced that result. Without versioning, debugging becomes guesswork because we cannot confidently tell whether a failure came from stale artifacts, a partial re-index, or a pipeline change.

### User answer review 3 — sys.path fix and packaging
**What was correct:**
- You noticed the script was checking paths.

**What was weak or wrong:**
- Your answer did not explain why the import problem happened.
- The `sys.path` fix is not about creating files if they do not exist.
- You did not answer the production-grade packaging part.

**Stronger interview-quality answer:**
The `sys.path` fix was needed because when the script was executed directly from the `scripts` folder, Python did not automatically treat the project root as an import base for the `app` package. So `from app.core.config import get_settings` failed. The quick fix was to add the project root to `sys.path`. A more production-grade approach later would be to package the project properly, for example with a `pyproject.toml` and an installable package layout, so imports work consistently without manual path injection.

### User answer review 4 — Why .gitignore matters more in RAG
**What was correct:**
- Correctly identified secret leakage risk.

**What was weak or incomplete:**
- Too narrow.
- In RAG projects, the bigger issue is not just API keys, but also local artifacts, vector indexes, processed text, logs, and cached embeddings.

**Stronger interview-quality answer:**
`.gitignore` is especially important in a RAG project because the system generates many local artifacts beyond source code, such as processed documents, chunk outputs, embedding caches, vector store files, logs, and evaluation outputs. If those are committed accidentally, the repository becomes noisy, reproducibility suffers, and sensitive internal document content may leak. So `.gitignore` is not only about API keys; it is also about keeping local pipeline artifacts, caches, and vector indexes out of source control.

### User answer review 5 — Delaying centralized logging
**What was correct:**
- You recognized debugging would become harder.

**What was weak or incomplete:**
- Too vague.
- You did not explain specific failure cases in ingestion and retrieval.
- You did not explain how missing log consistency harms root-cause analysis.

**Stronger interview-quality answer:**
If centralized logging was delayed until Stage 5 or 6, ingestion and retrieval debugging would become much harder because different modules would likely use inconsistent prints or ad hoc logs. That would make it difficult to trace which files failed parsing, which chunking logic was used, which settings were active, what retrieval candidates were returned, and why an answer was unsupported or incorrect. In a multi-stage RAG pipeline, missing consistent logging slows root-cause analysis and makes failures difficult to reproduce reliably.

---

## Stage 1.1 — Environment contract alignment

### Question 1
Why should `.env.example` be kept aligned with the `Settings` class instead of being updated later only when a runtime error appears?

**Strong answer:**
`.env.example` should stay aligned with the `Settings` class because it defines the operational contract for developers and future deployments. If it drifts from the code, onboarding becomes error-prone, runtime failures happen later than necessary, and different developers may run the system with inconsistent assumptions. Keeping it aligned early reduces setup friction and makes configuration expectations explicit.

### Question 2
What production risk appears if the code supports a config variable but `.env.example` does not document it?

**Strong answer:**
An undocumented variable creates hidden configuration dependencies. The system may work only on one developer's machine, fail in CI/CD or deployment, or behave inconsistently across environments because required settings were never clearly documented.

### Question 3
Why is an empty placeholder value in `.env.example` better than committing a real API key?

**Strong answer:**
Placeholder values document the required configuration surface without exposing secrets. Real credentials should never be committed because they can leak through source control history and require immediate rotation if exposed.

### Question 4
Why do version fields like `ARTIFACT_VERSION` and `INDEX_VERSION` belong in `.env.example` even if they currently have simple defaults in code?

**Strong answer:**
They belong in `.env.example` because versioning is part of the runtime contract, not just an internal code default. Exposing them in the environment makes it easier to change versions intentionally across local runs, experiments, and deployments without editing source code.

---

## Stage 2 — Filename parsing baseline

### Question 1
Why is filename parsing an important first ingestion step in this project instead of starting directly with DOCX text extraction?

**Strong answer:**
Filename parsing is an important first ingestion step because the filename already carries release-lineage metadata that is critical to correctness in this domain. Before extracting text, we need to reliably identify document family and release number so later processing can group releases correctly and support cumulative retrieval behavior.

### Question 2
What production risk appears if filename parsing is inconsistent or too loosely defined?

**Strong answer:**
If filename parsing is inconsistent, documents may be grouped into the wrong family, releases may be misordered, and retrieval may mix unrelated evidence or miss valid historical context. That can directly reduce answer correctness and trustworthiness.

### Question 3
Why is it useful to return a structured object like `ParsedDocumentName` instead of a loose dictionary?

**Strong answer:**
Returning a structured object makes the ingestion contract explicit, easier to test, and less error-prone than passing around loosely defined dictionaries. It improves readability, reduces key-name mistakes, and supports future extension of metadata fields.

### Question 4
Why is writing a test for an invalid filename just as important as testing a valid filename?

**Strong answer:**
Invalid-input tests are important because production pipelines fail at edge cases, not just happy paths. Testing invalid filenames ensures the parser fails explicitly and predictably instead of silently producing wrong metadata that would contaminate downstream indexing and retrieval.

### Question 5
If later you discover multiple filename conventions in the corpus, what is the right engineering response?

**Strong answer:**
The right response is to make the parsing logic explicit and versioned, not to silently broaden the regex in an uncontrolled way. We should inspect the corpus, define supported patterns intentionally, add tests for each pattern, and log unsupported cases so ingestion quality remains measurable.

---

## Stage 2 — Evaluation of user answers

### User answer review 1 — Why filename parsing comes first
**What was correct:**
- You recognized that filename conventions are important for lineage tracking.
- You correctly connected release numbers to old vs newer changes.

**What was weak or incomplete:**
- The answer was still too informal and repetitive.
- You did not explicitly explain why this should happen before text extraction.
- You did not mention grouping by document family.

**Stronger interview-quality answer:**
Filename parsing is an important first ingestion step because the filename already contains critical release-lineage metadata such as document family and release number. We need that metadata before full text extraction so that later ingestion outputs can be grouped correctly across releases and support cumulative retrieval behavior. If we delay that logic, downstream processing may lose an early and reliable domain signal that is essential for correctness.

### User answer review 2 — Risk of inconsistent parsing
**What was correct:**
- You correctly identified that release ordering would break.
- You correctly connected it to the lineage model.

**What was weak or incomplete:**
- You did not mention wrong grouping of document families.
- You did not explain the downstream production consequence: retrieval mixing unrelated evidence.

**Stronger interview-quality answer:**
If filename parsing is inconsistent or too loosely defined, the system may misgroup documents into the wrong family, misread release ordering, or fail to recognize related variants under the same release. That would break release-lineage reasoning and could cause retrieval to mix unrelated evidence or miss valid historical context, which directly harms answer correctness and trust.

### User answer review 3 — Structured object vs dictionary
**What was correct:**
- You understood that a structured object organizes fields better.
- You recognized the benefit of separating family and release information clearly.

**What was weak or incomplete:**
- "well mannered way" is not interview-quality language.
- You did not mention type safety, explicit contracts, or easier testing.

**Stronger interview-quality answer:**
Returning a structured object like `ParsedDocumentName` is better than using a loose dictionary because it makes the ingestion contract explicit, typed, and easier to test. It reduces the chance of inconsistent key names, makes downstream code more readable, and supports future metadata extension without relying on fragile dictionary conventions.

### User answer review 4 — Why invalid tests matter
**What was correct:**
- You understood that invalid-case testing is useful for future automated or sanity testing.

**What was weak or incomplete:**
- The answer was too vague.
- You did not explain that invalid tests protect against silent metadata corruption.

**Stronger interview-quality answer:**
Testing invalid filenames is just as important as testing valid ones because production pipelines often fail at edge cases, not happy paths. If invalid patterns are not tested, the parser may silently produce incorrect metadata, which would contaminate downstream indexing and retrieval. Explicit failure behavior makes ingestion safer and easier to debug.

### User answer review 5 — Multiple filename conventions
**What was correct:**
- You understood that parsing logic may need to expand when real corpus patterns differ.

**What was weak or incomplete:**
- Too vague.
- You did not mention inspecting the corpus first, intentionally defining supported patterns, adding tests, or logging unsupported cases.
- "fit then in those areas" is unclear and not interview-quality.

**Stronger interview-quality answer:**
If we discover multiple filename conventions in the corpus, the correct engineering response is to inspect the corpus carefully, define supported patterns explicitly, and update the parser in a controlled way with tests for each pattern. We should not silently broaden the regex without evidence because that can introduce ambiguous parsing. Unsupported patterns should be logged so ingestion quality remains measurable.

---

## Stage 2.1 — Safe DOCX discovery

### Question 1
Why is filtering Office temporary lock files an important ingestion step instead of a minor cleanup detail?

**Strong answer:**
Filtering Office temporary lock files is important because they are not real source documents and can trigger avoidable ingestion failures. In production pipelines, small input-noise issues can create misleading errors, wasted debugging time, and unstable batch runs, so they should be handled explicitly and early.

### Question 2
Why should the loader fail explicitly for a missing input directory instead of silently returning an empty list?

**Strong answer:**
Failing explicitly for a missing input directory is better because an empty list could hide a configuration or deployment mistake. Explicit failure makes the problem visible immediately and prevents silent ingestion runs that appear successful while processing nothing.

### Question 3
Why return a structured `DiscoveredDocxFile` object instead of plain file paths only?

**Strong answer:**
Returning a structured object keeps the ingestion contract extensible and explicit. Even if we currently store only path, filename, and temporary-file status, this design makes it easier to attach future discovery metadata without rewriting downstream code.

### Question 4
What downstream problem could happen if temporary files are not filtered before filename parsing and DOCX extraction?

**Strong answer:**
If temporary files are not filtered, the parser may fail on malformed names or the extractor may try to read incomplete lock files that are not valid source documents. That adds noisy failures and makes ingestion debugging less reliable.

### Question 5
In a real enterprise folder, what other file discovery issues would you expect beyond temporary Office files?

**Strong answer:**
I would expect mixed file types, duplicated exports, partially downloaded files, renamed legacy documents, unsupported formats, nested directories, and hidden system files. A production-minded loader should define what is accepted, reject unsupported inputs explicitly, and log discovery statistics so input quality is observable.

---

## Stage 2.1 — Evaluation of user answers

### User answer review 1 — Why filtering lock files matters
**What was correct:**
- You recognized that temporary lock files appear when a document is open.
- You correctly understood that they should not be ingested as real documents.

**What was weak or incomplete:**
- Too short and too informal.
- You did not explain the production consequence: noisy failures and unstable ingestion runs.

**Stronger interview-quality answer:**
Filtering Office temporary lock files is important because those files are not real source documents. They are transient artifacts created while documents are open, and if we try to ingest them, we can trigger avoidable parsing or extraction failures. In a production ingestion pipeline, small input-noise issues like this can waste debugging time and make batch runs unreliable.

### User answer review 2 — Missing input directory
**What was correct:**
- You understood that if the input directory is missing, ingestion cannot proceed correctly.
- You correctly saw this as a stopping condition.

**What was weak or incomplete:**
- "unresourceful" is not clear interview language.
- You did not explain why silently returning an empty list is dangerous.

**Stronger interview-quality answer:**
The loader should fail explicitly for a missing input directory because silently returning an empty list could hide a configuration or deployment mistake. That would make the pipeline appear to run successfully while actually processing no documents. Explicit failure makes the problem visible immediately and prevents false confidence in ingestion results.

### User answer review 3 — Structured object vs plain paths
**What was correct:**
- You were directionally correct that structured outputs help maintainability.

**What was weak or incomplete:**
- Too vague.
- You did not explain ingestion contracts, extensibility, or clearer downstream interfaces.
- "versionize" is not good interview wording.

**Stronger interview-quality answer:**
Returning a structured `DiscoveredDocxFile` object is better than returning plain file paths because it creates a clearer ingestion contract and makes the loader easier to extend. Even if we currently store only the path, filename, and temporary-file status, we can later add more discovery metadata without changing every downstream function signature.

### User answer review 4 — Downstream problem from unfiltered temp files
**What was correct:**
- You tried to connect the issue to downstream effects.

**What was weak or wrong:**
- Duplicate embeddings is not the main or immediate risk here.
- The more direct problem is parsing or extraction failure on files that are not valid source documents.

**Stronger interview-quality answer:**
If temporary files are not filtered before filename parsing and DOCX extraction, the parser may fail on malformed names or the extractor may try to read incomplete lock files that are not valid source documents. That introduces noisy ingestion failures and makes debugging less reliable. Duplicate embeddings could happen in some cases, but it is not the primary risk.

### User answer review 5 — Other enterprise discovery issues
**What was correct:**
- You did not answer this one.

**What was weak or missing:**
- Missing answer.

**Stronger interview-quality answer:**
In a real enterprise folder, I would expect mixed file types, duplicate exports, partially downloaded files, unsupported formats, renamed legacy documents, hidden system files, and nested subdirectories. A production-minded loader should define accepted input rules clearly, reject unsupported files explicitly, and log discovery statistics so input quality issues are visible.

---

## Stage 2.2 — DOCX paragraph extraction baseline

### Question 1
Why do we start with paragraph extraction only instead of trying to handle tables, images, and screenshots in the same first extractor?

**Strong answer:**
We start with paragraph extraction only because it gives us the smallest correct and testable ingestion unit. If we try to handle paragraphs, tables, and image-related complexity all at once, debugging becomes harder and failure causes become less clear. A staged extractor is more reliable and easier to validate.

### Question 2
Why is it useful to keep both `paragraphs` and `full_text` in the extracted output instead of storing only one representation?

**Strong answer:**
Keeping both representations supports different downstream needs. Paragraph-level output is useful for debugging, section-aware processing, and later chunking logic, while `full_text` is convenient for quick inspection, exports, or baseline processing. This preserves optionality without requiring re-extraction.

### Question 3
Why do we track both `paragraph_count` and `non_empty_paragraph_count`?

**Strong answer:**
Tracking both counts helps us measure document quality and extraction behavior. Total paragraph count shows structural density, while non-empty paragraph count reflects usable textual content. This is useful for debugging and detecting unusually sparse or malformed documents.

### Question 4
Why should the extractor fail explicitly for a missing file or wrong suffix instead of trying to continue silently?

**Strong answer:**
Explicit failure is better because silent continuation hides bad inputs and creates misleading downstream behavior. If the extractor accepts wrong files or missing paths without complaint, ingestion results become untrustworthy and debugging becomes harder.

### Question 5
What business or production signal can you get from a document with many paragraphs but very few non-empty paragraphs?

**Strong answer:**
A document with many paragraphs but very little usable text may indicate formatting-heavy content, extraction issues, template noise, or low informational value. In production, that is a signal to inspect document quality because such files may degrade retrieval quality or require special handling later.

---

## Test execution and import behavior — Evaluation of user answers

### User answer review 1 — Why `ModuleNotFoundError` happened
**What was correct:**
- You recognized that the problem was related to Python not resolving the project root correctly.

**What was weak or incomplete:**
- "doesn't have access to root directory" is too vague for interview quality.
- You did not explain import resolution or Python module search paths.

**Stronger interview-quality answer:**
`ModuleNotFoundError: No module named 'app'` happened because Python did not automatically include the project root in its module search path for that execution context. Even though the `app/` folder exists, Python only imports packages from locations it knows about through `sys.path`, so the import failed when the root directory was not being treated as an import base.

### User answer review 2 — Why `conftest.py` is a better short-term fix
**What was correct:**
- You correctly identified that one shared place is easier to maintain than repeating logic in every test.

**What was weak or incomplete:**
- You did not mention pytest behavior specifically.
- You did not mention reducing duplication and keeping tests cleaner.

**Stronger interview-quality answer:**
`tests/conftest.py` is a better short-term fix because pytest loads it automatically, which lets us apply shared test configuration in one place instead of duplicating `sys.path` manipulation in every test file. That keeps tests cleaner, reduces repetition, and makes future maintenance easier.

### User answer review 3 — Why `python -m pytest` is correct
**What was correct:**
- You asked for explanation instead of guessing. That is better than inventing a weak answer.

**What was weak or missing:**
- Missing answer.

**Stronger interview-quality answer:**
`python -m pytest ...` is the correct way to run these tests because pytest discovers tests, loads `conftest.py`, applies fixtures and shared configuration, and executes the files within pytest's test runner context. Running a test file directly with `python test_x.py` bypasses that test-runner behavior, so imports, fixtures, and configuration may not work the same way.

### User answer review 4 — Long-term production-grade solution
**What was correct:**
- You asked for explanation instead of giving a vague answer.

**What was weak or missing:**
- Missing answer.

**Stronger interview-quality answer:**
A more production-grade long-term solution is to package the project properly, for example with a `pyproject.toml` and an installable package layout, so the application can be installed in editable mode and imports work consistently across development, testing, and deployment. That is more reliable than manually changing `sys.path`.

### User answer review 5 — Risk of inconsistent test execution paths
**What was correct:**
- You asked for explanation instead of guessing.

**What was weak or missing:**
- Missing answer.

**Stronger interview-quality answer:**
If different developers run tests in different ways and imports only work in some execution paths, the project becomes fragile and environment-dependent. Tests may pass on one machine and fail on another, onboarding becomes confusing, and CI/CD may behave differently from local runs. That reduces trust in the test suite and wastes debugging time.

---

## Stage 2.2 — Evaluation of user answers

### User answer review 1 — Why start with paragraph extraction only
**What was correct:**
- Correct. You identified the main reason: smallest correct and testable ingestion unit.

**What was weak or incomplete:**
- Too short.
- You did not explain debugging clarity or why handling tables/images at the same time would reduce reliability.

**Stronger interview-quality answer:**
We start with paragraph extraction only because it gives us the smallest correct and testable ingestion unit. If we try to handle paragraphs, tables, screenshots, and other structure all at once, debugging becomes harder and failure causes become less clear. A staged extraction approach is more reliable and easier to validate before adding complexity.

### User answer review 2 — Why keep both `paragraphs` and `full_text`
**What was correct:**
- Correct direction. You recognized they support different downstream needs.

**What was weak or incomplete:**
- Too short.
- You did not explain what those downstream needs are.

**Stronger interview-quality answer:**
Keeping both `paragraphs` and `full_text` is useful because they serve different purposes. Paragraph-level output supports debugging, structure-aware processing, and later chunking logic, while `full_text` is convenient for quick inspection, exports, and simple baseline processing. Storing both preserves flexibility without requiring re-extraction.

### User answer review 3 — Why track both counts
**What was correct:**
- Good. This answer was mostly correct.
- You connected the two counts to structural density and usable text content.

**What was weak or incomplete:**
- You could be slightly more explicit about debugging and anomaly detection.

**Stronger interview-quality answer:**
We track both `paragraph_count` and `non_empty_paragraph_count` because they provide different signals about document quality and extraction behavior. Total paragraph count reflects structural density, while non-empty paragraph count reflects usable content. The difference between them can help identify sparse, formatting-heavy, or malformed documents that may need special handling later.

### User answer review 4 — Why fail explicitly
**What was correct:**
- Correct. You identified that silent continuation creates misleading downstream behavior.

**What was weak or incomplete:**
- Too short.
- You did not connect it clearly to ingestion trustworthiness.

**Stronger interview-quality answer:**
The extractor should fail explicitly for a missing file or wrong suffix because silent continuation hides invalid inputs and creates misleading downstream behavior. If ingestion appears to succeed while actually skipping bad files or accepting unsupported ones, the pipeline becomes untrustworthy and debugging becomes much harder.

### User answer review 5 — Production signal from sparse documents
**What was correct:**
- You correctly identified that the document may contain little usable value and may be formatting-heavy.

**What was weak or incomplete:**
- "I guess" weakens the answer.
- You did not clearly connect it to business or retrieval impact.

**Stronger interview-quality answer:**
A document with many paragraphs but very few non-empty paragraphs may indicate formatting-heavy content, extraction problems, template noise, or low informational value. In production, that is a useful quality signal because such documents may degrade retrieval quality, waste embedding cost, or require special handling before indexing.

---

## Stage 2.3 — DOCX table extraction baseline

### Question 1
Why is table extraction a separate step instead of being merged invisibly into the paragraph extractor?

**Strong answer:**
Table extraction is a separate step because tables have different structure, failure modes, and downstream value compared with paragraphs. Keeping them separate makes ingestion easier to test, debug, and improve without mixing multiple content types into one opaque extractor.

### Question 2
Why do we preserve both structured `rows` and a flattened `text_representation` for each table?

**Strong answer:**
Structured rows preserve the original table layout for debugging and future structured processing, while flattened text gives a simple baseline representation that can be indexed or inspected quickly. Keeping both avoids losing important structure while still supporting simpler downstream workflows.

### Question 3
Why normalize cell text instead of storing raw cell strings exactly as extracted?

**Strong answer:**
Normalizing cell text reduces noisy whitespace and makes extracted table content more consistent for testing, debugging, and downstream indexing. Without normalization, superficial formatting differences can make the extracted output harder to compare and less reliable.

### Question 4
What production problem can happen if table content is ignored in enterprise functional specification documents?

**Strong answer:**
If table content is ignored, the system may miss key business rules, field mappings, screen columns, or parameter definitions that exist only in tabular form. That can directly reduce retrieval accuracy and cause answers to omit important grounded evidence.

### Question 5
What future limitations still remain even after adding this basic table extractor?

**Strong answer:**
This basic extractor still does not handle richer layout semantics such as merged cells, header inference, nested tables, table-to-section relationships, or screenshots embedded near tables. It is a good baseline, but more structure-aware handling may be needed later if evaluation shows gaps.

---

## Stage 2.3 — Evaluation of user answers

### User answer review 1 — Why table extraction is separate
**What was correct:**
- You correctly recognized that tables have a different structure from paragraphs.
- You understood that mixing them carelessly can reduce ingestion quality.

**What was weak or incomplete:**
- The answer was unfinished and not phrased cleanly.
- You did not mention separate failure modes or easier testing/debugging.

**Stronger interview-quality answer:**
Table extraction should be a separate step because tables have different structure, failure modes, and downstream value compared with paragraphs. Keeping them separate makes ingestion easier to test, debug, and improve without mixing multiple content types into one opaque extractor.

### User answer review 2 — Why preserve both `rows` and `text_representation`
**What was correct:**
- Good. You correctly identified the main tradeoff.
- You understood that structured rows preserve layout while flattened text is easier to search or index.

**What was weak or incomplete:**
- You could be more explicit that this supports different downstream workflows.

**Stronger interview-quality answer:**
We preserve both structured `rows` and flattened `text_representation` because they support different downstream uses. Structured rows preserve original table layout for debugging and future structured processing, while flattened text gives a simple baseline representation that can be indexed, searched, and inspected quickly.

### User answer review 3 — Why normalize cell text
**What was correct:**
- Directionally correct. You recognized normalization reduces noise and improves consistency.

**What was weak or incomplete:**
- Too short for interview quality.
- You did not explain why this matters for testing and downstream indexing.

**Stronger interview-quality answer:**
We normalize cell text to reduce noisy whitespace and make extracted table content more consistent for testing, debugging, and downstream indexing. Without normalization, superficial formatting differences can make outputs harder to compare and less reliable.

### User answer review 4 — Risk of ignoring table content
**What was correct:**
- You correctly understood that important business information may exist in tables.

**What was weak or incomplete:**
- You used vague phrases like "insights" and "explaination" instead of precise engineering consequences.
- You did not explicitly connect it to retrieval and grounded answering failures.

**Stronger interview-quality answer:**
If table content is ignored, the system may miss key business rules, field mappings, report columns, or parameter definitions that exist only in tabular form. That can directly reduce retrieval accuracy and cause grounded answers to omit important evidence.

### User answer review 5 — Remaining limitations
**What was correct:**
- Good. You identified merged cells and screenshots as limitations.
- You correctly recognized that basic extraction still has gaps.

**What was weak or incomplete:**
- You mentioned Excel attachments inside tables, which is possible but less central than layout semantics.
- You missed table header inference, nested tables, and table-to-section relationships.

**Stronger interview-quality answer:**
This basic table extractor still has important limitations. It does not yet handle richer layout semantics such as merged cells, header inference, nested tables, table-to-section relationships, or screenshots located near tables. It is a solid baseline, but more structure-aware handling may be needed if evaluation shows these gaps affect retrieval quality.

---

## Step 55 preparation — Dense vs lexical retrieval comparison

### Question 1
Why is dense semantic retrieval often weak for acronyms, report IDs, and field names?

**Strong answer:**
Dense embeddings are optimized for semantic similarity, not exact token identity. Acronyms, report IDs like `B-01`, field names, and short labels often have little surrounding context and may be represented weakly in embedding space. Similar business-context chunks can outrank the exact identifier chunk, so dense retrieval may retrieve generally relevant report text while missing the exact identifier or field label.

### Question 2
Why should lexical retrieval be implemented before hybrid retrieval?

**Strong answer:**
Lexical retrieval should be implemented before hybrid retrieval because hybrid search should be justified by measurement, not assumed. A lexical baseline gives an independent exact-match retrieval signal. Once dense and lexical results are compared on the same evaluation set, we can identify dense-only wins, lexical-only wins, both-win cases, and both-fail cases before designing any fusion logic.

### Question 3
If lexical retrieval returns boilerplate documents at the top, what upstream pipeline issue does that suggest?

**Strong answer:**
It suggests the indexed corpus still contains high-frequency noise such as document-control tables, review tables, traceability matrices, headers, footers, or front-matter content. It may also mean chunks are too broad or not section-aware, causing useful terms to be mixed with low-value text. The fix may involve stronger normalization, section-aware chunking, metadata filters, or domain-noise handling before retrieval fusion.

### Question 4
How would you compare dense retrieval and lexical retrieval fairly using the existing evaluation set?

**Strong answer:**
A fair comparison changes only the retrieval method. The evaluation cases, corpus, filters, `top_k`, expected text markers, artifact version, and index version should stay constant. Then compare metrics such as Hit@k, Top-1 correctness, MRR, release correctness, source-kind correctness, and failure categories such as dense-only win, lexical-only win, both win, and both fail.

```python
def compare_retrievers(eval_cases, dense_retriever, lexical_retriever, top_k: int = 5):
    rows = []
    for case in eval_cases:
        dense_results = dense_retriever(case.query, case.filters, limit=top_k)
        lexical_results = lexical_retriever(case.query, case.filters, limit=top_k)

        dense_passed = evaluate_retrieval_results(case, dense_results).passed
        lexical_passed = evaluate_retrieval_results(case, lexical_results).passed

        rows.append({
            "case_id": case.case_id,
            "dense_passed": dense_passed,
            "lexical_passed": lexical_passed,
            "winner": classify_winner(dense_passed, lexical_passed),
        })
    return rows
```

### Question 5
If both dense and lexical fail for `B-01 report layout`, what exact debugging checklist should be followed before changing retrieval logic?

**Strong answer:**
Before changing retrieval logic, verify the evidence path in order: confirm the source document contains the `B-01` layout, check whether the evidence is inside normal DOCX text, a table, an embedded Excel object, or a screenshot, inspect the processed ingestion artifact, inspect normalized text to ensure it was not removed, inspect chunked/retrieval-ready artifacts, verify the embedding cache and Qdrant index contain the unit, and only then decide whether the issue is retrieval, ingestion scope, chunking, metadata filtering, or unsupported content. If the evidence is inside an embedded Excel attachment, the correct diagnosis is ingestion scope limitation, not dense retrieval failure.

---

## Step 55 preparation — Evaluation of user answers

### User answer review 1 — Dense retrieval weakness
**What was correct:**
- You correctly identified that dense retrieval is based on semantic relations rather than strict exact-match behavior.

**What was weak or incomplete:**
- You did not explicitly mention exact identifiers like `B-01`, acronyms, field names, or short labels being fragile in embedding space.

**Stronger interview-quality answer:**
Dense retrieval can be weak for acronyms, report IDs, and field names because embeddings optimize semantic similarity rather than exact token identity. Exact identifiers may have little context and can be outranked by semantically similar but identifier-missing chunks.

### User answer review 2 — Lexical before hybrid
**What was correct:**
- You correctly said lexical retrieval should be compared first so the decision to use hybrid is evidence-based.

**What was weak or incomplete:**
- You should explicitly say hybrid should not be added until dense and lexical failure modes are measured independently.

**Stronger interview-quality answer:**
We implement lexical retrieval before hybrid because hybrid should be justified by measurement. Lexical retrieval provides a separate exact-match baseline, and only after comparing dense and lexical behavior should we design fusion.

### User answer review 3 — Boilerplate at top
**What was correct:**
- You correctly connected the issue to possible chunking weakness.

**What was weak or incomplete:**
- You missed the more direct cause: boilerplate/noise surviving normalization and being indexed.

**Stronger interview-quality answer:**
If lexical retrieval returns boilerplate at the top, it likely means noisy document-control text, traceability tables, headers, footers, or review tables remain in the indexed corpus. Section-aware chunking may help, but normalization and metadata filtering should also be inspected.

### User answer review 4 — Fair dense vs lexical comparison
**What was correct:**
- You correctly interpreted what it means when dense wins or lexical wins.

**What was weak or incomplete:**
- You did not answer the experiment-design part. A fair comparison requires holding the evaluation set, filters, corpus, `top_k`, expectations, artifact version, and index version constant while changing only the retriever.

**Stronger interview-quality answer:**
To compare dense and lexical fairly, I would run both retrievers on the same evaluation cases, same corpus, same filters, same `top_k`, and same expected evidence markers. Then I would compare Hit@k, Top-1 correctness, MRR, release/source-kind correctness, and classify each case as dense-only win, lexical-only win, both win, or both fail.

### User answer review 5 — Both dense and lexical fail for B-01 layout
**What was correct:**
- You correctly identified unsupported material as the likely cause because the layout appears to be inside an embedded Excel object.

**What was weak or incomplete:**
- You still did not state the full debugging checklist. In production RAG, you must trace evidence from source document to extraction, normalization, chunking, embedding, indexing, and retrieval before changing retrieval logic.

**Stronger interview-quality answer:**
If both dense and lexical fail for `B-01 report layout`, I would first verify whether the evidence exists in the source DOCX and identify whether it is normal text, a table, an embedded Excel object, or screenshot. Then I would inspect the processed artifact, normalized artifact, retrieval-ready artifact, embedding cache, and Qdrant payloads. If the evidence exists only in an embedded Excel object, the issue is ingestion scope limitation, not retrieval logic.

---

## Step 55 — Lexical retrieval baseline implementation

### Question 1
Why did we preserve identifiers like `B-01` as one token instead of splitting them into `b` and `01`?

**Strong answer:**
We preserve identifiers like `B-01` as one token because the exact identifier carries business meaning. Splitting it into `b` and `01` would weaken exact-match retrieval, increase false positives, and make it harder to distinguish report IDs from generic text.

### Question 2
Why does lexical search load `.retrieval_ready.json` artifacts instead of reading raw DOCX files directly?

**Strong answer:**
Lexical search should operate on the same retrieval units used by embeddings and vector indexing. Loading `.retrieval_ready.json` keeps dense and lexical comparisons fair because both retrievers search the same cleaned, chunked, metadata-rich corpus rather than different document representations.

### Question 3
What did the real `B-01 report layout` lexical smoke test prove, and what did it not prove?

**Strong answer:**
It proved that the lexical retriever can find the exact paragraph marker containing `B-01`, `report`, and `layout` in the R24 retrieval-ready artifact. It did not prove that the actual report layout details are available, because the paragraph points to an attached sample report. If the detailed layout is inside an embedded Excel object, that remains an ingestion-scope limitation.

### Question 4
Where can this simple lexical baseline break in production?

**Strong answer:**
It can break on typos, synonyms, formatting variants, pluralization, very noisy boilerplate, repeated high-frequency terms, and cases where the evidence exists only in unsupported content such as attachments or screenshots. It is useful as an exact-match baseline, but it is not a full semantic retriever or a replacement for ingestion quality.

### Question 5
Why are we not implementing hybrid retrieval immediately after this lexical baseline?

**Strong answer:**
We should not implement hybrid retrieval until we compare dense and lexical results on the same evaluation cases. Hybrid should be justified by measurable improvement, not added because it sounds advanced. The next disciplined step is to evaluate lexical behavior and compare it against dense retrieval.

---

## Step 55 — Evaluation of user answers

### User answer review 1 — Preserving `B-01` as one token
**What was correct:**
- You understood that preserving the original identifier matters during search.

**What was weak or incomplete:**
- You framed this as preserving document order and semantics, but the more precise issue is preserving exact identifier identity.
- Splitting `B-01` into `b` and `01` increases false positives and weakens report-ID matching.

**Stronger interview-quality answer:**
Preserving `B-01` as one token matters because it is a business identifier, not just normal text. If we split it into `b` and `01`, the retriever may match unrelated chunks containing either token and lose the exact report-ID signal. Keeping it intact improves exact-match retrieval for report names, field IDs, and enterprise labels.

### User answer review 2 — Searching `.retrieval_ready.json` instead of raw DOCX
**What was correct:**
- You correctly said the retrieval-ready artifacts are already cleaned, normalized, and faster to search.

**What was weak or incomplete:**
- You did not mention fairness: dense and lexical retrieval should search the same retrieval units.
- You did not mention metadata filtering from the retrieval-ready schema.

**Stronger interview-quality answer:**
Lexical search uses `.retrieval_ready.json` because those artifacts contain the same cleaned, chunked, metadata-rich units used for embedding and vector indexing. This makes dense-vs-lexical comparison fair and lets lexical search use release/source metadata filters instead of re-reading raw DOCX files differently.

### User answer review 3 — What the `B-01 report layout` smoke test proved
**What was correct:**
- You recognized that retrieving a `B-01` result does not automatically mean the complete answer is available.
- You noticed that repeated references can make retrieval ambiguous in larger corpora.

**What was weak or incomplete:**
- The answer was vague: "relationship" and "essence" are not precise enough for an interview.
- The key limitation is not primarily latency; it is that the actual layout may be inside an embedded attachment not extracted into text.

**Stronger interview-quality answer:**
The smoke test proved that lexical retrieval can find the exact paragraph marker containing `B-01`, `report`, and `layout` in the R24 retrieval-ready artifact. It did not prove that the actual layout details are available, because the paragraph says the sample report is attached. If the detailed layout is inside an embedded Excel object, the limitation is ingestion scope, not lexical retrieval.

### User answer review 4 — Where the lexical baseline can break
**What was correct:**
- You correctly identified large corpora and repeated reference keywords as risks.

**What was weak or incomplete:**
- You missed typos, synonyms, formatting variants, pluralization, boilerplate, high-frequency terms, and unsupported attachments/screenshots.

**Stronger interview-quality answer:**
This lexical baseline can break when many documents share the same keywords, when boilerplate dominates, when users use synonyms or typos, when formatting variants change the exact token, and when evidence is inside unsupported attachments or screenshots. It is useful for exact-match signals, but it is not robust semantic retrieval.

### User answer review 5 — Dense-vs-lexical comparison before hybrid
**What was correct:**
- You correctly said we should analyze both methods before combining them.

**What was weak or incomplete:**
- You should explicitly say that hybrid retrieval must be justified by measured improvement, not assumed.
- You should mention dense-only wins, lexical-only wins, both-win, and both-fail categories.

**Stronger interview-quality answer:**
We compare lexical and dense retrieval before hybrid because hybrid should be evidence-driven. By running both on the same evaluation cases, we can identify dense-only wins, lexical-only wins, both-win cases, and both-fail cases. Only then can we design a fusion strategy that addresses measured failure modes instead of adding complexity blindly.

---

## Step 56 — Dense vs lexical retrieval comparison report

### Question 1
Why does a `lexical_only` outcome not automatically mean lexical retrieval is better than dense retrieval overall?

**Strong answer:**
`lexical_only` means lexical passed that specific evaluation case while dense did not. It does not prove lexical is globally better because dense may still perform better on semantic or paraphrased queries. It also does not prove answer completeness; lexical may match exact markers while missing the deeper evidence needed for a grounded answer.

### Question 2
What did the dense-only realignment-summary case show about lexical retrieval?

**Strong answer:**
It showed that lexical retrieval can over-rank chunks that share exact query terms but are not the best evidence. In the realignment-summary case, lexical search ranked an annexure/layout-related chunk above the requirements-summary chunk because of exact term overlap. Dense retrieval handled the semantic intent better.

### Question 3
Why is the `B-01 report layout` lexical-only case dangerous to interpret naively?

**Strong answer:**
It is dangerous because lexical retrieval found exact `B-01` and `layout` markers, but that does not prove the actual layout details are extracted. The paragraph or table may only point to an attached sample report. If the detailed layout is inside an embedded Excel object, retrieval succeeded only at finding a marker, while ingestion still lacks the underlying evidence.

### Question 4
What must stay constant when comparing dense and lexical retrieval fairly?

**Strong answer:**
The evaluation cases, corpus, filters, top-k limit, expected evidence markers, artifact version, and index version should stay constant. Only the retrieval method should change. Otherwise, differences in results could come from changed inputs rather than retrieval behavior.

### Question 5
Why is the next step error analysis rather than immediate hybrid retrieval?

**Strong answer:**
The comparison report tells us that dense and lexical have complementary failures, but it does not yet tell us the best fusion strategy. Error analysis is needed to inspect why each dense-only, lexical-only, and both-fail case happened. Without that, hybrid retrieval could simply combine noise from both systems instead of improving evidence quality.

---

## Step 56 — Evaluation of user answers

### User answer review 1 — Why `lexical_only` does not mean lexical is globally better
**What was correct:**
- You correctly identified that lexical retrieval may simply be matching keywords or patterns.
- You correctly noticed that repeated keywords and boilerplate can make lexical results misleading.

**What was weak or incomplete:**
- You did not contrast this with dense retrieval's strength on semantic or paraphrased queries.
- You did not mention that passing a marker-based eval case does not prove answer completeness.

**Stronger interview-quality answer:**
`lexical_only` means lexical passed that specific evaluation case while dense did not. It does not prove lexical is globally better because dense may still perform better on semantic, paraphrased, or context-heavy queries. Lexical can also pass by matching surface markers while still missing the deeper evidence required for a complete grounded answer.

### User answer review 2 — Dense-only realignment-summary case
**What was correct:**
- You understood that both dense and lexical can retrieve useful results when the evidence exists in the ingested corpus.

**What was weak or wrong:**
- Your answer missed the actual finding from the report.
- The case was `dense_only`, not both-pass. Dense passed the top-1 requirement, while lexical failed top-1 by ranking an annexure/layout-related chunk above the requirements-summary chunk.

**Stronger interview-quality answer:**
The dense-only realignment-summary case showed that lexical retrieval can over-rank chunks with exact query-term overlap even when they are not the best evidence. Dense retrieval found the semantic requirements-summary chunk, while lexical search ranked an annexure/layout chunk first because it shared terms like `branch`, `reports`, and `realignment`.

### User answer review 3 — Naive interpretation of `B-01 report layout`
**What was correct:**
- You correctly identified that repeated `B-01` references across documents can create noisy retrieval.

**What was weak or incomplete:**
- You missed the most important issue: lexical retrieval found a marker, not necessarily the actual layout evidence.
- The actual layout may live in an embedded Excel object or attachment outside current ingestion scope.

**Stronger interview-quality answer:**
The `B-01 report layout` lexical-only case is dangerous because lexical retrieval found exact `B-01` and `layout` markers, but that does not prove the detailed layout was extracted. The retrieved paragraph or table may only point to an attached sample report. If the actual layout is inside an embedded Excel object, the retrieval result is only a pointer, not the underlying evidence.

### User answer review 4 — Fair comparison constants
**What was correct:**
- You asked for explanation instead of guessing.

**What was missing:**
- You need to know that a fair experiment changes one variable at a time: retrieval method only.

**Stronger interview-quality answer:**
When comparing dense and lexical retrieval fairly, the evaluation cases, corpus, filters, top-k limit, expected evidence markers, artifact version, and index version must stay constant. Only the retrieval method changes. Otherwise, differences could be caused by different inputs rather than the retrieval strategy itself.

### User answer review 5 — Why error analysis before hybrid
**What was correct:**
- You asked the right question before jumping into implementation.

**What was missing:**
- Hybrid retrieval should not be implemented just because dense and lexical are complementary.
- We first need to inspect the actual failure modes so fusion does not combine noise.

**Stronger interview-quality answer:**
The next step is error analysis because the comparison report shows dense and lexical have complementary failures, but it does not tell us how to combine them safely. We need to inspect the dense-only, lexical-only, and both-fail cases to understand whether failures come from scoring, chunking, boilerplate, missing evidence, unsupported attachments, or weak evaluation labels. Only after that should we design hybrid fusion.

---

## Step 57 — Retrieval comparison error analysis

### Question 1
Why did the error analyzer exclude `both_pass` cases by default?

**Strong answer:**
The immediate goal is failure analysis, so excluding `both_pass` cases keeps attention on dense-only, lexical-only, and both-fail cases. Both-pass cases are still useful regression checks, but they are not the main source of retrieval improvement decisions.

### Question 2
Why is `r24_b01_report_layout_missing_embedded_attachment` labeled high severity?

**Strong answer:**
It is high severity because lexical retrieval appears to pass a case expected to fail by matching markers like `B-01` and `layout`, while the actual layout evidence may still be inside an unsupported embedded attachment. This can create false confidence: the retriever found a pointer, not necessarily the full evidence needed for a grounded answer.

### Question 3
What does `lexical_exact_term_false_positive` mean in this project?

**Strong answer:**
It means lexical retrieval ranked a chunk highly because it shared exact query terms, but the chunk did not satisfy the expected evidence requirement. In the realignment-summary case, lexical matched terms like `branch`, `reports`, and `realignment`, but ranked an annexure/layout chunk above the actual requirements-summary evidence.

### Question 4
Why is `expected_unanswerable` a useful error-analysis label instead of just calling it a retrieval failure?

**Strong answer:**
`expected_unanswerable` separates desired abstention behavior from true retrieval defects. If the corpus does not contain mobile-login evidence, retrieval should not be expected to find valid evidence. This case is useful for testing safe refusal, not for tuning retrieval to force an answer.

### Question 5
What should happen before hybrid fusion after this error analysis?

**Strong answer:**
Before hybrid fusion, we should tighten weak marker-based expectations, inspect whether the `B-01` layout evidence exists in extracted artifacts, and decide whether lexical false positives need filtering through metadata, section awareness, or better normalization. Hybrid should be designed only after we know which failures are retrieval-ranking problems versus ingestion/evaluation problems.

---

## Step 57 — Evaluation of user answers

### User answer review 1 — Why `both_pass` cases are excluded by default
**What was correct:**
- You noticed that `both_pass` exists in the comparison report.

**What was weak or misunderstood:**
- The comparison report includes `both_pass` cases because it summarizes all dense-vs-lexical outcomes.
- The error-analysis report excludes `both_pass` cases by default because its purpose is failure analysis.
- `both_pass` can still be included with `--include-both-pass`, but it is not the default because it would dilute focus from dense-only, lexical-only, and both-fail cases.

**Stronger interview-quality answer:**
`both_pass` cases are included in the comparison report, but the error analyzer excludes them by default because the immediate goal is failure analysis. We want to focus on dense-only, lexical-only, and both-fail cases where retrieval behavior needs explanation or improvement. Both-pass cases are still useful as regression checks and can be included with `--include-both-pass` if needed.

### User answer review 2 — Why B-01 layout is high severity
**What was correct:**
- You correctly said the system did not find enough relevant evidence for accurate full results.

**What was weak or incomplete:**
- The deeper issue is not just failure to find evidence; lexical appeared to pass even though it may only have found a marker or pointer.
- This is dangerous because it can create false confidence in retrieval quality.

**Stronger interview-quality answer:**
The B-01 layout case is high severity because lexical retrieval appears to pass by finding `B-01` and layout-related markers, but the actual layout details may still be inside an unsupported embedded attachment. That means the system may think it found evidence when it only found a pointer to evidence. This is dangerous for grounded answering because it can produce false confidence.

### User answer review 3 — `lexical_exact_term_false_positive`
**What was correct:**
- You correctly understood that this means something looks like a match but is not actually the right evidence.

**What was weak or incomplete:**
- You need to be more precise: lexical matched exact query terms, but those terms appeared in the wrong chunk.

**Stronger interview-quality answer:**
`lexical_exact_term_false_positive` means lexical retrieval ranked a chunk highly because it shared exact query terms, but the chunk did not satisfy the expected evidence requirement. In this project, the realignment case matched terms like `branch`, `reports`, and `realignment`, but ranked an annexure/layout chunk instead of the requirements-summary evidence.

### User answer review 4 — `expected_unanswerable`
**What was correct:**
- Good. You correctly connected this label to production robustness and handling unknowns.

**What was weak or incomplete:**
- You should explicitly connect it to safe refusal and avoiding false optimization.

**Stronger interview-quality answer:**
`expected_unanswerable` is useful because not every retrieval miss is a defect. If the corpus does not contain the answer, the correct behavior is safe refusal rather than forcing retrieval to find something. This label helps separate true retrieval bugs from cases that should test abstention and hallucination prevention.

### User answer review 5 — What should happen before hybrid fusion
**What was correct:**
- You correctly said weak results should be tightened with proof and stronger evidence.

**What was weak or incomplete:**
- You should name the concrete work: tighten marker-based eval labels, inspect B-01 evidence availability, and filter lexical false positives before fusion.

**Stronger interview-quality answer:**
Before hybrid fusion, we should tighten marker-based evaluation labels, inspect whether the full B-01 layout evidence exists in extracted artifacts, and decide whether lexical false positives need filtering through metadata, section awareness, or normalization. Hybrid should only be designed after we know which issues are retrieval-ranking problems versus ingestion or evaluation-label problems.

---

## Step 58 — Tightened marker-only unsupported evidence evaluation

### Question 1
Why is finding `B-01` and `layout` not enough to count as a successful retrieval answer for the B-01 layout case?

**Strong answer:**
Because `B-01` and `layout` may only be markers or references to an attached sample report. Successful answer evidence requires the actual layout details to be extracted and available, not merely a sentence pointing to `B-01 Branch End of Day Report.xlsx`.

### Question 2
Why did the B-01 case move from `lexical_only` to `both_fail` after tightening the evaluation labels?

**Strong answer:**
Before tightening, lexical retrieval passed because it found exact marker terms. After adding unsupported evidence markers like `Sample Report` and `.xlsx`, the evaluator recognized that the retrieved text points to an unsupported attachment rather than containing full layout evidence. Therefore neither dense nor lexical should be credited with a true evidence pass.

### Question 3
What is the difference between `expected_marker_contains_any` and `expected_text_contains_any`?

**Strong answer:**
`expected_marker_contains_any` records weaker marker/reference evidence that may indicate a relevant area was found. `expected_text_contains_any` is intended for actual evidence needed to answer the question. Separating them helps avoid treating references or pointers as complete answer evidence.

### Question 4
Why is `unsupported_evidence_contains_any` useful for production RAG evaluation?

**Strong answer:**
It lets the evaluator detect when retrieved text points to unsupported content such as attachments, screenshots, or external files. This is important because retrieval may find a reference to evidence without actually retrieving the evidence itself. In production, that should trigger abstention or an unsupported-content warning, not a confident answer.

### Question 5
What should happen next before hybrid retrieval is implemented?

**Strong answer:**
We should inspect the processed and retrieval-ready artifacts to confirm whether the B-01 layout details exist as extracted text or only as attachment references. If the actual evidence is unavailable, the system should abstain for layout-detail questions or explicitly say the answer depends on unsupported attachment content. Hybrid retrieval should wait until evaluation labels distinguish true evidence from marker-only evidence.

---

## Step 58 — Evaluation of user answers

### User answer review 1 — Why `B-01` and `layout` are not enough
**What was correct:**
- You correctly identified that the real goal is to retrieve the actual matching report layout, not merely words that look related.
- You correctly connected this to accuracy risk in corporate/enterprise information systems.

**What was weak or incomplete:**
- You should explicitly say that marker words may only point to an unsupported Excel attachment.

**Stronger interview-quality answer:**
Finding `B-01` and `layout` is not enough because those terms may only be markers or references to an attached sample report. A successful retrieval result must contain the actual layout details needed to answer the question. If the text only says `Sample Report: B-01 Branch End of Day Report.xlsx`, then retrieval found a pointer to evidence, not the evidence itself.

### User answer review 2 — Why B-01 moved from `lexical_only` to `both_fail`
**What was correct:**
- You correctly understood that the failure is expected after tightening the labels.

**What was weak or inaccurate:**
- It was not only because the expected marker/reference was missing. In fact, lexical did find markers.
- The key reason is that the retrieved markers were attachment references, which are now treated as unsupported marker-only content.

**Stronger interview-quality answer:**
The B-01 case moved from `lexical_only` to `both_fail` because lexical retrieval found marker terms, but those markers pointed to unsupported attachment evidence such as `Sample Report` and `.xlsx`. After tightening the labels, marker-only evidence is no longer counted as a true pass. Since neither dense nor lexical retrieved the actual layout details, both should fail this case.

### User answer review 3 — `expected_marker_contains_any` vs `expected_text_contains_any`
**What was correct:**
- You noticed both fields use substring matching, so the naming can feel confusing.

**What was wrong or incomplete:**
- `expected_marker_contains_any` is not the same as evidence.
- It is weaker than actual answer evidence. It means “we found a reference/pointer/marker.”

**Stronger interview-quality answer:**
`expected_marker_contains_any` is for weak marker or reference evidence, such as finding `B-01` or `layout`, which may only indicate that retrieval found the right neighborhood. `expected_text_contains_any` is for stronger answer evidence that should actually support the answer. Separating them prevents a pointer like `Sample Report: B-01...xlsx` from being treated as full layout evidence.

### User answer review 4 — Why `unsupported_evidence_contains_any` is useful
**What was correct:**
- You correctly recognized that keyword-only matching can produce shallow or boilerplate-like behavior.

**What was weak or incomplete:**
- The field is not meant to add weight to findings.
- It is a guardrail that marks retrieved content as unsupported when it points to attachments/screenshots/files outside current ingestion scope.

**Stronger interview-quality answer:**
`unsupported_evidence_contains_any` is useful because it detects when retrieved text points to evidence that the current pipeline has not actually extracted, such as Excel attachments, PDFs, screenshots, or sample reports. This prevents the evaluator from rewarding a result that only references unavailable evidence. In production, that should trigger abstention or an unsupported-content warning.

### User answer review 5 — Whether hybrid search can start now
**What was correct:**
- You correctly stated that the B-01 layout is inside an Excel object and is out of scope for the current text/table-only ingestion version.
- You correctly concluded that `false` is expected for this case under the current system scope.

**What still needs discipline:**
- Hybrid search can start now only if we explicitly preserve the unsupported-evidence behavior and do not treat B-01 layout as a success target.
- Hybrid should be evaluated against the tightened labels and must not optimize toward attachment-marker matches.

**Stronger interview-quality answer:**
Since the actual B-01 layout lives inside an embedded Excel object and current ingestion only covers DOCX paragraphs and tables, this case should remain an expected unsupported-evidence failure. We can move toward hybrid retrieval now, but hybrid must be evaluated using the tightened labels so it does not reward marker-only attachment references. Hybrid should improve dense/lexical ranking where evidence exists; it should not pretend unsupported attachment content has been extracted.

---

## Step 59 — Hybrid retrieval baseline with simple score fusion

### Question 1
Why did we normalize dense and lexical scores before fusing them?

**Strong answer:**
Dense and lexical scores live on different scales, so raw scores are not directly comparable. Normalizing each retriever's scores by its own maximum score for the query makes a simple weighted fusion possible without one retriever dominating just because its score range is larger.

### Question 2
Why is `hybrid_only_count=0` important?

**Strong answer:**
It means the current hybrid baseline did not solve any case that both dense and lexical failed individually. Hybrid preserved existing wins, but it did not create a new retrieval capability on this small evaluation set. We should not overclaim improvement from hybrid until evaluation shows new wins or better ranking.

### Question 3
Why is it good that the B-01 attachment case remained `all_fail`?

**Strong answer:**
Because the actual B-01 layout is inside an unsupported embedded Excel attachment. If hybrid had passed this case by combining marker evidence, it would have rewarded a false success. Remaining `all_fail` shows the tightened labels are protecting the system from treating attachment references as extracted evidence.

### Question 4
What does the realignment-summary `dense_and_hybrid` case tell us?

**Strong answer:**
It shows hybrid preserved the dense retrieval win where lexical alone failed top-1. Dense found the semantic requirements-summary chunk, and hybrid kept that chunk strong enough to pass. This is useful, but it is preservation of a dense win, not proof that hybrid is broadly better.

### Question 5
What should happen before making hybrid the default retriever?

**Strong answer:**
We should inspect hybrid failure/non-improvement cases, try controlled weight experiments, verify hybrid does not promote lexical noise, and expand the evaluation set. Hybrid should become default only if it consistently improves retrieval quality without increasing false positives or unsupported-evidence mistakes.

---

## Step 59 — Evaluation of user answers

### User answer review 1 — Why normalize dense and lexical scores
**What was correct:**
- Correct. You understood that dense and lexical scores are on different scales.
- You correctly connected normalization to making weights meaningful.

**What could be stronger:**
- Mention that without normalization, lexical scores could dominate simply because their numeric range is larger.

**Stronger interview-quality answer:**
We normalize dense and lexical scores because they are not naturally comparable. Dense cosine-style scores and lexical matching scores use different scales. If we fused raw scores directly, one retriever could dominate only because of score magnitude. Normalization makes the weighted fusion more interpretable.

### User answer review 2 — Why `hybrid_only_count=0` matters
**What was correct:**
- You recognized that dense and lexical are already doing reasonably well on the current small eval set.

**What was weak or inaccurate:**
- You said hybrid is doing well, but `hybrid_only_count=0` specifically means hybrid did not solve any case that both base retrievers failed.
- That means we should not overclaim hybrid improvement yet.

**Stronger interview-quality answer:**
`hybrid_only_count=0` means hybrid did not create any new wins beyond dense and lexical on this evaluation set. It preserved existing wins, but it did not solve a case that both base retrievers missed. So hybrid may be safe so far, but we cannot claim it is materially better until it shows new wins or better ranking across a larger evaluation set.

### User answer review 3 — Why B-01 remaining `all_fail` is good
**What was correct:**
- You correctly understood that this is a failed case and that preserving failure is desirable.

**What was weak or incomplete:**
- It is not mainly about “getting the semantics of dense.”
- The key issue is safety: hybrid must not convert unsupported attachment-marker evidence into a false pass.

**Stronger interview-quality answer:**
It is good that the B-01 attachment case remained `all_fail` because the actual layout evidence is inside an unsupported Excel attachment. If hybrid had passed this case by combining marker references, it would have created a false success. Remaining failed shows the tightened labels are protecting the system from pretending unsupported content was extracted.

### User answer review 4 — What `dense_and_hybrid` means for realignment summary
**What was correct:**
- You recognized that hybrid followed dense behavior and preserved expected behavior.

**What was weak or incomplete:**
- You overstated it as “covering all test cases” and “looks better.”
- The exact interpretation is narrower: hybrid preserved a dense win where lexical failed.

**Stronger interview-quality answer:**
The realignment-summary `dense_and_hybrid` case shows that hybrid preserved the dense retriever's semantic win. Lexical alone failed top-1 by ranking a less useful exact-term chunk, but dense found the requirements-summary chunk and hybrid kept that chunk ranked correctly. This is good, but it does not prove hybrid is broadly better.

### User answer review 5 — Before making hybrid default
**What was correct:**
- Correct. You identified weight bias analysis as the next necessary step.

**What was weak or incomplete:**
- Weight experiments are necessary but not sufficient. We also need more eval cases and checks that lexical-heavy settings do not promote noise.

**Stronger interview-quality answer:**
Before making hybrid the default retriever, we should run controlled weight experiments, inspect whether lexical-heavy fusion promotes noise, verify unsupported-evidence cases remain failures, and expand the evaluation set. Hybrid should only become default if it consistently improves retrieval quality without increasing false positives or unsafe evidence use.

---

## Step 60 — Hybrid weight experiment runner

### Question 1
Why did the experiment runner retrieve dense and lexical candidates once per case instead of once per weight setting?

**Strong answer:**
Dense query embedding calls are comparatively expensive and slow. Retrieving dense and lexical candidates once per case, then reusing them across weight settings, makes the experiment cheaper, faster, and more controlled because only the fusion weights change.

### Question 2
Why is the lexical-heavy `0.2 dense / 0.8 lexical` result important?

**Strong answer:**
It degraded from 7/7 expected outcomes to 6/7 and introduced a `hybrid_missed_dense` case. That shows lexical-heavy fusion can overpower useful dense semantic evidence and promote exact-term noise. This is evidence that lexical weighting should be used carefully.

### Question 3
Why did several weight settings tie as best?

**Strong answer:**
The current evaluation set is small and may not be sensitive enough to distinguish moderate weight settings. The tie means multiple settings are safe on the current cases, but it does not prove they are equally good in production. We need more evaluation cases before selecting a default.

### Question 4
Why is `unsafe_expected_failure_pass_count=0` important?

**Strong answer:**
It means none of the tested weight settings converted expected-failure cases, such as unsupported attachment evidence or unanswerable questions, into false passes. This is important because retrieval improvements should not come at the cost of unsafe hallucination-prone behavior.

### Question 5
What should happen before choosing the default hybrid weight?

**Strong answer:**
We should expand the evaluation set with more exact-identifier, semantic paraphrase, boilerplate-risk, table-heavy, unsupported-content, and unanswerable cases. Then rerun the weight experiments and choose a default only if one setting consistently improves quality without increasing false positives or unsafe expected-failure passes.

---

## Step 60 — Evaluation of user answers

### User answer review 1 — Candidate retrieval once per case
**What was correct:**
- You correctly recognized that case-wise retrieval makes comparison easier to inspect.

**What was weak or incomplete:**
- The strongest production reason is cost/control: dense query embeddings should not be repeatedly called for each weight setting.
- Reusing candidates ensures only the fusion weights change.

**Stronger interview-quality answer:**
The experiment runner retrieves dense and lexical candidates once per case because dense query embedding calls are slower and cost-bearing. Reusing the same candidate sets across all weight settings makes the experiment cheaper, faster, and fairer because only the fusion weights change.

### User answer review 2 — Lexical-heavy degradation
**What was correct:**
- Correct. You identified that the lexical-heavy setting passed only `6/7` expected outcomes.

**What was weak or incomplete:**
- You should explicitly state what this implies: too much lexical weight can overpower useful dense semantic evidence.

**Stronger interview-quality answer:**
The `0.2 dense / 0.8 lexical` result is important because it degraded from `7/7` expected outcomes to `6/7`. That means lexical-heavy fusion can overpower dense semantic evidence and promote exact-term noise. This is a warning not to set lexical weight too high without stronger evaluation support.

### User answer review 3 — Several settings tied as best
**What was correct:**
- Good. You correctly identified that the current evaluation set is too small to distinguish moderate weight settings confidently.

**What was weak or incomplete:**
- You should explicitly avoid calling any tied setting “optimal.”

**Stronger interview-quality answer:**
Several weight settings tied because the current evaluation set is small and not sensitive enough to separate moderate dense/lexical mixes. A tie means those settings are safe on the current cases, not that they are equally good in production. More evaluation cases are needed before selecting a true default.

### User answer review 4 — `unsafe_expected_failure_pass_count=0`
**What was correct:**
- You understood that this checks whether a weight setting causes erroneous behavior elsewhere.

**What was weak or incomplete:**
- The specific danger is converting expected-failure cases into false passes.

**Stronger interview-quality answer:**
`unsafe_expected_failure_pass_count=0` is important because none of the tested weight settings turned expected-failure cases into false successes. This means unsupported attachment cases and unanswerable questions remained protected, so the retrieval experiment did not improve apparent recall by creating unsafe evidence mistakes.

### User answer review 5 — Choosing `0.6 dense / 0.4 lexical` for now
**What was correct:**
- You correctly said release-lineage evaluation needs more cases later.
- You correctly recognized the current corpus is small, with only a limited set of ingested documents.
- Choosing `0.6 dense / 0.4 lexical` as a **provisional engineering default** is reasonable because it tied on expected outcomes and keeps dense retrieval slightly dominant.

**What still needs discipline:**
- Do not call `0.6 / 0.4` the “best” weight. It is only a provisional default.
- It must remain configurable and easy to change.
- It must be revisited after expanding the corpus and evaluation set.

**Stronger interview-quality answer:**
Given the current small corpus and tied experiment results, we can use `0.6 dense / 0.4 lexical` as a provisional default because it preserves dense semantic strength while adding lexical signal. However, it should not be treated as globally optimal. The weights should be config-driven and revisited after adding more release-lineage, exact-identifier, table-heavy, unsupported-content, and paraphrase evaluation cases.

---

## Step 61 — Config-driven provisional hybrid retrieval defaults

### Question 1
Why should the provisional `0.6 dense / 0.4 lexical` choice be config-driven instead of hardcoded?

**Strong answer:**
Because the weight is a provisional engineering default, not a proven global optimum. Making it config-driven lets us change weights, rerun experiments, or switch retrieval modes without editing source code. This is important as the corpus and evaluation set grow.

### Question 2
Why is `RETRIEVAL_MODE` useful even before wiring hybrid into answer generation?

**Strong answer:**
It defines the runtime contract early and makes the intended retrieval strategy explicit. Even before the query path uses it, tests and configuration can validate supported modes such as `dense`, `lexical`, and `hybrid`, reducing ambiguity before integration.

### Question 3
Why should invalid retrieval modes fail explicitly?

**Strong answer:**
Invalid retrieval modes should fail explicitly because silently falling back to another mode can hide configuration mistakes. In retrieval systems, using the wrong mode can change evidence selection, citations, and answer safety, so bad config should be visible immediately.

### Question 4
Why must hybrid weights reject the `0 / 0` case?

**Strong answer:**
If both weights are zero, hybrid scoring has no signal from either retriever. Allowing that would produce meaningless rankings. Explicit validation prevents silent ranking bugs and makes invalid experiments fail early.

### Question 5
Why did we not wire answer generation to hybrid in this step?

**Strong answer:**
This step only makes the retrieval defaults explicit and validated. Wiring answer generation changes runtime behavior and should be done separately with tests to ensure evidence sufficiency, citations, refusal behavior, and trace logging still work correctly. One step at a time avoids hidden regressions.

---

## Step 61 — Evaluation of user answers

### User answer review 1 — Config-driven provisional weight
**What was correct:**
- You correctly said configuration makes the setting easier to maintain and change.

**What was weak or incomplete:**
- This is not about model settings only; it is specifically retrieval behavior.
- You should mention reversibility and avoiding hardcoded experimental conclusions.

**Stronger interview-quality answer:**
The `0.6 dense / 0.4 lexical` choice should be config-driven because it is only a provisional retrieval default, not a proven global optimum. Keeping it in config makes the decision reversible, easy to experiment with, and safe to change across environments without editing source code.

### User answer review 2 — Why `RETRIEVAL_MODE` is useful before wiring
**What was correct:**
- You correctly connected retrieval mode to debugging and knowing which mode was used.

**What was weak or incomplete:**
- Before answer generation uses it, `RETRIEVAL_MODE` mainly defines the runtime contract and lets us validate supported values.

**Stronger interview-quality answer:**
`RETRIEVAL_MODE` is useful before wiring because it defines the runtime contract early. It makes the intended retrieval strategy explicit and lets the system validate supported modes like `dense`, `lexical`, and `hybrid`. Later, when answer generation uses the retrieval router, traces can record which mode selected the evidence.

### User answer review 3 — Invalid retrieval modes fail explicitly
**What was correct:**
- Correct. You identified that invalid cases should hard fail to avoid bad outputs.

**What could be stronger:**
- Mention silent fallback risk.

**Stronger interview-quality answer:**
Invalid retrieval modes should fail explicitly because silent fallback can hide configuration mistakes. If the system accidentally runs dense retrieval when hybrid was intended, or vice versa, evidence selection and citations can change. Explicit failure makes the misconfiguration visible immediately.

### User answer review 4 — What `0 / 0` hybrid weights means
**What was missing:**
- You asked for explanation, which is better than guessing.

**Explanation:**
`0 / 0` means:
```text
HYBRID_DENSE_WEIGHT=0
HYBRID_LEXICAL_WEIGHT=0
```

That would make the hybrid score meaningless:
```python
hybrid_score = dense_score * 0 + lexical_score * 0
```

Every result would get zero contribution from both retrievers. The ranking would become arbitrary or misleading.

**Stronger interview-quality answer:**
Hybrid weights must reject `0 / 0` because that means neither dense nor lexical retrieval contributes to the final score. A hybrid score with both weights set to zero has no retrieval signal and can produce meaningless rankings. Explicit validation prevents silent ranking bugs.

### User answer review 5 — Why answer generation was not wired yet
**What was missing:**
- You asked for explanation, which is appropriate.

**Explanation:**
Changing answer generation to use hybrid is a runtime behavior change. It affects:
- which chunks are retrieved
- evidence sufficiency
- citations
- refusal behavior
- answer traces
- token usage
- cost and latency

Step 61 only created validated configuration. Step 62 should add a tested retrieval router first. After that, answer generation can use the router safely.

**Stronger interview-quality answer:**
We did not wire answer generation to hybrid in Step 61 because configuration and runtime behavior should be changed separately. Answer generation depends on retrieved evidence, citations, sufficiency checks, and trace logging. If we changed all of that at once, debugging regressions would be harder. The safer path is to first create a tested retrieval router, then wire answer generation to that router in a later step.

---

## Step 62 — Retrieval mode service/router

### Question 1
Why is the retrieval router dependency-injected instead of directly constructing Qdrant and OpenAI clients inside it?

**Strong answer:**
Dependency injection keeps the router focused on routing logic rather than infrastructure setup. It makes the router easy to test with fake dense and lexical search functions, avoids real API/vector-store dependencies in unit tests, and keeps client construction in higher-level orchestration code.

### Question 2
Why does hybrid mode use `max(limit, hybrid_candidate_limit)` for base retrievers?

**Strong answer:**
Hybrid fusion needs enough candidates from each base retriever before it can combine scores. If the final limit is smaller than the candidate limit, we still want more candidates for fusion. If the final limit is larger, the candidate limit must not accidentally return fewer candidates than requested.

### Question 3
Why does the router normalize all routed results to `QdrantSearchResult`?

**Strong answer:**
Dense, lexical, and hybrid result classes may differ internally, but downstream evaluation, citation, and answer-generation code expects a common result shape. Normalizing to `QdrantSearchResult` keeps downstream code simpler and avoids mode-specific handling.

### Question 4
Why is this router not yet wired into answer generation?

**Strong answer:**
The router itself changes only retrieval selection. Wiring it into answer generation changes runtime behavior for evidence sufficiency, citations, refusal handling, traces, and possibly cost/latency. Keeping those changes separate makes regression testing and debugging safer.

### Question 5
What should the next service layer do after this router?

**Strong answer:**
The next service layer should build real dense and lexical search callables using Qdrant, embeddings, and retrieval-ready artifacts, then call the router according to `RETRIEVAL_MODE`. That service becomes the integration point between configuration and actual query retrieval.

---

## Step 62 — Evaluation of user answers

### User answer review 1 — Dependency injection in the retrieval router
**What was correct:**
- You correctly identified modularity and calling the required retrieval mode as needed.

**What was weak or incomplete:**
- The stronger reason is testability and separation of infrastructure from routing logic.
- The router should not own OpenAI/Qdrant construction because that would make unit tests slower, more fragile, and harder to isolate.

**Stronger interview-quality answer:**
The retrieval router is dependency-injected so it only handles routing logic. Dense and lexical search functions are passed in from outside, which makes the router easy to test without real OpenAI calls, Qdrant setup, or local artifact dependencies. Infrastructure setup belongs in a higher-level retrieval service.

### User answer review 2 — Why hybrid uses `max(limit, hybrid_candidate_limit)`
**What was missing:**
- You asked for explanation, which is better than guessing.

**Explanation:**
`limit` is the final number of results the caller wants. `hybrid_candidate_limit` is how many candidates each base retriever should fetch before fusion.

Example:
```text
limit = 5
hybrid_candidate_limit = 10
```

Hybrid fetches 10 dense + 10 lexical candidates, fuses them, then returns top 5. This gives fusion more evidence to work with.

But if:
```text
limit = 20
hybrid_candidate_limit = 10
```

Then using only 10 candidates would be wrong because the caller asked for 20 final results. So we use:
```python
candidate_limit = max(limit, hybrid_candidate_limit)
```

**Stronger interview-quality answer:**
Hybrid uses `max(limit, hybrid_candidate_limit)` because fusion needs enough candidates from each base retriever, but it should never fetch fewer candidates than the final requested limit. If final `limit` is 5 and candidate limit is 10, we fetch 10 candidates for better fusion. If final `limit` is 20, we fetch at least 20 candidates so the router can return the requested number.

### User answer review 3 — Why normalize routed results to `QdrantSearchResult`
**What was missing:**
- You asked for explanation, which is appropriate.

**Explanation:**
Dense, lexical, and hybrid results may come from different classes:
```text
QdrantSearchResult
LexicalSearchResult
HybridSearchResult
```

But downstream code should not need three branches for three result types. The system already uses `QdrantSearchResult`-like objects for:
- evaluation
- citation building
- evidence sufficiency
- answer generation inputs

So the router normalizes all modes into one common result shape.

**Stronger interview-quality answer:**
The router normalizes all routed results to `QdrantSearchResult` so downstream code can treat dense, lexical, and hybrid retrieval outputs the same way. This avoids mode-specific branching in evaluation, citation generation, evidence sufficiency, and answer-generation orchestration.

### User answer review 4 — Why router is not yet wired into answer generation
**What was correct:**
- You correctly said wiring will happen later.

**What was weak or incomplete:**
- You should explain why separating it matters: answer generation has safety-critical dependencies.

**Stronger interview-quality answer:**
The router is not wired into answer generation yet because that would change runtime evidence selection and could affect sufficiency checks, citations, refusal behavior, traces, latency, and cost. The safer sequence is to test the router first, then build a query retrieval service, and only then wire answer generation to that tested service.

### User answer review 5 — What the next service layer should do
**What was correct:**
- You correctly recognized that retrieved evidence eventually feeds answer generation.

**What was wrong or too far ahead:**
- The next layer should not send directly to another LLM yet.
- First, it must build real dense and lexical search callables and invoke the router.

**Stronger interview-quality answer:**
The next service layer should construct real dense and lexical search callables using Qdrant, query embeddings, and retrieval-ready artifact search. It should load validated retrieval config, call the router, and return routed retrieval results. Only after that service is tested should answer generation consume its output.

---

## Step 63 — Query retrieval service using the router

### Question 1
Why does the query retrieval service build dense and lexical callables instead of directly branching on retrieval mode itself?

**Strong answer:**
The router already owns mode selection. The service should focus on wiring real dependencies into reusable dense and lexical callables. This keeps routing logic centralized and prevents duplicate `if dense/lexical/hybrid` branching across the codebase.

### Question 2
Why is it important that dense and lexical callables use the same query and metadata filters?

**Strong answer:**
Fair hybrid retrieval requires both base retrievers to search the same query scope. If dense and lexical use different filters, releases, document families, or source kinds, their results are not comparable and fusion may mix inconsistent evidence.

### Question 3
Why does this service still avoid calling the LLM?

**Strong answer:**
This service is responsible only for evidence retrieval. LLM answering depends on sufficiency checks, prompts, citations, usage tracking, cost tracking, and trace logging. Keeping retrieval separate makes it easier to test and debug before changing answer behavior.

### Question 4
Why do we test dense, lexical, and hybrid modes separately in the service tests?

**Strong answer:**
Each mode exercises a different dependency path: dense uses query embeddings and Qdrant, lexical uses local retrieval-ready artifacts, and hybrid uses both plus fusion. Testing them separately catches mode-specific integration bugs before the service is used by answer generation.

### Question 5
What should be wired next after this service?

**Strong answer:**
The next step should update a manual query/search or answer smoke-test path to call this retrieval service, then verify evidence sufficiency, citation building, refusal behavior, and answer traces still work correctly with routed retrieval results.

---

## Step 63 — Evaluation of user answers

### User answer review 1 — Why build callables instead of branching directly
**What was correct:**
- You correctly identified modularity and maintainability.

**What was weak or incomplete:**
- The stronger point is separation of responsibilities: the router owns mode selection; the service owns dependency wiring.

**Stronger interview-quality answer:**
The query retrieval service builds dense and lexical callables because the router already owns retrieval-mode selection. The service's job is to wire real dependencies like Qdrant, query embeddings, artifact paths, and metadata filters into those callables. This keeps routing logic centralized and avoids duplicating dense/lexical/hybrid branching in multiple places.

### User answer review 2 — Same query and metadata filters
**What was correct:**
- Correct. You understood dense and lexical outputs are merged after normalization, so they must be based on the same query and filters.

**What could be stronger:**
- Mention evidence consistency and avoiding cross-release/source-kind mixing.

**Stronger interview-quality answer:**
Dense and lexical callables must use the same query and metadata filters so the results are comparable before fusion. If dense searches R24 paragraphs but lexical searches all releases or tables, hybrid fusion could mix inconsistent evidence. Same query and filters preserve release/source constraints and make fusion fair.

### User answer review 3 — Why the service avoids LLM calls
**What was correct:**
- You correctly connected this to modularity, logging, fixes, and isolated testing.

**What was weak or incomplete:**
- You should explicitly say retrieval and generation are separate stages with different responsibilities and failure modes.

**Stronger interview-quality answer:**
The query retrieval service avoids LLM calls because retrieval and generation are separate stages. Retrieval finds evidence; generation uses evidence to answer. Keeping them separate makes retrieval easier to test, debug, log, and evaluate without involving prompt behavior, token usage, cost, or citation validation.

### User answer review 4 — Why test dense, lexical, and hybrid modes separately
**What was correct:**
- You correctly said separate testing lets us compare behavior.

**What was weak or incomplete:**
- The point is not only “what fits best”; each mode has a different dependency path and failure mode.

**Stronger interview-quality answer:**
We test dense, lexical, and hybrid modes separately because each mode exercises different dependencies and failure modes. Dense uses query embeddings and Qdrant, lexical uses local retrieval-ready artifacts, and hybrid uses both plus fusion. Separate tests catch mode-specific bugs before answer generation depends on this service.

### User answer review 5 — What should be wired next
**What was missing:**
- You asked for production guidance, which is the right move.

**Production-scale sequence:**
1. Wire the manual query search script to use `retrieve_query_evidence`.
2. Keep evidence sufficiency checks after retrieval.
3. Log retrieval mode, weights, candidate limit, filters, and top evidence.
4. Then update answer smoke test to use the retrieval service.
5. Only after that, wire the API/UI layer.

**Stronger interview-quality answer:**
After the query retrieval service, the next step should update the manual query search script to use `retrieve_query_evidence`. That lets us verify routed dense/lexical/hybrid retrieval, evidence sufficiency, filters, and logging before changing answer generation. In production-minded systems, manual query inspection comes before API/UI wiring because it is easier to debug retrieval failures at the CLI level.

---

## Step 64 — Manual query search script uses retrieval service

### Question 1
Why was the manual query search script updated before the answer smoke test?

**Strong answer:**
Manual query search is the lowest-risk place to verify routed retrieval behavior. It lets us inspect retrieval mode, filters, scores, and evidence sufficiency before changing LLM answer generation, citations, refusals, and traces.

### Question 2
Why does lexical-only mode not require a Qdrant collection?

**Strong answer:**
Lexical-only retrieval reads local `.retrieval_ready.json` artifacts and does not need query embeddings or vector search. Dense and hybrid modes require Qdrant because they use dense vector retrieval.

### Question 3
Why should the query search script log retrieval mode and hybrid weights?

**Strong answer:**
Retrieval mode and weights determine which evidence is selected. Logging them makes manual debugging reproducible and helps explain why a query returned certain chunks, especially when comparing dense, lexical, and hybrid behavior.

### Question 4
Why should evidence sufficiency still run after routed retrieval?

**Strong answer:**
Changing retrieval mode does not guarantee the evidence is answerable. Dense, lexical, and hybrid retrieval can all return weak or irrelevant neighbors. Evidence sufficiency remains the safety gate before any answer generation.

### Question 5
What should be updated next after manual query search uses the retrieval service?

**Strong answer:**
The answer smoke-test script should use the retrieval service next. That will verify that routed retrieval still works with sufficiency checks, grounded prompt construction, citations, refusal behavior, usage/cost logging, and answer trace persistence.

---

## Step 64 — Evaluation of user answers

### User answer review 1 — Why manual query search before answer smoke test
**What was correct:**
- Correct. You identified modularization and debugging clarity.
- You correctly said we should validate the base retrieval behavior before moving to the next stage.

**What could be stronger:**
- Mention that manual query search isolates retrieval from LLM generation.

**Stronger interview-quality answer:**
The manual query search script was updated before the answer smoke test because it is the lowest-risk place to validate routed retrieval. It lets us inspect retrieval mode, filters, scores, and sufficiency without involving LLM generation, citations, usage tracking, or answer traces. This makes retrieval bugs easier to isolate.

### User answer review 2 — Why lexical-only mode does not require Qdrant
**What was correct:**
- You remembered that lexical has misses, but that is not the reason it does not require Qdrant.

**What was wrong:**
- Lexical-only mode does not require Qdrant because it does not use vector search at all.
- It searches local `.retrieval_ready.json` artifacts.

**Stronger interview-quality answer:**
Lexical-only mode does not require a Qdrant collection because it searches local `.retrieval_ready.json` artifacts using token matching. It does not embed the query and does not run vector search. Dense and hybrid modes require Qdrant because they use dense vector retrieval.

### User answer review 3 — Logging retrieval mode and hybrid weights
**What was correct:**
- Correct. You connected logging to debugging and future weight tuning.

**What could be stronger:**
- Mention reproducibility: without logged mode/weights, we cannot explain why a query returned certain evidence.

**Stronger interview-quality answer:**
The query search script should log retrieval mode and hybrid weights because those settings directly affect evidence selection. If a result looks wrong, we need to know whether it came from dense, lexical, or hybrid retrieval and what weights were active. This makes manual debugging and experiment reproduction possible.

### User answer review 4 — Evidence sufficiency after routed retrieval
**What was correct:**
- You correctly recognized that retrieval comes first and evidence testing comes after.

**What was weak or incomplete:**
- You should explicitly say sufficiency is a safety gate regardless of retrieval mode.

**Stronger interview-quality answer:**
Evidence sufficiency should still run after routed retrieval because changing retrieval mode does not guarantee the evidence is answerable. Dense, lexical, and hybrid retrieval can all return weak, irrelevant, or unsupported evidence. Sufficiency remains the safety gate before answer generation.

### User answer review 5 — What should be updated next
**What was correct:**
- You asked for guidance instead of guessing.

**Clarification:**
We do **not** need to update evidence sufficiency yet. It already works on the normalized routed result shape.

The next best step is to update the answer smoke-test script because that is the smallest end-to-end answer path. It will verify that retrieval service output still works with:
- evidence sufficiency
- grounded prompt construction
- citation generation
- citation validation
- usage/cost logging
- answer trace export

**Stronger interview-quality answer:**
The next step should update the answer smoke-test script to use `retrieve_query_evidence`. Evidence sufficiency should remain in place after retrieval. This lets us test the full retrieval-to-answer flow with routed retrieval before changing FastAPI, UI, or broader application behavior.

---

## Step 65 — Answer smoke-test script uses retrieval service

### Question 1
Why is the answer smoke-test script a better next integration point than the FastAPI or UI layer?

**Strong answer:**
The answer smoke-test script is the smallest end-to-end answer path. It verifies routed retrieval, evidence sufficiency, grounded prompt construction, citation handling, usage/cost logging, and trace export without adding API/UI complexity. This isolates retrieval-to-answer bugs before they become harder to debug in a user-facing layer.

### Question 2
Why must evidence sufficiency remain after routed retrieval instead of being replaced by retrieval mode selection?

**Strong answer:**
Retrieval mode only controls how evidence is selected. It does not prove the evidence is sufficient, relevant, or answerable. Dense, lexical, and hybrid retrieval can all return weak or misleading chunks, so sufficiency remains the safety gate before generation.

### Question 3
Why does the answer smoke-test script still create a Qdrant client even when lexical-only mode does not require a collection?

**Strong answer:**
The retrieval service signature currently accepts a Qdrant client because it supports dense, lexical, and hybrid modes through one interface. In lexical-only mode the collection existence check is skipped, so the client is not used for vector search. This keeps the integration simple for now, but a later refinement could lazily create Qdrant only for dense/hybrid modes.

### Question 4
What production problem is solved by logging retrieval mode, hybrid weights, candidate limit, filters, and answer trace path?

**Strong answer:**
Those logs make answer runs reproducible and debuggable. If an answer is wrong, we need to know the active retrieval mode, fusion weights, candidate pool size, metadata filters, and saved trace artifact to determine whether the failure came from retrieval, sufficiency, prompting, citation handling, or missing evidence.

### Question 5
What failure did the Step 65 tests expose, and what broader testing lesson does it teach?

**Strong answer:**
The first targeted test exposed a brittle CLI help assertion: argparse wrapped the description and split the expected phrase across lines. The broader lesson is that tests should verify meaningful behavior without depending on unstable formatting details like terminal wrapping, timestamps, UUIDs, or log layout unless those are the actual contract.

---

## Step 65 — Evaluation of user answers

### User answer review 1 — Why answer smoke-test before API/UI
**What was correct:**
- Directionally correct. You recognized that catching answer-generation mistakes earlier prevents them from becoming harder to isolate later.
- You correctly implied that debugging at this smaller integration point avoids disturbing other layers.

**What needs improvement:**
- Say explicitly that the smoke script is the smallest end-to-end retrieval-to-answer path.
- Mention that FastAPI/UI adds unrelated complexity: HTTP contracts, request validation, frontend state, UI rendering, and user interaction.

**Stronger interview-quality answer:**
The answer smoke-test script is a better next integration point because it is the smallest end-to-end answer path. It verifies routed retrieval, evidence sufficiency, grounded answer generation, citation handling, usage/cost logging, and trace export without adding FastAPI or UI complexity. This makes retrieval-to-answer bugs easier to isolate before exposing the flow through user-facing layers.

### User answer review 2 — Why sufficiency remains after routed retrieval
**What was correct:**
- You understood that retrieval happens before evidence sufficiency.

**What was weak or incorrect:**
- Your wording was unclear: “Once the route has been fixed then we should go for the evidence suffice” does not explain the safety reason.
- The core point is not just sequence. The core point is that retrieval mode does not prove answerability.

**Stronger interview-quality answer:**
Evidence sufficiency must remain after routed retrieval because routing only decides how chunks are retrieved: dense, lexical, or hybrid. It does not prove the retrieved chunks are relevant, strong enough, or actually answer the question. Sufficiency is the safety gate that prevents weak retrieval results from being passed to the LLM and becoming hallucinated answers.

### User answer review 3 — Why Qdrant client is still created in lexical-only mode
**What was correct:**
- You correctly associated dense retrieval with the vector DB.

**What was wrong or incomplete:**
- In lexical-only mode, retrieval does **not** fetch from the vector DB.
- Lexical-only retrieval searches local `.retrieval_ready.json` artifacts.
- The script still creates a Qdrant client because the current shared retrieval service interface accepts a client for all modes. The collection existence check is skipped for lexical-only mode, so vector search is not required.

**Stronger interview-quality answer:**
The answer smoke-test script still creates a Qdrant client because `retrieve_query_evidence(...)` currently has one shared interface for dense, lexical, and hybrid modes, and that interface accepts a Qdrant client. However, lexical-only mode does not use Qdrant for retrieval; it searches local `.retrieval_ready.json` artifacts. The script skips the collection existence check for lexical mode, so a missing Qdrant collection does not block lexical-only retrieval. A future optimization could lazily create the Qdrant client only for dense/hybrid modes.

### User answer review 4 — Why detailed retrieval and trace logging matters
**What was correct:**
- Correct. You recognized that logging is critical for production debugging.
- You correctly said logs help pinpoint which stage caused the problem.

**What needs improvement:**
- Be more specific. In RAG, logs must help distinguish retrieval configuration failures, weak evidence, prompt/generation issues, citation problems, and missing corpus evidence.
- Mention reproducibility: without mode/weights/filters/trace path, you cannot recreate the answer run.

**Stronger interview-quality answer:**
Logging retrieval mode, hybrid weights, candidate limit, filters, and answer trace path makes answer runs reproducible and debuggable. If an answer is wrong, those logs help identify whether the failure came from retrieval configuration, weak evidence, prompt behavior, citation validation, or missing indexed content. Without these logs, debugging becomes guesswork.

### User answer review 5 — Step 65 test failure and broader testing lesson
**What was missing:**
- You did not answer the question. Asking for explanation is acceptable, but this must become part of your interview vocabulary.

**Explanation:**
The first targeted Step 65 test failed because the test asserted that the help output contained the exact phrase:

```text
configured retrieval mode
```

But `argparse` wrapped the longer description across lines, so the phrase was split and the assertion failed even though the CLI help itself worked. This is a brittle test: it depends on formatting behavior rather than meaningful product behavior.

**Broader testing lesson:**
Do not write tests that rely on unstable formatting details unless formatting is the contract. CLI wrapping, timestamps, UUIDs, log spacing, and ordering can change without breaking real behavior. Tests should focus on stable behavior: command runs, expected options exist, correct service is called, sufficiency is preserved, client cleanup happens, and invalid modes fail safely.

**Stronger interview-quality answer:**
The Step 65 tests exposed a brittle CLI help assertion. The script help worked, but `argparse` wrapped the longer description and split the expected phrase across lines. The broader lesson is that tests should validate meaningful behavior rather than unstable formatting details. In production-minded tests, we should avoid depending on terminal wrapping, timestamps, UUIDs, or log formatting unless those are explicit contracts.

### Required revision before moving on
Revise answers **2, 3, and 5** in your own words before we continue to the next implementation step. Keep them precise and production-minded.

---

## Step 65 — Evaluation of revised user answers

### Revised answer review 2 — Evidence sufficiency after routed retrieval
**Your revised answer:**
> Retreival is the candidate selection and evidence sufficiency is the safety gate, without that our answers would be weak and same would be send to LLM.

**What was correct:**
- Much stronger. You now stated the key distinction: retrieval is candidate selection, sufficiency is the safety gate.
- You correctly connected missing sufficiency to weak evidence being sent to the LLM.

**What could still be sharper:**
- Say “weak or irrelevant evidence” rather than only “weak answers.” The weakness starts at the evidence layer.
- Mention that this applies regardless of dense, lexical, or hybrid mode.

**Interview-quality version:**
Evidence sufficiency must remain after routed retrieval because retrieval is only candidate selection. Dense, lexical, or hybrid retrieval can still return weak, irrelevant, or incomplete chunks. Sufficiency is the safety gate that prevents poor evidence from being sent to the LLM and becoming a hallucinated or unsupported answer.

### Revised answer review 3 — Lexical-only retrieval and Qdrant client
**Your revised answer:**
> Because lexical uses local .retreival_ready.json so if the things are found here we can fetch without creating the client

**What was correct:**
- Correct core concept: lexical retrieval uses local `.retrieval_ready.json` artifacts, not Qdrant vector search.

**What was still wrong or incomplete:**
- In the current implementation, the smoke script **still creates** a Qdrant client because `retrieve_query_evidence(...)` has one shared interface for dense, lexical, and hybrid modes.
- The important point is that lexical mode does **not require a Qdrant collection** and does **not use Qdrant for vector retrieval**.
- “Without creating the client” is a future optimization, not what the current code does.

**Interview-quality version:**
Lexical-only retrieval does not depend on Qdrant because it searches local `.retrieval_ready.json` artifacts using token matching. The current smoke script still creates a Qdrant client because the shared retrieval service interface accepts a client for all modes, but lexical mode skips the collection requirement and does not run vector search. Later, we could optimize this by lazily creating Qdrant only for dense or hybrid modes.

### Revised answer review 4 — Production debugging logs
**Your revised answer:**
> It helps deep dive into the debugging, to answer specifics to understand and postmortem the things in detail. We get the weak evidences, filters, meta data it can help to tinker the things and move foward

**What was correct:**
- Better. You connected logging to postmortem analysis and debugging specific parts of the system.
- You mentioned weak evidence, filters, and metadata, which are the right RAG debugging signals.

**What could still be sharper:**
- “Tinker” is too informal for interviews. Use “adjust,” “diagnose,” “reproduce,” or “tune.”
- Mention exact failure categories: retrieval configuration, candidate limit, filters, evidence sufficiency, prompt/citation issues, and missing indexed content.

**Interview-quality version:**
Logging retrieval mode, hybrid weights, candidate limit, filters, metadata, and trace path supports reproducible debugging and postmortem analysis. If an answer is wrong, we can diagnose whether the issue came from retrieval configuration, weak evidence, incorrect filters, prompt behavior, citation validation, or missing indexed content. This lets us fix the specific failure point without changing unrelated parts of the architecture.

### Remaining required answer before Step 66
You still need to answer Step 65 Question 5 in your own words:

**What failure did the Step 65 tests expose, and what broader testing lesson does it teach?**

Your answer must mention:
- brittle CLI help assertion
- `argparse` line wrapping
- testing stable behavior instead of unstable formatting details

### Final revised answer review 5 — Step 65 test failure and testing lesson
**Your revised answer:**
> The targeted test exposed a lower CLI help text caused by argparse line wraps. We need to verify this behaviour to avoid unstable formattings

**What was correct:**
- Accepted. You identified the core issue: the CLI help assertion was brittle because `argparse` line wrapping changed the output format.
- You correctly connected this to avoiding unstable formatting assumptions in tests.

**What needs improvement:**
- Use “brittle CLI help assertion,” not “lower CLI help text.”
- Say “stable behavior” explicitly: command runs, expected options exist, service wiring works, and cleanup happens.

**Interview-quality version:**
The Step 65 targeted test exposed a brittle CLI help assertion. The script help worked, but `argparse` wrapped the description and split the expected phrase across lines. The broader lesson is that tests should validate stable behavior, not unstable formatting details like terminal wrapping, timestamps, UUIDs, or log layout unless those details are the actual contract.

### Step 65 answer gate status
Accepted. Step 65 interview review is complete, and we can move to the next implementation step.

---

## Step 66 — Reusable answer orchestration service

### Question 1
Why did we add an answer orchestration service instead of jumping directly to FastAPI or Streamlit?

**Strong answer:**
Because no API/UI query path exists yet, and the retrieval-to-answer flow should be reusable and tested before exposing it through user-facing layers. The orchestration service gives API, CLI, and future UI code one backend contract for routed retrieval, sufficiency, grounded generation, and trace writing.

### Question 2
Why should API/UI code not duplicate retrieval, sufficiency, answer generation, and trace logic?

**Strong answer:**
Duplicating orchestration logic across API, CLI, and UI creates inconsistent behavior and makes bugs harder to fix. A shared service keeps the safety gate, citations, refusals, and trace artifacts consistent across entrypoints.

### Question 3
Why does `run_grounded_answer_query(...)` accept injected embedding and LLM clients?

**Strong answer:**
Dependency injection keeps the service testable without real API calls and lets higher-level infrastructure decide how clients are created. This separates orchestration behavior from external service setup and makes unit tests faster, cheaper, and more reliable.

### Question 4
Why is retrieval metadata now stored in the answer trace?

**Strong answer:**
Retrieval metadata makes answer runs reproducible and debuggable. If an answer is wrong, we need to know the retrieval mode, hybrid weights, candidate limit, result limit, and sufficiency thresholds used during that run to diagnose whether the failure came from retrieval configuration, evidence quality, or generation.

### Question 5
Why must the orchestration service still write a trace when evidence is insufficient?

**Strong answer:**
Insufficient-evidence cases are exactly the failures we need to debug. Writing a trace preserves the query, filters, retrieved weak evidence, sufficiency reason, retrieval settings, and refusal response so we can explain why the system abstained and improve retrieval or ingestion if needed.

---

## Step 66 — Evaluation of user answers

### User answer review 1 — Why orchestration service before FastAPI/Streamlit
**What was correct:**
- Correct. You understood that the retrieval-to-answer flow should be tested and reusable before exposing it through FastAPI.

**What could be stronger:**
- Say explicitly that no API/UI query path exists yet.
- Mention that API/UI layers add unrelated complexity and should call a tested backend service rather than own the orchestration.

**Stronger interview-quality answer:**
We added the answer orchestration service before FastAPI or Streamlit because the retrieval-to-answer flow needs one reusable, tested backend contract first. No API/UI query path exists yet, and jumping directly to UI would mix core RAG behavior with HTTP or frontend concerns. The service lets future CLI, API, and UI layers call the same routed retrieval, sufficiency, generation, and trace-writing flow.

### User answer review 2 — Why API/UI should not duplicate orchestration logic
**What was correct:**
- Correct. You identified inconsistency and redundancy as the core risks.
- You correctly recognized that retrieval, sufficiency, and traces already belong in the shared backend flow.

**What could be stronger:**
- Mention safety consistency: refusal behavior, citation handling, and trace format should not differ by entrypoint.

**Stronger interview-quality answer:**
API/UI code should not duplicate retrieval, sufficiency, answer generation, and trace logic because duplication creates inconsistent behavior across entrypoints. If CLI, API, and UI each implement their own flow, refusal behavior, citation handling, thresholds, and trace artifacts can drift. A shared orchestration service keeps the safety gate and debugging artifacts consistent.

### User answer review 3 — Why injected embedding and LLM clients
**What was correct:**
- Good. You understood this keeps the service independent and easier to test.
- You connected the design to future flexibility.

**What could be stronger:**
- Be more precise: dependency injection avoids real API calls in unit tests and lets infrastructure layers own client creation/lifecycle.
- “Top later” should be “top layer.”

**Stronger interview-quality answer:**
`run_grounded_answer_query(...)` accepts injected embedding and LLM clients so the orchestration logic can be tested without real API calls. Higher-level infrastructure can decide how clients are created, configured, retried, and closed. This keeps the service focused on orchestration and makes future API/UI changes possible without rewriting the core query flow.

### User answer review 4 — Why retrieval metadata is stored in traces
**What was correct:**
- Correct. You identified debugging and reproducibility.
- You mentioned weights, thresholds, evidence failures, and quality signals, which are exactly the right categories.

**What could be stronger:**
- Mention mode and candidate limit explicitly because they directly affect which evidence was selected.

**Stronger interview-quality answer:**
Retrieval metadata is stored in the answer trace so every answer run can be reproduced and debugged. If an answer is wrong, we need to know the retrieval mode, hybrid weights, candidate limit, result limit, filters, and sufficiency thresholds. Those details help determine whether the failure came from retrieval configuration, weak evidence, incorrect filtering, or generation.

### User answer review 5 — Why write traces for insufficient evidence
**What was correct:**
- Correct. You understood that insufficient-evidence cases need trace logging because they explain why the system refused.
- You correctly listed filters, evidence, reasoning, settings, and modes as important debugging context.

**What could be stronger:**
- Say explicitly that refusals are not failures to hide; they are production events to audit.
- Mention that traces help improve ingestion/retrieval later.

**Stronger interview-quality answer:**
The orchestration service must still write a trace when evidence is insufficient because refusals are production events that need to be auditable. The trace preserves the query, filters, weak retrieved evidence, sufficiency reason, retrieval settings, and refusal response. This lets us explain why the system abstained and decide whether the issue is missing corpus evidence, retrieval weakness, filtering, or ingestion limitations.

### Step 66 answer gate status
Accepted. You can move to Step 67.

---

## Step 67 — Answer smoke-test script uses answer orchestration service

### Question 1
Why was the answer smoke-test script refactored to call `run_grounded_answer_query(...)` instead of keeping its own retrieval and generation flow?

**Strong answer:**
The script should not duplicate core RAG orchestration logic. By calling the shared service, the smoke test uses the same retrieval, sufficiency, generation, and trace path that future API/UI layers should use, reducing behavior drift across entrypoints.

### Question 2
What responsibilities should remain in the CLI script after this refactor?

**Strong answer:**
The CLI should own CLI-specific concerns: parsing arguments, loading settings, configuring logging, creating/closing the Qdrant client, checking whether the selected mode requires a Qdrant collection, and printing human-readable output. The service should own retrieval-to-answer orchestration.

### Question 3
Why was it important to preserve the mode-aware Qdrant collection check in the script?

**Strong answer:**
Dense and hybrid modes require Qdrant because they use vector retrieval, while lexical-only mode uses local retrieval-ready artifacts and should not fail just because a Qdrant collection is missing. Preserving this check keeps failure behavior correct by retrieval mode.

### Question 4
What production risk is reduced when CLI, future API, and future UI share the same orchestration service?

**Strong answer:**
Sharing one service reduces behavior drift. Without it, CLI might refuse weak evidence while API answers, or traces/citations/thresholds might differ by entrypoint. One service keeps safety, citations, sufficiency, and trace behavior consistent.

### Question 5
What should be the next integration step after this CLI refactor?

**Strong answer:**
The next step should add a minimal FastAPI query contract or schema layer that calls `run_grounded_answer_query(...)`. The API should handle request validation, response formatting, and safe error handling without duplicating retrieval/generation internals.

---

## Step 67 — Evaluation of user answers

### User answer review 1 — Why refactor smoke script to orchestration service
**What was correct:**
- Directionally correct. You understood that once the reusable retrieval-to-answer flow exists, the script should not reinvent it.
- “No point reinventing the wheel” is the right instinct.

**What was weak or incomplete:**
- You said “retrieval has been done,” but the actual refactor is broader than retrieval. It centralizes retrieval, sufficiency, generation, and trace writing.
- You should explicitly mention avoiding behavior drift across CLI/API/UI.

**Stronger interview-quality answer:**
The answer smoke-test script was refactored to call `run_grounded_answer_query(...)` because the script should not duplicate the core RAG orchestration flow. The shared service already owns routed retrieval, evidence sufficiency, grounded answer generation, and trace writing. Calling it from the CLI reduces behavior drift and ensures the CLI exercises the same backend path future API/UI layers should use.

### User answer review 2 — What responsibilities remain in the CLI script
**What was correct:**
- You described why manual inspection is useful before API/UI wiring, which is true.

**What was wrong or off-target:**
- You answered a previous-step question, not this exact one.
- The question asked what the CLI script should still own after orchestration moves into the service.

**Stronger interview-quality answer:**
After the refactor, the CLI script should own CLI-specific responsibilities: parsing command-line arguments, loading settings, configuring logging, creating and closing the Qdrant client, checking whether the selected retrieval mode requires a Qdrant collection, and printing human-readable output. The orchestration service should own retrieval, sufficiency, answer generation, and trace writing.

### User answer review 3 — Why preserve mode-aware Qdrant check
**What was correct:**
- You recognized that logging/knowing which retrieval mode was used matters.

**What was weak or incorrect:**
- The question was not mainly about comparing modes.
- The key reason is dependency correctness: dense/hybrid require Qdrant, lexical-only does not.

**Stronger interview-quality answer:**
The mode-aware Qdrant collection check must be preserved because dense and hybrid retrieval require Qdrant vector search, while lexical-only retrieval uses local `.retrieval_ready.json` artifacts. Lexical-only mode should not fail just because a Qdrant collection is missing. Dense and hybrid should fail early if the collection is missing.

### User answer review 4 — Production risk reduced by shared orchestration service
**What was correct:**
- Strong answer. You connected shared orchestration to modularity, debugging, reproducibility, and fixing failures in the correct layer.
- You correctly described the orchestration service as the engine for multiple flows.

**What could be sharper:**
- Explicitly name behavior drift: CLI, API, and UI should not have different refusal/citation/trace behavior.

**Stronger interview-quality answer:**
When CLI, API, and UI share the same orchestration service, we reduce behavior drift across entrypoints. Without a shared service, one layer might answer weak evidence while another refuses, or citations and traces might differ. A shared orchestration engine keeps retrieval, sufficiency, refusal behavior, citations, and trace logging consistent and reproducible.

### User answer review 5 — Next integration step after CLI refactor
**What was missing:**
- You asked for guidance instead of guessing, which is acceptable.

**Best next step explained:**
The best next step is **not** Streamlit yet. The disciplined next step is a minimal FastAPI query contract or schema layer that calls `run_grounded_answer_query(...)`.

Why:
- Backend API is the boundary between the core RAG engine and any UI.
- API request/response schemas force us to define the product contract clearly.
- API tests can validate request validation, response formatting, and safe error handling without frontend complexity.
- Streamlit should call a stable backend contract, not directly own retrieval/generation behavior.

**Stronger interview-quality answer:**
The next integration step should be a minimal FastAPI query contract or schema layer that calls `run_grounded_answer_query(...)`. The API should handle request validation, response formatting, and safe errors while leaving retrieval, sufficiency, generation, and trace writing inside the orchestration service. Streamlit should come after the backend contract is stable.

### Required revision before Step 68
Revise answers **2, 3, and 5** in your own words before we continue:
1. What responsibilities should remain in the CLI script after this refactor?
2. Why was it important to preserve the mode-aware Qdrant collection check in the script?
3. What should be the next integration step after this CLI refactor?

---

## Step 67 — Evaluation of revised user answers

### Revised answer review 1 — CLI responsibilities
**Your revised answer:**
> It should be able to parse the configs, modes like hybrid/dense, number of top-k for evidence, settings loading,

**What was correct:**
- Better. You identified argument/config parsing, retrieval mode, top-k/limit, and settings loading.

**What was still incomplete:**
- You missed logging setup, Qdrant client lifecycle, mode-aware Qdrant collection check, and human-readable output.
- Also say what the CLI should **not** own anymore: retrieval, sufficiency, generation, and trace writing.

**Interview-quality version:**
The CLI script should own command-line concerns: parsing arguments such as query, mode-related settings, top-k limit, filters, and sufficiency threshold; loading settings; configuring logging; creating and closing the Qdrant client; checking whether the selected retrieval mode requires a Qdrant collection; and printing human-readable results. It should not own retrieval, evidence sufficiency, answer generation, or trace-writing logic because those belong in the orchestration service.

### Revised answer review 2 — Mode-aware Qdrant collection check
**Your revised answer:**
> It is best to have a mode-aware Qdrant as based on our evals we found lexcial isn't the best so moving forward we are using dense or hybrid we can further check and take a call on that

**What was correct:**
- You remembered that the current provisional default is dense/hybrid-oriented rather than lexical-only.

**What was still wrong or incomplete:**
- This answer still misses the main point.
- The check is not about lexical being worse or better.
- The check is about dependency requirements:
  - dense requires Qdrant
  - hybrid requires Qdrant
  - lexical does **not** require Qdrant collection
- Lexical mode is still supported and must not fail just because Qdrant is missing.

**Interview-quality version:**
The mode-aware Qdrant collection check is important because different retrieval modes have different infrastructure dependencies. Dense and hybrid retrieval require Qdrant because they use vector search. Lexical-only retrieval searches local `.retrieval_ready.json` artifacts and should not fail just because a Qdrant collection is missing. This keeps failure behavior correct for each retrieval mode.

### Revised answer review 3 — Next integration step
**Your revised answer:**
> After this we may start building a fastAPI, next LLM output integration, after that create more test cases

**What was correct:**
- Directionally correct. FastAPI is the next integration layer.

**What was still too broad:**
- “FastAPI” is not specific enough.
- “Next LLM output integration” is not the immediate need because answer generation already exists behind the orchestration service.
- The next step should be a **minimal FastAPI query contract/schema layer** that calls `run_grounded_answer_query(...)`.

**Interview-quality version:**
The next integration step should be a minimal FastAPI query contract or schema layer that calls `run_grounded_answer_query(...)`. The API should validate request inputs, format grounded-answer responses, expose citations and trace/debug fields where appropriate, and handle errors safely without duplicating retrieval, sufficiency, generation, or trace-writing internals. More test cases and UI can come after the backend API contract is stable.

### Required final revision before Step 68
Revise answers **2 and 3** one more time in your own words:

1. Why was it important to preserve the mode-aware Qdrant collection check in the script?
2. What should be the next integration step after this CLI refactor?

Your answer for Q2 must mention: `dense/hybrid require Qdrant`, `lexical uses local retrieval-ready artifacts`, and `lexical should not fail if Qdrant collection is missing`.

Your answer for Q3 must mention: `minimal FastAPI query contract/schema`, `calls run_grounded_answer_query`, `request validation`, and `response formatting/safe errors`.

### Final revised answer review 2 — Mode-aware Qdrant collection check
**Your revised answer:**
> Its more important that dense/hybrid requires Qdrant client while lexical uses the local retreival ready json.

**What was correct:**
- Accepted with minor improvement. You now correctly stated the key dependency split: dense/hybrid require Qdrant, while lexical uses local retrieval-ready JSON artifacts.

**What was still missing:**
- Explicitly say lexical-only mode should not fail if the Qdrant collection is missing.

**Interview-quality version:**
The mode-aware Qdrant collection check is important because dense and hybrid retrieval require Qdrant vector search, while lexical-only retrieval uses local `.retrieval_ready.json` artifacts. Lexical-only mode should not fail just because a Qdrant collection is missing, but dense and hybrid should fail early if Qdrant is not ready.

### Final revised answer review 3 — Next integration step
**Your revised answer:**
> It should be the FastAPI with minimal setup to ensure the pipeline or query/retreival things are working

**What was correct:**
- Directionally correct. FastAPI is the right next integration layer.

**What was still incomplete:**
- You did not mention the minimal FastAPI query contract/schema.
- You did not mention that the API should call `run_grounded_answer_query(...)`.
- You did not mention request validation, response formatting, or safe error handling.

**Interview-quality version:**
The next integration step should be a minimal FastAPI query contract or schema layer that calls `run_grounded_answer_query(...)`. The API should validate request inputs, format the grounded answer response with citations and trace/debug information where appropriate, and handle errors safely without duplicating retrieval, sufficiency, generation, or trace-writing internals.

### Required final answer before Step 68
Revise only this answer:

**What should be the next integration step after this CLI refactor?**

Your answer must mention:
- minimal FastAPI query contract/schema
- calls `run_grounded_answer_query(...)`
- request validation
- response formatting and safe errors

### Final answer review — Next integration step
**Your final answer:**
> Yes next integration step after CLI refactor is to proceed with minimal FastAPI with contract/schema which calls run_ground_answer_query, validate request, response formatting, looks for unexpected errors and handle them

**What was correct:**
- Accepted. You now included the key points: minimal FastAPI contract/schema, calling the orchestration service, request validation, response formatting, and safe error handling.

**What to sharpen:**
- Use the exact function name: `run_grounded_answer_query(...)`.
- Say “safe error handling” instead of only “looks for unexpected errors.”

**Interview-quality version:**
The next integration step should be a minimal FastAPI query contract/schema that calls `run_grounded_answer_query(...)`. The API should validate incoming request fields, format grounded-answer responses with citations and trace/debug details where appropriate, and handle errors safely without duplicating retrieval, sufficiency, generation, or trace-writing internals.

### Step 67 answer gate status
Accepted. You can move to Step 68.

---

## Step 68 — Minimal FastAPI query contract

### Question 1
Why should the FastAPI route call `run_grounded_answer_query(...)` instead of directly calling retrieval and LLM services?

**Strong answer:**
The API should be an entrypoint boundary, not a duplicate orchestration layer. Calling `run_grounded_answer_query(...)` keeps retrieval, sufficiency, generation, and trace writing centralized in one tested service, reducing behavior drift between CLI, API, and future UI.

### Question 2
Why is request validation part of the API layer?

**Strong answer:**
The API is the external contract, so it should reject malformed requests before they reach retrieval or generation. Validating blank queries, invalid limits, invalid source kinds, and invalid thresholds prevents avoidable downstream errors and gives clients clear feedback.

### Question 3
Why does dense/hybrid mode return `503` when the Qdrant collection is missing, while lexical-only mode does not?

**Strong answer:**
Dense and hybrid retrieval depend on Qdrant vector search, so a missing collection means the required backend dependency is unavailable and `503 Service Unavailable` is appropriate. Lexical-only retrieval uses local retrieval-ready artifacts, so it should not fail because the Qdrant collection is missing.

### Question 4
Why should unexpected API errors return a generic safe message instead of the raw exception?

**Strong answer:**
Raw exceptions can leak implementation details, file paths, secrets, credentials, or internal architecture. The API should log the detailed exception internally but return a safe generic error to the client.

### Question 5
Why should the API response include trace ID, sufficiency, citations, retrieval mode, and retrieval metadata?

**Strong answer:**
Those fields make answers auditable and debuggable. Clients and developers can see whether the answer was supported, which citations were used, which retrieval mode selected evidence, what thresholds/config were active, and where to find the local trace artifact for postmortem analysis.

---

## Step 68 — Evaluation of user answers

### User answer review 1 — Why API calls orchestration service
**What was correct:**
- Correct. You understood that FastAPI should not duplicate the RAG engine.
- You named the major responsibilities correctly: retrieval, sufficiency, and generation.

**What could be stronger:**
- Also mention trace writing and consistency across CLI/API/future UI.

**Stronger interview-quality answer:**
The FastAPI route should call `run_grounded_answer_query(...)` because the API should not duplicate the RAG orchestration engine. Retrieval, evidence sufficiency, answer generation, and trace writing are already centralized in the orchestration service. The API should act as a validated HTTP boundary and keep behavior consistent with CLI and future UI.

### User answer review 2 — Why request validation belongs in API layer
**What was correct:**
- Correct. “Garbage in, garbage out” is the right instinct.
- You understood invalid requests should be caught before causing downstream errors.

**What could be stronger:**
- Be more specific: blank query, invalid limit, invalid source kind, invalid threshold.

**Stronger interview-quality answer:**
Request validation belongs in the API layer because the API is the external contract. It should reject malformed input such as blank queries, invalid limits, unsupported source kinds, and invalid thresholds before the request reaches retrieval or generation. This gives clients clear feedback and avoids unnecessary downstream failures.

### User answer review 3 — Why dense/hybrid missing Qdrant gives 503 but lexical-only does not
**What was correct:**
- You correctly said dense/hybrid need Qdrant collection access.
- You correctly said lexical uses local retrieval-ready JSON artifacts.
- You correctly connected missing Qdrant for dense/hybrid to `503`.

**What needs correction:**
- Current implementation may still create a Qdrant client in lexical mode because the API follows the shared client lifecycle, but lexical mode skips the **collection check** and does not need Qdrant vector search.
- Do not say lexical does not need to create a Qdrant client as the current implementation guarantee. Say lexical does not require the Qdrant collection/vector search.

**Stronger interview-quality answer:**
Dense and hybrid modes return `503` when the Qdrant collection is missing because they require Qdrant vector search. Lexical-only retrieval uses local `.retrieval_ready.json` artifacts, so it should not fail just because the Qdrant collection is missing. In the current API implementation, a client may still be created, but lexical mode skips the collection check and does not depend on vector retrieval.

### User answer review 4 — Why unexpected errors return generic safe message
**What was correct:**
- Correct. You understood raw exceptions may expose internal details to external clients.
- You connected this to safe API behavior.

**What could be stronger:**
- Mention secrets, file paths, credentials, and implementation details.
- Mention internal logging keeps debuggability while client response stays safe.

**Stronger interview-quality answer:**
Unexpected API errors should return a generic safe message because raw exceptions can leak secrets, credentials, file paths, implementation details, or internal architecture. The server should log the detailed exception internally for debugging, but the client should receive a safe generic error.

### User answer review 5 — Why include trace/debug fields in API response
**What was correct:**
- Correct. You connected trace fields to debugging and production observability.
- You recognized logging and traceability are critical in production systems.

**What could be stronger:**
- Be careful: not all internal logs should be exposed to end users. The API response should expose useful trace/debug identifiers and metadata appropriate for a local/debug mode.
- Mention auditability, citation trust, and reproducing answer runs.

**Stronger interview-quality answer:**
The API response should include trace ID, sufficiency, citations, retrieval mode, and retrieval metadata because those fields make the answer auditable and debuggable. They show whether the answer was supported, which evidence was cited, which retrieval mode selected evidence, and what config influenced the run. In a production system, we should expose only appropriate debug metadata to clients while keeping sensitive internal logs server-side.

### Required revision before Step 69
Revise only Question 3:

**Why does dense/hybrid mode return `503` when the Qdrant collection is missing, while lexical-only mode does not?**

Your answer must mention:
- dense/hybrid require Qdrant vector search
- lexical uses local `.retrieval_ready.json` artifacts
- lexical skips/does not require the Qdrant collection check
- current implementation may still create a client, but lexical does not depend on vector retrieval

### Final revised answer review 3 — Dense/hybrid `503` vs lexical-only behavior
**Your revised answer:**
> implementation distinguishes between semantic/hybrid and lexical retrieval methods. Dense and hybrid modes necessitate a Qdrant vector database, requiring established collections, while pure lexical search operates independently using local .retrieval_ready.json artifacts. The current implementation may initialize a Qdrant client regardless of mode, but lexically-focused pipelines bypass the collection verification step

**What was correct:**
- Accepted. You now correctly explained the infrastructure dependency split.
- You stated dense/hybrid require Qdrant vector search and existing collections.
- You stated lexical uses local `.retrieval_ready.json` artifacts.
- You captured the nuance that a client may still be initialized, but lexical bypasses collection verification.

**What to sharpen:**
- Use “lexical-only retrieval” instead of “lexically-focused pipelines” for clearer interview wording.
- Explicitly connect missing dense/hybrid collection to `503 Service Unavailable`.

**Interview-quality version:**
Dense and hybrid modes return `503` when the Qdrant collection is missing because they require Qdrant vector search, so the required backend dependency is unavailable. Lexical-only retrieval uses local `.retrieval_ready.json` artifacts and does not require the Qdrant collection check. The current implementation may still create a Qdrant client because of the shared API lifecycle, but lexical retrieval does not depend on vector search.

### Step 68 answer gate status
Accepted. You can move to Step 69.

---

## Step 69 — Minimal FastAPI health endpoint

### Question 1
Why should `/health` avoid running retrieval, embeddings, LLM calls, or Qdrant collection checks?

**Strong answer:**
`/health` should be a cheap liveness/config endpoint. If it runs expensive or failure-prone dependencies, it becomes slow and noisy. Full dependency readiness can be a separate endpoint later.

### Question 2
Why is it useful for `/health` to expose active retrieval mode and hybrid weights?

**Strong answer:**
Retrieval mode and weights directly affect answer behavior. Exposing them in health output lets developers and future UI confirm which retrieval configuration is active without running a query.

### Question 3
Why does `/health` report whether Qdrant is required for the current mode instead of checking collection existence?

**Strong answer:**
The endpoint is intentionally lightweight. Reporting whether Qdrant is required explains the dependency expectation without performing vector-store I/O. Actual Qdrant readiness can be checked by a separate readiness endpoint or query flow.

### Question 4
Why should invalid retrieval configuration return a safe generic error from `/health`?

**Strong answer:**
Invalid config details may reveal internal environment values or deployment mistakes. The server should log details internally but return a safe generic message to the client.

### Question 5
What is the difference between a liveness health check and a readiness check?

**Strong answer:**
A liveness check answers “is the backend process running and minimally configured?” A readiness check answers “are all required dependencies ready to serve traffic?” Readiness may include Qdrant collection existence, model API reachability, index health, or artifact availability.

---

## Step 69 — Evaluation of user answers

### User answer review 1 — Why `/health` avoids expensive dependency checks
**What was correct:**
- Correct. You understood `/health` should be a minimal endpoint/config check.
- You correctly connected retrieval, embeddings, LLM calls, and Qdrant checks to latency and unnecessary slowness.

**What could be stronger:**
- Also mention noisy failures: health should not fail just because an optional downstream dependency is temporarily unavailable unless the endpoint is explicitly a readiness check.

**Stronger interview-quality answer:**
`/health` should avoid retrieval, embeddings, LLM calls, and Qdrant collection checks because it is meant to be a cheap liveness/config endpoint. If it calls expensive or failure-prone dependencies, it becomes slow, noisy, and less useful for basic backend monitoring. Full dependency checks should be handled by a separate readiness endpoint.

### User answer review 2 — Why expose retrieval mode and hybrid weights
**What was correct:**
- Correct. You understood that retrieval mode and hybrid weights affect system behavior.
- You correctly connected this to developers and future UI users seeing the active retrieval configuration.

**What could be stronger:**
- Say this helps diagnose environment/config drift without running a query.

**Stronger interview-quality answer:**
It is useful for `/health` to expose retrieval mode and hybrid weights because those settings directly affect which evidence is selected. Developers and future UI clients can confirm the active configuration and detect config drift without running a full query.

### User answer review 3 — Why report Qdrant requirement instead of checking collection existence
**What was correct:**
- Correct. You understood that `/health` should remain lightweight.
- You correctly said Qdrant readiness can be checked by a separate endpoint when needed.

**What could be stronger:**
- Use the terms “liveness” and “readiness” clearly.
- Mention avoiding vector-store I/O in liveness checks.

**Stronger interview-quality answer:**
`/health` reports whether Qdrant is required for the active retrieval mode instead of checking collection existence because it is a lightweight liveness/config endpoint. Checking collection existence would perform vector-store I/O and turn health into a readiness check. A separate readiness endpoint can verify Qdrant, indexes, artifacts, and model APIs later.

### User answer review 4 — Why invalid retrieval config returns safe generic error
**What was correct:**
- Correct. You recognized invalid configuration may reveal internal environment values or developer mistakes.
- You correctly said details should be logged internally while returning a safe generic message to users.

**What could be stronger:**
- Mention secrets, deployment details, and internal config names as examples of sensitive leakage.

**Stronger interview-quality answer:**
Invalid retrieval configuration should return a safe generic error because raw config details can expose environment values, internal deployment mistakes, or sensitive configuration. The server should log details internally for debugging, but clients should receive a safe generic error.

### User answer review 5 — Liveness vs readiness
**What was correct:**
- Correct. You clearly distinguished liveness from readiness.
- You correctly said readiness can include Qdrant, API access, and artifact availability.

**What could be stronger:**
- Tighten wording: liveness checks whether the process is alive; readiness checks whether it can serve real traffic.

**Stronger interview-quality answer:**
A liveness check answers whether the backend process is alive and minimally configured. A readiness check answers whether the service can actually serve traffic, including dependencies such as Qdrant collection/index health, model API availability, and required local artifacts.

### Step 69 answer gate status
Accepted. You can move to Step 70.

---

## Step 70 — API smoke-test client script

### Question 1
Why should the API smoke-test script call `/health` before optionally calling `/query`?

**Strong answer:**
Calling `/health` first verifies the backend is reachable and shows active retrieval configuration before running a potentially expensive query. It separates cheap liveness/config validation from retrieval and LLM work.

### Question 2
Why is `POST /query` optional in the smoke-test script?

**Strong answer:**
Queries may trigger retrieval, embeddings, LLM calls, traces, latency, and cost. Making query execution opt-in prevents accidental model/API usage when the user only wants to verify backend liveness.

### Question 3
Why does the script test the API over HTTP instead of importing and calling Python services directly?

**Strong answer:**
The purpose is to test the API boundary that future UI or external clients will use. Calling Python services directly would bypass request validation, HTTP response formatting, status codes, and API error handling.

### Question 4
Why should HTTP error handling avoid exposing response bodies or raw server details?

**Strong answer:**
Error responses can contain internal details or sensitive strings. The smoke script should report status-level failure clearly without leaking server response bodies into logs or exceptions.

### Question 5
Why should we document API run/smoke commands before building Streamlit?

**Strong answer:**
The backend should be operable and verifiable independently before a UI depends on it. Documentation gives a reproducible way to start the API and test `/health` and `/query`, making later UI debugging easier.

---

## Step 70 — Evaluation of user answers

### User answer review 1 — Why call `/health` before `/query`
**What was correct:**
- Correct. You understood `/health` verifies reachability and active configuration quickly.
- You correctly connected it to a fast check before running heavier query behavior.

**What could be stronger:**
- Mention that this separates cheap liveness/config validation from retrieval/LLM work.

**Stronger interview-quality answer:**
The smoke-test script calls `/health` first to verify the API is reachable and to inspect active configuration before running any expensive query path. This separates cheap liveness/config validation from retrieval, embedding, and LLM work.

### User answer review 2 — Why `/query` is optional
**What was correct:**
- Correct. You clearly identified that `/query` can trigger retrieval, embeddings, LLM calls, and evidence validation.
- You correctly connected optional query execution to avoiding cost when only backend liveness is needed.

**Stronger interview-quality answer:**
`POST /query` is optional because it can trigger retrieval, embedding calls, LLM generation, evidence sufficiency checks, traces, latency, and cost. If the user only wants to verify backend availability, `/health` is enough.

### User answer review 3 — Why test over HTTP instead of importing services
**What was correct:**
- Correct. You understood that direct Python service calls bypass the actual API boundary.
- You correctly named request validation, response formatting, status codes, and API error handling.

**Stronger interview-quality answer:**
The API smoke test uses HTTP because that is the boundary future UI and external clients will use. Importing Python services directly would bypass request validation, JSON serialization, response formatting, status codes, and safe API error handling.

### User answer review 4 — Why avoid exposing response bodies/raw server details
**What was correct:**
- Correct. You identified sensitive internal information as the risk.
- You correctly said a generic failure response is safer.

**What could be stronger:**
- Mention examples: secrets, file paths, stack traces, config values, implementation details.

**Stronger interview-quality answer:**
HTTP error handling should avoid exposing response bodies or raw server details because they may contain secrets, stack traces, file paths, config values, or implementation details. The smoke script should report clear status-level failure without leaking sensitive internals.

### User answer review 5 — Why document API commands before Streamlit / whether to skip Streamlit
**What was correct:**
- Correct. You understood backend API operation should be independently reproducible before UI depends on it.
- Your pushback on Streamlit is reasonable. A browser-accessible FastAPI app plus docs may be enough for a backend-focused portfolio milestone.

**What needs refinement:**
- We do not need to permanently reject Streamlit now. The disciplined decision is to document and harden FastAPI first, then decide whether UI adds value.

**Stronger interview-quality answer:**
We should document API run and smoke-test commands before any UI because the backend must be independently operable and verifiable. If FastAPI already provides a browser-accessible OpenAPI interface and clear smoke tests, we can defer Streamlit until it provides real additional demo value. The immediate priority is a reliable backend contract.

### Step 70 answer gate status
Accepted. You can move to Step 71.

---

## Step 71 — Backend-first API verification documentation

### Question 1
Why is backend run/smoke-test documentation part of the production contract instead of just project polish?

**Strong answer:**
Backend run and smoke-test documentation is part of the production contract because it defines how developers and operators start, verify, and debug the service reproducibly. Without it, failures become ad hoc and UI integration may hide whether the problem is startup, routing, retrieval, Qdrant, LLM calls, or evidence insufficiency.

### Question 2
Why should health-only smoke testing be documented as the first check before query smoke testing?

**Strong answer:**
Health-only smoke testing should come first because it verifies backend reachability and active configuration without triggering retrieval, embeddings, LLM generation, Qdrant checks, trace writing, latency, or cost. It isolates cheap liveness/config validation from expensive query behavior.

### Question 3
Why is it important to document that query smoke testing may incur retrieval, embedding, generation, trace, latency, and cost?

**Strong answer:**
It is important because `POST /query` is not a harmless ping. It may call model APIs, perform retrieval, write local traces, and consume time and money. Documenting this prevents accidental cost and teaches users to treat query execution as an intentional end-to-end RAG test.

### Question 4
Why should the README explain how to interpret `503` in dense/hybrid mode?

**Strong answer:**
Dense and hybrid retrieval depend on Qdrant vector search. If the required collection is unavailable, the API returns `503` because the dependency needed to serve the query is not ready. Documenting this points debugging toward indexing and collection readiness instead of treating the endpoint as randomly broken.

### Question 5
Why should smoke-test client errors avoid printing raw server response bodies?

**Strong answer:**
Raw server response bodies can expose secrets, stack traces, local paths, internal config values, or implementation details. A smoke-test client should report clear status-level failure while keeping sensitive diagnostics in server logs or local trace artifacts.

---

## Step 71 — Evaluation of user answers

### User answer review 1 — Documentation as production contract
**Your answer:**
> Smoketest docs is part of production baseline as it defines the how operation starts, what configurations were defined, how well we can reproduce it. Without it just becomes adhoc, we won't be sure error happened at which layer and difficult to reproduce the configs

**What was correct:**
- Correct. You connected smoke-test documentation to operational reproducibility.
- You correctly said it prevents ad hoc debugging and helps identify which layer failed.
- You correctly mentioned configuration reproducibility.

**What could be stronger:**
- Use cleaner interview language: “how the service starts,” “how it is verified,” and “which dependency or layer failed.”
- Mention UI integration explicitly: backend verification should not depend on a future UI.

**Interview-quality version:**
Backend run and smoke-test documentation is part of the production contract because it defines how the service is started, verified, and debugged reproducibly. Without it, failures become ad hoc and it becomes difficult to tell whether the issue is startup, routing, retrieval, Qdrant readiness, LLM calls, or evidence sufficiency. A UI should depend on a documented backend contract, not become the first debugging tool.

### User answer review 2 — Health-only smoke test first
**Your answer:**
> Because it verifies the backend connectivity, sets the configuration layers wtihout triggering the retrevial, embeddings, LLM, evidence etc it stays light weight and useful

**What was correct:**
- Correct. You said health-only smoke testing verifies backend connectivity and active configuration.
- You correctly said it avoids retrieval, embeddings, LLM work, and evidence processing.
- You understood it should stay lightweight.

**What could be stronger:**
- Say “liveness/config check” explicitly.
- Mention that it separates cheap checks from cost-bearing query behavior.

**Interview-quality version:**
Health-only smoke testing should be the first check because it verifies backend reachability and active configuration as a cheap liveness/config check. It does not trigger retrieval, embeddings, LLM generation, evidence sufficiency checks, trace writing, latency, or cost. This isolates basic backend availability before testing the heavier query path.

### User answer review 3 — Query smoke testing may incur cost
**Your answer:**
> We here are referring to the post query module, where it calls a light API call, performed retrevial, embeds. Documenting this prevents accidental cost and teaches users to treat query execution as an intentional use

**What was correct:**
- You correctly identified that `POST /query` performs retrieval and embeddings.
- You correctly connected documentation to preventing accidental cost.
- You correctly said query execution should be intentional.

**What was weak or incorrect:**
- Do **not** call `POST /query` a “light API call.” It is the heavy RAG path.
- You did not mention LLM generation, trace writing, latency, or local artifacts clearly.
- The important distinction is `/health` is lightweight; `/query` may be cost-bearing.

**Interview-quality version:**
It is important to document query smoke testing because `POST /query` is not a harmless ping. It can run retrieval, call embeddings, call the LLM, perform sufficiency checks, write local answer traces, add latency, and incur API cost. Documenting that boundary prevents accidental spend and makes query smoke testing an intentional end-to-end RAG validation.

### User answer review 4 — Interpreting `503` in dense/hybrid mode
**Your answer:**
> When the required data is unavailable in the corpus we generally give out '503' it serves the purpose that the pipeline was not broken instead the ingestion did not contains the required info, It helps to rectify or understand the system behaviour

**What was correct:**
- You tried to connect status codes to system behavior and debugging.

**What was wrong:**
- This answer is incorrect for the current implementation.
- `503` in dense/hybrid mode does **not** mean the required answer data is missing from the corpus.
- Missing evidence or weak evidence should produce an insufficient-evidence/refusal response, not `503`.
- `503` means a required service dependency is unavailable: the Qdrant collection needed for dense/hybrid vector retrieval is missing.

**Interview-quality version:**
The README should explain `503` in dense/hybrid mode because dense and hybrid retrieval require a Qdrant collection for vector search. If that collection is missing, the backend dependency needed to serve the query is unavailable, so `503 Service Unavailable` is appropriate. This is different from insufficient evidence: if the corpus lacks the answer, the system should return a safe refusal or insufficient-evidence response, not `503`.

### User answer review 5 — Avoid raw server response bodies
**Your answer:**
> Raw server response bodies can expose secrets, stack traces, local paths, internal config values, or implementation details.  So we need to avoid giving out sensitive information

**What was correct:**
- Correct. This was your strongest answer.
- You named the right leakage risks: secrets, stack traces, local paths, config values, and implementation details.

**What could be stronger:**
- Mention the client should report status-level failure while detailed diagnostics stay in server logs/local traces.

**Interview-quality version:**
Smoke-test client errors should avoid printing raw server response bodies because those bodies may contain secrets, stack traces, local file paths, internal config values, or implementation details. The client should report a clear status-level failure while detailed diagnostics remain in server logs or local trace artifacts.

### Required revision before Step 72
Revise answers **3 and 4** only:

1. Why is it important to document that query smoke testing may incur retrieval, embedding, generation, trace, latency, and cost?
2. Why should the README explain how to interpret `503` in dense/hybrid mode?

Your revised answer for Question 3 must mention:
- `POST /query` is not lightweight / not a harmless ping
- retrieval
- embeddings
- LLM generation
- trace writing
- latency/cost

Your revised answer for Question 4 must mention:
- dense/hybrid require Qdrant vector search
- missing Qdrant collection means dependency unavailable
- `503 Service Unavailable`
- missing corpus evidence should be insufficient-evidence/refusal, not `503`

### Revised answer review 3 — Query smoke testing may incur cost
**Your revised answer:**
> It is important because `POST /query` is not a harmless ping. It may call model APIs, perform retrieval, write local traces, and consume time and money. Documenting this prevents accidental cost and teaches users to treat query execution as an intentional end-to-end RAG test.

**What was correct:**
- Much better. You correctly said `POST /query` is not a harmless ping.
- You correctly mentioned retrieval, local trace writing, time, money, accidental cost, and intentional end-to-end RAG testing.

**What is still missing:**
- You did not explicitly mention embeddings.
- You did not explicitly mention LLM generation.
- “Model APIs” is directionally correct, but in an interview you should name the concrete cost-bearing stages.

**Interview-quality version:**
It is important to document query smoke testing because `POST /query` is not a harmless ping. It may run retrieval, call the embedding API, call the LLM for generation, perform evidence sufficiency checks, write local answer traces, add latency, and incur API cost. Documenting this prevents accidental spend and makes query execution an intentional end-to-end RAG validation.

### Revised answer review 4 — Interpreting `503` in dense/hybrid mode
**Your revised answer:**
> Dense and hybrid retrieval depend on Qdrant vector search. If the required collection is unavailable, the API returns `503` because the dependency needed to serve the query is not ready. Documenting this points debugging toward indexing and collection readiness instead of treating the endpoint as randomly broken.

**What was correct:**
- Good correction. You now correctly explained that dense/hybrid retrieval depends on Qdrant vector search.
- You correctly connected a missing collection to an unavailable dependency.
- You correctly connected `503` to indexing and collection readiness debugging.

**What is still missing:**
- You did not explicitly distinguish this from missing corpus evidence.
- Missing answer evidence should result in insufficient-evidence/refusal behavior, not `503`.

**Interview-quality version:**
The README should explain `503` in dense/hybrid mode because dense and hybrid retrieval require Qdrant vector search. If the required Qdrant collection is missing, the dependency needed to serve the query is unavailable, so `503 Service Unavailable` is appropriate. This is different from missing corpus evidence: if retrieval runs but the corpus does not contain enough evidence, the system should return an insufficient-evidence/refusal response, not `503`.

### Required final revision before Step 72
Revise answers **3 and 4 one more time**, but keep them short.

For Question 3, explicitly include:
- embedding API
- LLM generation

For Question 4, explicitly include:
- missing corpus evidence should produce insufficient-evidence/refusal, not `503`

### Final revised answer review 3 — Query smoke testing cost
**Your final revised answer:**
> `POST /query` is not a harmless ping. It may run retrieval, call the embedding API, call the LLM for generation, write local traces, and add latency/cost.

**What was correct:**
- Accepted. You now explicitly named the cost-bearing stages: retrieval, embedding API, LLM generation, local trace writing, latency, and cost.
- This is now production-minded because it distinguishes cheap health checks from intentional end-to-end RAG execution.

**Interview-quality version:**
`POST /query` is not a harmless ping. It may run retrieval, call the embedding API, call the LLM for generation, write local answer traces, and add latency and cost. Therefore query smoke testing should be opt-in and documented as an intentional end-to-end RAG validation.

### Final revised answer review 4 — `503` interpretation
**Your final revised answer:**
> Dense/hybrid require Qdrant vector search. Missing Qdrant collection means dependency unavailable. API returns `503 Service Unavailable`. Missing corpus evidence should produce insufficient-evidence/refusal, not `503`.

**What was correct:**
- Accepted. You now correctly separated infrastructure readiness from evidence sufficiency.
- You stated dense/hybrid require Qdrant vector search.
- You stated a missing Qdrant collection means dependency unavailable and should return `503 Service Unavailable`.
- You correctly stated missing corpus evidence should produce insufficient-evidence/refusal, not `503`.

**Interview-quality version:**
Dense and hybrid retrieval require Qdrant vector search. If the required Qdrant collection is missing, the dependency is unavailable, so the API returns `503 Service Unavailable`. That is different from missing corpus evidence: if retrieval runs but evidence is weak or absent, the system should return an insufficient-evidence/refusal response, not `503`.

### Step 71 answer gate status
Accepted. You can move to Step 72.

---

## Step 72 — Minimal FastAPI readiness endpoint

### Question 1
Why should `/ready` be separate from `/health`?

**Strong answer:**
`/health` should stay a cheap liveness/config check, while `/ready` checks whether required dependencies and artifacts are available to serve real traffic. Separating them prevents basic liveness from becoming slow, noisy, or dependent on optional infrastructure.

### Question 2
Why does `/ready` check Qdrant only for dense and hybrid modes?

**Strong answer:**
Dense and hybrid retrieval require Qdrant vector search, so readiness must verify the required collection exists. Lexical-only retrieval searches local `.retrieval_ready.json` artifacts and should not fail because Qdrant is missing.

### Question 3
Why does `/ready` check `.retrieval_ready.json` artifacts for lexical and hybrid modes?

**Strong answer:**
Lexical retrieval searches local retrieval-ready artifacts. Hybrid retrieval uses lexical retrieval as one of its candidate sources. If those artifacts are missing, lexical/hybrid retrieval cannot serve the active mode correctly.

### Question 4
Why should `/ready` avoid embedding API calls and LLM generation even though it checks model configuration?

**Strong answer:**
Readiness should verify configuration and local dependency availability without creating token usage, latency, cost, or external API noise. A deeper model API reachability check can be separate if needed, but `/ready` should not become an accidental query or generation path.

### Question 5
Why is a `/ready` `503` different from an insufficient-evidence/refusal response from `/query`?

**Strong answer:**
`/ready` returning `503` means a required dependency or artifact is unavailable, such as a missing Qdrant collection or missing retrieval-ready files. An insufficient-evidence/refusal response means the query pipeline ran but retrieved evidence was weak or absent. Infrastructure readiness and answer sufficiency are different failure categories.

---

## Step 72 — Evaluation of user answers

### User answer review 1 — Why `/ready` is separate from `/health`
**Your answer:**
> `/health` should stay a cheap liveness/config check, while `/ready` checks whether required dependencies and artifacts are available to serve real traffic. Separating them prevents basic liveness from becoming slow, noisy, or dependent on optional infrastructure.

**What was correct:**
- Accepted. You clearly separated liveness/config checks from dependency/artifact readiness.
- You correctly explained why `/health` should remain cheap and non-noisy.
- You correctly connected `/ready` to serving real traffic.

**Interview-quality version:**
`/health` should stay a cheap liveness/config check, while `/ready` checks whether required dependencies and artifacts are available to serve real traffic. Separating them prevents basic liveness from becoming slow, noisy, or dependent on optional infrastructure.

### User answer review 2 — Why Qdrant is checked only for dense/hybrid
**Your answer:**
> Dense and hybrid retrieval require Qdrant vector search, so readiness must verify the required collection exists. Lexical-only retrieval searches local `.retrieval_ready.json` artifacts and should not fail because Qdrant is missing.

**What was correct:**
- Accepted. You correctly stated the infrastructure dependency split.
- Dense/hybrid require Qdrant vector search.
- Lexical-only retrieval uses local retrieval-ready artifacts and should not fail because Qdrant is unavailable.

**Interview-quality version:**
Dense and hybrid retrieval require Qdrant vector search, so readiness must verify the required collection exists. Lexical-only retrieval searches local `.retrieval_ready.json` artifacts and should not fail because Qdrant is missing.

### User answer review 3 — Why retrieval-ready artifacts are checked for lexical/hybrid
**Your answer:**
> Lexical retrieval searches local retrieval-ready artifacts. Hybrid retrieval uses lexical retrieval as one of its candidate sources. If those artifacts are missing, lexical/hybrid retrieval cannot serve the active mode correctly.

**What was correct:**
- Accepted. You correctly linked lexical retrieval to local `.retrieval_ready.json` artifacts.
- You correctly explained that hybrid depends on lexical candidates as one source.
- You correctly identified missing artifacts as a readiness failure for lexical/hybrid modes.

**Interview-quality version:**
Lexical retrieval searches local retrieval-ready artifacts. Hybrid retrieval uses lexical retrieval as one of its candidate sources. If those artifacts are missing, lexical or hybrid retrieval cannot serve the active mode correctly.

### User answer review 4 — Why `/ready` avoids embedding API calls and LLM generation
**Your answer:**
> Readiness should verify configuration and local dependency availability without creating token usage, latency, cost, or external API noise. A deeper model API reachability check can be separate if needed, but `/ready` should not become an accidental query or generation path.

**What was correct:**
- Accepted. This is production-minded.
- You correctly separated configuration presence checks from external model API reachability.
- You correctly mentioned token usage, latency, cost, and external API noise.
- You correctly said deeper model reachability can be a separate check later.

**Interview-quality version:**
Readiness should verify configuration and local dependency availability without creating token usage, latency, cost, or external API noise. A deeper model API reachability check can be separate if needed, but `/ready` should not become an accidental query or generation path.

### User answer review 5 — `/ready` `503` vs `/query` insufficient evidence
**Your answer:**
> `/ready` returning `503` means a required dependency or artifact is unavailable, such as a missing Qdrant collection or missing retrieval-ready files. An insufficient-evidence/refusal response means the query pipeline ran but retrieved evidence was weak or absent. Infrastructure readiness and answer sufficiency are different failure categories.

**What was correct:**
- Accepted. Strong answer.
- You correctly distinguished infrastructure readiness from answer sufficiency.
- You correctly said `/ready` `503` means dependency/artifact unavailable.
- You correctly said `/query` insufficient evidence means the pipeline ran but evidence was weak or absent.

**Interview-quality version:**
`/ready` returning `503` means a required dependency or artifact is unavailable, such as a missing Qdrant collection or missing retrieval-ready files. An insufficient-evidence/refusal response means the query pipeline ran but retrieved evidence was weak or absent. Infrastructure readiness and answer sufficiency are different failure categories.

### Step 72 answer gate status
Accepted. You can move to Step 73.

---

## Step 73 — API smoke-test client optional readiness check

### Question 1
Why should the API smoke-test client make `/ready` optional instead of always calling it?

**Strong answer:**
`/health` should remain the cheapest default smoke path. `/ready` performs dependency/artifact checks that are useful before real queries but are not necessary for a basic liveness check. Making readiness opt-in keeps fast health checks cheap while still supporting a stronger pre-query verification path.

### Question 2
Why should the smoke-test client call `/ready` before `/query` when `--check-ready` is supplied?

**Strong answer:**
Readiness should run before query execution because it can catch missing dependencies or artifacts, such as a missing Qdrant collection or missing retrieval-ready files, before triggering retrieval, embeddings, LLM generation, traces, latency, or cost.

### Question 3
Why should a failed `/ready` check block `POST /query` in the smoke-test client?

**Strong answer:**
If readiness fails, the active retrieval mode cannot be served reliably. Continuing to `/query` would waste time/cost and produce a predictable infrastructure failure. Blocking the query keeps the operational failure clear and prevents unnecessary model or retrieval work.

### Question 4
Why should readiness failure errors avoid printing the raw `/ready` response body?

**Strong answer:**
Readiness response bodies may contain internal dependency details, local paths, config hints, or operational information. The smoke-test client should report a clear status-level failure while keeping detailed diagnostics in backend logs or controlled local output.

### Question 5
What is the correct operational sequence after Step 73?

**Strong answer:**
The correct sequence is `GET /health` for cheap liveness/config, optional `GET /ready` for dependency/artifact readiness, and only then optional `POST /query` for cost-bearing RAG execution. This separates liveness, readiness, and answer generation into distinct operational checks.


## Step 73 - API smoke-test client optional readiness check

### Questions asked
1. Why should the API smoke-test client make /ready optional instead of always calling it?
2. Why should the smoke-test client call /ready before /query when --check-ready is supplied?
3. Why should a failed /ready check block POST /query in the smoke-test client?
4. Why should readiness failure errors avoid printing the raw /ready response body?
5. What is the correct operational sequence after Step 73?

### Correct answers / evaluation
Accepted: Step 73 answer gate passed.

What was correct:
- Q1: Correct. You separated cheap liveness from stronger readiness checks.
- Q2: Correct. You placed readiness before query execution to avoid retrieval, embedding, LLM, trace, latency, and cost when dependencies are missing.
- Q3: Correct. You connected failed readiness to reliability, failure containment, and wasted work avoidance.
- Q4: Correct. You identified security and log-hygiene risk from leaking internal dependency details, local paths, config hints, or operational information.
- Q5: Correct. The operational sequence is health first, optional readiness second, optional query last.

Area to improve:
- In interview answers, explicitly name that readiness may touch stateful dependencies such as Qdrant or local readiness artifacts, while health must stay safe for frequent orchestrator or load-balancer probes.

Stronger interview-quality version:
- The smoke client should default to health only because liveness probes must be cheap, frequent, and dependency-light. Readiness is opt-in because it may inspect stateful serving prerequisites such as Qdrant collections and local retrieval-ready artifacts. When requested, readiness must run before query execution so predictable infrastructure failures are caught before cost-bearing retrieval, embedding, generation, and trace-writing work. A readiness failure should block query execution and report only a safe status-level error, while detailed diagnostics remain in controlled backend logs. The correct sequence is GET health, optional GET ready, then optional POST query.

## Step 74 - Lexical /query Qdrant dependency boundary

### Questions asked
1. Why is it a production bug if lexical /query creates a Qdrant client even though lexical retrieval does not need Qdrant?
2. Why should the query route decide whether to create Qdrant before calling orchestration instead of letting lower layers fail later?
3. Why is qdrant_client: QdrantClient | None acceptable only if dense/hybrid modes explicitly reject None?
4. What failure modes did the Step 74 tests intentionally simulate, and why are they useful?
5. What is the correct operational interpretation of lexical readiness/query behavior after Step 74?

### Correct answers / evaluation
Pending: awaiting user answers.

### User answer evaluation - Step 74 - 2026-05-15

#### Overall verdict
Not yet interview-pass. The intuition is mostly correct, especially around lexical mode using local retrieval-ready JSON artifacts and Qdrant downtime fallback. However, the answers are too vague for a production ML/RAG interview. The weakest parts are Q2 and Q4: dependency-boundary ownership must be precise, and a stress test must describe executable failure injection, not just the reason lexical mode should avoid Qdrant.

#### What was correct
- Correctly recognized that lexical retrieval can use local retrieval-ready JSON artifacts and does not need Qdrant.
- Correctly recognized that Qdrant-backed dense/hybrid modes have different infrastructure requirements from lexical mode.
- Correctly connected lexical mode to business continuity when Qdrant is unavailable.
- Correctly understood that HTTP 503 communicates a service/dependency failure rather than normal query success.

#### Areas to improve
- Q1 needs consequences: unnecessary Qdrant creation can create false outages, latency, operational coupling, and misleading failures for a mode that should work locally.
- Q2 must name the right boundaries: API route owns infrastructure lifecycle, retrieval service enforces defensive invariants, retrieval router selects mode. Do not say vaguely "all layers"; each layer has a specific responsibility.
- Q3 must explain fail-fast behavior: 503 prevents embedding calls, retrieval attempts, LLM generation, trace side effects, and misleading answers when required vector-store state is missing.
- Q4 is incomplete. You explained why lexical should not call vector DB, but the question asked how to prove it. A strong answer must mention monkeypatching/fake clients that raise if Qdrant or embeddings are called, then asserting lexical succeeds while dense/hybrid fail safely.
- Q5 is acceptable but should mention degraded quality, cost control, reliability/SLA, and explicit degraded-mode behavior.

#### Stronger interview-quality answer
Lexical-only retrieval should not instantiate Qdrant because lexical mode is designed to operate from local retrieval-ready artifacts. If the API route creates a Qdrant client anyway, Qdrant downtime can cause a false outage for a path that should still work, adds avoidable latency, couples cheap fallback retrieval to vector-store infrastructure, and may trigger unnecessary operational noise.

The boundary should be enforced at more than one layer, but with clear ownership. The API route should own infrastructure lifecycle and only create a Qdrant client for dense or hybrid modes. The retrieval service should defensively reject `qdrant_client=None` for dense/hybrid so callers cannot bypass the invariant. The router should choose the retrieval strategy, but it should not create infrastructure clients.

HTTP 503 is appropriate when dense or hybrid mode is configured but the Qdrant collection is missing because the service dependency is unavailable or not initialized. It fails fast before embedding calls, retrieval, LLM generation, and trace side effects. This avoids wasted cost and avoids returning misleading low-evidence or hallucinated answers.

To stress the boundary, I would monkeypatch Qdrant client creation to raise if called, use a fake embedding client that raises if embeddings are requested, run lexical `/query`, and assert the request succeeds with `qdrant_client=None`. Then I would run dense and hybrid modes with a fake Qdrant client whose `collection_exists` returns false and assert HTTP 503, no orchestration call, and no leaked internal details.

The business impact is that lexical mode becomes a cheaper degraded fallback during vector-store incidents. The product can still answer some exact-match or keyword-style queries, while dense/hybrid provide stronger semantic retrieval when Qdrant is healthy. This improves reliability and cost control, but the system should communicate that retrieval quality may be lower in degraded lexical-only operation.

#### Gate
Do not move to Step 75 yet. Rewrite answers to Q2 and Q4 with concrete ownership and executable stress-test details.

### Gate acceptance - Step 74 - 2026-05-15

#### Rewritten answers accepted
The rewritten Q2 and Q4 are now interview-quality.

#### Why accepted
- Q2 now clearly separates ownership:
  - API route owns infrastructure lifecycle and creates Qdrant only for dense/hybrid.
  - Retrieval service defensively rejects `qdrant_client=None` for dense/hybrid.
  - Router selects retrieval strategy but does not create infrastructure clients.
- Q4 now gives an executable stress-test design:
  - Monkeypatch Qdrant client creation to raise if called.
  - Use a fake embedding client that raises if embeddings are requested.
  - Run lexical `/query` and assert success with `qdrant_client=None`.
  - Run dense/hybrid with a fake missing collection and assert HTTP 503, no orchestration call, and no leaked internals.

#### Gate
Step 74 interview gate accepted. Proceed to Step 75 when ready.

## Step 75 - Retrieval-mode dependency matrix documentation

### Questions asked
1. Why is a dependency matrix useful in a production RAG API instead of relying only on code behavior?
2. Explain the difference between `/health`, `/ready`, and `/query` from a cost and failure-boundary perspective.
3. In lexical mode, why should missing corpus evidence be treated differently from missing lexical artifacts?
4. In hybrid mode, why does `/ready` need to check both Qdrant collection existence and local lexical artifacts?
5. How would this matrix help during an incident where Qdrant is unavailable but the business still wants partial service?

### Correct answer key
1. A matrix makes operational dependencies explicit for engineers and operators. It prevents accidental coupling, clarifies expected failure modes, supports onboarding, and gives tests a stable documentation contract.
2. `/health` is cheap liveness/config and should not touch dependencies. `/ready` checks required dependencies/artifacts for the active mode without model calls or real retrieval. `/query` is cost-bearing and may run retrieval, embeddings, LLM generation, and trace writing depending on mode and evidence sufficiency.
3. Missing lexical artifacts means the system cannot serve lexical retrieval and should fail readiness. Missing corpus evidence for a specific user query means retrieval ran but found insufficient support, so the correct behavior is a safe refusal/insufficient-evidence response.
4. Hybrid combines dense and lexical signals. If either Qdrant state or lexical artifacts are missing, the configured hybrid retrieval path is degraded or invalid, so readiness should fail before user queries.
5. The matrix shows that lexical mode can be used as a cheaper degraded path when Qdrant is down, while dense/hybrid should fail fast. This supports clear incident communication, cost control, and explicit tradeoffs between availability and retrieval quality.

### User answer evaluation - Step 75 - 2026-05-15

#### Overall verdict
Not yet interview-pass. Your intuition is directionally correct, especially around documentation, cost, dependency checks, hybrid requiring both dense and lexical dependencies, and lexical as degraded service during Qdrant incidents. However, the answers are still too imprecise for production RAG interview depth. The main weak areas are Q2 and Q3.

#### What was correct
- Q1 correctly identified maintainability, onboarding, debugging, and expected-failure documentation as reasons for a dependency matrix.
- Q2 correctly identified `/health` as cheapest and fastest, `/ready` as dependency/artifact checks, and `/query` as the expensive path.
- Q4 correctly identified hybrid as a combination of dense and lexical retrieval requiring both Qdrant and local lexical artifacts.
- Q5 correctly identified lexical as a lower-cost degraded path during Qdrant incidents, with possible quality degradation.

#### Areas to improve
- Q1 should explicitly mention that code behavior alone is not enough for operators, incident responders, and future engineers. A matrix becomes an operational contract and can be protected by tests.
- Q2 incorrectly says `/query` "checks" LLM config/retrieval/embeddings. Stronger: `/query` actually executes retrieval and may call embeddings, LLM generation, and trace writing. Also, `/health`, `/ready`, and `/query` are not always a strict mandatory order for every caller; rather, the recommended safe operational flow is health first, readiness second, query last.
- Q3 is incomplete. Missing lexical artifacts means the system is not ready to serve lexical retrieval and should fail readiness. Missing corpus evidence means artifacts exist and retrieval ran, but the specific user query lacked enough evidence; that should be a safe refusal, not readiness failure.
- Q4 should mention readiness fails before user queries because hybrid retrieval would be incomplete or degraded if either dense or lexical side is missing.
- Q5 should include incident communication: switch to lexical mode intentionally, communicate degraded semantic quality, monitor refusal/answer quality, and avoid embedding/vector-store spend until Qdrant is restored.

#### Stronger interview-quality answer
A dependency matrix is useful because production RAG systems have multiple runtime dependencies and cost boundaries. Code behavior alone is not enough for operators, reviewers, and incident responders. The matrix makes expected dependencies, failure modes, and degraded-mode behavior explicit. It also gives us a documentation contract that can be protected with tests, reducing accidental coupling such as making lexical retrieval depend on Qdrant.

`/health` is the cheapest liveness/config endpoint. It should not run retrieval, embeddings, LLM calls, Qdrant checks, or trace writing. `/ready` is a dependency/artifact readiness endpoint. It checks whether the active retrieval mode can be served, such as lexical artifacts for lexical/hybrid or Qdrant collection existence for dense/hybrid, but it should not execute user retrieval or model calls. `/query` is the cost-bearing path: it may run retrieval, embeddings for dense/hybrid retrieval, LLM generation when evidence is sufficient, and local trace writing. Operationally, health then readiness then query is the safe smoke-test sequence, but the endpoints have distinct contracts rather than being the same kind of check.

In lexical mode, missing lexical artifacts means the system lacks the required local retrieval index/artifact and should fail readiness with `503`. Missing corpus evidence is different: the artifacts exist and retrieval runs, but the specific user query does not have enough supporting evidence. That should produce an insufficient-evidence safe refusal, not a readiness failure or server crash.

In hybrid mode, `/ready` must check both Qdrant collection existence and local lexical artifacts because hybrid retrieval fuses dense and lexical signals. If either side is missing, the configured hybrid path is incomplete. Failing readiness early prevents user-facing queries from running with partially broken retrieval and avoids wasted embedding/LLM cost.

During a Qdrant incident, the matrix tells operators that dense/hybrid depend on Qdrant and should fail fast, while lexical can be used as a lower-cost degraded fallback if local lexical artifacts are available. The business can keep partial service running, but should communicate reduced semantic retrieval quality, monitor refusals and answer quality, and switch back to dense/hybrid once Qdrant is restored.

#### Gate
Do not move to Step 76 yet. Rewrite Q2 and Q3 only. Be precise about endpoint contracts and the difference between missing artifacts versus missing evidence.

### Gate acceptance - Step 75 - 2026-05-15

#### Rewritten answers accepted
The rewritten Q2 and Q3 are now interview-quality.

#### Why accepted
- Q2 now correctly separates endpoint contracts:
  - `/health` is cheap liveness/config and does not touch retrieval, embeddings, LLMs, Qdrant, or traces.
  - `/ready` checks active-mode dependencies/artifacts without user retrieval or model calls.
  - `/query` is the cost-bearing path and may execute retrieval, embeddings for dense/hybrid, LLM generation when evidence is sufficient, and trace writing.
  - health -> readiness -> query is a safe smoke-test sequence, not a universal mandatory order.
- Q3 now clearly separates failure classes:
  - Missing lexical artifacts means the system lacks required local retrieval state and should fail readiness with `503`.
  - Missing corpus evidence means retrieval ran but the specific query lacks enough support, so the correct behavior is insufficient-evidence safe refusal.

#### Gate
Step 75 interview gate accepted. Proceed to Step 76 when ready.


<!-- Active interview history archived through Step 93 on 2026-07-29 -->

## Step 76 - Readiness retrieval-mode dependency boundary tests

### Questions asked
1. Why should dense readiness pass even when `.retrieval_ready.json` lexical artifacts are missing?
2. Why should hybrid readiness fail if either Qdrant or lexical artifacts are missing?
3. Why is it useful for hybrid readiness to still check Qdrant even when lexical artifacts are already missing?
4. What kind of production incident would be hidden if lexical readiness accidentally created a Qdrant client?
5. Why did Step 76 add tests instead of changing production readiness code?

### Correct answer key
1. Dense retrieval depends on vector-store state and embeddings, not local lexical artifacts. Requiring lexical artifacts in dense mode would create unnecessary coupling and false readiness failures.
2. Hybrid fuses dense and lexical evidence. If either Qdrant or local lexical artifacts are missing, the configured hybrid retrieval path is incomplete and should fail readiness before user queries.
3. Reporting both dependency states helps operators diagnose all missing prerequisites in one readiness response instead of fixing one dependency only to discover another later.
4. It would hide the fact that lexical mode is supposed to be a degraded local fallback. Qdrant downtime could incorrectly make lexical readiness fail, causing false outages and unnecessary incident escalation.
5. The production route already implemented the correct dependency boundary. The risk was regression or undocumented assumptions, so tests were the right minimal change to lock the contract down.

### User answer evaluation - Step 76 - 2026-05-20

#### Overall verdict
Not yet interview-pass. Q1 and Q2 are directionally correct. Q3 is incorrect, Q4 misses the main production incident, and Q5 does not answer the "why" question. You are recognizing dependencies, but you must be sharper about readiness diagnostics and degraded-mode reliability.

#### What was correct
- Q1 correctly says dense retrieval depends on Qdrant/vector search and not `.retrieval_ready.json` lexical artifacts.
- Q2 correctly says hybrid combines dense and lexical dependencies, so either missing side should make hybrid readiness fail.
- Q5 correctly lists the behaviors Step 76 tested.

#### Areas to improve
- Q1 should avoid saying dense readiness "gets embeddings". Readiness should not call embedding APIs. Dense readiness checks configuration and Qdrant collection existence only.
- Q3 is wrong. Lexical artifacts are not "searched in vector DB". Lexical artifacts are local JSON artifacts used for lexical search. Hybrid readiness checks Qdrant even when artifacts are missing so the readiness response reports all dependency states in one call.
- Q4 is incomplete. The biggest incident is false outage/degraded-mode failure: lexical mode should survive Qdrant downtime. If lexical readiness creates Qdrant, Qdrant downtime could incorrectly mark lexical mode not ready. Cost overhead is secondary.
- Q5 lists test cases but does not explain why tests were chosen instead of production code changes. The reason is that production code already had correct behavior; the risk was regression, so tests were the minimal and correct change.

#### Stronger interview-quality answer
Dense readiness should pass without `.retrieval_ready.json` files because dense retrieval relies on vector-store state, not lexical artifacts. `/ready` should check that dense mode has valid runtime/model configuration and that the required Qdrant collection exists. It should not run embedding calls or retrieval during readiness.

Hybrid readiness should fail if either Qdrant or lexical artifacts are missing because hybrid retrieval fuses dense and lexical evidence. If Qdrant is missing, the dense side cannot run. If lexical artifacts are missing, the lexical side cannot run. The configured hybrid path is incomplete, so readiness should fail before user queries.

Hybrid readiness should still check Qdrant even when lexical artifacts are already missing because operators need complete dependency diagnostics in one response. Otherwise they might fix lexical artifacts, rerun readiness, and only then discover Qdrant is also missing. Reporting both states reduces incident time and avoids sequential debugging.

If lexical readiness accidentally created a Qdrant client, Qdrant downtime could cause lexical mode to fail readiness even though lexical mode should be a local degraded fallback. That would create a false outage, unnecessary escalation, and possibly prevent partial service. Extra overhead is a concern, but reliability and false dependency coupling are the main risks.

Step 76 added tests instead of production code changes because the readiness implementation already enforced the correct dependency boundaries. The missing piece was regression protection. Tests lock down the contract so future changes cannot accidentally make dense depend on lexical artifacts, lexical depend on Qdrant, or hybrid hide one of its dependency states.

#### Gate
Do not move to Step 77 yet. Rewrite Q3, Q4, and Q5 only. Be precise: local lexical artifacts are not vector DB data; tests were added because production behavior was already correct and needed regression protection.

### Gate acceptance - Step 76 - 2026-05-20

#### Rewritten answers accepted
The rewritten Q3, Q4, and Q5 are accepted.

#### Why accepted
- Q3 now recognizes that hybrid readiness should report both lexical-artifact and Qdrant dependency states in one response because hybrid depends on both dense and lexical retrieval paths.
- Q4 now recognizes that lexical mode uses local JSON artifacts and should remain independent of Qdrant. If lexical readiness touched Qdrant, Qdrant downtime could cause a false readiness failure for the degraded lexical path.
- Q5 now recognizes that production readiness code already existed and Step 76 added regression protection as an extra guardrail.

#### Wording correction
Use "Qdrant collection check" or "Qdrant client lifecycle" instead of "Qdrant creation". Readiness may create a client to check collection existence, but it should not create a collection.

#### Gate
Step 76 interview gate accepted. Proceed to Step 77 when ready.

## Step 77 - Query Qdrant dependency failure returns safe 503

### Questions asked
1. Why should a Qdrant client creation failure in dense/hybrid `/query` return `503` instead of `500`?
2. Why must query orchestration not run after the required Qdrant dependency check fails?
3. Why do we preserve an explicit `503` message for a missing collection but use a generic safe `503` for unexpected Qdrant exceptions?
4. What sensitive information might leak if raw Qdrant exceptions were returned to API clients?
5. Why is closing the Qdrant client still important when the collection check raises?

### Correct answer key
1. Dense/hybrid require Qdrant as an external dependency. If that dependency is unavailable or cannot be checked, the service is temporarily unable to serve that mode, which is a `503 Service Unavailable`, not an internal application bug.
2. Once the required vector-store dependency is known to be unavailable, running retrieval/orchestration could waste embedding or LLM cost, write misleading traces, or produce low-quality/hallucinated answers.
3. A missing collection is an expected operational failure with a clear fix: run indexing. Unexpected Qdrant exceptions may contain internals, local paths, or secrets, so clients should receive a safe generic dependency-failure message while logs keep details for operators.
4. Raw exceptions may expose local filesystem paths, collection names, host/port details, stack traces, credentials, API keys, or deployment configuration.
5. Client cleanup prevents resource leaks, file locks, stale handles, or degraded behavior in later requests, especially for persistent/local vector-store clients.

## Step 78 - Answer smoke script lexical Qdrant independence

### Questions asked
1. Why should the answer smoke script avoid creating a Qdrant client in lexical mode?
2. Why is it still correct for dense and hybrid smoke tests to create a Qdrant client and check collection existence before orchestration?
3. Why should lexical mode pass `qdrant_client=None` into shared answer orchestration instead of passing a fake or unused Qdrant client?
4. What production or local-development problem could happen if lexical smoke testing still touched Qdrant?
5. Why does the new regression test make Qdrant client creation raise instead of only asserting the final orchestration arguments?

### Correct answer key
1. Lexical retrieval uses local `.retrieval_ready.json` artifacts and does not need vector-store state. Creating Qdrant in lexical mode adds an unnecessary dependency and violates the retrieval-mode dependency matrix.
2. Dense retrieval depends on vector search, and hybrid retrieval depends on both dense and lexical paths. If the Qdrant collection is missing, those modes should fail before retrieval/orchestration starts.
3. Passing `None` makes the dependency contract explicit: lexical mode has no Qdrant dependency. The downstream retrieval service already rejects `None` for dense/hybrid and allows it for lexical, so this is cleaner than passing an unused client.
4. Qdrant could be down, locked, corrupted, missing, or expensive to initialize. If lexical smoke testing touched it, a local degraded-mode test could fail even though lexical retrieval should still work from local artifacts.
5. Making Qdrant construction raise turns accidental coupling into an immediate test failure. Final-argument assertions prove what was passed to orchestration, but the raising fake proves the script did not touch the dependency at all.

### User answer evaluation - Step 78 - 2026-06-11

#### Overall verdict
Partially accepted, but not interview-pass yet. Q1, Q3, and Q4 are directionally correct. Q2 is too vague and slightly imprecise. Q5 was not answered yet.

#### What was correct
- Q1 correctly identifies that lexical retrieval uses existing local JSON artifacts, so Qdrant is unnecessary.
- Q3 correctly says `qdrant_client=None` represents the absence of a Qdrant dependency in lexical mode.
- Q4 correctly identifies the degraded-mode failure risk: lexical could fail unnecessarily if Qdrant is unavailable.

#### Areas to improve
- Q1 should name the artifact type precisely: lexical retrieval uses local `.retrieval_ready.json` artifacts, not just any JSON files.
- Q2 should say dense and hybrid depend on an existing Qdrant collection for vector search. The smoke test should check collection existence before orchestration so it fails fast and avoids wasted embedding/LLM work. Avoid saying "Qdrant creation"; the script creates a client, not the collection.
- Q4 should mention local file locks, corrupted/missing local Qdrant state, downtime, or unnecessary dependency coupling as concrete failure modes.
- Q5 still needs an answer.

#### Stronger interview-quality answer
Lexical answer smoke testing should avoid creating a Qdrant client because lexical retrieval reads local `.retrieval_ready.json` artifacts and does not perform vector search. Touching Qdrant would add an unnecessary dependency and could make the degraded lexical path fail for reasons unrelated to lexical retrieval.

Dense and hybrid smoke tests should still check Qdrant because dense retrieval needs vector search, and hybrid retrieval includes the dense side. If the required collection is missing, the smoke test should fail before orchestration so it avoids wasted embedding or LLM calls and gives the operator a clear indexing prerequisite.

`qdrant_client=None` is the right lexical contract because it makes the dependency boundary explicit. Lexical mode has no Qdrant dependency, while the shared retrieval service still rejects `None` for dense and hybrid modes.

If lexical smoke tests touched Qdrant, a missing, locked, corrupted, or unavailable local Qdrant store could make lexical testing fail even though local lexical artifacts are sufficient. That would create a false failure and weaken degraded-mode reliability.

A test that raises on Qdrant creation is stronger because it proves the unwanted dependency was never touched. Checking final orchestration arguments only proves what was eventually passed downstream; the script could still have created Qdrant earlier and then passed `None`, hiding the accidental side effect.

#### Gate
Do not move to Step 79 yet. Rewrite Q2 and answer Q5. Use the phrase "Qdrant collection check" or "Qdrant client creation", not "Qdrant creation".

### Gate acceptance - Step 78 - 2026-06-11

#### Rewritten answers accepted
The rewritten Q2 and Q5 are accepted.

#### Why accepted
- Q2 now correctly explains that dense retrieval needs vector search and hybrid includes the dense side, so the smoke test should check the required Qdrant collection before orchestration.
- Q5 now correctly explains why a raising fake is stronger than checking final orchestration arguments: it proves the unwanted dependency was never touched, rather than only proving the final value passed downstream.

#### Wording correction
Use "Qdrant client creation" instead of "Qdrant creation" when describing this test. The script creates a client and checks an existing collection; it does not create the Qdrant collection.

#### Gate
Step 78 interview gate accepted. Proceed to Step 79 when ready.

## Step 79 - Query-search script lexical Qdrant independence

### Questions asked
1. Why did `scripts/run_qdrant_query_search.py` need the same lexical Qdrant-independence fix as the answer smoke script?
2. What is the difference between the query-search script and the answer-smoke script in terms of cost-bearing behavior?
3. Why should dense and hybrid query-search runs fail before `retrieve_query_evidence(...)` if the Qdrant collection is missing?
4. Why did the Step 79 test include both a hybrid case and a lexical case?
5. What failure would be hidden if the lexical test only asserted `qdrant_client is None` after retrieval was called?

### Correct answer key
1. Both scripts route through the configured retrieval mode. Lexical mode uses local `.retrieval_ready.json` artifacts, so creating a Qdrant client in either script would violate the same dependency matrix.
2. The query-search script runs retrieval and sufficiency only; it does not generate an answer or call the chat LLM. Dense/hybrid query search can still call the embedding API for dense retrieval, while lexical query search should avoid both Qdrant and embedding calls.
3. Dense retrieval requires vector-store search, and hybrid includes the dense side. If the required Qdrant collection is missing, the script should fail fast before retrieval so it avoids embedding work and returns a clear indexing prerequisite.
4. The hybrid case proves the required dependency path still works: create client, check collection, pass client, close client. The lexical case proves the degraded local path does not touch Qdrant at all.
5. The script could still create or touch a Qdrant client before eventually passing `None` into retrieval. A raising fake catches that unwanted side effect immediately, while final-argument assertions only inspect the downstream call.

### User answer evaluation - Step 79 - 2026-06-12

#### Overall verdict
Accepted. This is interview-pass.

#### What was correct
- Q1 correctly connects the query-search script and answer-smoke script to the same retrieval-mode dependency matrix.
- Q2 correctly separates retrieval/sufficiency cost from answer-generation cost. You also correctly noted that dense/hybrid query search can still call embeddings, while lexical should avoid both Qdrant and embeddings.
- Q3 correctly explains fail-fast behavior before `retrieve_query_evidence(...)` when dense/hybrid Qdrant collection state is missing.
- Q4 correctly explains why both hybrid and lexical tests were needed: one preserves the required dependency path, the other protects the degraded local path.
- Q5 correctly explains why a raising fake catches side effects that final-argument assertions could miss.

#### Sharpening note
For Q3, add one production phrase: failing before retrieval also gives the operator a clear indexing prerequisite. The strongest version is: "Fail before retrieval so we avoid embedding work and return a clear signal to run indexing."

#### Gate
Step 79 interview gate accepted. Proceed to Step 80 when ready.

## Step 80 - Retrieval comparison script Qdrant client cleanup

### Questions asked
1. Why is `run_retrieval_comparison.py` still allowed to require Qdrant, unlike lexical-only scripts?
2. Why is `try/finally` the right pattern for closing the Qdrant client in this script?
3. What local-development problem can happen if a persistent local Qdrant client is not closed after an exception?
4. Why should the script still raise the original dense retrieval error instead of swallowing it after cleanup?
5. Why did Step 80 add both a success-path cleanup test and a failure-path cleanup test?

### Correct answer key
1. The script explicitly compares dense vector retrieval against lexical artifact retrieval. Dense retrieval requires Qdrant, so this script has a legitimate vector-store dependency.
2. `try/finally` guarantees cleanup after normal completion and after exceptions from collection checks, dense retrieval, lexical retrieval, comparison building, or report writing.
3. An unclosed persistent local Qdrant client can leave file handles or locks behind, causing later indexing, querying, or evaluation runs to fail or behave inconsistently.
4. Cleanup should not hide the real failure. The caller/operator still needs the original error to diagnose retrieval, embedding, artifact, or report-writing problems.
5. The success-path test proves normal behavior still closes the client. The failure-path test proves cleanup is not only a happy-path property and still happens when retrieval fails mid-run.

### User answer evaluation - Step 80 - 2026-07-22

#### Overall verdict
Not yet interview-pass. Q1 is directionally correct but unclear. Q2 is incorrect because `finally` performs cleanup; it does not switch retrieval to a lexical fallback. Q3, Q4, and Q5 were not answered.

#### What was correct
- Q1 recognizes that dense retrieval performs vector search through Qdrant, whereas a lexical-only script can use local lexical artifacts without Qdrant.
- Q2 recognizes that `finally` runs when work in the `try` block fails, but its purpose needs correction.

#### Areas to improve
- Q1 must state why this particular script legitimately needs Qdrant: it compares dense Qdrant retrieval with lexical retrieval.
- Q2 must separate resource cleanup from fallback behavior. `finally` closes the client after success or failure; it does not run lexical retrieval as a recovery path.
- Q3 should identify file handles, local storage locks, stale resources, and failures or inconsistent behavior in later indexing/query/evaluation runs.
- Q4 should explain that cleanup and error reporting are separate responsibilities. The original exception must remain visible so operators can diagnose the actual failure.
- Q5 should distinguish the guarantees: the success test protects normal cleanup and behavior, while the failure test proves exception-safe cleanup.

#### Gate
Do not proceed to Step 81. Rewrite Q2 and answer Q3-Q5. Q1 is accepted with the wording correction provided in chat.

### Follow-up evaluation - Step 80 - 2026-07-22

#### Accepted answers
- Q3 accepted: an unclosed persistent Qdrant client can retain stale resources, inconsistent local state, file handles, or database locks, which can make later runs fail or hang.
- Q4 accepted: cleanup must release resources while preserving the original exception so operators can diagnose and fix the actual failure. Prefer "original exception/error" over "failed log" because logging alone is not the failure signal propagated to the caller.
- Q5 accepted with correction: the success-path test proves normal comparison behavior and cleanup; the failure-path test proves exception-safe client cleanup. The failure test does not "cleanse logging or hangs"—it verifies that `close()` runs so stale handles and locks are less likely to affect later runs.

#### Remaining gate requirement
Q2 was not rewritten. Explain in your own words why `try/finally` is the correct client-cleanup pattern and explicitly state that `finally` does not provide a lexical fallback.

### Second follow-up evaluation - Step 80 - 2026-07-22

#### Q2 verdict
Not yet accepted. Saying `finally` removes "unwanted debris" recognizes cleanup generally, but does not identify the Qdrant client, explain that `finally` runs after both success and exceptions, or correct the earlier lexical-fallback misconception.

#### Required correction
State all three points: the comparison work belongs in `try`; `finally` always closes the Qdrant client after success or failure; and `finally` performs cleanup only—it does not switch to lexical retrieval when dense retrieval fails.

### Third follow-up evaluation - Step 80 - 2026-07-22

#### Q2 verdict
Mechanically correct but incomplete. The answer correctly places comparison work in `try` and states that `finally` closes the Qdrant client after success or exceptions. It does not explicitly correct the earlier misconception that `finally` can switch to lexical retrieval.

#### Final required clause
Confirm that `finally` performs resource cleanup only and does not implement a lexical fallback or recover the failed comparison.

### Gate acceptance - Step 80 - 2026-07-26

#### Final answer accepted
The final clarification is accepted. `finally` performs resource cleanup only: it closes the Qdrant client after success or failure. It does not switch to lexical retrieval, retry dense retrieval, or recover the failed comparison.

#### Why accepted
- The answer now separates cleanup from fallback behavior.
- Together with the earlier response, it correctly explains the `try/finally` lifecycle guarantee.
- Q1-Q5 now meet the Step 80 interview gate after the recorded wording corrections.

#### Gate
Step 80 interview gate accepted. Proceed to Step 81 when ready.

## Step 81 - Hybrid retrieval evaluation Qdrant client cleanup

### Questions asked
1. Why does hybrid retrieval evaluation legitimately require Qdrant even though one of its retrieval branches is lexical?
2. Which operations are protected by the new `try/finally` lifecycle boundary, and why should report writing remain inside that boundary?
3. Why is removing the manual `client.close()` from only the missing-collection branch an improvement rather than a loss of cleanup?
4. Why should a dense retrieval exception still propagate after the Qdrant client is closed?
5. What different guarantees do the success-path and failure-path regression tests provide?

### Correct answer key
1. Hybrid evaluation measures dense, lexical, and fused retrieval. Its dense branch requires vector search against the Qdrant collection, so the overall evaluation has a legitimate Qdrant dependency even though lexical retrieval itself does not.
2. The boundary covers collection checking, dense retrieval, lexical retrieval, fusion, case evaluation, aggregate report building, and report writing. Report writing remains inside because filesystem or serialization failures must also trigger Qdrant cleanup.
3. The manual close protected only one known failure branch. A single `finally` provides one centralized cleanup guarantee for normal completion and every exception that occurs after client creation, reducing duplicated and easy-to-miss cleanup logic.
4. Closing the client releases resources but does not resolve the retrieval failure. Propagating the original exception preserves the real diagnostic signal and prevents a failed or incomplete evaluation from appearing successful.
5. The success-path test proves normal evaluation behavior still works and closes the client. The failure-path test proves cleanup is exception-safe and that the original dense retrieval error is not swallowed.

### User answer evaluation - Step 81 - 2026-07-26

#### Overall verdict
Accepted. This is interview-pass.

#### What was correct
- Q1 correctly recognizes that hybrid evaluation combines dense Qdrant retrieval with lexical retrieval, so the overall evaluation legitimately requires Qdrant.
- Q2 correctly identifies that the lifecycle boundary covers the collection check, both retrieval branches, fusion, case evaluation, report building, and report writing. It also correctly explains that report failures must trigger cleanup.
- Q3 correctly explains that a manual close in one known branch is incomplete because many later operations can fail. Centralized `finally` cleanup covers those unknown failure paths.
- Q4 correctly separates resource cleanup from fixing the underlying dense retrieval failure.
- Q5 correctly distinguishes normal-path cleanup from exception-safe cleanup and preservation of the original failure.

#### Sharpening notes
- For Q1, say explicitly that Qdrant is required by the dense vector-search branch; lexical retrieval itself remains Qdrant-independent.
- For Q4 and Q5, use "propagate the original exception" rather than only "log the error." Logging is useful, but propagation is what prevents the failed evaluation from appearing successful to the caller or automation.
- Avoid the phrase "safe exception." The important guarantees are that cleanup runs and the original exception is not swallowed or replaced.

#### Stronger interview-quality answer
Hybrid evaluation requires Qdrant because it evaluates dense retrieval, lexical retrieval, and their fused hybrid result. The dense branch performs vector search against an existing Qdrant collection, while the lexical branch remains independent and reads local lexical artifacts.

The `try/finally` boundary covers the collection check, dense and lexical retrieval, fusion, case evaluation, aggregate report construction, and report writing. Report writing stays inside because serialization or filesystem failures must also close the already-open persistent Qdrant client.

Removing the branch-specific manual close improves the design because it replaces cleanup for one known error with one centralized guarantee. The `finally` block runs after success and after any exception raised anywhere in the protected workflow.

A dense retrieval exception must propagate because closing the client only releases resources; it does not repair the failed retrieval. Preserving the original exception gives operators and automation the real diagnostic signal and prevents an incomplete evaluation from appearing successful.

The success-path test proves the normal evaluation and reporting flow still works and closes the client. The failure-path test proves cleanup is exception-safe and that the original dense retrieval exception still propagates.

#### Gate
Step 81 interview gate accepted. Proceed to Step 82 when ready.

## Step 82 - Hybrid weight experiment Qdrant client cleanup

### Questions asked
1. Why does a hybrid weight experiment require Qdrant even though it reuses the same dense candidates across multiple weight settings?
2. Why should candidate retrieval happen once per evaluation case rather than once per weight setting?
3. Which failures after candidate retrieval still need to trigger Qdrant client cleanup?
4. Why must an experiment report not be treated as successful when only some cases or weight settings were evaluated before an exception?
5. What two independent properties does the failure-path regression test verify?

### Correct answer key
1. Each experiment still needs dense vector candidates before it can test fusion weights. Reusing those candidates reduces repeated work, but their initial retrieval still requires the Qdrant collection and embedding-backed dense search.
2. Fusion weights change how existing dense and lexical scores are combined; they do not change the base retrieval queries. Retrieving once avoids repeated embedding calls, vector searches, latency, and cost while making weight comparisons fair because every setting uses identical candidates.
3. Lexical retrieval, weight parsing/evaluation, hybrid fusion, result ranking, logging, serialization, filesystem creation, and report writing can all fail after the client opens. Any such failure must still execute `finally` and close the client.
4. A partial report can produce misleading model-selection decisions because settings may not have been evaluated on identical cases. The exception should propagate, and automation must see the run as failed rather than consuming incomplete metrics.
5. It proves exception-safe resource cleanup and error transparency: the Qdrant client closes even when dense candidate retrieval fails, while the original exception is preserved and propagated.

### User answer evaluation - Step 82 - 2026-07-26

#### Overall verdict
Accepted. This is interview-pass.

#### What was correct
- Q1 correctly identifies that the initial dense candidate retrieval still requires Qdrant and embeddings, while candidate reuse avoids repeating that work for every weight setting.
- Q2 correctly connects candidate reuse to both efficiency and experimental fairness: identical candidate pools isolate fusion weights as the variable under test.
- Q3 correctly states that failures during retrieval, ranking, reporting, or file I/O must still reach `finally` and close the client.
- Q4 correctly identifies partial reports as misleading evaluation artifacts and explains why exceptions must propagate to automation.
- Q5 correctly identifies both independent guarantees: exception-safe client cleanup and preservation of the original failure.

#### Sharpening note
For Q3, explicitly include hybrid fusion and weight-setting evaluation among the protected failure points. The answer already captures the broader principle correctly.

#### Gate
Step 82 interview gate accepted. Proceed to Step 83 when ready.

## Step 83 - Dense retrieval evaluation Qdrant client cleanup

### Questions asked
1. Why is `run_retrieval_eval.py` legitimately Qdrant-dependent rather than retrieval-mode conditional?
2. Why must result evaluation and serialization remain inside the same `try/finally` boundary as dense search?
3. If the fifth evaluation case raises, what should happen to the first four in-memory results, the output report, the exception, and the Qdrant client?
4. Why does client cleanup not remove the need for an explicit Qdrant collection-existence check?
5. What does the failure-path test prove that a success-path cleanup test cannot prove?

### Correct answer key
1. This script is specifically the dense retrieval baseline evaluator. It always calls embedding-backed vector search against Qdrant; it is not a general lexical/dense/hybrid router.
2. Evaluation, result serialization, aggregate report construction, and JSON writing can fail after the client opens. Keeping them inside the boundary guarantees that every post-creation exception still closes the persistent client.
3. The partial in-memory results should not be published as a successful complete report. The fifth-case exception should propagate unchanged, and `finally` should close the Qdrant client. Any deliberately supported partial-report behavior would require an explicit, separately designed contract.
4. Cleanup answers what happens after a client is opened; collection validation answers whether the required indexed dependency is ready before work begins. Without an explicit check, the script may fail later with a less actionable backend error or start avoidable embedding work.
5. It proves exception safety: when dense search raises, cleanup still runs and the original exception is preserved. A success-path test only proves cleanup after normal completion.

### User answer evaluation - Step 83 - 2026-07-26

#### Overall verdict
Accepted. This is interview-pass.

#### What was correct
- Q1 correctly distinguishes the dedicated dense evaluator from retrieval-mode routing.
- Q2 correctly explains why all post-client-creation work belongs inside the lifecycle boundary.
- Q3 correctly rejects publishing partial results as a complete report, preserves the exception, and guarantees cleanup.
- Q4 correctly separates pre-flight dependency validation from resource cleanup and identifies wasted embedding work and vague backend failures.
- Q5 correctly identifies exception safety as the guarantee unique to the failure-path test.

#### Gate
Step 83 interview gate accepted. The user requested that the remaining Qdrant lifecycle fixes be completed as one consolidated sweep rather than one file per step.

## Step 84 - Consolidated persistent Qdrant client lifecycle sweep

### Questions asked
1. Why should `run_qdrant_indexing.py` close its Qdrant client even when an upsert or collection-creation operation raises?
2. Why is an early `return` inside the check script's `try` block still safe with `finally`?
3. Why were API routes and previously hardened retrieval scripts audited but not edited again?
4. Why should direct Qdrant clients created inside unit tests not automatically be treated as production lifecycle defects?
5. What does the combination of call-site inventory, targeted failure tests, broader tests, and the full suite prove—and what does it still not prove?

### Correct answer key
1. Indexing failures can occur after the persistent client opens and may leave local handles or locks behind. `finally` releases the client while preserving the original indexing exception for diagnosis and recovery.
2. Python executes `finally` before completing a `return` from the associated `try`. The missing-collection path therefore remains a normal early return but still closes the client.
3. They already implemented the required invariant. Re-editing compliant code would add churn and regression risk without improving behavior; the audit verifies coverage while tests protect the existing paths.
4. Test-created clients are scoped fixtures or local test resources rather than long-running production call sites. They should still be closed when persistent resources are used, but in-memory or deliberately reopened test clients require case-specific handling rather than a blanket production rewrite.
5. Together they provide strong evidence that all currently discovered production factory call sites use cleanup and that tested success/failure behavior remains correct across the repository. They do not mathematically prove the absence of dynamically created clients, future regressions outside the tests, process-kill safety, operating-system crashes, or failures inside `client.close()` itself.

### User answer evaluation - Step 84 - 2026-07-26

#### Overall verdict
Accepted. This is interview-pass.

#### What was correct
- Q1 correctly explains that `finally` releases persistent Qdrant locks and handles while preserving the original indexing failure.
- Q2 correctly states Python's control-flow guarantee that `finally` runs before an early `return` completes.
- Q3 correctly applies minimal-change discipline: audit compliant paths and avoid unnecessary edits that add regression risk.
- Q4 correctly rejects a blanket rewrite of test resources and recognizes that fixture lifecycle depends on whether the client is persistent, in-memory, deliberately reopened, or otherwise scoped.
- Q5 correctly limits the claim to currently audited production call sites and recognizes that ordinary tests cannot guarantee cleanup after process termination or operating-system crashes.

#### Sharpening note
The strongest Q5 answer also mentions that the sweep cannot prove future call sites will remain compliant, detect every dynamically created client, or guarantee behavior if `client.close()` itself raises.

#### Gate
Step 84 interview gate accepted. The consolidated persistent Qdrant lifecycle sweep is complete.

## Step 85 - Dense retrieval evaluation fail-fast collection check

### Questions asked
1. Why should the evaluator check collection existence before calling `search_query_text(...)`?
2. What cost or side effect might occur if dense search starts before discovering that the collection is missing?
3. Why is the collection check placed inside, rather than before, the `try/finally` boundary?
4. Why does the missing-collection test make the search fake raise if it is called?
5. What is the difference between an actionable dependency error and a raw backend search error in production operations?

### Correct answer key
1. The evaluator has a hard dependency on indexed vector data. A pre-flight check fails at the actual missing prerequisite and provides a clear instruction before retrieval begins.
2. Dense search may generate a query embedding first, causing avoidable API cost and latency. It may also emit misleading traces or fail later with a less clear Qdrant error.
3. `collection_exists(...)` itself can return false or raise after the client has opened. Keeping it inside ensures `finally` closes the client on both outcomes.
4. The raising fake proves search was never touched, not merely that the script eventually raised the expected error. It catches accidental embedding/search work before validation.
5. An actionable dependency error identifies the unavailable prerequisite and recovery action, such as running indexing. A raw backend error may expose implementation details, vary by backend version, and force operators to infer the real cause.

### User answer evaluation - Step 85 - 2026-07-26

#### Overall verdict
Accepted. This is interview-pass.

#### What was correct
- Q1 correctly frames the collection check as early validation of required indexed vector data.
- Q2 correctly identifies embedding cost, latency, misleading traces, and unclear downstream errors as avoidable consequences.
- Q3 correctly explains that the collection check itself belongs inside the lifecycle boundary because it can fail after client creation.
- Q4 correctly identifies the raising fake as proof that retrieval and embedding work were never touched.
- Q5 correctly distinguishes a recovery-oriented prerequisite error from a raw backend diagnostic that forces operator inference.

#### Gate
Step 85 interview gate accepted. Proceed to Step 86 when ready.

## Step 86 - Evaluation script missing-collection regression coverage

### Questions asked
1. Why was Step 86 a test-only change rather than a production-code change?
2. Why is asserting only the final missing-collection error insufficient to prove fail-fast behavior?
3. What does a raising search fake prove that a mock call-argument assertion may not prove as directly?
4. Why should retrieval comparison and hybrid evaluation fail completely when Qdrant is missing even though their lexical branch could still run?
5. What consistency benefit comes from using the same actionable indexing prerequisite across all dense/hybrid evaluation CLIs?

### Correct answer key
1. The production scripts already checked collection existence before retrieval and closed clients in `finally`. The uncovered risk was regression, so tests were the minimal change that protected the existing contract without unnecessary churn.
2. A script could perform an embedding call or partial retrieval work and then raise the same final error. Error-text validation alone does not establish execution order or absence of cost-bearing side effects.
3. The fake turns any accidental search invocation into an immediate failure, proving the dependency was not touched. Call assertions can also work, but a raising fake directly enforces the forbidden-operation boundary throughout execution.
4. These scripts are defined to compare or evaluate dense and hybrid behavior, not provide a degraded lexical-only result. Running only lexical would change the experiment, make reports incomparable, and could misleadingly label an incomplete evaluation as valid.
5. A consistent error gives operators and automation one recognizable failure class and one recovery action: run indexing. It reduces diagnostic ambiguity and simplifies runbooks and CI handling.

### User answer evaluation - Step 86 - 2026-07-26

#### Overall verdict
Accepted. This is interview-pass.

#### What was correct
- Q1 correctly applies minimal-change discipline and identifies regression protection as the missing requirement.
- Q2 correctly separates final output validation from proof that no earlier cost-bearing or partial work occurred.
- Q3 correctly explains how a raising fake enforces the forbidden-operation boundary.
- Q4 correctly rejects a silent lexical fallback because it would alter the experiment and legitimize incomplete data.
- Q5 correctly connects standardized errors to both automated handling and operator recovery.

#### Gate
Step 86 interview gate accepted. Proceed to Step 87 when ready.

## Step 87 - Typed UI API client boundary

### Questions asked
1. Why should the Streamlit page call a dedicated API client instead of using `httpx` directly throughout the UI code?
2. Why does the client validate successful JSON against the existing Pydantic response models?
3. Why should timeout, connection, HTTP, and invalid-response failures have different internal error codes but safe generic user messages?
4. Why does an injected `httpx.Client` remain caller-owned while a client created internally is closed by `RagApiClient`?
5. What backend contract drift will the malformed/schema-invalid response tests detect before the visual UI is built?

### Correct answer key
1. A dedicated boundary centralizes URLs, timeouts, serialization, validation, and error policy. The UI remains presentation-focused, while HTTP behavior is independently testable and reusable by other frontends.
2. HTTP 200 does not guarantee a usable contract. Pydantic catches missing fields, wrong types, and incompatible shapes before presentation code accesses them, turning silent UI corruption into an explicit invalid-response failure.
3. Internal codes let the UI choose appropriate behavior, telemetry, and retry guidance. User messages should remain safe and stable because raw response bodies or network exceptions may contain hosts, paths, dependency state, or other implementation details.
4. Resource ownership should follow construction ownership. The wrapper must close resources it creates, but closing a caller-supplied client could break other components that intentionally share that client.
5. They detect non-JSON success bodies and successful JSON that no longer matches required health/readiness/query fields. This protects the UI from backend version drift even when HTTP status remains 200.

### User answer evaluation - Step 87 - 2026-07-26

#### Overall verdict
Accepted with wording corrections. This is interview-pass.

#### What was correct
- Q1 correctly identifies centralization of URL, timeout, validation, and independently testable HTTP behavior.
- Q2 correctly rejects HTTP 200 as sufficient proof and identifies missing fields, wrong types, and invalid response shapes.
- Q3 correctly connects internal codes to retry/warning/failure behavior and safe messages to sensitive-detail protection.
- Q4 contains the correct ownership principle: the wrapper closes resources it creates and must not close a caller-owned client that may be shared elsewhere.
- Q5 correctly identifies malformed JSON and schema/type mismatches that can occur even with HTTP 200.

#### Areas to sharpen
- Q1 should say the dedicated client centralizes `httpx` behavior; it does not represent behavior separate from `httpx`.
- Q2 should say contract validation prevents invalid data from reaching the UI, rather than "validation corruption."
- Q4 should omit "pragma case." The precise rule is construction ownership: the component that constructs the resource owns its lifecycle unless the contract explicitly transfers ownership.
- Q5 protects the UI from backend contract drift; it does not prevent the backend error itself.

#### Stronger interview-quality answer
A dedicated API client centralizes base URLs, timeouts, request serialization, response validation, and error policy. Streamlit remains a thin presentation layer, while the underlying `httpx` behavior can be tested independently and reused by other frontends.

Pydantic validation is required because HTTP 200 only means the request succeeded at the protocol level. The body may still omit required fields, contain incorrect types, or use an incompatible schema. Validation catches that drift before invalid data reaches UI rendering code.

Distinct internal codes allow the UI and telemetry to distinguish timeout, unavailability, readiness, HTTP, and contract failures. User-facing messages remain generic because raw response bodies and network exceptions may expose hosts, paths, dependency state, or configuration.

Resource lifecycle follows construction ownership. `RagApiClient` closes an HTTP client it creates internally. A client injected by the caller remains caller-owned because it may be shared with other components, and closing it inside the wrapper could break them.

Malformed and schema-invalid tests detect non-JSON success bodies, missing fields, incorrect types, and incompatible response shapes. They protect the UI from backend contract drift even when the server still returns HTTP 200.

#### Gate
Step 87 interview gate accepted. Proceed to Step 88 when ready.

## Step 88 - Streamlit grounded-query interface

### Questions asked
1. Why should every UI query perform readiness validation even if the user already clicked **Check backend**?
2. Why is insufficient evidence rendered as a valid refusal result, while backend unavailability is rendered as an operational error?
3. Why should the UI omit blank optional filters rather than send empty strings to the API?
4. Why does the UI display the trace ID but not the local trace output path?
5. What does live browser QA prove that unit tests of `_build_query_request(...)` and `_run_ready_query(...)` cannot prove?

### Correct answer key
1. Readiness is time-sensitive and may change after the earlier manual check. Rechecking at submission prevents stale UI state from starting cost-bearing retrieval/generation after a dependency has failed.
2. Insufficient evidence is an expected grounded-RAG outcome: the system worked and correctly refused to hallucinate. Backend unavailability means the operation could not execute because an operational dependency failed.
3. Empty strings can be interpreted as real filter values, cause false zero-result retrieval, or create different cache/trace semantics. Omitting them preserves the API's optional-field contract.
4. The trace ID is a safe correlation identifier for support and debugging. The output path exposes server filesystem structure and is not useful to a browser user who cannot access the backend filesystem.
5. Live QA proves the Streamlit app starts, renders the intended layout, exposes usable controls, and displays safe validation/error states in the actual framework. Unit tests prove logic and ordering but not visual composition, widget wiring, runtime imports, or framework rendering.

### User answer evaluation - Step 88 - 2026-07-26

#### Overall verdict
Not yet interview-pass. Q1, Q4, and Q5 are accepted. Q2 and Q3 require correction. The gate is paused at the user's request.

#### What was correct
- Q1 correctly explains that readiness is time-sensitive and must be rechecked before cost-bearing retrieval.
- Q4 correctly distinguishes a safe correlation ID from a sensitive and browser-useless server filesystem path.
- Q5 correctly identifies runtime startup, layout, widget presence, and actual framework rendering as evidence supplied by browser QA.

#### Areas to improve
- Q2 incorrectly describes insufficient evidence as an operational-efficiency problem. It is a successful grounded-RAG outcome: retrieval and sufficiency evaluation ran, but the evidence did not meet the answer threshold, so refusal prevents hallucination. Backend unavailability is the operational failure.
- Q3 should not claim that empty filters directly hallucinate results. They can be treated as literal filter values, cause false zero-result retrieval, alter cache/trace semantics, and violate the intended optional-field contract. Hallucination prevention is handled by sufficiency and grounded generation, not by trimming filters alone.
- Q5 should separate the evidence clearly: unit tests prove pure logic and ordering; live QA proves Streamlit startup, widget wiring, visual composition, and rendered error states.

#### Stronger interview-quality answer
Insufficient evidence is a valid refusal because the RAG pipeline executed successfully and determined that retrieved evidence did not meet the sufficiency threshold. Refusing is the grounded behavior that prevents unsupported generation. Backend unavailability is different: the pipeline could not execute because a required operational dependency was unavailable.

Blank optional filters should be omitted because an empty string may be interpreted as a literal filter value, producing false zero-result retrieval or inconsistent cache and trace semantics. Omitting it preserves the API contract that the filter is absent.

#### Gate
Step 88 remains paused. When the user is ready, rewrite Q2 and Q3 only. Do not proceed to Step 89 until those corrections are accepted and the user's questions or requested changes are addressed.

### Gate acceptance - Step 88 - 2026-07-26

#### Rewritten answers accepted
The rewritten Q2 and Q3 are accepted.

#### Why accepted
- Q2 now correctly separates an expected evidence-insufficiency refusal from an operational dependency failure.
- Q3 now correctly explains that empty strings may become literal filters, cause false zero-result retrieval, or change cache/trace semantics, while omission preserves the optional-field contract.

#### Gate
Step 88 interview gate accepted. Project progression remains paused at the user's request while their questions and requested changes are addressed. Do not begin Step 89 yet.

## Step 89 - Reproducible uv project migration

### Questions asked
1. Why is committing `.venv` the wrong solution for reproducible deployment, while committing `uv.lock` is correct?
2. What different responsibilities do `pyproject.toml` and `uv.lock` have?
3. Why should `pytest` be in a development dependency group rather than runtime dependencies?
4. Why use `uv run --locked` and `uv sync --locked` in CI instead of plain `uv run` or `uv sync`?
5. Why is a generated `requirements.txt` acceptable for a legacy hosting platform while a manually maintained second requirements file is risky?

### Correct answer key
1. `.venv` contains platform-specific interpreters, paths, binaries, and installed artifacts; it is large, non-portable, and disposable. `uv.lock` records exact universal dependency resolution while allowing each target environment to build compatible local artifacts.
2. `pyproject.toml` declares project metadata, supported Python versions, direct dependency constraints, and dependency groups. `uv.lock` records the exact resolved direct and transitive versions, sources, markers, and hashes used for reproducible installation.
3. Tests are not needed to run FastAPI or Streamlit in production. Keeping pytest in `dev` reduces production attack surface, image size, install time, and unnecessary dependency conflicts while preserving the development workflow.
4. Plain project commands may automatically relock when metadata changes. `--locked` makes CI fail when `pyproject.toml` and `uv.lock` disagree instead of silently changing dependency resolution during a supposedly reproducible build.
5. A generated export is a deterministic adapter derived from the canonical lock and can be regenerated. A manually maintained second file can drift from `pyproject.toml` or `uv.lock`, producing different local, CI, and production environments.

### User answer evaluation - Step 89 - 2026-07-26

#### Overall verdict
Pass. All five answers identify the correct production and reproducibility
contracts.

#### Precision improvements
- A universal lock can represent resolution for multiple supported platforms,
  but it does not promise compatibility with every possible operating system,
  architecture, or unavailable wheel.
- Because this project does not use Docker, the production benefit of excluding
  pytest should be described as a smaller deployed environment, faster
  installation, fewer conflicts, and reduced attack surface rather than only a
  smaller image.
- `pyproject.toml` remains the source of dependency intent, while `uv.lock` is
  the source of the exact resolved environment. A generated requirements export
  is only a compatibility adapter.

#### Gate
Step 89 interview gate accepted. Proceed to Step 90.

## Step 90 - Locked GitHub Actions CI

### Questions asked
1. Why pin both the `setup-uv` action and the installed uv version?
2. Why run `uv lock --check` before `uv sync --locked` even though locked synchronization also detects inconsistency?
3. Why should the normal CI suite avoid requiring model API credentials or a live Qdrant collection?
4. Why does dependency caching improve speed without weakening lockfile reproducibility?
5. What can the local workflow contract tests prove, and what can only an actual GitHub Actions run prove?

### Correct answer key
1. Pinning the action by commit protects the CI bootstrap code from an
   unexpected mutable-tag change. Pinning the uv executable separately keeps
   lock parsing, synchronization, and command behavior reproducible because the
   action and the tool it installs are different supply-chain layers.
2. `uv lock --check` provides an early, explicit lock-freshness gate with a
   focused error before installation work begins. `uv sync --locked` is still
   required to ensure environment creation cannot silently update the lock.
3. The normal suite should use fakes and dependency boundaries so pull requests
   remain deterministic, safe for forks, inexpensive, and independent of
   network or backend availability. Credentialed live-system tests belong in a
   separately controlled integration workflow.
4. The cache only reuses downloaded or built package artifacts. The committed
   lockfile still determines which exact artifacts may be installed, and uv
   validates them rather than treating the cache as dependency authority.
5. Local contract tests prove that the workflow file contains the intended
   triggers, permissions, pins, timeout, and locked commands. Only a hosted run
   proves GitHub can parse the YAML, obtain the actions, restore the cache,
   provision Linux/Python, install dependencies, and execute the suite in the
   real runner environment.

### Gate
Step 90 implementation is complete. Await the user's answers before proceeding
to the conversation-model step.

### User answer evaluation - Step 90 - 2026-07-27

#### Overall verdict
Pass. All five answers clearly distinguish the CI controls and the guarantees
each layer provides.

#### What was strong
- The two pins were correctly treated as separate bootstrap and tool
  supply-chain controls.
- Lock freshness was correctly separated from immutable environment creation.
- Ordinary PR validation was correctly kept deterministic, safe for forks,
  inexpensive, and independent of live credentials and services.
- Cached artifacts were correctly distinguished from lockfile authority.
- Static workflow-contract evidence was correctly separated from real hosted
  runner evidence.

#### Gate
Step 90 interview gate accepted. Proceed to Step 91.

## Step 91 - Conversation domain and local persistence

### Questions asked
1. Why should conversation persistence be hidden behind a `ConversationStore`
   protocol instead of calling SQLite directly from FastAPI or Streamlit?
2. Why are conversation summaries stored as versioned checkpoints with a
   `summarized_through_sequence` value?
3. Why must messages have per-conversation sequence numbers instead of relying
   only on timestamps?
4. Why does archiving keep a conversation readable but prevent further writes?
5. What failure modes are covered by durability, isolation, forward-only
   summary, and use-after-close tests?

### Correct answer key
1. The protocol separates domain/application behavior from storage technology,
   keeps API and UI code thin, makes tests deterministic, and allows a future
   Oracle adapter to replace SQLite without changing conversation orchestration.
2. The checkpoint identifies exactly which messages the summary covers, while
   the version records each replacement. This prevents duplicate context,
   supports traceability, and makes stale or backward-moving summary writes
   detectable.
3. Timestamps can collide, have different precision across databases, and do
   not by themselves express a stable order within one conversation. A unique
   sequence number provides deterministic reconstruction and summary
   boundaries.
4. Archive is a reversible lifecycle state, unlike destructive deletion.
   Readability preserves history and auditability, while blocking writes avoids
   silently reviving a conversation the UI considers closed.
5. Durability proves history survives store reopening; isolation proves one
   conversation cannot read another's messages; forward-only tests prevent
   stale summaries from replacing newer coverage or claiming nonexistent
   messages; use-after-close tests give a deliberate application error instead
   of leaking a low-level SQLite failure.

### Gate
Step 91 implementation is complete. Await the user's answers before proceeding
to context budgeting and rolling summarization.

### User answer evaluation - Step 91 - 2026-07-27

#### Overall verdict
Pass, with precision improvements required in Q2 and Q3. The answers identify
the correct production contracts and are sufficient to proceed.

#### What was strong
- Q1 correctly separates application behavior from storage technology and
  recognizes that SQLite can later be replaced without coupling the API or UI
  to the new database.
- Q2 correctly identifies checkpoint coverage, duplicate-context prevention,
  and traceability.
- Q3 correctly recognizes timestamp collisions and the need for deterministic
  message ordering.
- Q4 clearly distinguishes reversible archival from destructive deletion and
  connects read-only history to auditability.
- Q5 correctly maps durability, isolation, forward-only summaries, and
  use-after-close behavior to the failures each test is intended to expose.

#### Precision improvements
- The `summarized_through_sequence` value identifies the exact final message
  covered by a summary. The version is a separate monotonic revision number for
  summary replacements. Together they detect stale writes, avoid sending both
  summarized and raw copies of the same history, and support traceability.
- Sequence numbers are scoped to one conversation. Their main guarantee is a
  stable total order and an exact summary boundary. Timestamps can collide and
  database timestamp precision can differ; simultaneous users are not the only
  reason timestamps are insufficient.
- The store protocol also improves deterministic unit testing by allowing a
  fake or alternative adapter, not only a future production database.

#### Stronger interview-quality answer
Persistence belongs behind a `ConversationStore` protocol so domain and
application behavior do not depend on SQLite APIs or schema details. FastAPI
and Streamlit remain thin consumers of the same contract, tests can substitute
a deterministic adapter, and a future Oracle implementation can replace the
local SQLite adapter without rewriting conversation orchestration.

A summary checkpoint records the exact message sequence through which history
has been compressed. Its version records successive replacements. This makes
summary coverage auditable, prevents duplicate raw and summarized context, and
allows stale or backward-moving updates to be rejected.

Per-conversation sequence numbers provide deterministic ordering and exact
summary boundaries. Timestamps may collide, vary in precision across databases,
or fail to express a stable order for messages created close together.

Archiving is a reversible lifecycle transition. Keeping archived conversations
readable preserves history and auditability, while rejecting new writes avoids
silently reopening a conversation the application considers closed.

Durability tests catch data loss across store reopen; isolation tests catch
cross-conversation leakage; forward-only checkpoint tests catch stale summaries
and claims beyond existing messages; use-after-close tests ensure a deliberate
domain-level error is raised instead of exposing an incidental SQLite failure.

#### Gate
Step 91 interview gate accepted. Proceed to Step 92.

## Step 92 - Token-aware context budgeting and rolling summarization

### Questions asked
1. Why must conversation history receive only the context capacity left after
   reserving tokens for system instructions, retrieved evidence, and the answer?
2. Why should rolling summarization be triggered by tokens rather than a fixed
   message count, and why keep recent messages verbatim?
3. How do `summarized_through_sequence` and summary versioning prevent duplicate
   context and stale rolling-summary updates?
4. Why is an injectable approximate token counter acceptable for preflight
   budgeting, and where can it still break in production?
5. Why must conversation summaries remain separate from documentary RAG
   evidence, and what should happen if mandatory recent messages or the
   generated summary exceed their allocation?

### Correct answer key
1. The model context window is shared by every prompt component. Allowing
   history to consume it all can truncate system rules, crowd out fresh
   evidence, leave no output capacity, or cause a hard context-length failure.
   Explicit reserves make the prompt allocation predictable and protect
   grounding and answer generation.
2. Equal message counts can contain radically different token volumes, so
   token-triggering reflects the actual model constraint. Recent messages retain
   exact wording, entities, and follow-up references; older history is more
   safely compressed because some detail loss is acceptable there.
3. The checkpoint records the final message represented by the summary, so
   those messages are not also sent verbatim. The version records successive
   replacements. Forward-only persistence rejects coverage moving backward or
   claiming future messages, making stale writes detectable.
4. A deterministic approximation provides cheap preflight control without a
   tokenizer dependency, and dependency injection allows replacement with an
   exact model tokenizer. It can misestimate model-specific tokenization,
   message framing, Unicode, or provider overhead, so production needs a safety
   margin, observed usage, and eventually the selected model's tokenizer.
5. Chat memory can contain user assumptions, prior model errors, and prompt
   injection; it helps interpret intent but cannot prove functional-spec facts.
   Claims still require fresh retrieved evidence and citations. If required
   recent context or a summary cannot fit, the system must fail explicitly or
   request a smaller input rather than silently discard context, invade reserved
   capacity, or save an invalid checkpoint.

### Gate
Step 92 implementation is complete. Await the user's answers before proceeding
to the conversation API.

### User answer evaluation - Step 92 - 2026-07-27

#### Overall verdict
Pass. All five answers identify the essential context-management, grounding,
and failure-handling contracts.

#### What was strong
- Q1 correctly protects instructions, fresh evidence, and response capacity
  from unbounded conversation history.
- Q2 correctly ties compaction to the real token constraint and distinguishes
  exact recent context from lossy older memory.
- Q3 correctly connects checkpoint coverage and versioning to duplicate,
  stale, and invalid updates.
- Q4 correctly treats approximation as replaceable preflight logic and calls
  for safety margins plus observed production usage.
- Q5 correctly separates conversational context from evidence and requires an
  explicit failure when mandatory context cannot fit.

#### Precision improvements
- `summarized_through_sequence` is the control that prevents the same messages
  from appearing in both summarized and raw form. The summary version records
  replacements for traceability; forward-only checkpoint validation rejects
  backward or future coverage.
- Approximation can diverge because of model-specific tokenization, chat-message
  framing, Unicode behavior, and provider-added overhead. Monitoring real usage
  should eventually inform replacement with the selected model's exact
  tokenizer.
- Explicit overflow handling must avoid silent truncation, consuming evidence
  or answer reserves, and saving a summary checkpoint that does not accurately
  represent the retained context.

#### Stronger interview-quality answer
Conversation history receives only the remaining model-window capacity because
system rules, newly retrieved evidence, and answer output are mandatory prompt
components. Explicit reserves prevent memory from weakening grounding or
causing a hard context-length failure.

Compaction is token-triggered because equal message counts can have very
different sizes. Recent messages remain verbatim to preserve exact wording,
entities, and follow-up references, while older history can tolerate controlled
loss through summarization.

The summary checkpoint identifies the exact final message represented by the
summary, preventing those messages from also being sent raw. The version tracks
successive replacements, while forward-only validation rejects stale or future
coverage.

A cheap deterministic estimate is suitable for preflight budgeting when it is
behind a replaceable contract. Because it may miss model tokenization, framing,
Unicode, and provider overhead, production should retain a safety margin,
measure actual usage, and move to an exact model tokenizer where required.

Conversation memory may contain assumptions, prior model errors, or injected
instructions, so it can clarify intent but cannot prove document facts. Fresh
retrieval and citations remain mandatory. If required context cannot fit, the
system must fail explicitly rather than silently truncate history, invade
reserved capacity, or persist an invalid summary checkpoint.

#### Gate
Step 92 interview gate accepted. Proceed to Step 93.

## Step 93 - Conversation API and multi-turn message submission

### Questions asked
1. Why should the conversation routes receive a `ConversationStore` through a
   FastAPI dependency instead of opening SQLite directly inside every handler?
2. Why does message submission persist the user message before running the
   cost-bearing grounded query, but persist the assistant message only after a
   completed answer or safe refusal?
3. Why is conversation memory included in the generation prompt but explicitly
   prohibited from acting as documentary evidence?
4. Why are missing, archived, context-overflow, dependency-unavailable, and
   unexpected failures mapped to different HTTP status classes?
5. What do the conversation API tests prove about isolation and failure
   handling, and what important production guarantees do they not yet prove?

### Correct answer key
1. Dependency injection centralizes store construction and cleanup, keeps
   handlers storage-agnostic, supports deterministic replacement in tests, and
   allows a future Oracle adapter without changing endpoint behavior. Direct
   per-handler SQLite code would duplicate lifecycle and schema coupling.
2. Persisting the accepted user input preserves auditability and enables retry
   after retrieval or model failure. Delaying the assistant write prevents a
   nonexistent answer from being recorded. The resulting user-only partial turn
   is honest but requires UI retry and future idempotency handling.
3. Memory can clarify pronouns, referenced entities, user intent, and prior
   constraints, but it may contain assumptions, prior model errors, or injected
   instructions. Functional-spec claims must therefore still be supported by
   newly retrieved evidence and validated citations.
4. `404` means the resource does not exist, `409` means its archived lifecycle
   conflicts with mutation, `413` means bounded context cannot fit, and `503`
   means a required retrieval dependency is unavailable. Unexpected internal
   failures use a safe `500`. Distinct statuses let clients choose the correct
   recovery behavior without exposing exception details.
5. Tests prove local route wiring, validation, per-conversation history
   isolation, archive enforcement, safe overflow, reuse of grounded query
   execution, trace persistence, and honest partial-turn behavior under a fake
   downstream failure. They do not prove hosted concurrency, multi-process
   SQLite write safety, authentication/authorization, idempotent retries, real
   model follow-up quality, exact tokenization, or live Qdrant/model behavior.

### Gate
Step 93 implementation is complete. Await the user's answers before proceeding
to the Streamlit multi-turn chat UI.

### User answer evaluation - Step 93 - 2026-07-29

#### Overall verdict
Not yet interview-pass. Q1, Q3, and Q5 are accepted with precision
improvements. Q2 and Q4 require correction before proceeding.

#### What was strong
- Q1 recognizes centralized API handling and the ability to replace SQLite with
  Oracle or another persistence technology.
- Q3 correctly identifies memory's role in resolving pronouns, entities,
  intent, and constraints while requiring current document retrieval for
  factual claims that may change across releases.
- Q5 correctly identifies lifecycle, archival, isolation, and simulated failure
  evidence, then distinguishes those tests from real high-concurrency,
  pagination, memory, latency, and streaming behavior.
- The delivery analogy in Q2 correctly conveys why a system must not record a
  downstream outcome before that outcome actually exists.

#### Areas to improve
- Q1 should name FastAPI dependency injection rather than routing alone. The
  dependency centralizes store construction and cleanup, keeps handlers
  storage-agnostic, and lets tests substitute a deterministic store.
- Q2's final sentence says the message should be stored only after completion
  or safe refusal. That reverses the implemented contract. The accepted user
  message is persisted before the cost-bearing query; only the assistant
  message waits for a completed grounded answer or safe refusal. A downstream
  failure therefore leaves an honest, retryable user-only partial turn.
- Q4 must map the statuses rather than merely say different errors help
  debugging: `404` is missing conversation, `409` is an archived-state conflict,
  `413` is context too large, `503` is a required dependency unavailable, and
  safe `500` is unexpected internal failure. The client recovery differs for
  each.
- Q5 could additionally name authentication/authorization, idempotent retries,
  multi-process database safety, real Qdrant/model behavior, and exact
  tokenization as unproven production guarantees.

#### Stronger interview-quality answer
The store is provided through a FastAPI dependency so connection construction
and cleanup are centralized, route handlers depend on the `ConversationStore`
contract rather than SQLite, tests can inject a deterministic store, and a
future Oracle adapter can replace the local implementation without changing
endpoint behavior.

The user message is persisted as soon as valid input is accepted, before the
cost-bearing query, so a later retrieval or model failure does not erase the
user's attempt and the client can retry it. The assistant message is persisted
only after a grounded answer or safe refusal actually exists. This avoids
recording a false successful outcome and deliberately permits an auditable
user-only partial turn.

Conversation memory can clarify references, entities, intent, and constraints,
but it may contain stale assumptions, previous model errors, or injected
instructions. Therefore it cannot prove functional-spec facts; current
retrieval and validated citations remain authoritative.

`404` means the conversation does not exist, `409` means archival conflicts
with a requested mutation, `413` means required context cannot fit its budget,
`503` means a required retrieval dependency is unavailable, and a safe `500`
represents an unexpected internal failure. These distinctions let clients
choose between correcting an ID, starting or selecting another chat, reducing
context, retrying later, or reporting a server fault.

The tests prove local validation, endpoint wiring, durable turn storage,
archive enforcement, conversation isolation, trace linkage, safe overflow, and
partial-turn behavior under simulated failures. They do not prove concurrent
multi-user behavior, authentication and authorization, idempotent retries,
pagination at scale, multi-process SQLite safety, streaming latency, exact
token accounting, or live Qdrant and model behavior.

#### Gate
Step 93 remains open. Rewrite Q2 and Q4 only. Do not proceed to Step 94 until
both corrections are accepted.

### Gate acceptance - Step 93 - 2026-07-29

#### Rewritten answers accepted
The rewritten Q2 and Q4 are accepted.

#### Why accepted
- Q2 now correctly states that accepted user input is persisted for audit and
  retry before retrieval or model execution, while assistant output is delayed
  until a grounded answer or safe refusal actually exists. It also recognizes
  the resulting UI retry and idempotency requirement.
- Q4 now correctly separates `404` missing resources, `409` lifecycle
  conflicts such as archived writes, `413` context-capacity failures, `503`
  unavailable dependencies, and safe `500` internal failures without leaking
  implementation details.

#### Gate
Step 93 interview gate accepted. Proceed to Step 94.

## Step 94 - Streamlit multi-turn chat UI

### Questions asked
1. Why must the backend conversation store remain the source of truth while
   Streamlit session state is only a UI cache?
2. Why should the UI call `/ready` before submitting a cost-bearing conversation
   message, and what race condition still exists after a successful readiness
   response?
3. Why should a user-only partial turn be displayed explicitly instead of
   hidden or automatically paired with an error-shaped assistant message?
4. What value does the optional evidence/debug panel provide, and what
   information should still be hidden from users?
5. What did the unit tests and live browser QA prove separately, and which
   production UI guarantees remain unproven?

### Correct answer key
1. Durable backend storage survives Streamlit reruns, browser refreshes, and
   process restarts and can be shared by API consumers. Session state is tied to
   one UI session and may disappear or become stale. It may cache display-only
   details, but conversation selection and history must be reconciled with the
   backend on each rerun.
2. Readiness prevents known dependency or artifact failures from starting
   retrieval, embedding, or generation cost. It is still a point-in-time check:
   Qdrant, artifacts, networking, or model services may fail between readiness
   and the subsequent message request, so the submission path must retain its
   own safe failure handling.
3. The persisted user message is an honest record of accepted input and enables
   audit and retry. Hiding it loses history; inventing an assistant error message
   misrepresents a completed model outcome. An explicit partial state tells the
   user what happened and supports later idempotent retry design.
4. The panel exposes citations, release labels, scores, sufficiency, trace IDs,
   context-budget state, summary checkpoints, usage, and cost for trust,
   evaluation, and debugging. It should not expose raw backend bodies, stack
   traces, secrets, local filesystem paths, credentials, or unrestricted
   internal logs.
5. Unit tests prove typed request/response handling, state-selection logic,
   filter validation, readiness ordering, partial-turn detection, safe error
   mapping, and presence of required UI contracts. Browser QA proves the real
   processes start, Streamlit renders, controls are wired, readiness appears,
   lifecycle reruns work, and no browser errors occur in the tested path. It does
   not prove concurrent users, accessibility completeness, responsive layouts,
   long-history performance, streaming quality, idempotent retries, live model
   answers, or cross-browser behavior.

### Gate
Step 94 implementation is complete. Await the user's answers before proceeding
to Step 95 evaluation coverage.

### User answer evaluation - Step 94 - 2026-07-29

#### Overall verdict
Pass. All five answers correctly identify the UI state, reliability,
auditability, observability, and validation boundaries.

#### What was strong
- Q1 clearly separates durable backend history from transient UI state and
  correctly requires reconciliation after each Streamlit rerun.
- Q2 correctly treats readiness as a cost-saving point-in-time signal rather
  than a guarantee that dependencies will remain available during submission.
- Q3 correctly connects accepted user input to audit and retry while rejecting
  hidden failures or fabricated assistant outcomes.
- Q4 gives a strong observability boundary: expose grounding, trace, budget,
  usage, and cost signals while withholding secrets and unsafe internal details.
- Q5 accurately separates deterministic contract/logic tests from live
  rendering and wiring evidence, then names the major production guarantees that
  remain unproven.

#### Precision improvements
- Backend reconciliation should also clear or replace a session-selected
  conversation when that conversation has been archived or removed elsewhere.
- A successful `/ready` result does not cover every cost-bearing dependency:
  model APIs are intentionally not called by readiness, and even checked
  dependencies can fail immediately afterward.
- Persistent debug details across sessions would require a backend API or trace
  lookup contract; current Streamlit state only retains details returned during
  the active UI session.

#### Stronger interview-quality answer
Durable backend storage is authoritative because it survives Streamlit reruns,
browser refreshes, and process restarts. Session state is appropriate only for
replaceable display caches and must be reconciled with the backend, including
clearing stale selections.

Readiness blocks work when known required configuration, artifacts, or Qdrant
state is unavailable, reducing wasted cost. It is a point-in-time check and does
not call model APIs, so submission still needs independent safe handling for
dependencies that fail or change after the check.

An accepted user message is an auditable retryable event. If downstream work
fails, the UI should display the resulting partial turn explicitly rather than
erase history or represent an assistant response that never existed.

The debug panel should expose citations, release lineage, retrieval scores,
sufficiency, trace IDs, context budgeting, token usage, and estimated cost.
Secrets, credentials, raw error bodies, stack traces, local paths, and
unrestricted logs must remain hidden.

Unit tests establish typed contracts, validation, state-selection logic,
readiness ordering, partial-turn detection, and safe error mapping. Browser QA
establishes real Streamlit startup, rendering, control wiring, lifecycle reruns,
and browser-console health for the tested path. Neither proves high concurrency,
full accessibility, responsive behavior, long-history performance, streaming,
idempotent retry, real model quality, or cross-browser compatibility.

#### Gate
Step 94 interview gate accepted. Proceed to Step 95.

## Step 95 - Complete, token-bounded prompt evidence

### Questions asked
1. Why must the evidence sent to the LLM be separate from the short citation
   preview returned to the API and Streamlit UI?
2. Why does the evidence budget admit complete ranked units instead of
   truncating text exactly at the remaining character or token limit?
3. Why should an oversized highest-ranked evidence unit cause a safe refusal
   before the LLM call?
4. Which part of the reported BOR problem does Step 95 fix, and why can the
   “17 reduced to 4” answer still fail?
5. What do the new regressions prove, and what answer-level evaluation is still
   required before claiming the BOR scenario is solved end to end?

### Correct answer key
1. The model needs complete factual evidence, while clients need compact,
   bounded display metadata. Reusing a 240-character preview as prompt evidence
   can remove a decisive acronym or table row and force an unnecessary refusal.
2. Mid-unit truncation can separate labels from values or omit later rows,
   changing a table's meaning. Whole-unit admission preserves integrity and
   makes budget behavior deterministic; exact model tokenization can later
   replace the lightweight estimate.
3. Calling the model with a knowingly incomplete highest-ranked unit risks an
   unsupported or misleading answer and wastes cost. A deliberate application
   refusal is observable, testable, and safer than accidental truncation.
4. It fixes the prompt-construction loss: BOR and B-04 facts beyond character
   240 can now reach the model. It does not fix hybrid ranking; the trace showed
   that the decisive R24 realignment tables were outside hybrid top-5 for the
   second natural query.
5. The tests prove full selected evidence reaches the prompt, previews stay
   short, whole-unit budgeting works, and oversized evidence avoids an LLM call.
   We still need a natural-language answer-level case asserting the supported
   facts “17 currently” and “4 after R24,” plus retrieval recall/rank checks for
   all required supporting units.

### Gate
Step 95 implementation is complete. Await the user's answers before proceeding
to the retrieval-ranking and current-state evaluation step.

## Step 96 - Weighted Reciprocal Rank Fusion

### Questions asked
1. Why is adding normalized dense cosine scores to lexical relevance scores
   unsafe even when each list is divided by its own maximum?
2. How does weighted RRF combine retrievers, and why is the rank constant
   important for a short candidate list?
3. Why was the final RRF score normalized to `[0,1]`, and how is that different
   from normalizing the dense and lexical input scores?
4. Why is a lexical-heavy weight defensible for this corpus, and what evaluation
   risk would make that choice unsafe to generalize?
5. What did the failed live answer prove even after retrieval improved?

### Correct answer key
1. The raw scores have different definitions and distributions. Dividing each
   by its query maximum does not calibrate them; it merely makes each winner
   equal to one. With fixed weights, a mediocre dense-only result could still
   outrank lexical rank 1 regardless of its absolute lexical relevance.
2. Each result contributes `weight / (k + rank)` for every retriever that found
   it. Smaller `k` preserves more separation between early ranks; large `k`
   makes ranks within a short list nearly equal and can over-reward any overlap.
3. Final normalization preserves ordering while producing a bounded score
   compatible with logging and the existing sufficiency threshold. It does not
   compare or combine the retrievers' raw score magnitudes; only RRF rank
   contributions have already been combined.
4. Exact report IDs, abbreviations, action words, and table headers are
   unusually important in functional specifications, and lexical retrieval
   ranked the decisive tables first. The weight must still be evaluated across
   semantic paraphrases and unanswerable cases; optimizing only the BOR example
   would overfit and could damage dense-retrieval recall.
5. It proved that retrieval recall is necessary but insufficient. Even with
   detailed R24 change evidence in the prompt, the model applied the wrong
   meaning of “current,” so explicit latest-deployed-release semantics,
   release-aware query scoping, and answer-level fact assertions are required.

### Gate
Step 96 is complete. Await the user's answers before implementing current-state
temporal synthesis and conversation-aware release scoping.

### User answer evaluation - Step 96 - 2026-07-30

#### Overall verdict
Pass. All five answers correctly explain score calibration, RRF mechanics,
bounded final scoring, corpus-specific weighting, and the separation between
retrieval recall and temporal answer synthesis.

#### What was strong
- Q1 precisely identifies that per-query maximum division does not calibrate
  dense and lexical scores or make their magnitudes comparable.
- Q2 gives the correct weighted RRF equation and accurately explains how the
  rank constant changes top-rank separation.
- Q3 correctly distinguishes bounded output-score normalization from combining
  incompatible retriever input scores.
- Q4 justifies lexical emphasis using exact identifiers and table language while
  naming paraphrase and hard-negative validation as the protection against
  overfitting.
- Q5 correctly interprets the failed live answer as a temporal-policy defect
  after retrieval recall improved.

#### Precision improvements
- Final RRF normalization also preserves ranking because every fused result is
  divided by the same query-level maximum possible RRF score.
- RRF weights and the rank constant should be selected against a representative
  evaluation set containing semantic paraphrases, exact identifiers,
  multi-table questions, and unsupported queries rather than one BOR case.
- Release-aware retrieval alone is insufficient: answer generation must treat
  the latest relevant deployed release as the effective current state and
  distinguish its pre-change baseline from its resulting state.

#### Stronger interview-quality answer
Dense cosine similarity and lexical relevance have different statistical
meanings and distributions. Dividing each list by its maximum only makes both
winners equal to one; it does not calibrate the remaining values. Weighted RRF
avoids that comparison by summing `weight / (k + rank)` contributions. For
short candidate lists, a smaller `k` retains meaningful separation among early
ranks, while a large `k` compresses them and disproportionately rewards overlap.

The combined RRF score can then be divided by the maximum possible RRF score to
produce a bounded diagnostic and sufficiency value without changing result
ordering or comparing raw retriever magnitudes. Lexical-heavy weighting is
defensible for identifier- and table-heavy functional specifications, but it
must be validated across semantic paraphrases, hard negatives, and unsupported
questions to avoid optimizing only the BOR case.

The failed live answer shows that evidence recall is necessary but not
sufficient. The model received relevant R24 change evidence but interpreted the
pre-change “Existing Functionality” section as current. The next layer must
resolve release scope and explicitly define current state as the result after
applying the latest relevant deployed release.

#### Gate
Step 96 interview gate accepted. Proceed to Step 97 current-state temporal
synthesis and conversation-aware release scoping after resolving the reported
Codex diff-viewer issue.

## Step 97 - Current-state temporal synthesis and release scoping

### Questions asked
1. Why may conversation memory resolve the release referenced by “it” while
   still being forbidden as factual evidence for the answer?
2. Why does a current-state query retrieve a wider candidate set before
   selecting the latest release, instead of immediately filtering to the
   globally highest release?
3. Why must release labels be ordered numerically, and what failure would
   lexicographic ordering cause?
4. Why is `Existing Functionality` in R24 treated as the pre-change baseline
   rather than the current deployed result in this project?
5. What does the answer-level evaluation prove beyond retrieval recall, and
   what future corpus change would invalidate the “all releases are deployed”
   assumption?

### Correct answer key
1. Memory is appropriate for intent resolution because it records the user's
   conversational reference, but it may be stale, summarized, or unsupported.
   It can select R24 for retrieval; the model may make functional claims only
   from newly retrieved R24 evidence and citations.
2. The globally highest release may belong to an unrelated topic. Broad
   retrieval first establishes a relevance-bounded candidate set; the highest
   release among those candidates is then selected. Expanding once also allows
   baseline and change tables to survive before the final top-k is applied,
   without paying for a second query embedding.
3. String ordering can place `R9` after `R10` because it compares characters.
   Parsing the numeric suffix produces the actual lineage order:
   `R2 < R9 < R10 < R24`.
4. The corpus represents deployed release changes. R24's Existing Functionality
   describes the state entering that change, while its realignment tables define
   the result after deployment. Therefore 6/17 is historical baseline context
   and 2/4 is the current state.
5. Retrieval recall proves that evidence was available; answer evaluation also
   verifies temporal wording, derived counts, report IDs, release isolation,
   and citation coverage. If drafts, proposals, rejected changes, or partially
   deployed releases enter the corpus, release number alone can no longer prove
   effective current state; lifecycle/deployment metadata becomes mandatory.

### Gate
Step 97 implementation is complete. Await the user's answers before proceeding
to the next step.

### User answer evaluation - Step 97 - 2026-07-30

#### Overall verdict
Pass. All five answers correctly explain the boundary between conversational
intent and evidence, relevance-bounded release selection, numeric lineage,
baseline-versus-resulting-state interpretation, and answer-level evaluation.

#### What was strong
- Q1 correctly allows memory to resolve references while requiring fresh R24
  evidence and citations for every functional claim.
- Q2 accurately explains both reasons for wider candidate retrieval: selecting
  the latest relevant release and retaining complementary baseline/change
  evidence without a second embedding request.
- Q3 gives the correct numeric ordering and clearly rejects lexicographic
  comparison.
- Q4 applies the project's deployed-corpus rule correctly: 6/17 is the R24
  incoming baseline and 2/4 is the resulting current state.
- Q5 distinguishes evidence availability from answer correctness and correctly
  identifies when deployment lifecycle metadata becomes mandatory.

#### Precision improvements
- The selected release is the highest release among relevance-bounded retrieved
  candidates, not the highest release anywhere in the corpus.
- Lifecycle metadata would be required not only for non-final releases, but also
  for drafts, rejected changes, partial deployments, rollbacks, and releases
  deployed only to selected environments or customers.
- Answer evaluation should verify that cited evidence entails the associated
  claims, in addition to checking that required citation units are present.

#### Stronger interview-quality answer
Conversation memory may resolve a reference such as “it” because it captures
the user's prior intent, but it may be stale, summarized, or unsupported. It may
therefore scope retrieval to R24 but cannot prove any functional claim; those
claims require newly retrieved R24 evidence and citations.

Current-state retrieval first gathers a wider relevance-bounded candidate set,
then selects the highest numeric release represented in that set. This avoids a
globally newer but unrelated document, preserves baseline and change tables,
and avoids a second query-embedding request. Numeric parsing is essential
because string ordering can misorder `R9` and `R10`.

For this deployed corpus, R24's Existing Functionality section describes the
state entering the release, while its realignment tables define the resulting
production state. Thus 6/17 is historical baseline context and 2/4 is current.
Answer-level evaluation must verify temporal framing, counts, identifiers,
release isolation, citation coverage, and ultimately citation entailment—not
just retrieval recall. If drafts, rejected changes, partial deployments,
rollbacks, or environment-specific releases enter the corpus, explicit
lifecycle and deployment metadata becomes mandatory.

#### Gate
Step 97 interview gate accepted.

## Step 98 - Conversation/RAG reliability evaluation

### Questions asked
1. Why should end-to-end reliability tests use deterministic retrieval and LLM
   doubles while keeping the real FastAPI, SQLite, context, and persistence
   boundaries?
2. Why is checking only the text of an abstention insufficient, and which
   structured fields must agree with it?
3. What summary drift can the identifier-level evaluator detect, and what
   important semantic drift can it not prove?
4. Why should a failed or overflowed turn retain the accepted user message but
   not create an assistant message?
5. What production claims cannot be made from 300 passing deterministic tests,
   and which additional evaluation layers would be needed?

### Correct answer key
1. Dependency doubles make the gate fast, reproducible, offline, and free from
   model/network variability. Keeping actual application boundaries proves
   routing, context assembly, durability, isolation, and response persistence;
   mocked unit-only wiring would not prove those integrations.
2. Clients and metrics use structured state, not prose alone. A refusal must
   have `is_answered=false`, a non-empty safe `refusal_reason`, and no unsafe
   generated claim; citations, if present, must refer only to admitted evidence.
3. It catches invented `R<n>`, `T-<n>`, and `B-<n>` identifiers and loss of
   explicit release scope. It cannot prove that paraphrased facts, counts,
   negation, temporal meaning, or causal relationships are semantically
   faithful; those need richer assertions or model/human entailment review.
4. The accepted input is an auditable retryable event. Persisting an assistant
   message before grounded completion would create a false success record.
   Clients must explicitly represent the resulting partial turn and use
   idempotency/retry controls in a production workflow.
5. Passing tests do not establish concurrency capacity, latency, streaming
   smoothness, browser/accessibility quality, cross-browser behavior, live-model
   stability, retry correctness, or citation entailment. Add load/soak tests,
   browser and accessibility QA, fault injection, live golden-set evaluation,
   model-based or human entailment review, and production monitoring.

### Gate
Step 98 implementation is complete. Await the user's answers before proceeding
to Step 99.

### User answer evaluation - Step 98 - 2026-07-30

#### Overall verdict
Pass. All five answers are technically correct, concise, and appropriately
separate deterministic integration confidence from claims that require live,
browser, concurrency, or semantic evaluation.

#### What was strong
- Q1 correctly explains both sides of the test boundary: deterministic doubles
  remove external variability, while real application layers prove integration
  behavior rather than isolated mocks.
- Q2 correctly treats abstention as a structured contract and prevents unsafe
  generated content or unadmitted citations from leaking through refusal text.
- Q3 precisely distinguishes identifier drift detection from semantic
  faithfulness.
- Q4 correctly models accepted input as an auditable retryable event and avoids
  false assistant-success records.
- Q5 correctly limits the production claims supported by the suite and names
  the major missing evaluation layers.

#### Precision improvements
- Deterministic doubles must preserve dependency contracts closely; otherwise
  they can make tests pass while real provider schemas, timeouts, or error
  behavior have changed. A small live contract/smoke suite should complement
  them.
- Retryable user messages need an idempotency key or turn state in production;
  otherwise a client retry can duplicate the accepted input.
- “QA” should be made measurable: specify load/soak targets, latency
  percentiles, streaming time-to-first-token and inter-token gaps,
  accessibility standards, browser matrix, and golden-set grounding metrics.
- Monitoring observes failures after deployment; it does not replace
  pre-production load, fault-injection, browser, or entailment evaluation.

#### Stronger interview-quality answer
Deterministic retrieval and LLM doubles make the main regression gate fast,
offline, reproducible, and cost-free, while retaining the real FastAPI, SQLite,
context, grounding, and persistence boundaries proves application integration.
The doubles must remain contract-faithful and should be complemented by a small
live provider smoke suite.

An abstention is a machine-readable state, not merely polite prose. It requires
`is_answered=false`, a safe non-empty `refusal_reason`, no unsupported generated
claim, and citations restricted to admitted evidence. Identifier checks can
detect invented or lost release/report IDs, but cannot prove semantic
faithfulness for paraphrases, counts, negation, temporal meaning, or causality.

Accepted user input remains durable for audit and retry, while an assistant
message is written only after grounded completion or safe refusal. Production
retries additionally require idempotency or explicit turn state. Finally, 300
deterministic tests do not establish concurrency capacity, latency percentiles,
streaming quality, accessibility, browser compatibility, live-model stability,
retry behavior, or citation entailment. Those require measurable load/soak,
browser/accessibility, fault-injection, live golden-set, entailment-review, and
production-observability layers.

#### Gate
Step 98 interview gate accepted.

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

### User answer evaluation

Pass. All five answers meet the Step 110 production rubric.

1. Correctly limits unreviewed cases to regression discovery rather than an
   SME-backed release decision.
2. Correctly separates release grouping from exact document-source identity and
   cross-module confusion detection.
3. Correctly identifies the factual-entailment limits of structural checking.
4. Correctly covers live model cost, local traces/reports, Qdrant reads, and
   the diagnostic value of run metadata.
5. Correctly explains why evidence-bearing refusals are safe only when the
   response remains explicitly non-answering and machine-readable.

### Gate

Step 110 implementation and interview gate accepted. Await the user's explicit
choice: an `--allow-unreviewed` draft baseline or an SME-approved quality-gate
run.

### Draft-baseline result review

The user executed the explicitly labelled draft baseline. It produced 14/30
structural passes, 16/30 failures, and only 1/6 expected abstentions. The result
is correctly treated as diagnostic evidence rather than a quality gate.

### Follow-up interview questions

1. Why is an empty `document_id` on an otherwise valid R2/R24 citation a
   grounding defect rather than a cosmetic reporting issue?
2. Why must we distinguish missing document-ID metadata from retrieval of the
   wrong release before choosing a remediation?
3. What safety/business risk is exposed when five unsupported questions are
   answered instead of refused, even if the answers sound plausible?
4. Why does `estimated_llm_cost=0.0` in this report not demonstrate that the
   draft run had zero real provider cost?
5. What evidence must a repaired rerun produce before we may mark the 30 cases
   SME-approved and use them as a release-quality gate?

### Correct-answer rubric

1. Exact-source identity is necessary for filters, auditability, coverage, and
   citations. A blank ID cannot prove which same-release FDD supported the
   answer, so grounded lineage attribution is incomplete.
2. Metadata backfill/re-indexing can repair a known correct point with absent
   identity; wrong release retrieval requires investigation of query planning,
   filters, ranking, corpus content, and evaluation expectations. Treating both
   as one problem risks an ineffective fix.
3. The system can fabricate functional guidance or lead users to act on a
   nonexistent feature. This violates graceful failure and makes citations look
   like support for a claim the corpus cannot ground.
4. Cost reporting relies on configured per-token prices. Zero configuration
   produces a zero estimate even when the embedding/LLM provider was called;
   provider usage, invoices, or correctly configured price inputs are needed.
5. The rerun must show exact document/release citations for supported cases,
   safe abstention for unsupported cases, no unresolved retrieval confusion,
   trace/retrieval evidence for every result, and SME review of claim
entailment before approval.

## Step 111 — Direct-support decision and six-case abstention rerun

### Interview questions

1. Why can a high retrieval score for “investment limit” still be insufficient
   to answer a question about an “interest rate”?
2. Why is a required `DECISION: ANSWER`/`DECISION: REFUSE` header safer than
   inferring answer state from unconstrained model prose?
3. What should happen when a model omits or misspells the decision header, and
   why?
4. Why is selecting explicit case IDs safer than a prefix `--max-cases` limit
   when rerunning paid failure cases?
5. The six cases now refuse safely but lack explicit follow-up questions. Why
   is that a usability gap rather than a grounding-safety failure?

### Correct-answer rubric

1. Similar vocabulary/retrieval relevance does not establish the requested
   attribute, value, entity, or relationship. Direct evidence must support the
   actual material claim, not merely a nearby topic.
2. A structured header gives the service a machine-readable, auditable state
   that the UI, API, and evaluation can enforce. Free prose can appear hesitant
   while the system still incorrectly marks it as answered.
3. Refuse safely and do not return the unconstrained content. A malformed
   contract cannot be trusted as grounded support; fail closed protects users.
4. A prefix limit depends on ordering and can run unrelated cases or omit a
   requested one. Explicit IDs provide reviewable scope and predictable cost.
5. The system correctly avoids an unsupported claim and records a refusal, so
   grounding safety holds. It still fails the user-assistance goal because the
   user is not given a clear next question to ask.

### User answer evaluation

Pass. All five answers meet the Step 111 production rubric. The user correctly
identified direct support as distinct from similarity, machine-readable
decision-state control, fail-closed malformed-output handling, bounded paid-run
scope, and the difference between safety and recovery usability.

## Step 112 — Enforced helpful recovery guidance for refusals

### Interview questions

1. Why is a deterministic generic follow-up safer than having the application
   generate a specific next question from details that are not in the evidence?
2. Why must the below-score refusal path receive the same follow-up guidance as
   the model-level `DECISION: REFUSE` path?
3. What would be wrong with changing `is_answered=true` merely because a
   refusal includes helpful related citations and a suggested question?
4. Why must we inspect the persisted report rather than rely only on terminal
   logs to confirm the six responses contain the new section?
5. Which two independent defects from the original 30-case report remain after
   the abstention repair, and why should they be investigated separately?

### Correct-answer rubric

1. Specific follow-ups can become unsupported recommendations or leak an
   invented interpretation. A generic prompt directs users to documented scope
   without asserting facts absent from evidence.
2. Both paths are user-visible safe refusals. Inconsistent guidance creates a
   confusing UX and leaves the common low-score failure path less usable.
3. Answer state records whether the requested functional claim was supported,
   not whether the response was helpful. Marking it answered would again break
   API/UI semantics and evaluation safety.
4. Logs report control flow but may truncate or omit generated content. The
   persisted trace/report is the durable local artifact consumed by later
   evaluation and audit, so it must contain the contract.
5. Exact `document_id` payload/citation gaps in legacy R2/R24 evidence and
   wrong-release retrieval in positive confusion/current-state cases remain.
   Metadata backfill/re-indexing and retrieval/query-planning evaluation address
   different root causes and need separate evidence.

### User answer evaluation

Pass. All five answers meet the Step 112 production rubric. The user correctly
identified the safety boundary of generic guidance, consistent refusal behavior,
answer-state semantics, persisted artifact verification, and the separate
metadata versus retrieval-selection defects.

### Gate

Step 112 is accepted. The next investigation is limited to positive-case
citation identity and release selection; the retrieval algorithm must not be
changed until those failure classes are measured separately.

## Step 113 — R21 table retrieval linkage diagnosis

### Interview questions

1. Why does the presence of `table_chunk_10` in the retrieval-ready artifact
   prove table ingestion worked but not that the user query can retrieve it?
2. Why is copying the entire preceding paragraph permanently into the cited
   table text a weaker design than preserving original table text with separate
   parent/section retrieval context?
3. Why should the repair use a new versioned Qdrant collection rather than
   mixing context-enriched table vectors into `functional_specs_v2`?
4. What exact test would prove the repair fixed this question without merely
   improving a lexical score?
5. Why is a generic weighted-RRF tuning change premature given this evidence?

### Correct-answer rubric

1. Extraction proves the unit exists; retrieval also depends on query/unit
   vocabulary, embedding representation, ranking, candidate limit, and context
   linkage. A correct but isolated unit can rank too low.
2. It blurs primary-source boundaries, duplicates prose across points, bloats
   prompts, and can make citations misleading. Separate original display text
   and structured retrieval context retain provenance.
3. Changed retrieval text changes embeddings and deterministic point IDs. A new
   generation prevents old and new evidence representations from mixing and
   keeps rollback/evaluation honest.
4. The exact R21 query must retrieve the context-linked table in the bounded
   evidence set, answer all eleven supported fields, cite the R21 table unit,
   and pass a negative query that still refuses unsupported attributes.
5. The failure is localized to a known parent/table vocabulary disconnect.
   Global ranking changes could regress unrelated queries and would not repair
   the missing relationship; measure linkage first, then compare only if needed.

### User answer evaluation

Pass. All five answers meet the Step 113 production rubric. The user correctly
identified retrieval as a contextual ranking process, protected citation
provenance, required versioned activation, defined an end-to-end proof, and
rejected premature global fusion tuning.

## Step 114 — Deterministic parent-table retrieval relationship model

### Interview questions

1. Why must `retrieval_text` and citeable `text` remain separate fields rather
   than replacing the table source text with enriched text everywhere?
2. Why is an original DOCX paragraph index a more reliable parent-table link
   than matching the table to whichever paragraph happens to have similar text?
3. How does the backward-compatible fallback for old artifacts avoid breaking
   current retrieval while still making their metadata limitations visible?
4. Why does the R21 table improving from rank 244 to rank 2 in an in-memory
   lexical probe prove a targeted mechanism but not authorize v2 activation?
5. What must the controlled migration verify before API/UI traffic moves to a
   new collection containing context-enriched table vectors?

### Correct-answer rubric

1. Retrieval enrichment is derived context, while citation text is the original
   source evidence. Conflating them can misattribute prose, obscure what the
   table actually says, and weaken auditability.
2. Original order directly represents the source structure. Text similarity is
   ambiguous, fails with repeated headings, and can attach an unrelated parent.
3. Old artifacts keep their original search behavior through `text` fallback,
   so they remain readable/indexable. Their missing relationship fields remain
   explicit rather than fabricated, enabling planned migration and evaluation.
4. It proves the context representation addresses this lexical candidate gap in
   controlled memory. v2 still has old vectors/artifacts, and dense/hybrid,
   citations, negative cases, collection integrity, and rollback remain
   unverified.
5. Reprocess intended archived sources with a recorded manifest, confirm source
   hashes/unit counts/context links/embedding dimensions/exact Qdrant points,
   run R21 positive and negative plus broader regression evaluation, verify API
   configuration/readiness, and retain the prior collection for rollback.

### User answer evaluation

Pass. All five answers meet the Step 114 production rubric. The user correctly
preserved citation provenance, preferred source-order identity, described
backward compatibility, limited the in-memory result to its evidence, and
defined a complete staged activation gate.

### Gate

Step 114 is accepted. Build a controlled archived-source staging workflow next;
do not overwrite v2 artifacts or switch API/UI configuration during its setup.

## Step 115 — Isolated all-FDD staged rebuild workflow

### Interview questions

1. Why is it safe to reuse a vector only when the retrieval text, embedding
   model, and artifact/cache compatibility version agree, even though the
   `document_id` and Qdrant point ID still differ?
2. Why must the staging script reject an existing `functional_specs_v3`
   collection rather than upsert into it or automatically delete it after a
   failed run?
3. A real staged run reports 937 units, 780 cached vectors, and 157 newly
   embedded vectors. What do these figures mean operationally, and what would
   you investigate if the newly embedded count were unexpectedly close to 937?
4. Why is an exact payload/point-ID verification stronger than merely seeing a
   Qdrant collection count of 937?
5. After the staged run verifies successfully, what separate evidence is still
   required before setting `QDRANT_COLLECTION_NAME=functional_specs_v3` for the
   API/UI?

### Correct-answer rubric

1. A vector represents exactly the embedded retrieval text under one embedding
   model and compatibility contract. Reusing it when any of those changes risks
   a stale/mismatched vector. Citeability remains distinct: each occurrence
   needs its own document/release/unit payload and deterministic point ID.
2. Existing state can be a successful prior generation or partial/corrupt data.
   Upserting mixes runs; deletion destroys investigation/rollback evidence and
   is especially unsafe on local Qdrant. A new named generation is reviewable
   and recoverable.
3. Total is the complete staged evidence coverage, cached is avoided embedding
   API cost for unchanged inputs, and newly embedded is the changed/new
   retrieval representation. A near-937 miss rate indicates an unintended
   cache-version/model/text change, missing seed cache, or preprocessing drift
   and must be investigated before accepting cost or quality impact.
4. Count proves only cardinality. Exact verification proves every intended
   record has its deterministic ID and expected document identity, release,
   original citeable text, retrieval representation, and parent relationship;
   it detects stale, duplicate, or wrong-payload points.
5. Run the R21 table positive and unsupported-negative tests against the staged
   artifacts/collection, broader FDD retrieval and grounded-answer evaluation,
   citation/release correctness checks, readiness/configuration inspection,
   rollback rehearsal with v2 retained, and SME review of material answers.

### User answer evaluation

Pass. All five answers meet the Step 113 production rubric. The user correctly
identified retrieval as a contextual ranking process, preserved citation
provenance, required collection versioning for changed vectors, defined an
end-to-end repair test, and rejected premature global fusion tuning.

### Gate

Step 113 is accepted. Implement explicit parent-table context relationships;
do not modify the weighted-RRF algorithm in this step.
