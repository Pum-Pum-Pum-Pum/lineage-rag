# Project Plan — Enterprise Functional Spec RAG Assistant

## 1. Project Objective
Build a production-minded enterprise RAG system for functional specification documents so that you learn, step by step, how a real LLM application is designed, evaluated, debugged, and improved.

This project is not just about making a chatbot work. It is about learning how to build a grounded, maintainable, interview-worthy AI system with realistic enterprise constraints.

---

## 2. Why this project matters for learning and interviews
This project helps you practice the exact skills that are valuable in applied LLM, RAG, AI engineer, and enterprise GenAI roles.

By the end of the project, you should be able to explain:
- how document ingestion works in messy enterprise settings
- why chunking matters and how chunking affects retrieval quality
- how embeddings, vector search, and lexical search work together
- how grounded prompting reduces hallucinations
- how to make a system say "I don't know" safely
- how to evaluate retrieval quality and answer faithfulness
- how to control token usage, latency, and cost
- how to think about release lineage, re-indexing, caching, versioning, and maintainability
- how to debug failures in a production-minded RAG system

This means the project is both:
- a working portfolio project
- a structured learning roadmap

---

## 3. Architecture direction
We are using a hybrid-cloud architecture:
- Local document storage
- Local processed artifact storage
- Local vector DB
- Internal OpenAI-compatible model APIs for embeddings and chat/generation
- Local backend and local UI
- Conversation-scoped chat history with bounded model context

The application should remain modular so future codebase retrieval, MCP servers,
and Oracle Database persistence can be added through adapters without coupling
them directly to the UI, conversation logic, or core RAG pipeline.

### Why this architecture
This gives you the best balance of:
- strong model quality
- faster development
- local control over artifacts and indexes
- realistic enterprise design tradeoffs
- reproducibility and debuggability

### Retrieval architecture evolution
The retrieval path should evolve in stages instead of starting with maximum complexity:
- baseline dense semantic retrieval first
- lexical retrieval baseline second for comparison
- hybrid retrieval only after baseline evaluation
- optional reranking only if evaluation shows measurable gain

### Reproducibility expectation
The system should support:
- local artifact versioning
- index versioning
- repeatable indexing runs
- debug-friendly intermediate outputs
- locked Python environments through `pyproject.toml` and `uv.lock`

### Deployment constraint
Docker deployment is out of scope for this project because the target Oracle
environment does not allow it. Deployment work should use a locked `uv`
environment and an approved native process or service mechanism. The exact
packaging target will be selected when the Oracle runtime environment is
confirmed.

---

## 4. Recommended stack
- Python
- FastAPI
- Streamlit
- Internal OpenAI-compatible APIs for embeddings and generation
- Qdrant local vector DB
- python-docx
- pydantic-settings
- pytest
- uv with `pyproject.toml` and `uv.lock`
- rank-bm25 or equivalent lexical retrieval utility
- tenacity for bounded retries
- tiktoken for token estimation where needed
- structured logging / JSON logging support

---

## 5. Working principles for the build
- one step at a time
- implement before over-abstracting
- prioritize evaluation early
- optimize retrieval before adding advanced orchestration
- production-minded decisions over hype
- build in a way that supports interview explanation later
- design for cumulative release-aware retrieval, not only latest-document retrieval
- justify every retrieval upgrade with measurable gain
- keep local artifacts for debugging and reproducibility

---

## 6. Document reality we must design for
Your functional specification documents are not simple one-version documents.
They evolve across releases over time.

### Key characteristics
- documents follow a naming convention such as `FS_FCIS_14.4.0.0.0$ASNB_R1`, `..._R2`, `..._R10`
- documents in the same family belong to the same functional lineage
- later releases are usually incremental enhancement documents
- older releases remain valid unless something is explicitly changed later
- the current truth may therefore be distributed across multiple releases

### Important implication
This means the assistant must not assume:
- latest release always contains the full truth

Instead, it must support:
- base behavior from older releases
- additive enhancements from later releases
- more recent overrides when behavior explicitly changes
- historical trace when the user asks how functionality evolved

### Example interpretation
If a feature was introduced in R1 and never changed, then R1 still contains the current truth.
If a screen was introduced in R1 and additional columns were added in R2, R7, and R16, then the current truth is cumulative across those releases.

This is why the project must be designed as a **cumulative release-lineage RAG system**.

### Source content constraints
The source documents are DOCX-based but may contain:
- paragraphs
- tables
- images
- screenshots

For v1:
- paragraph text extraction is in scope
- table extraction is in scope because tables may contain important business rules
- image and screenshot presence should be detected and recorded as metadata markers
- full OCR or vision reasoning is out of scope for v1
- questions that depend only on screenshot or image evidence may require abstention

---

## 7. Release-lineage retrieval design
The system should support three mental modes, even if we do not build all of them fully in the first MVP.

### Mode 1 — Current functional view
Default mode.
The system should assemble the current behavior of a functionality by combining:
- original base behavior from older valid releases
- later enhancements and additions
- later overrides where behavior changed

### Mode 2 — Change trace view
For questions like:
- what changed from R1 to R10
- how did this screen evolve over time

The system should present a timeline-style answer with citations by release.

### Mode 3 — Point-in-time view
For questions like:
- what was the behavior in R2

The system should answer using only the state known up to that release.

### MVP expectation
For the first version, we will focus on:
- cumulative current-state retrieval and answering
- explicit release citations
- basic historical trace support when relevant

We will not claim perfect automatic semantic change reconciliation in v1.

---

## 8. Release-aware metadata requirements
Metadata is now a first-class design requirement.
Each chunk should eventually carry fields like:
- `document_name`
- `document_family`
- `release_label`
- `release_number`
- `source_type`
- `section_title`
- `chunk_id`
- `chunk_index`
- `ingestion_timestamp`
- `content_hash`
- `artifact_version`
- `index_version`

Later, if needed, we may add:
- `feature_area`
- `screen_name`
- `module_name`
- `change_type` such as introduced, enhanced, modified, deprecated, unknown
- `image_presence`
- `table_count`

### Why this matters
Without strong release metadata:
- retrieval may mix outdated and current evidence blindly
- citations become less useful
- change trace becomes weak
- current-state synthesis becomes unreliable
- reproducibility becomes harder

---

## 9. Retrieval strategy evolution
The retrieval system should be built in explicit phases.

### Phase 1 — Dense semantic baseline
- use embedding-based retrieval only
- measure retrieval quality before adding complexity
- inspect where dense retrieval fails on exact terms, abbreviations, field names, and screen labels

### Phase 2 — Lexical baseline
- implement keyword or BM25-style retrieval
- compare lexical retrieval against dense retrieval on the same evaluation set
- use this to identify cases where exact-match signals matter more than semantic similarity

### Phase 3 — Hybrid retrieval
- combine dense and lexical retrieval using a simple fusion strategy
- evaluate whether hybrid improves recall, citation quality, and answer completeness
- do not assume hybrid is better until metrics show it is better

### Phase 4 — Optional reranking
- rerank only a small candidate set if ranking quality remains weak
- add reranking only if evaluation shows measurable improvement worth the added latency and cost

### Principle
Every retrieval upgrade must be justified by measurable evaluation gain, not hype.

---

## 10. Observability, debugging, and reproducibility requirements
The system should be easy to inspect, explain, and debug.

### Minimum expectations
- assign request IDs for query flows
- log retrieval inputs and selected chunks
- log document family and release metadata used in answers
- log prompt template version
- log model name and embedding model version
- log latency for ingestion, indexing, retrieval, and generation
- log abstention reasons when the system refuses to answer
- record index version and artifact version for every run
- keep local debug artifacts for failed or interesting cases

### Why this matters
Production-minded RAG systems must let you answer:
- why did the system return this answer
- which chunks were retrieved and selected
- which index version was active
- whether the failure came from ingestion, retrieval, prompting, or missing evidence

---

## 11. Reliability and failure-handling requirements
The system must fail safely, not just answer safely.

### Required failure modes to handle
- embedding API timeout or transient failure
- generation API timeout or transient failure
- corrupt or weakly parsed DOCX files
- partial indexing failures
- vector DB unavailability
- malformed or missing metadata
- empty retrieval results
- overlong prompt context
- unsupported content such as screenshot-only evidence in v1

### Minimum reliability behavior
- bounded retries for transient API failures
- explicit logging of failed documents and failed requests
- safe fallback when retrieval is empty or insufficient
- partial progress preservation during indexing when possible
- clear backend error messages without leaking secrets

---

## 12. Security, privacy, and data-governance assumptions
Even for a local-first MVP, the design should show enterprise discipline.

### Security and governance expectations
- use environment variables for internal API credentials
- never store secrets in source control
- avoid logging sensitive content unnecessarily
- keep processed artifacts and indexes local unless intentionally exported
- version indexes and processed artifacts for reproducibility
- define a rollback strategy for bad indexing runs
- preserve lineage of old releases even when newer releases are added

### Why this matters
Enterprise credibility is weaker if the project ignores secrets handling, data governance, and reproducibility.

---

## 13. Stage-by-stage project roadmap
The project will be learned and implemented in small stages. Each stage should teach one clear idea and produce one clear artifact.

---

### Stage 1 — Project setup and architecture foundation
#### Goal
Set up the workspace so the project starts clean, modular, and production-minded.

#### What we do in this stage
- create the project folder structure
- define the main modules
- define config handling
- define what data goes where
- define how indexing and querying are separated
- define artifact and index version naming conventions

#### Why this stage matters
Many weak projects fail before the AI part because the codebase becomes messy. This stage teaches disciplined engineering from day one.

#### What you learn
- project organization
- modular thinking
- separation of concerns
- production-minded structure
- reproducibility planning

#### Output of this stage
- clean folder structure
- `docs/project_plan.md`
- `.clinerules`
- base files like `README.md`, `requirements.txt`, `.env.example`
- initial versioning and logging conventions

#### Interview angle
Be able to explain:
- why you separated ingestion, retrieval, vector store, llm, and ui modules
- why production systems should separate indexing flow from query flow
- why reproducibility starts at project structure level, not only at model level

---

### Stage 2 — Document ingestion from functional specification files
#### Goal
Read functional specification Word documents and extract usable raw content.

#### What we do in this stage
- load DOCX files from local storage
- extract paragraph text
- extract table content in structured text form
- preserve section structure where possible
- detect images and screenshots and record their presence as metadata
- parse release information from document filenames
- group related releases into document families
- store raw extracted output locally for debugging
- explicitly log unsupported or weakly extracted content

#### Why this stage matters
If ingestion is weak, everything later becomes weak. Bad ingestion leads to bad chunks, bad retrieval, and hallucinated answers. In your case, bad ingestion also means losing release lineage and silently missing important table content.

#### What you learn
- document extraction problems in enterprise settings
- hidden noise in source files
- why preserving structure improves downstream retrieval
- why naming conventions can be critical metadata signals
- why table handling matters in business documents

#### Output of this stage
- ingestion script
- extracted text artifacts in `data/processed/`
- extracted table artifacts
- parsed release and family identifiers
- image or screenshot presence markers
- unsupported-content and ingestion-quality logs
- understanding of what the source docs really look like

#### Interview angle
Be able to explain:
- why enterprise document ingestion is harder than toy PDF or DOCX demos
- what can go wrong during text extraction
- why filename parsing is important in your domain
- why table extraction matters
- how you scoped screenshots and images honestly in v1

---

### Stage 3 — Text cleaning, normalization, and metadata extraction
#### Goal
Clean the extracted text and prepare consistent metadata for downstream chunking and retrieval.

#### What we do in this stage
- normalize whitespace
- remove repeated noise where needed
- preserve headings and section labels
- attach metadata such as document family, release number, section title, chunk lineage, timestamps, source type, and content hashes
- design metadata so later releases can be linked to earlier releases in the same functional family
- validate metadata quality before chunking

#### Why this stage matters
Retrieval quality depends heavily on clean and structured input. Metadata is also critical for citations, filtering, debugging, cumulative release-aware answering, and future re-indexing.

#### What you learn
- why preprocessing still matters in LLM systems
- why metadata is a first-class design decision
- how traceability is built into enterprise AI systems
- why release lineage is part of correctness, not just organization

#### Output of this stage
- cleaned text artifacts
- metadata schema definition
- normalized document representation ready for chunking
- metadata validation rules

#### Interview angle
Be able to explain:
- why metadata matters in vector retrieval
- how poor preprocessing affects answer quality
- how release metadata changes retrieval behavior
- why metadata validation matters before indexing

---

### Stage 4 — Section-aware chunking and context-window strategy
#### Goal
Split documents into meaningful chunks that are good for retrieval and safe for the model context window.

#### What we do in this stage
- detect sections and headings where possible
- chunk by section first
- split oversized sections into smaller paragraph-aware chunks
- preserve useful table context in chunkable form
- add overlap only where it helps
- ensure every chunk inherits release-lineage metadata
- track chunk lineage for debugging and later cumulative assembly

#### Why this stage matters
Chunking is one of the most important parts of RAG. A large 10-page section should not be passed as one unit. Large chunks hurt retrieval quality and increase token cost.

#### What you learn
- chunk size tradeoffs
- context-window limitations
- why naive fixed chunking is often weak
- how section-aware chunking improves grounding and citations
- why chunk lineage matters when releases evolve over time

#### Output of this stage
- chunked document artifacts
- chunking config decisions
- chunk metadata including section references, source types, and release metadata

#### Interview angle
Be able to explain:
- why chunking matters more than many beginners expect
- how you would handle very large sections safely
- what tradeoff exists between small chunks and large chunks
- why chunk metadata must preserve release identity

---

### Stage 5 — Embeddings and local vector indexing
#### Goal
Convert chunks into embeddings and store them in a local vector database.

#### What we do in this stage
- call the embedding API for each chunk
- store vectors and metadata in local Qdrant
- create a repeatable indexing pipeline
- prepare local persistence for debugging and re-indexing
- preserve document family and release-aware metadata in the index
- record artifact and index versions

#### Why this stage matters
This is where the document corpus becomes searchable semantically. Without a clean indexing pipeline, the system cannot retrieve relevant evidence reliably.

#### What you learn
- what embeddings represent
- how vector search works
- why local vector storage is useful even with cloud models
- why indexing should be treated as a pipeline, not a one-off script
- why metadata-rich indexing is important in enterprise RAG

#### Output of this stage
- local vector collection
- indexing script
- chunk-to-vector mapping with metadata
- index version record

#### Interview angle
Be able to explain:
- how embeddings help semantic search
- why Qdrant was chosen over simpler options
- why vector DB choice affects maintainability and filtering
- why metadata-aware indexing matters for cumulative truth assembly

---

### Stage 6 — Baseline retrieval implementation and comparison
#### Goal
Build retrieval in measurable phases instead of jumping directly to hybrid complexity.

#### What we do in this stage
- implement dense semantic retrieval first
- inspect dense retrieval scores and failure cases
- implement lexical retrieval baseline for exact-match comparison
- compare dense and lexical retrieval on the same evaluation slice
- implement simple hybrid fusion only after baseline comparison
- keep retrieval outputs debuggable at every stage

#### Why this stage matters
Retrieval is the heart of RAG. In enterprise document systems, exact terms, abbreviations, field names, and screen labels often matter. Dense retrieval alone may miss those cases, but hybrid should be added only when justified by evidence.

#### What you learn
- top-k retrieval logic
- retrieval debugging
- recall vs noise tradeoffs
- why more context is not always better
- why enterprise retrieval often needs domain-aware logic, not just similarity search
- how to compare retrieval strategies rigorously

#### Output of this stage
- dense retrieval service
- lexical retrieval baseline
- hybrid retrieval prototype
- retrieval comparison notes
- debug view of retrieved chunks

#### Interview angle
Be able to explain:
- why top-k matters
- how retrieval failures lead to hallucinations
- what dense retrieval missed
- what lexical retrieval recovered
- why hybrid was added only after measurement

---

### Stage 7 — Cumulative release-lineage retrieval pipeline
#### Goal
Retrieve the most relevant chunks for a user question while respecting release history and cumulative truth.

#### What we do in this stage
- retrieve relevant chunks across releases in the same document family
- inspect retrieval scores together with release metadata
- prepare evidence for current-state answers by combining base and enhancement chunks
- avoid blindly treating the latest release as the only truth source
- support point-in-time retrieval behavior for later extension

#### Why this stage matters
In your case, naive top-k retrieval is not enough because the current truth may span multiple releases.

#### What you learn
- cumulative evidence assembly
- release-aware filtering
- retrieval logic beyond naive similarity search
- why domain constraints shape retrieval design

#### Output of this stage
- retrieval service with release-aware logic
- evidence preparation for cumulative answering
- release-aware debug artifacts

#### Interview angle
Be able to explain:
- why latest-document-only retrieval would fail in your document system
- how cumulative retrieval differs from naive retrieval
- how release-aware filtering changes retrieved evidence

---

### Stage 8 — Grounded answering with citations and cumulative synthesis
#### Goal
Generate answers only from retrieved evidence and assemble the current functional view across releases.

#### What we do in this stage
- build a strict prompt template
- pass only retrieved context to the model
- instruct the model not to invent missing information
- tell the model that later releases may enhance or modify older valid functionality
- format answer plus citations with release labels
- clearly distinguish current behavior from historical notes when relevant
- include prompt versioning for reproducibility

#### Why this stage matters
This stage turns retrieval into a usable assistant. It also teaches the difference between generic chat and grounded LLM application design.

#### What you learn
- prompt design for grounded generation
- citation-based trust building
- limiting hallucinations through system design
- how to guide an LLM to synthesize cumulative truth more carefully

#### Output of this stage
- answer generation service
- answer plus evidence format
- citation formatting logic with release labels
- prompt version tracking

#### Interview angle
Be able to explain:
- why grounded prompting is different from generic prompting
- why citations matter for enterprise trust
- why cumulative multi-release synthesis is harder than single-document Q and A
- why RAG is often better than fine-tuning for knowledge access problems

---

### Stage 9 — Graceful failure: handling "I don't know"
#### Goal
Ensure the system refuses unsupported answers safely when evidence is weak or missing.

#### What we do in this stage
- add prompt instructions for abstention
- define simple evidence sufficiency rules
- include explicit unanswerable test cases
- return an insufficient-evidence response when needed
- log abstention reasons for debugging

#### Why this stage matters
A production-grade assistant must fail safely. Hallucinating confidently is unacceptable in enterprise settings.

#### What you learn
- abstention design
- failure-mode thinking
- how to make systems trustworthy, not just fluent

#### Output of this stage
- abstention logic
- unanswerable query test cases
- safe fallback response design
- abstention reason logs

#### Interview angle
Be able to explain:
- how you prevent hallucinated answers
- why saying "I don't know" is a strength, not a weakness
- what signals you use to detect weak evidence

---

### Stage 10 — Token usage tracking, latency, and cost awareness
#### Goal
Track token usage, latency, and estimated API cost for embeddings and answer generation.

#### What we do in this stage
- log tokens used per embedding request
- log tokens used per chat request
- estimate cost by model and request type
- log latency for retrieval and generation stages
- store usage logs locally for analysis

#### Why this stage matters
In enterprise environments, a system must be not only correct but also cost-aware and measurable.

#### What you learn
- API usage observability
- latency and quality tradeoffs
- cost and quality tradeoffs
- why token budgeting matters in prompt design

#### Output of this stage
- token usage logger
- latency records
- local usage records
- cost-awareness notes for experiments

#### Interview angle
Be able to explain:
- why token tracking is important
- how retrieval size affects cost
- how latency changes system design decisions
- how you would control operating cost without hurting quality too much

---

### Stage 11 — FastAPI backend, conversational memory, and Streamlit chatbot UI
#### Goal
Expose the system through a usable multi-turn local chat interface with bounded,
conversation-scoped memory.

#### What we do in this stage
- build FastAPI endpoints for query and health
- connect retrieval and generation flow
- create a Streamlit chatbot UI
- create conversation and message domain models
- persist conversation history behind a `ConversationStore` interface
- include recent turns when resolving follow-up questions
- implement a token-aware context budget
- summarize older turns when the configured context threshold is reached
- retain recent turns verbatim while using a rolling summary for older turns
- reserve context space for system instructions, newly retrieved evidence, and
  the generated answer
- show answers, citations, release labels, and optional debug details
- add a debug evidence panel for retrieved chunks and scores
- add conversation controls such as New Chat, conversation selection, and
  archive or clear actions

#### Conversation-memory boundary
- Memory is scoped to one conversation and is not a source of documentary truth.
- Functional-spec claims must still be grounded in newly retrieved evidence and
  citations.
- Summaries preserve user intent, decisions, constraints, referenced entities,
  and unresolved questions, but must not replace RAG evidence.
- Summary compaction is triggered by token budget rather than a fixed number of
  messages.
- The durable conversation store is the source of truth; Streamlit session state
  is only a UI cache.
- User-level long-term memory across chats is out of scope because the
  application has no login or user-identity feature.

#### Why this stage matters
A usable multi-turn interface turns the system into a real product artifact
rather than a backend-only, single-question experiment. Bounded summarization
also prevents chat history from growing until it crowds out retrieved evidence
or exceeds the model context window.

#### What you learn
- API-driven AI system design
- connecting services cleanly
- building a usable LLM application interface
- why debugging views matter for AI products
- token-aware conversation context management
- summary drift and conversation-isolation failure modes

#### Output of this stage
- FastAPI service
- multi-turn conversation API contract
- conversation-store abstraction and local implementation
- context builder and rolling-summary policy
- Streamlit chat interface with conversation controls
- local demo-ready assistant
- debug evidence panel

#### Interview angle
Be able to explain:
- why FastAPI was used
- how UI and backend responsibilities are separated
- why a debug evidence panel helps trust and debugging
- what would change for a multi-user deployment later
- why recent messages and older summaries have different context roles
- why conversation memory cannot be treated as authoritative RAG evidence

---

### Stage 12 — Evaluation set and retrieval improvement experiments
#### Goal
Measure the system rather than trusting intuition.

#### What we do in this stage
- create an evaluation dataset with answerable and unanswerable questions
- add release-specific evaluation cases
- define retrieval metrics
- define answer quality rubric
- define citation correctness checks
- define abstention metrics
- compare dense, lexical, and hybrid retrieval strategies
- inspect retrieval failures and failure patterns

#### Required evaluation categories
- base-only truth case where R1 is still current because nothing changed later
- incremental enhancement case where current truth spans multiple releases
- explicit modification case where later release overrides earlier behavior
- historical comparison case where the user asks how a feature changed over time
- point-in-time case where truth must be restricted to a past release
- unanswerable case where no release contains the answer
- unsupported-content case where screenshot-only evidence should trigger abstention

#### Required retrieval metrics
- hit@k
- recall@k
- MRR or nDCG where practical
- family-level retrieval correctness
- release-level retrieval correctness

#### Required answer and safety metrics
- groundedness or faithfulness rubric
- citation correctness rate
- answer completeness rubric
- current-state synthesis correctness
- historical trace correctness
- abstention precision
- abstention recall

#### Required efficiency metrics
- retrieval latency
- generation latency
- end-to-end latency
- token usage per query
- cost per query

#### Evaluation labeling policy
- define gold evidence chunks where possible
- define acceptable citations for each query
- define additive vs override release cases
- define point-in-time truth rules
- define what counts as partially correct vs incorrect

#### Why this stage matters
A demo is not enough. Evaluation is what makes the project interview-worthy and production-minded.

#### What you learn
- retrieval metrics
- evaluation discipline
- error analysis for RAG systems
- how to improve the system based on evidence
- why cumulative release-aware evaluation is harder than generic RAG evaluation

#### Output of this stage
- evaluation dataset
- baseline metrics
- experiment notes
- retrieval improvement decisions
- error analysis summary

#### Interview angle
Be able to explain:
- how you evaluated the RAG system
- why evaluation must include unanswerable cases
- why evaluation must include release-lineage cases
- why dense vs lexical vs hybrid comparison matters
- what the major failure modes were and how you improved them

---

### Stage 13 — Caching and incremental re-indexing
#### Goal
Improve maintainability, efficiency, and real-world update behavior.

#### What we do in this stage
- add chunk and content hashing
- avoid re-embedding unchanged chunks
- detect changed documents
- re-index only what changed
- preserve release lineage during re-indexing
- store local cache artifacts for repeatable runs
- maintain index version naming and audit logs
- support safe rebuild vs incremental update modes

#### Why this stage matters
In real systems, documents change. You do not want to rebuild everything every time. This stage teaches scalable update thinking.

#### What you learn
- embedding cache design
- incremental indexing strategy
- why re-indexing is usually more important than retraining
- how document lineage complicates update workflows

#### Output of this stage
- caching strategy
- changed-document detection
- re-index script
- index audit log

#### Interview angle
Be able to explain:
- how you would handle new and updated documents
- why re-indexing comes before retraining in many enterprise RAG systems
- what caching saves in cost and latency
- how you preserve release history when new releases arrive

---

### Stage 14 — Testing, polishing, and interview packaging
#### Goal
Turn the project into a polished portfolio artifact.

#### What we do in this stage
- add unit tests for filename parsing, normalization, and chunking
- add tests for table extraction and metadata validation
- add smoke tests for retrieval
- add regression tests for known failure queries
- add tests for citation formatting and abstention behavior
- add tests for follow-up questions, conversation isolation, context-budget
  overflow, summary triggering, and summary drift
- improve README and project docs
- document tradeoffs and limitations
- prepare project explanation notes for interviews
- create short, medium, and deep-dive explanation formats

#### Why this stage matters
Strong candidates do not just build. They explain, test, and justify what they built.

#### What you learn
- engineering polish
- communicating technical tradeoffs
- presenting systems clearly under interview pressure
- turning experiments into a defensible portfolio artifact

#### Output of this stage
- basic test suite
- improved documentation
- architecture diagram
- known limitations section
- interview-ready explanation of the project
- demo script for interviews

#### Interview angle
Be able to explain:
- your architecture in 1 to 2 minutes
- your major tradeoffs
- why you staged retrieval complexity instead of jumping to hybrid immediately
- what you would do next if given more time

---

## 14. Explicit production-minded requirements
- Handle "I don't know" gracefully for missing or weak evidence
- Never fabricate citations
- Log token usage, latency, and estimated cost per request
- Use section-aware chunking for large functional spec sections
- Preserve useful table content during ingestion and chunking
- Detect image or screenshot presence even if full vision support is out of scope
- Bound prompt context to fit context window limits
- Support conversation-scoped multi-turn questions
- Compact older chat turns with token-triggered rolling summaries
- Keep recent turns verbatim and reserve context capacity for retrieved evidence
- Keep conversation history separate from authoritative document evidence
- Prevent context or summaries from leaking between conversations
- Store vectors and artifacts locally
- Separate indexing flow from query flow
- Support re-indexing when documents change
- Make the system debuggable with clear intermediate artifacts
- Keep the design modular so models or vector stores can be swapped later
- Preserve release lineage in indexing, retrieval, and answer generation
- Support cumulative current-state answers across multiple releases
- Version indexes and processed artifacts for reproducibility
- Measure dense, lexical, and hybrid retrieval instead of assuming one is best

---

## 15. Learning goal
Use this project to become interview-ready for:
- AI / ML engineer roles
- applied LLM / RAG roles
- enterprise GenAI roles

The real learning goal is not only to build the system, but to be able to justify every major design decision.

---

## 16. What we are deliberately not doing early
To keep learning disciplined, we are not starting with:
- SQL generation
- agent-style orchestration
- LangGraph workflows
- PL/SQL linking
- SR document processing
- user accounts, login, or user-level long-term memory across conversations
- Docker packaging or Docker deployment
- direct coupling between the chat service and future MCP or Oracle integrations
- full OCR or vision reasoning for screenshots
- fine-tuning or retraining the model
- perfect automatic semantic diffing across releases
- reranking before baseline retrieval evaluation

These may come later, but only after ingestion, retrieval, grounding, and evaluation are strong.

---

## 17. Final project outcome
By the end of the first major version, the system should:
- ingest functional specification documents
- understand document families and release lineage
- preserve table content and detect non-text content presence
- retrieve grounded evidence
- compare dense, lexical, and hybrid retrieval strategies
- combine valid information across releases when needed
- answer with citations and release labels
- refuse unsupported answers safely
- track token usage, latency, and cost
- support local vector storage and future re-indexing
- provide a usable local UI with debug visibility
- support bounded multi-turn conversation memory and rolling summaries
- isolate history between conversations without implementing user-level memory
- include evaluation artifacts and interview-ready documentation
- support reproducible indexing and debugging workflows

This makes the project both a practical LLM portfolio project and a structured learning journey.

---

## 18. Near-term roadmap after Step 89
The remaining work should proceed in this order, one practical step at a time:

1. **Completed:** Step 89 interview gate for the reproducible `uv` migration.
2. **Completed:** CI automation using locked `uv` commands.
3. **Completed:** Conversation, message, and summary models plus a
   `ConversationStore` interface and local SQLite implementation.
4. **Completed:** Token-aware context budgeting and rolling summarization.
5. **Completed:** FastAPI conversation creation, history retrieval, archival,
   and multi-turn message submission.
6. **Completed:** Streamlit multi-turn chat UI with conversation controls,
   durable history, citations/debug details, readiness state, and safe errors.
7. **Completed:** Separate complete token-bounded LLM prompt evidence from short
   API/UI citation previews, with safe oversized-unit refusal and BOR/B-04
   regression protection.
8. **Completed:** Replace normalized-score hybrid fusion with weighted RRF,
   require multi-unit evidence coverage in evaluation, and preserve decisive
   teller/branch tables for the demonstrated R24 ranking pattern.
9. **Completed:** Add production-deployed latest-release temporal semantics,
   conversation-aware release scoping, expanded current-state evidence
   candidates, and answer-level assertions for the R24 current state of
   2 teller and 4 branch reports.
10. **Completed:** Add end-to-end and evaluation coverage for follow-up resolution,
   conversation isolation, summary drift, context overflow, abstention, and
   evidence grounding.
11. **In progress:** Add security, audit, observability, and native
   Oracle-compatible deployment packaging without Docker.
   - **Completed:** privacy-safe structured API audit events, validated request
     correlation, defensive response headers, and answer-trace correlation.
   - **Completed:** optional durable HMAC-chained local audit journal, offline
     integrity verification, trusted checkpoint support for suffix-deletion
     detection, and production preflight enforcement.
   - **Completed:** synthetic privacy-safe local benchmark for per-record
     HMAC/append/flush/fsync latency, single-writer throughput, storage growth,
     and chain-verification cost.
   - **Completed:** storage-neutral `AuditSink` boundary with explicit durability
     semantics and a synchronous HMAC-JSONL adapter; future grouped, database,
     network collector, and centralized adapters remain separate choices.
   - **Remaining:** identity-aware authentication/authorization and rate
     controls once the deployment identity boundary is selected; centralized
     append-only audit shipping/retention, checkpoint custody, and key rotation.
   - **Completed:** deterministic native Python 3.12 deployment bundle,
     per-file SHA-256 manifest, external mutable-state contract, and offline
     deployment preflight. OS service installation remains target-specific.
12. Keep codebase retrieval, MCP integrations, and Oracle persistence/vector
   adapters as future modular extensions after the core conversation experience
   is stable and measured.

---

## 19. Approved knowledge-expansion roadmap after Step 103

### Current verified baseline

- The active corpus contains two deployed delta FDDs.
- The system supports release-aware grounded RAG, cumulative current-state
  synthesis, conversation-scoped memory, weighted-RRF hybrid retrieval,
  validated citations, safe refusals, and local answer traces.
- FastAPI request auditing is decoupled behind a storage-neutral `AuditSink`;
  the current HMAC-JSONL adapter remains synchronous and
  `durable_on_return`.
- The full regression suite passes 327 tests with one existing non-failing
  upstream Starlette `TestClient`/HTTPX deprecation warning.
- The latest deterministic native bundle contains 102 files and has SHA-256
  `3ffa4df318a23c164570bc65c41f0bc03bc5a427cc343c7452754ea6cfc07059`.
- Security and deployment hardening is paused after Step 103. Grouped audit
  commits, authentication, authorization, centralized audit retention,
  platform supervision, and production deployment work are deferred until the
  knowledge capabilities below are stable and evaluated.

### Architecture decision

Continue as one modular product with shared API, UI, conversation,
observability, and evaluation foundations. Keep FDDs, custom code, Oracle
schema metadata, and SQL examples in separate knowledge lanes with their own
ingestion, indexes, retrieval thresholds, citations, and evaluation suites.
Do not create three independent applications or prematurely split the system
into microservices.

### Phase 1 - Expand FDD lineage RAG

- Add 3-5 deployed delta FDDs.
- Preserve cumulative release semantics and current-state synthesis.
- Build versioned staging artifacts and indexes before activation, retaining
  the prior active version for rollback.
- Require SME-reviewed document-specific, cross-release, conflict, abstention,
  and citation evaluation cases.
- Gate progression on valid ingestion, retrieval recall@10 of at least `0.90`,
  correct release selection, valid citations, safe refusals, and at least `90%`
  SME-reviewed answer correctness.

### Phase 2 - Custom JavaScript and PL/SQL understanding

- Ingest only FDD-linked custom modules from manual TortoiseSVN exports pinned
  to repository revisions.
- Generate deterministic content manifests and identify added, modified,
  deleted, and renamed files between snapshots.
- Index an explicit JavaScript and PL/SQL source allowlist. Exclude hidden
  kernel Java, generated, minified, vendor, binary, credential, and
  secret-bearing configuration files.
- Parse JavaScript with Tree-sitter and PL/SQL with a pinned ANTLR grammar.
  Preserve parser versions and diagnostics, and use conservative file/line
  fallback chunks when syntax is unsupported.
- Store symbols, dependency edges, snapshot IDs, SVN revisions, repository
  paths, content hashes, and line ranges.
- Maintain curated FDD-release to SVN-snapshot mappings as the authority for
  cross-source version alignment.
- Support evidence-backed explanation and impact analysis only. Do not generate
  code changes at this stage.
- Treat hidden Java kernel behavior and unresolved dynamic SQL as explicit
  unknowns rather than inferred implementation evidence.

### Phase 2B - Combined and bounded agentic analysis

- Keep FDD and code indexes, retrieval thresholds, citations, and evaluation
  suites separate.
- Add explicit `fdd`, `code`, and `combined` user modes.
- In combined mode, retrieve each evidence type independently and present
  separate documented-functionality and visible-custom-implementation sections.
- Introduce bounded FDD-search, code-search, and impact-graph tools only after
  the deterministic combined workflow passes evaluation.
- Do not enable automatic routing initially.

### Phase 3 - Oracle Text-to-SQL

- Run a separate MCP metadata service using approved application schemas from
  the test Oracle database.
- Expose schema metadata only. Do not expose arbitrary query execution, DDL,
  DML, row access, or unrestricted database tools.
- Maintain separate versioned Oracle schema and curated SQL-example stores.
- Use more than 50 SME-reviewed natural-language intent and Oracle `SELECT`
  examples with leakage-resistant retrieval, development, and held-out test
  partitions.
- Generate, explain, cite, and validate exactly one Oracle `SELECT` statement
  without executing it.
- Validate statement type, approved objects, identifiers, joins, ambiguity,
  synonyms, and prompt-injection cases.
- Defer fine-tuning until retrieval, prompting, schema linking, and held-out
  error analysis demonstrate a persistent model behavior gap.

### Phase sequencing and gates

Proceed sequentially: FDD expansion, code retrieval, deterministic combined
analysis, bounded tool orchestration, and then Text-to-SQL. Each phase must
pass its evaluation gate before implementation begins on the next phase.

---

## 20. Approved Phase 2 custom PL/SQL architecture

Phase 2 begins with complete curated custom-code snapshots under
`data/raw_code/<snapshot-request>/source`. Snapshot requests bind a module set
to an SVN revision, application build, reviewer, optional prior snapshot, and
optional expected changed packages. Local snapshot publication is
content-addressed, atomic, no-overwrite, and separate from active retrieval.

The current allowlist is case-insensitive `.sql`, `.spc`, `.prc`, `.fnc`, and
`.ddl`; `.spc` was added through the versioned capability policy when the first
real package specification was supplied.
FDD and code extension mappings are centralized in the versioned
`config/ingestion_sources.toml` capability policy. New extensions may map to an
existing implemented handler without changing Python; new formats require a
new handler and cannot be enabled by configuration alone. Code snapshot
manifests bind the normalized policy SHA-256 and report policy changes from the
base snapshot.
Files above 5 MiB are warned about and later parsed with resource isolation;
size alone is not a rejection condition. Streaming validation rejects binary,
unsupported, symlinked, and potential secret-bearing inputs without exposing
secret values. Complete manifests determine added, modified, deleted,
unchanged, and unambiguous exact-rename changes; the reviewer does not need to
provide changed line ranges.

PL/SQL parsing will preserve original citeable source, conditional-compilation
regions and exact line maps. It will use full, segmented, fallback, or failed
parser states rather than silently treating degraded parsing as complete.
Unquoted Oracle identifiers canonicalize to uppercase while quoted identifiers
retain exact case. Overload-safe symbol identity will include a deterministic
parameter discriminator, with a separate full declaration hash for semantic
diffs.

Code retrieval units will keep exact source separate from compact derived
package context and linked declaration units. Static DDL and synonym targets
will carry explicit resolution states; unavailable kernel code, unresolved
dynamic SQL, compiler branches, and external schemas remain qualified unknowns.

FDD and code evidence remain in separate artifacts, indexes, Qdrant
collections, thresholds, citations, and evaluation suites. Reviewed lineage
mappings may target modules, files, or overload-specific symbols. Explicit
`fdd`, `code`, and later `combined` modes are required; automatic routing and
code generation remain out of scope until the deterministic combined gate is
passed.

Implementation status after Step 155: the ANTLR 4.13.2 runtime and a
commit/hash-pinned grammars-v4 PL/SQL grammar are integrated. Conditional
directives are recorded without changing citation source, parsing runs in
resource-bounded isolated workers with explicit degradation states, and local
no-overwrite parse generations contain exact-source retrieval units with
selective linked package context. Overload-safe symbol identity, dependency
extraction, DDL structure, code indexing, answering, and FDD/code mappings
remain in Step 156 and later.

Implementation status after Step 158: Oracle-aware quoted/unquoted identifier
handling, overload-safe symbol and occurrence identities, signature collision
diagnostics, snapshot-scoped dependency resolution, explicit dynamic-SQL and
kernel boundaries, DDL structures, and conservative synonym resolution are
implemented. The versioned analysis policy is hash-bound to a new isolated
parser generation. Code embeddings, lexical/Qdrant generations, retrieval,
answering, and FDD/code lineage mappings remain unimplemented and gated by
Steps 159 and later.

### Phase 2 completion status after Step 237

The approved initial PL/SQL scope is complete. The delivered local capability now
includes immutable curated snapshots, resource-bounded parsing, overload-safe
symbols, dependency and DDL analysis, deterministic code artifacts, isolated
lexical/Qdrant generations, weighted-RRF code retrieval, source-line citations,
reviewed FDD-to-code lineage, explicit code/combined modes, safe unknown handling,
manual UAT, paid grounded-answer evaluation, and deliberate local activation with
rollback evidence.

The bounded deterministic tool extension has a reviewed manifest, safety cases,
manual UAT, and an effective **10/10 SME-accepted paid answer set** after one
authorized targeted replay. It remains offline and is not automatically routed or
exposed through a new API/UI tool surface by this completion decision.

Deferred work remains explicit rather than being treated as missing Phase 2 scope:

- JavaScript parsing/indexing after the PL/SQL architecture is stable on a broader
  curated corpus;
- bounded-tool API/UI exposure and any automatic routing under a separate
  activation contract;
- larger-corpus dependency and retrieval evaluation;
- concurrency, latency, sustained-provider, monitoring, disaster-recovery,
  authentication, authorization, and production deployment evidence;
- Phase 3 Oracle metadata and Text-to-SQL work.
