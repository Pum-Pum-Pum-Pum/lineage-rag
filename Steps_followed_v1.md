--FastAPI::
python -m uvicorn app.api.main:app --host 127.0.0.1 --port 8000 --reload

--Streamlit::
python -m streamlit run app/ui/streamlit_app.py --server.address 127.0.0.1 --server.port 8501

# Steps Followed

## Step 1 — Project setup and architecture foundation

### Description
Established the Stage 1 foundation for the enterprise RAG project by defining centralized configuration, centralized logging, a setup validation script, and interview-tracking artifacts.

### Code snippets used

#### Central settings bootstrap
```python
class Settings(BaseSettings):
    app_name: str = "Culling Blade Lineage GenAI RAG System"
    environment: str = "dev"
    log_level: str = "INFO"
    artifact_version: str = "v1"
    index_version: str = "fsrag_v1"
```

#### Logging bootstrap
```python
def configure_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
```

#### Setup validation script
```python
def main() -> None:
    settings = get_settings()
    configure_logging(settings.log_level)
    logger = get_logger("setup_check")
```

### Why this step matters
This establishes the foundation needed for reproducibility, cleaner configuration management, and easier debugging before implementing ingestion, retrieval, and generation logic.

### Additional setup refinement
- Added a `.gitignore` file to keep secrets, caches, local artifacts, vector store files, and temporary development files out of version control.
- Strengthened secret-related ignore rules for environment files, key material, certificates, and local logs.
- Aligned `.env.example` with the centralized `Settings` contract so local setup matches the code-defined configuration surface.

## Step 2 — Release-aware filename parsing baseline

### Description
Started Stage 2 with the smallest correct ingestion building block: parsing release-aware DOCX filenames into structured metadata.

### Code snippets used

#### Filename parsing pattern
```python
FILENAME_PATTERN = re.compile(
    r"^(?P<document_family>.+?)_R(?P<release_number>\d+)$",
    re.IGNORECASE,
)
```

#### Parsed output schema
```python
@dataclass(frozen=True)
class ParsedDocumentName:
    document_name: str
    document_family: str
    release_label: str
    release_number: int
    source_type: str = "docx"
```

### Why this step matters
Filename parsing is the first release-lineage signal in your domain. If this is wrong, document grouping, release-aware retrieval, and cumulative truth assembly will all become unreliable later.

### Refinement after real filename examples
- Updated the parser to support real corpus filenames where the release token is followed by additional descriptive suffix text, for example `_PNB_Branch Online Reports(BOR)_v1.2`.
- Added `variant_suffix` so multiple files under the same family and release can still be distinguished without losing the main release-lineage fields.

## Step 3 — Safe DOCX file discovery baseline

### Description
Implemented the next smallest ingestion step: discovering valid DOCX files while excluding temporary Microsoft Office lock files such as `~$...docx`.

### Code snippets used

#### Discovery rules
```python
SUPPORTED_DOCX_SUFFIX = ".docx"
TEMP_DOCX_PREFIX = "~$"
```

#### Discovery output schema
```python
@dataclass(frozen=True)
class DiscoveredDocxFile:
    file_path: Path
    file_name: str
    is_temporary: bool
```

### Why this step matters
Real enterprise document folders are noisy. If ingestion does not filter temporary lock files and non-DOCX files early, later parsing and extraction logic will fail on avoidable input noise.

## Step 4 — DOCX paragraph text extraction baseline

### Description
Implemented the first real content extraction step: reading paragraph text from a DOCX file and returning both raw paragraph-level output and a joined text view.

### Code snippets used

#### Extraction output schema
```python
@dataclass(frozen=True)
class ExtractedDocxText:
    file_name: str
    paragraph_count: int
    non_empty_paragraph_count: int
    paragraphs: list[str]
    full_text: str
```

#### Core extraction logic
```python
document = Document(path)
paragraphs = [paragraph.text.strip() for paragraph in document.paragraphs]
non_empty_paragraphs = [paragraph for paragraph in paragraphs if paragraph]
full_text = "\n".join(non_empty_paragraphs)
```

### Why this step matters
This is the first true content-ingestion step. We intentionally start with paragraph text only so extraction logic remains testable and debuggable before adding tables, images, and more complex structure handling.

### Test execution refinement
- Added `tests/conftest.py` to make the project root importable during pytest runs, so test modules can reliably import the `app` package across local environments and editor-driven test execution.

## Step 5 — DOCX table extraction baseline

### Description
Implemented the first table extraction step so business-critical tabular content can be ingested separately from paragraph text.

### Code snippets used

#### Table output schema
```python
@dataclass(frozen=True)
class ExtractedTable:
    table_index: int
    row_count: int
    column_count: int
    rows: list[list[str]]
    text_representation: str
```

#### Table normalization logic
```python
def _normalize_cell_text(text: str) -> str:
    return " ".join(text.split())
```

### Why this step matters
Your functional specification documents may contain important business rules in tables. If table content is ignored, retrieval quality and answer correctness can degrade even if paragraph extraction works well.

## Step 6 — Unified DOCX ingestion artifact

### Description
Combined filename parsing, paragraph extraction, and table extraction into one document-level ingestion artifact so one DOCX file can be inspected as a single coherent extracted unit.

### Code snippets used

#### Unified artifact schema
```python
@dataclass(frozen=True)
class IngestedDocxArtifact:
    document_name: str
    parsed_name: ParsedDocumentName
    extracted_text: ExtractedDocxText
    extracted_tables: ExtractedDocxTables
```

#### Composition logic
```python
parsed_name = parse_document_filename(path)
extracted_text = extract_docx_text(path)
extracted_tables = extract_docx_tables(path)
```

### Why this step matters
This creates the first real document-level ingestion artifact. It gives us one structured object per DOCX file before we move to normalization, metadata enrichment, and processed artifact export.

## Step 7 — Processed ingestion artifact export

### Description
Added local JSON export for unified DOCX ingestion artifacts so extracted document state can be inspected and debugged before normalization and chunking.

### Code snippets used

#### JSON export logic
```python
output_file.write_text(
    json.dumps(asdict(artifact), indent=2, ensure_ascii=False),
    encoding="utf-8",
)
```

### Why this step matters
Local artifact export is important in production-minded RAG systems because it gives you an inspectable checkpoint between raw document ingestion and later processing stages. This makes debugging much easier when extraction quality is uncertain.

## Step 8 — Runnable ingestion mini-pipeline

### Description
Added a runnable ingestion script that discovers DOCX files, builds unified ingestion artifacts, and writes them as JSON into `data/processed/`.

### Code snippets used

#### Pipeline flow
```python
discovered_files = discover_docx_files(settings.raw_specs_dir)

for discovered in discovered_files:
    artifact = ingest_docx_file(discovered.file_path)
    output_file = write_ingested_artifact_to_json(artifact, settings.processed_dir)
```

### Why this step matters
This is the first executable ingestion flow that operates on real documents in `data/raw_specs/` and produces persistent JSON artifacts in `data/processed/`. It turns isolated components into a usable mini-pipeline.

## Step 9 — Text normalization baseline

### Description
Implemented the first normalization step to remove obvious empty and template-noise paragraphs from the ingested document artifact before chunking.

### Code snippets used

#### Noise detection logic
```python
def is_noise_paragraph(paragraph: str) -> bool:
    stripped = paragraph.strip()
    if not stripped:
        return True
```

#### Normalization result schema
```python
@dataclass(frozen=True)
class NormalizedTextResult:
    original_non_empty_paragraph_count: int
    cleaned_paragraph_count: int
    removed_paragraph_count: int
    cleaned_paragraphs: list[str]
    cleaned_full_text: str
```

### Why this step matters
This is the first real cleaning step. It improves signal quality before chunking and embeddings by removing obvious boilerplate and empty layout artifacts that would otherwise contaminate retrieval later.

### Refinement after real chunk inspection
- Extended normalization to remove front-matter headings like `Document Control` and `Table of Contents`.
- Added TOC-like line detection so entries such as `1\tIntroduction\t9` do not survive into retrieval chunks.
- Extended normalization again to remove front-page metadata lines such as client name, release name, author/date/version fields, and footer-style corporate contact text.
- Added `Oracle Corporation` as an exact-match footer noise rule after inspecting the remaining last chunk in the real R24 chunk artifact.

## Step 10 — Normalized document artifact

### Description
Combined the raw ingestion artifact and normalized text output into one normalized document artifact so downstream chunking can use cleaned text while debugging still keeps the raw extraction state.

### Code snippets used

#### Normalized artifact schema
```python
@dataclass(frozen=True)
class NormalizedDocxArtifact:
    raw_artifact: IngestedDocxArtifact
    normalized_text: NormalizedTextResult
```

#### Composition logic
```python
normalized_text = normalize_ingested_text(artifact)
return NormalizedDocxArtifact(
    raw_artifact=artifact,
    normalized_text=normalized_text,
)
```

### Why this step matters
This creates a cleaner contract for the next stage. Chunking should operate on cleaned content, but production debugging still needs access to the original extracted state. This artifact preserves both views together.

## Step 11 — Paragraph-aware chunking baseline

### Description
Implemented the first chunking baseline using normalized paragraphs. This creates fixed-size paragraph-aware chunks before introducing section-aware chunking.

### Code snippets used

#### Chunk schema
```python
@dataclass(frozen=True)
class TextChunk:
    chunk_id: str
    chunk_index: int
    paragraph_start_index: int
    paragraph_end_index: int
    paragraph_count: int
    text: str
```

#### Chunking logic
```python
for chunk_index, start in enumerate(range(0, len(paragraphs), max_paragraphs_per_chunk)):
    end = min(start + max_paragraphs_per_chunk, len(paragraphs))
```

### Why this step matters
This is the first transition from cleaned document text to retrieval-ready units. It is intentionally simple so chunk behavior can be tested before more advanced section-aware chunking is introduced.

## Step 12 — Chunked artifact export

### Description
Added JSON export for chunked document output so chunk boundaries and chunk text can be inspected on real documents before embeddings.

### Code snippets used

#### Chunked JSON export logic
```python
output_file.write_text(
    json.dumps(asdict(chunked_document), indent=2, ensure_ascii=False),
    encoding="utf-8",
)
```

### Why this step matters
This gives a second important checkpoint in the pipeline: not just what was extracted, but exactly how the cleaned document was chunked. That is critical for debugging retrieval quality later.

## Step 13 — End-to-end mini-pipeline with chunk artifact export

### Description
Extended the runnable ingestion pipeline so it now writes both raw ingestion JSON artifacts and chunked JSON artifacts for real documents.

### Code snippets used

#### Extended pipeline flow
```python
artifact = ingest_docx_file(discovered.file_path)
normalized_artifact = build_normalized_artifact(artifact)
chunked_document = chunk_normalized_artifact(normalized_artifact)
chunk_output_file = write_chunked_document_to_json(chunked_document, settings.processed_dir)
```

### Why this step matters
This makes the mini-pipeline more production-minded because it now exposes both pre-chunk and post-chunk checkpoints for real documents. That gives better observability before embeddings and retrieval are introduced.

## Step 14 — Table-aware chunking baseline

### Description
Implemented the first table-aware chunking step so extracted table content also becomes retrieval-ready units instead of being left out of chunked outputs.

### Code snippets used

#### Table chunk schema
```python
@dataclass(frozen=True)
class TableChunk:
    chunk_id: str
    chunk_index: int
    table_index: int
    row_count: int
    column_count: int
    text: str
```

#### Table chunking logic
```python
for chunk_index, table in enumerate(artifact.raw_artifact.extracted_tables.tables):
    if not table.text_representation.strip():
        continue
```

### Why this step matters
This addresses a major gap in the current pipeline: paragraph chunking alone misses table-based business content. Table-aware chunking creates retrieval-ready units for table content before later fusion or ranking logic is introduced.

## Step 15 — Combined retrieval-ready artifact

### Description
Combined paragraph chunks and table chunks into a single retrieval-ready artifact while preserving source kind and release-aware metadata.

### Code snippets used

#### Retrieval-ready unit schema
```python
@dataclass(frozen=True)
class RetrievalReadyUnit:
    unit_id: str
    unit_index: int
    source_kind: str
    text: str
    document_family: str
    release_label: str
```

#### Fusion logic
```python
for chunk in paragraph_chunks.chunks:
    ... source_kind="paragraph" ...

for table_chunk in table_chunks.table_chunks:
    ... source_kind="table" ...
```

### Why this step matters
This creates the first combined retrieval-ready view of a document. It keeps paragraph and table evidence together in one artifact while still preserving source type for later debugging, filtering, and evaluation.

## Step 16 — Retrieval-ready artifact export and pipeline integration

### Description
Added JSON export for the combined retrieval-ready artifact and extended the runnable pipeline to write retrieval-ready JSON outputs for real documents.

### Code snippets used

#### Retrieval-ready JSON export logic
```python
output_file.write_text(
    json.dumps(asdict(artifact), indent=2, ensure_ascii=False),
    encoding="utf-8",
)
```

#### Pipeline integration
```python
chunked_tables = chunk_tables_from_artifact(normalized_artifact)
retrieval_ready_artifact = build_retrieval_ready_artifact(
    normalized_artifact,
    chunked_document,
    chunked_tables,
)
```

### Why this step matters
This closes the gap you identified: table content is now represented in a persisted retrieval-ready JSON artifact, not only in the raw ingestion artifact. That gives the pipeline a more complete pre-embedding checkpoint.

## Step 17 — Embedding record schema and pipeline contract

### Description
Defined a local embedding record schema and embedding batch contract for retrieval-ready units without calling the actual embedding API yet.

### Code snippets used

#### Embedding record schema
```python
@dataclass(frozen=True)
class EmbeddingRecord:
    unit_id: str
    unit_index: int
    source_kind: str
    document_family: str
    release_label: str
    text: str
    embedding_model: str
    embedding_status: str
    vector: list[float] | None = None
```

#### Contract builder
```python
def build_embedding_batch_contract(
    artifact: RetrievalReadyArtifact,
    embedding_model: str,
) -> EmbeddingBatch:
```

### Why this step matters
This is the correct transition into embeddings. Before calling any model API, we first define the exact unit-level schema and metadata contract that the embedding stage will consume and produce.

## Step 18 — Real embedding-call baseline

### Description
Implemented the first embedding client wrapper and embedding batch execution function so retrieval-ready units can be turned into vectors using the configured embedding model.

### Code snippets used

#### Embedding client bootstrap
```python
def get_embedding_client() -> OpenAI:
    settings = get_settings()
    return OpenAI(
        api_key=settings.openai_api_key,
        base_url=settings.openai_base_url or None,
    )
```

#### Batch embedding call
```python
response = embedding_client.embeddings.create(
    model=settings.openai_embedding_model,
    input=texts,
)
```

### Why this step matters
This is the first real model-integration step. It turns retrieval-ready units into vectors while preserving metadata and explicit embedding status, which is necessary before indexing into a vector store later.

## Step 19 — Embedding failure-mode guardrails

### Description
Added baseline robustness checks to the embedding client so empty batches are handled safely and mismatched embedding API response counts fail explicitly.

### Code snippets used

#### Empty batch guard
```python
if not texts:
    return EmbeddingBatch(
        document_name=batch.document_name,
        total_records=0,
        records=[],
    )
```

#### Response-count validation
```python
if len(response.data) != len(batch.records):
    raise RuntimeError(
        "Embedding response count does not match input record count: "
        f"expected={len(batch.records)}, received={len(response.data)}"
    )
```

### Why this step matters
This intentionally stresses a failure scenario before moving forward. If the embedding API returns fewer vectors than requested, silently zipping records and vectors would corrupt metadata-to-vector alignment. Explicit failure protects retrieval correctness and makes debugging safer.

## Step 20 — Embedding batch persistence baseline

### Description
Added local JSON persistence for embedding batches so embedded vectors and their metadata can be inspected and reused before vector-store indexing.

### Code snippets used

#### Embedding batch writer
```python
def write_embedding_batch_to_json(
    batch: EmbeddingBatch,
    output_directory: str | Path,
) -> Path:
```

#### JSON persistence logic
```python
output_file.write_text(
    json.dumps(asdict(batch), indent=2, ensure_ascii=False),
    encoding="utf-8",
)
```

### Why this step matters
Embedding API calls can be expensive and slow. Persisting the embedding batch creates a checkpoint between embedding generation and vector-store indexing, which improves debugging, reproducibility, and future caching/re-indexing behavior.

## Step 21 — Lightweight embedding cache-key foundation

### Description
Added a stable `content_hash` field to each embedding record so future caching and incremental re-indexing can detect unchanged retrieval units.

### Code snippets used

#### Content hash helper
```python
def compute_content_hash(text: str) -> str:
    normalized_text = text.replace("\r\n", "\n").replace("\r", "\n")
    return hashlib.sha256(normalized_text.encode("utf-8")).hexdigest()
```

#### Embedding record field
```python
content_hash: str
```

### Why this step matters
This is not a full embedding cache yet. It is the first cache-key building block. Later, the system can combine `content_hash`, embedding model, and preprocessing/artifact version to skip re-embedding unchanged units and reduce cost.

## Step 22 — Embedding cache key builder

### Description
Added a deterministic embedding cache key that combines `content_hash`, `embedding_model`, and `artifact_version` so cached embeddings can be safely invalidated when content, model, or preprocessing version changes.

### Code snippets used

#### Cache key helper
```python
def compute_embedding_cache_key(
    content_hash: str,
    embedding_model: str,
    artifact_version: str,
) -> str:
```

#### Stable payload hashing
```python
stable_payload = json.dumps(payload, sort_keys=True, separators=(",", ":"))
return hashlib.sha256(stable_payload.encode("utf-8")).hexdigest()
```

### Why this step matters
`content_hash` alone is not enough. The same text embedded with a different model or preprocessing version can produce different vectors. This cache key prevents unsafe reuse of stale embeddings in future caching and re-indexing workflows.

## Step 23 — Minimal embedding cache lookup interface

### Description
Added a minimal local cache lookup layer that reads persisted `.embeddings.json` artifacts and finds embedded records by `cache_key`.

### Code snippets used

#### Cache loader
```python
def load_embedding_cache(cache_directory: str | Path) -> dict[str, EmbeddingRecord]:
```

#### Cache lookup
```python
def find_cached_embedding(
    cache_key: str,
    cache_directory: str | Path,
) -> EmbeddingRecord | None:
```

#### Conflict guard
```python
if existing is not None and existing.vector != record.vector:
    raise RuntimeError(
        "Conflicting cached embeddings found for cache_key="
        f"{record.cache_key}"
    )
```

### Why this step matters
This is still not a full cache, but it gives the system a practical read path for persisted embedding artifacts. It also intentionally stresses a failure case: if the same cache key maps to different vectors, the cache is corrupt and should fail loudly.

## Step 24 — Cache-aware embedding execution

### Description
Integrated the local embedding cache lookup into `embed_batch` so cached records are reused and only uncached records call the embedding API.

### Code snippets used

#### Cache-aware split
```python
cache = load_embedding_cache(cache_directory) if cache_directory is not None else {}
updated_records: list[EmbeddingRecord | None] = [None] * len(batch.records)
uncached_records: list[tuple[int, EmbeddingRecord]] = []
```

#### Cached record reuse
```python
cached_record = cache.get(record.cache_key)
if cached_record is not None:
    updated_records[index] = replace(
        record,
        embedding_status="cached",
        vector=cached_record.vector,
    )
```

### Why this step matters
This is the first real cache behavior. The embedding pipeline can now skip API calls for unchanged records that already have persisted vectors, reducing cost and latency while preserving record order and metadata.

## Step 25 — Lightweight embedding cache statistics

### Description
Added lightweight cache statistics to embedding batches so cache reuse behavior can be inspected without building a full metrics system yet.

### Code snippets used

#### Embedding batch stats fields
```python
cached_count: int = 0
embedded_count: int = 0
cache_miss_count: int = 0
```

#### Cache-aware counts
```python
cached_count = 0

if cached_record is not None:
    cached_count += 1
```

### Why this step matters
Cache behavior must be measurable. These lightweight counts tell us how many records were reused from cache versus newly embedded, which is the first step toward tracking cache hit rate, API calls avoided, latency saved, and cost saved.

## Step 26 — Embedding cache hit-rate helper

### Description
Added a lightweight metrics helper to calculate cache hit rate and summarize cache behavior for an embedding batch.

### Code snippets used

#### Cache hit-rate helper
```python
def calculate_cache_hit_rate(batch: EmbeddingBatch) -> float:
    if batch.total_records == 0:
        return 0.0
    return batch.cached_count / batch.total_records
```

#### Metrics summary schema
```python
@dataclass(frozen=True)
class EmbeddingCacheMetrics:
    total_records: int
    cached_count: int
    embedded_count: int
    cache_miss_count: int
    cache_hit_rate: float
```

### Why this step matters
This turns raw cache counts into an interpretable metric. Cache hit rate tells us whether caching is actually reducing repeated embedding work, while safely handling empty batches without divide-by-zero errors.

## Step 27 — Embedding run summary artifact

### Description
Added a lightweight embedding run summary artifact that persists cache metrics, embedding model information, and artifact version information for one embedding batch.

### Code snippets used

#### Summary schema
```python
@dataclass(frozen=True)
class EmbeddingRunSummary:
    document_name: str
    total_records: int
    cached_count: int
    embedded_count: int
    cache_miss_count: int
    cache_hit_rate: float
    embedding_models: list[str]
    artifact_versions: list[str]
```

#### Summary writer
```python
def write_embedding_run_summary_to_json(
    summary: EmbeddingRunSummary,
    output_directory: str | Path,
) -> Path:
```

### Why this step matters
This creates a local observability checkpoint for embedding runs. Before vector-store indexing, we can inspect how much work was cached, how much required API embedding, and which model/artifact versions were involved.

### Failure intentionally discovered
- After a second smoke-test run, persisted records had `embedding_status="cached"` instead of `embedding_status="embedded"`.
- The Qdrant indexing script loaded only `embedded` records, so it created a collection with zero points.
- The cache loader was corrected to treat both `embedded` and `cached` records with non-null vectors as usable for indexing.
- A test initially assumed dictionary key order for cache contents. The assertion was corrected to compare sets because cache correctness should not depend on insertion order.

## Step 28 — Real embedding smoke-test script

### Description
Added a controlled script for running a tiny real embedding smoke test on only the first few retrieval-ready units from a real DOCX document.

### Code snippets used

#### Smoke-test limiter
```python
def limit_embedding_batch(batch: EmbeddingBatch, limit: int) -> EmbeddingBatch:
    if limit <= 0:
        raise ValueError("Smoke-test limit must be greater than 0")
```

#### Real smoke-test embedding flow
```python
embedded_batch = embed_batch(
    smoke_batch,
    cache_directory=embedding_cache_dir,
)
embedding_output = write_embedding_batch_to_json(embedded_batch, embedding_cache_dir)
summary = build_embedding_run_summary(embedded_batch)
summary_output = write_embedding_run_summary_to_json(summary, embedding_cache_dir)
```

### Why this step matters
This gives us a safe bridge from tested fake-client embedding behavior to a tiny real API validation. It intentionally limits API usage so we can verify configuration, vector dimensions, persistence, and cache metrics before indexing a full corpus.

### Refinement after table retrieval test
- Added `--source-kind` support to `scripts/run_embedding_smoke_test.py` so smoke tests can intentionally embed paragraph or table units.
- Added source-kind-specific output suffixes such as `.table.embeddings.json` to avoid overwriting paragraph-only smoke artifacts for the same document.
- This fixes the issue where `--source-kind table` retrieval returned no results because the R24 smoke test had embedded only the first paragraph units.

## Step 29 — Qdrant dependency setup

### Description
Updated `requirements.txt` to include the Qdrant Python client and the project dependencies currently used by the ingestion and embedding pipeline.

### Dependencies added
```text
openai
pydantic
pydantic-settings
python-docx
pytest
qdrant-client
```

### Why this step matters
The Qdrant Python client is needed to define collections, upsert vectors, and perform vector search from Python. We will start with local/in-memory Qdrant client mode before requiring a running Qdrant server.

## Step 30 — Qdrant local collection schema baseline

### Description
Added a Qdrant collection schema helper using in-memory local Qdrant client mode. The collection is configured for 3072-dimensional vectors with cosine distance.

### Code snippets used

#### Collection config
```python
@dataclass(frozen=True)
class QdrantCollectionConfig:
    collection_name: str
    vector_size: int
    distance: Distance = Distance.COSINE
```

#### In-memory local client
```python
def create_local_qdrant_client() -> QdrantClient:
    return QdrantClient(":memory:")
```

#### Collection creation
```python
client.create_collection(
    collection_name=config.collection_name,
    vectors_config=VectorParams(
        size=config.vector_size,
        distance=config.distance,
    ),
)
```

### Why this step matters
This introduces Qdrant safely without requiring Docker or a running server. It validates the core vector-store concept: a collection must define vector dimensionality and distance metric before vectors can be indexed.

## Step 31 — Docker-free persistent Qdrant local mode

### Description
Added a persistent local Qdrant client helper using `QdrantClient(path=...)` so vectors can be stored locally without Docker or a separate Qdrant server.

### Code snippets used

#### Persistent local Qdrant client
```python
def create_persistent_qdrant_client(path: str | Path) -> QdrantClient:
    storage_path = Path(path)
    storage_path.mkdir(parents=True, exist_ok=True)
    return QdrantClient(path=str(storage_path))
```

#### Local vector store ignore rule
```text
data/qdrant_local/
```

### Why this step matters
This supports your no-Docker organization constraint while still giving you persistent local vector storage. It is stronger than pure in-memory testing because collections survive process restarts, but it still avoids server setup complexity.

## Step 32 — Qdrant point mapping and upsert baseline

### Description
Added Qdrant point mapping and upsert logic so embedded records can be converted into vector-store points with metadata payloads.

### Code snippets used

#### Deterministic point ID
```python
def build_qdrant_point_id(record: EmbeddingRecord) -> str:
    return str(uuid5(NAMESPACE_URL, record.cache_key))
```

#### Payload mapping
```python
payload = {
    "unit_id": record.unit_id,
    "source_kind": record.source_kind,
    "document_family": record.document_family,
    "release_label": record.release_label,
    "cache_key": record.cache_key,
    "embedding_model": record.embedding_model,
    "text": record.text,
}
```

#### Upsert execution
```python
client.upsert(
    collection_name=collection_name,
    points=points,
)
```

### Why this step matters
This is the first actual vector-store indexing step. It preserves lineage metadata alongside vectors so later retrieval can filter, cite, and debug results correctly.

### Failure intentionally discovered
- The initial test expected Qdrant to return the raw vector `[0.1, 0.2]`.
- Qdrant returned a normalized vector because the collection uses cosine distance.
- The test was corrected to assert the cosine-normalized vector instead of assuming raw-vector storage behavior.

## Step 33 — Basic Qdrant vector search

### Description
Added a basic Qdrant vector search helper and tests that verify a query vector returns the expected nearest payload from local Qdrant.

### Code snippets used

#### Search result schema
```python
@dataclass(frozen=True)
class QdrantSearchResult:
    point_id: str
    score: float
    payload: dict[str, Any]
```

#### Search call
```python
results = client.query_points(
    collection_name=collection_name,
    query=query_vector,
    limit=limit,
    with_payload=True,
)
```

### Why this step matters
This proves that vectors inserted into Qdrant can be searched and that the expected payload metadata comes back with the nearest result. It is the first retrieval behavior in the vector-store layer.

## Step 34 — Metadata-filtered Qdrant search

### Description
Added optional metadata filtering to Qdrant vector search for `document_family`, `release_label`, and `source_kind`.

### Code snippets used

#### Metadata filter builder
```python
def build_metadata_filter(
    document_family: str | None = None,
    release_label: str | None = None,
    source_kind: str | None = None,
) -> Filter | None:
```

#### Qdrant field condition
```python
FieldCondition(
    key="document_family",
    match=MatchValue(value=document_family),
)
```

#### Filtered search call
```python
results = client.query_points(
    collection_name=collection_name,
    query=query_vector,
    limit=limit,
    query_filter=query_filter,
    with_payload=True,
)
```

### Why this step matters
This is the first lineage-aware retrieval control. Filtering by document family, release label, and source kind lets the system restrict evidence before generation, which is essential for citations, point-in-time retrieval, and cumulative release-aware answers.

## Step 35 — Persistent Qdrant indexing script

### Description
Added a persistent Qdrant indexing path that reads `.embeddings.json` artifacts from `data/cache/embeddings/`, ensures the configured Qdrant collection exists, and upserts embedded records into local persistent Qdrant.

### Code snippets used

#### Cache-to-batch loader
```python
def build_embedding_batch_from_cache(cache_directory: str | Path) -> EmbeddingBatch:
    cache = load_embedding_cache(cache_directory)
    records = sorted(cache.values(), key=lambda record: record.cache_key)
```

#### Indexing function
```python
def index_embedding_cache_directory(
    client: QdrantClient,
    collection_config: QdrantCollectionConfig,
    cache_directory: str | Path,
) -> QdrantUpsertSummary:
```

#### Runnable script
```python
python scripts/run_qdrant_indexing.py
```

### Why this step matters
This turns vector-store indexing from isolated unit-test behavior into a repeatable local pipeline step. It indexes persisted embeddings into Docker-free local Qdrant while preserving payload metadata for later retrieval.

## Step 36 — Persistent Qdrant index verification script

### Description
Added a small verification script to inspect the persistent local Qdrant index after running indexing.

### Code snippets used

#### Collection check
```python
if not client.collection_exists(collection_name):
    logger.warning("Collection does not exist: %s", collection_name)
```

#### Point count check
```python
count = client.count(collection_name).count
logger.info("Point count: %s", count)
```

#### Sample payload inspection
```python
points, _ = client.scroll(
    collection_name=collection_name,
    limit=1,
    with_payload=True,
    with_vectors=False,
)
```

### Why this step matters
Indexing is not complete until we verify the collection exists and contains expected payloads. This script gives a lightweight operational check before building query embedding and retrieval flows.

## Step 37 — Query embedding and Qdrant search script

### Description
Added a query embedding/search layer that embeds a user query, searches persistent local Qdrant, and prints scored payload results with optional metadata filters.

### Code snippets used

#### Query embedding
```python
response = client.embeddings.create(
    model=embedding_model,
    input=[cleaned_query],
)
```

#### Query-to-search flow
```python
query_vector = embed_query_text(
    query_text=query_text,
    embedding_model=embedding_model,
    embedding_client=embedding_client,
)
return search_vectors(...)
```

#### Runnable query search
```python
python scripts/run_qdrant_query_search.py --query "branch report" --limit 5
```

### Why this step matters
This is the first end-to-end retrieval path: user query text becomes a query vector, Qdrant searches indexed vectors, and payload evidence is returned. It is still baseline retrieval, but it proves the core retrieval loop works.

## Step 38 — Small retrieval evaluation set and runner

### Description
Added a small JSON retrieval evaluation set and a runner that executes queries against persistent local Qdrant, checks basic expectations, and logs pass/fail results.

### Code snippets used

#### Evaluation case schema
```python
@dataclass(frozen=True)
class RetrievalEvalCase:
    case_id: str
    query: str
    filters: RetrievalEvalFilters
    expectation: RetrievalEvalExpectation
    notes: str = ""
```

#### Evaluation runner command
```bash
python scripts/run_retrieval_eval.py --limit 5
```

### Why this step matters
Ad hoc retrieval checks are not enough. A small evaluation set lets us repeatedly test whether retrieval is returning expected release/source evidence, and it exposes known limitations such as missing embedded attachment content.

## Step 39 — Top-1 and expected-failure retrieval evaluation

### Description
Improved the retrieval evaluator to distinguish top-k recall from top-1 quality and to support expected unsupported-evidence failures.

### Code snippets used

#### Expectation fields
```python
expected_to_pass: bool = True
expected_top1_contains_any: list[str] | None = None
```

#### Outcome-as-expected logic
```python
passed = not failures
outcome_as_expected = passed == expectation.expected_to_pass
```

### Why this step matters
Retrieval evaluation must distinguish a genuinely good result from a case where useful evidence appears somewhere in top-k but rank 1 is weak. It also needs to mark known unsupported cases, like evidence inside embedded attachments, as expected failures rather than unexpected system regressions.

## Step 41 — Retrieval evaluation report export

### Description
Added JSON report export for retrieval evaluation runs so results, failures, scores, and payload previews can be saved under `data/eval/generated/`.

### Code snippets used

#### Report schema
```python
@dataclass(frozen=True)
class RetrievalEvalReport:
    total_cases: int
    passed_count: int
    expected_outcome_count: int
    cases: list[RetrievalEvalCaseReport]
```

#### Report writer
```python
def write_retrieval_eval_report_to_json(
    report: RetrievalEvalReport,
    output_path: str | Path,
) -> Path:
```

#### Runner output argument
```bash
python scripts/run_retrieval_eval.py --limit 5 --output-file data/eval/generated/retrieval_eval_report.json
```

### Why this step matters
Logs are temporary. Persisting retrieval evaluation reports makes retrieval behavior auditable and comparable across future changes to ingestion, chunking, embeddings, and ranking.

## Step 43 — Evidence sufficiency baseline

### Description
Added a transparent evidence sufficiency checker that decides whether retrieved results are strong enough to use for answer generation.

### Code snippets used

#### Decision schema
```python
@dataclass(frozen=True)
class EvidenceSufficiencyDecision:
    is_sufficient: bool
    reason: str
    result_count: int
    top_score: float | None
```

#### Baseline sufficiency checks
```python
if result_count < min_results:
    return EvidenceSufficiencyDecision(is_sufficient=False, ...)

if top_score is None or top_score < min_top_score:
    return EvidenceSufficiencyDecision(is_sufficient=False, ...)
```

### Why this step matters
Vector search always returns nearest neighbors, even for unsupported questions. Evidence sufficiency checks are needed before LLM answer generation so weak or irrelevant context does not become a hallucinated answer.

## Step 44 — Evidence sufficiency reporting in query search

### Description
Integrated the evidence sufficiency checker into the query search script so each manual query reports whether retrieved evidence is sufficient before answer generation.

### Code snippets used

#### Sufficiency check
```python
sufficiency = assess_evidence_sufficiency(
    results,
    min_results=1,
    min_top_score=args.min_top_score,
)
```

#### CLI threshold option
```bash
python scripts/run_qdrant_query_search.py --query "..." --min-top-score 0.30
```

### Why this step matters
Manual retrieval inspection now includes a safety signal. Before passing context to an LLM, the system can warn whether evidence looks sufficient or whether it should abstain.

## Step 46 — Grounded answer contract and citation baseline

### Description
Added the first grounded answer contract with citation objects and a safe insufficient-evidence response builder. This does not call the LLM yet.

### Code snippets used

#### Citation schema
```python
@dataclass(frozen=True)
class Citation:
    unit_id: str
    document_family: str | None
    release_label: str | None
    source_kind: str | None
    score: float
    text_preview: str
```

#### Insufficient evidence response
```python
def build_insufficient_evidence_response(
    request: GroundedAnswerRequest,
) -> GroundedAnswerResponse:
```

### Why this step matters
Grounded answer generation needs a stable contract before calling an LLM. The system must be able to cite retrieved evidence and safely refuse when evidence sufficiency fails.

## Step 47 — Grounded answer prompt template

### Description
Added a grounded prompt template that instructs the LLM to answer only from provided evidence, cite retrieved sources, preserve release labels, and refuse unsupported answers.

### Code snippets used

#### System prompt rule
```python
SYSTEM_PROMPT = """You are a grounded enterprise functional specification assistant.

Rules:
1. Answer only using the provided evidence.
2. Do not use outside knowledge.
3. Do not invent missing facts.
..."""
```

#### Evidence block citation format
```python
f"[C{index}]"
f"unit_id: {citation.unit_id}"
f"release_label: {citation.release_label}"
f"text: {citation.text_preview}"
```

### Why this step matters
Prompt design is part of the RAG safety boundary. Before calling an LLM, the prompt must explicitly constrain generation to retrieved evidence and require citations.

## Step 48 — LLM client wrapper baseline

### Description
Added a small LLM chat-completion wrapper that sends a grounded system/user prompt to the configured chat model and returns validated response text.

### Code snippets used

#### Client bootstrap
```python
def get_llm_client() -> OpenAI:
    settings = get_settings()
    return OpenAI(
        api_key=settings.openai_api_key,
        base_url=settings.openai_base_url or None,
    )
```

#### Chat completion call
```python
response = llm_client.chat.completions.create(
    model=selected_model,
    messages=[
        {"role": "system", "content": prompt.system_prompt},
        {"role": "user", "content": prompt.user_prompt},
    ],
)
```

### Why this step matters
This isolates model-calling behavior from answer orchestration. The wrapper is testable with fake clients and validates missing/empty model responses before the full answer-generation service is built.

## Step 49 — Grounded answer-generation service

### Description
Added an answer-generation service that orchestrates evidence sufficiency, refusal behavior, prompt construction, LLM calling, and citation output.

### Code snippets used

#### Service function
```python
def generate_grounded_answer(
    query: str,
    retrieved_results: list[QdrantSearchResult],
    sufficiency: EvidenceSufficiencyDecision,
    llm_client: Any | None = None,
    model: str | None = None,
) -> GroundedAnswerResponse:
```

#### Refusal gate
```python
if not sufficiency.is_sufficient:
    return build_insufficient_evidence_response(request)
```

#### LLM path
```python
prompt = build_grounded_prompt(request)
answer_text = generate_chat_completion(prompt=prompt, model=model, client=llm_client)
```

### Why this step matters
This is the first complete grounded-answer workflow. It ensures insufficient evidence triggers safe refusal before any LLM call, while sufficient evidence uses the grounded prompt and returns structured citations.

## Step 50 — Grounded answer smoke-test script

### Description
Added a runnable answer smoke-test script that executes retrieval, evidence sufficiency, grounded answer generation, and citation logging for one query.

### Code snippets used

#### Retrieval + sufficiency + answer flow
```python
retrieved_results = search_query_text(...)
sufficiency = assess_evidence_sufficiency(retrieved_results, ...)
response = generate_grounded_answer(
    query=args.query,
    retrieved_results=retrieved_results,
    sufficiency=sufficiency,
)
```

#### Command
```bash
python scripts/run_answer_smoke_test.py --query "branch reports realignment" --release-label R24 --source-kind paragraph
```

### Why this step matters
This is the first real end-to-end answer path. It is still a controlled smoke test, but it validates retrieval-to-answer orchestration before building FastAPI or UI layers.

## Step 51 — LLM usage tracking baseline

### Description
Added lightweight LLM usage tracking to the chat-completion wrapper and grounded answer response path.

### Code snippets used

#### Usage schema
```python
@dataclass(frozen=True)
class LLMUsage:
    model: str
    prompt_tokens: int | None
    completion_tokens: int | None
    total_tokens: int | None
```

#### Usage extraction
```python
def extract_llm_usage(response: Any, model: str) -> LLMUsage:
```

#### Smoke-test logging
```python
logger.info(
    "LLM usage | model=%s | prompt_tokens=%s | completion_tokens=%s | total_tokens=%s",
    response.usage.model,
    response.usage.prompt_tokens,
    response.usage.completion_tokens,
    response.usage.total_tokens,
)
```

### Why this step matters
Now that the system makes real LLM calls, we need token observability. This is the first step toward tracking cost, comparing prompts/models, and controlling production spend.

## Step 52 — LLM cost estimation baseline

### Description
Added lightweight LLM cost estimation using prompt/completion token usage and configurable input/output token prices.

### Code snippets used

#### Cost schema
```python
@dataclass(frozen=True)
class LLMCostEstimate:
    model: str
    input_cost: float | None
    output_cost: float | None
    total_cost: float | None
    currency: str = "USD"
```

#### Cost calculation
```python
input_cost = (usage.prompt_tokens / 1000) * input_cost_per_1k_tokens
output_cost = (usage.completion_tokens / 1000) * output_cost_per_1k_tokens
```

#### Config fields
```python
LLM_INPUT_COST_PER_1K_TOKENS=0.0
LLM_OUTPUT_COST_PER_1K_TOKENS=0.0
```

### Why this step matters
Token counts become actionable only when connected to pricing. This baseline lets us estimate answer cost per model once pricing values are configured, while keeping defaults at zero for safety.

## Step 53 — Answer trace artifact storage

### Description
Added local answer trace logging so each answer-generation run can be persisted with query, filters, retrieval results, sufficiency decision, final answer, citations, usage, and cost.

### Code snippets used

#### Trace schema
```python
@dataclass(frozen=True)
class AnswerTrace:
    request_id: str
    created_at_utc: str
    query: str
    filters: dict[str, str | None]
    sufficiency: EvidenceSufficiencyDecision
    answer_response: GroundedAnswerResponse
    retrieval_results: list[dict]
```

#### Trace writer
```python
def write_answer_trace(
    trace: AnswerTrace,
    output_directory: str | Path,
) -> Path:
```

### Why this step matters
Answer traces are essential for debugging grounded answers and refusals. They preserve the full request/response context locally before we expose this workflow through an API or UI.

## Step 54 — Post-generation citation validation

### Description
Added citation validation to detect invalid or missing citation IDs in generated answers before returning them as answered responses.

### Code snippets used

#### Citation extraction
```python
CITATION_PATTERN = re.compile(r"\[C(?P<number>\d+)\]")
```

#### Validation rule
```python
invalid_ids = [citation_id for citation_id in cited_ids if citation_id not in available_ids]
missing_citation = response.is_answered and bool(response.citations) and not cited_ids
```

#### Service guard
```python
citation_validation = validate_answer_citations(response)
if not citation_validation.is_valid:
    return GroundedAnswerResponse(is_answered=False, ...)
```

### Why this step matters
Prompt instructions are not guarantees. Citation validation prevents answers with fabricated or missing citations from being returned as trusted grounded answers.

## Step 45 — Config-driven retrieval sufficiency threshold

### Description
Moved the default retrieval evidence sufficiency threshold into centralized settings while keeping CLI override support for experiments.

### Code snippets used

#### Settings field
```python
retrieval_min_top_score: float = Field(default=0.30, alias="RETRIEVAL_MIN_TOP_SCORE")
```

#### Query script fallback logic
```python
min_top_score = (
    args.min_top_score
    if args.min_top_score is not None
    else settings.retrieval_min_top_score
)
```

### Why this step matters
Thresholds should not be hardcoded. Keeping the default in config improves maintainability, while keeping a CLI override supports evaluation experiments without editing source code.

## Step 42 — Expanded retrieval evaluation cases

### Description
Expanded the retrieval evaluation set with additional R24 paragraph cases and one expected unanswerable/no-evidence case.

### Cases added
```text
r24_existing_functionality_reports_count
r24_assumptions_extraction_logic_same
r24_unanswerable_mobile_app_login
```

### Why this step matters
The retrieval evaluation set needs more than one or two examples. Adding answerable, assumption-based, and unanswerable cases helps distinguish real retrieval quality from lucky manual queries.

## Step 40 — Tightened R24 realignment retrieval expectation

### Description
Refined the R24 branch reports realignment evaluation case to require paragraph evidence and stronger top-1 markers from the actual report-consolidation paragraph.

### Evaluation change
```json
"filters": {
  "release_label": "R24",
  "source_kind": "paragraph"
}
```

```json
"expected_top1_contains_any": [
  "The enhancements scoped",
  "multiple Teller reports",
  "multiple Branch reports"
]
```

### Why this step matters
The previous expectation was too broad and allowed a traceability matrix table to pass as top evidence. Tightening the case makes the evaluator better at detecting ranking quality issues before we change retrieval logic.

## Step 55 — Lexical retrieval baseline

### Description
Added a dependency-free lexical retrieval baseline over persisted `.retrieval_ready.json` artifacts so exact-match behavior can be compared against dense Qdrant retrieval before introducing hybrid search.

### Code snippets used

#### Identifier-preserving tokenizer
```python
TOKEN_PATTERN = re.compile(r"[A-Za-z0-9]+(?:[-_][A-Za-z0-9]+)*")

def tokenize(text: str) -> list[str]:
    return [match.group(0).lower() for match in TOKEN_PATTERN.finditer(text)]
```

#### Lexical result schema
```python
@dataclass(frozen=True)
class LexicalSearchResult:
    point_id: str
    score: float
    payload: dict[str, Any]
```

#### Artifact search entrypoint
```python
def search_lexical_artifacts(
    artifact_directory: str | Path,
    query_text: str,
    limit: int = 5,
    document_family: str | None = None,
    release_label: str | None = None,
    source_kind: str | None = None,
) -> list[LexicalSearchResult]:
```

### Why this step matters
Dense retrieval is often weak for exact identifiers, acronyms, field names, and report IDs such as `B-01`. This lexical baseline gives us a measurable exact-match retrieval path before hybrid retrieval is considered. If lexical succeeds where dense fails, the system has an exact-match retrieval gap. If both fail, the likely issue is ingestion, unsupported source content, metadata, or missing indexed evidence rather than retrieval scoring alone.

### Validation and intentional stress test
- Added `tests/test_lexical_search.py` covering tokenization, stopword handling, exact-identifier ranking, metadata filtering, artifact loading, invalid inputs, and missing artifact directories.
- Targeted test run passed: `7 passed`.
- Full regression suite passed: `108 passed`.
- Real artifact stress test for `B-01 report layout` returned the expected R24 paragraph marker chunk first: `chunk_6`, matching `b-01`, `layout`, and `report`.
- Important limitation discovered: lexical search can find the paragraph that says the layout is attached, but it still does not extract the actual embedded attachment layout. That remains an ingestion-scope limitation, not a lexical retrieval success for full layout details.

## Step 56 — Dense vs lexical retrieval comparison report

### Description
Added a comparison layer and runnable script that evaluate dense Qdrant retrieval and lexical artifact retrieval on the same retrieval evaluation cases, then classify each case as `both_pass`, `dense_only`, `lexical_only`, or `both_fail`.

### Code snippets used

#### Comparison outcome classifier
```python
def classify_retrieval_comparison(
    dense_passed: bool,
    lexical_passed: bool,
) -> str:
    if dense_passed and lexical_passed:
        return COMPARISON_BOTH_PASS
    if dense_passed:
        return COMPARISON_DENSE_ONLY
    if lexical_passed:
        return COMPARISON_LEXICAL_ONLY
    return COMPARISON_BOTH_FAIL
```

#### Case-level comparison report
```python
@dataclass(frozen=True)
class RetrievalComparisonCaseReport:
    case: RetrievalEvalCase
    dense_evaluation: RetrievalEvalResult
    lexical_evaluation: RetrievalEvalResult
    comparison_outcome: str
    dense_top_results: list[dict[str, Any]]
    lexical_top_results: list[dict[str, Any]]
```

#### Runnable comparison command
```bash
python scripts/run_retrieval_comparison.py --limit 5 --output-file data/eval/generated/retrieval_comparison_report.json
```

### Why this step matters
Hybrid retrieval should not be added blindly. This step gives a measurable comparison between dense semantic retrieval and lexical exact-match retrieval using the same evaluation cases, filters, and top-k limit. It tells us where dense retrieval is stronger, where lexical retrieval is stronger, where both work, and where both fail.

### Validation and real comparison result
- Added `app/retrieval/retrieval_comparison.py`.
- Added `scripts/run_retrieval_comparison.py`.
- Added `tests/test_retrieval_comparison.py` and `tests/test_retrieval_comparison_script.py`.
- Targeted Step 56 tests passed: `5 passed`.
- Full regression suite passed: `113 passed`.
- Real comparison report was written to `data/eval/generated/retrieval_comparison_report.json`.
- Real comparison summary:
  - `total_cases=7`
  - `dense_passed_count=5`
  - `lexical_passed_count=5`
  - `both_pass_count=4`
  - `dense_only_count=1`
  - `lexical_only_count=1`
  - `both_fail_count=1`

### Failure modes discovered
- Dense-only case: `r24_branch_reports_realignment_summary`. Dense retrieval found the semantic requirements-summary chunk, while lexical retrieval ranked an annexure/layout-related chunk first because of exact terms like `branch`, `reports`, and `realignment`.
- Lexical-only case: `r24_b01_report_layout_missing_embedded_attachment`. Lexical retrieval found exact `B-01`/layout markers better than dense retrieval, but this does **not** prove the actual report layout details are extracted; the detailed layout may still live in an embedded attachment.
- Both-fail case: `r24_unanswerable_mobile_app_login`, which is expected because the indexed R24 corpus does not contain mobile-login evidence.

### Production interpretation
This comparison proves that dense and lexical retrieval have complementary failure modes, but it also exposes a weakness in marker-based evaluation: matching words like `B-01` and `layout` is not the same as proving the actual layout evidence exists. The next disciplined step is error analysis of the comparison report before implementing hybrid fusion.

## Step 57 — Retrieval comparison error analysis

### Description
Added a deterministic error-analysis layer over the dense-vs-lexical comparison report. The analyzer classifies non-both-pass cases with root-cause labels and recommended next actions before any hybrid retrieval logic is designed.

### Code snippets used

#### Error-analysis case schema
```python
@dataclass(frozen=True)
class RetrievalErrorAnalysisCase:
    case_id: str
    query: str
    comparison_outcome: str
    expected_to_pass: bool
    severity: str
    root_cause_labels: list[str]
    rationale: str
    recommended_next_action: str
```

#### Comparison report analyzer
```python
def analyze_retrieval_comparison_report(
    comparison_report_payload: dict[str, Any],
    include_both_pass: bool = False,
) -> RetrievalErrorAnalysisReport:
```

#### Runnable analysis command
```bash
python scripts/run_retrieval_error_analysis.py \
  --comparison-report data/eval/generated/retrieval_comparison_report.json \
  --output-file data/eval/generated/retrieval_error_analysis_report.json
```

### Why this step matters
The comparison report showed complementary dense and lexical behavior, but comparison alone does not tell us how to build hybrid retrieval safely. Error analysis identifies whether failures come from lexical false positives, dense identifier misses, marker-only evidence, unsupported attachments, or expected unanswerable queries. This prevents premature hybrid search from combining noise with signal.

### Validation and generated artifact
- Added `app/retrieval/retrieval_error_analysis.py`.
- Added `scripts/run_retrieval_error_analysis.py`.
- Added `tests/test_retrieval_error_analysis.py` and `tests/test_retrieval_error_analysis_script.py`.
- Targeted Step 57 tests passed: `6 passed`.
- Full regression suite passed: `119 passed`.
- Generated `data/eval/generated/retrieval_error_analysis_report.json`.

### Real error-analysis summary
- `total_cases=7`
- `analyzed_case_count=3` because both-pass cases are excluded by default
- `high_severity_count=1`
- `medium_severity_count=1`
- `low_severity_count=1`

### Root-cause labels observed
```text
dense_exact_identifier_miss: 1
dense_ranking_failure: 1
expected_unanswerable: 1
lexical_exact_term_false_positive: 1
lexical_ranking_failure: 1
lexical_top1_failure: 1
marker_match_not_full_evidence: 1
unsupported_attachment_marker_match: 1
weak_marker_expectation: 1
```

### Production interpretation
- `r24_branch_reports_realignment_summary` is a lexical ranking problem: lexical over-ranked exact-term overlap from an annexure chunk instead of the requirements-summary chunk.
- `r24_b01_report_layout_missing_embedded_attachment` is a high-severity evaluation/ingestion warning: lexical found exact markers, but the full layout evidence may live in an embedded attachment; this should not be treated as a clean retrieval win.
- `r24_unanswerable_mobile_app_login` is a valid expected-unanswerable case and should remain a refusal/abstention regression check.

### Next decision implied
The next disciplined step is not general hybrid retrieval yet. First, tighten the problematic evaluation case and/or inspect whether full `B-01` layout evidence exists in extracted artifacts. Hybrid fusion should wait until marker-only passes are separated from true evidence passes.

## Step 58 — Tightened marker-only unsupported evidence evaluation

### Description
Tightened the retrieval evaluation contract so marker/reference matches can be separated from actual answer evidence and unsupported attachment-only evidence. The main target was the `r24_b01_report_layout_missing_embedded_attachment` case, which previously risked looking like a lexical win simply because lexical retrieval found `B-01`, `layout`, and sample-report markers.

### Code snippets used

#### Extended expectation schema
```python
@dataclass(frozen=True)
class RetrievalEvalExpectation:
    expected_marker_contains_any: list[str] | None = None
    expected_top1_contains_any: list[str] | None = None
    expected_text_contains_any: list[str] | None = None
    unsupported_evidence_contains_any: list[str] | None = None
```

#### Unsupported marker/reference-only detection
```python
if expectation.unsupported_evidence_contains_any:
    found_unsupported_markers = [
        marker
        for marker in expectation.unsupported_evidence_contains_any
        if marker in combined_text
    ]
    if found_unsupported_markers:
        failures.append(
            "Retrieved evidence appears to contain unsupported marker/reference-only content: "
            f"{found_unsupported_markers}"
        )
```

#### Tightened B-01 eval case
```json
"expected_marker_contains_any": ["B-01", "Branch End of Day", "layout"],
"unsupported_evidence_contains_any": ["attached sample report", "Sample Report", ".xlsx"]
```

### Why this step matters
Marker matches are not the same as answer evidence. In enterprise documents, text can say `Sample Report: B-01 Branch End of Day Report.xlsx`, but that only proves a reference to an attachment exists. It does not prove that the attachment's layout was extracted and indexed. This distinction prevents retrieval evaluation from rewarding marker-only evidence as if it were complete grounded evidence.

### Validation and regenerated artifacts
- Updated `app/retrieval/evaluation.py`.
- Updated `data/eval/retrieval_eval.json`.
- Updated `app/retrieval/retrieval_error_analysis.py` to classify expected unsupported evidence separately from expected unanswerable cases.
- Updated tests in `tests/test_retrieval_evaluation.py` and `tests/test_retrieval_error_analysis.py`.
- Targeted tests passed: `13 passed`.
- Full regression suite passed: `122 passed`.
- Regenerated:
  - `data/eval/generated/retrieval_eval_report.json`
  - `data/eval/generated/retrieval_comparison_report.json`
  - `data/eval/generated/retrieval_error_analysis_report.json`

### Result change after tightening
Before tightening, the B-01 case appeared as `lexical_only` in dense-vs-lexical comparison because lexical found marker terms.

After tightening:
```text
dense_passed_count=5
lexical_passed_count=4
both_pass_count=4
dense_only_count=1
lexical_only_count=0
both_fail_count=2
```

The B-01 case is now correctly classified as `both_fail` with error-analysis labels:
```text
expected_unsupported_evidence
marker_match_not_full_evidence
unsupported_attachment_marker_match
```

### Production interpretation
This is a critical correction. The system no longer treats finding attachment markers as a successful retrieval of actual layout evidence. That makes future hybrid retrieval safer, because hybrid will not be optimized toward a false lexical win.

### Next decision implied
The next step should inspect the processed and retrieval-ready artifacts for the B-01 layout path and decide whether embedded attachment extraction is in scope, or whether the assistant should explicitly abstain whenever a question depends on that attachment.

## Step 59 — Hybrid retrieval baseline with simple score fusion

### Description
Implemented the first hybrid retrieval baseline by fusing dense Qdrant retrieval results with lexical artifact retrieval results using normalized weighted score fusion. This is a baseline comparison mechanism, not a reranker and not an agent.

### Code snippets used

#### Hybrid result schema
```python
@dataclass(frozen=True)
class HybridSearchResult:
    point_id: str
    score: float
    payload: dict[str, Any]
```

#### Simple normalized score fusion
```python
def fuse_dense_and_lexical_results(
    dense_results: Sequence[Any],
    lexical_results: Sequence[Any],
    limit: int = 5,
    dense_weight: float = 0.5,
    lexical_weight: float = 0.5,
) -> list[HybridSearchResult]:
```

#### Hybrid evaluation command
```bash
python scripts/run_hybrid_retrieval_eval.py \
  --limit 5 \
  --candidate-limit 10 \
  --dense-weight 0.5 \
  --lexical-weight 0.5 \
  --output-file data/eval/generated/hybrid_retrieval_eval_report.json
```

### Why this step matters
This is the first measured hybrid retrieval experiment. It tests whether combining dense semantic evidence and lexical exact-match evidence improves retrieval quality under the tightened evaluation labels. The B-01 attachment case remains protected: hybrid is not allowed to count attachment-marker evidence as a true pass.

### Validation and generated artifact
- Added `app/retrieval/hybrid_search.py`.
- Added `app/retrieval/hybrid_evaluation.py`.
- Added `scripts/run_hybrid_retrieval_eval.py`.
- Added `tests/test_hybrid_search.py`, `tests/test_hybrid_evaluation.py`, and `tests/test_hybrid_retrieval_eval_script.py`.
- Targeted Step 59 tests passed: `7 passed`.
- Full regression suite passed: `129 passed`.
- Generated `data/eval/generated/hybrid_retrieval_eval_report.json`.

### Real hybrid evaluation summary
```text
total_cases=7
dense_passed_count=5
lexical_passed_count=4
hybrid_passed_count=5
all_pass_count=4
hybrid_only_count=0
dense_and_hybrid_count=1
lexical_and_hybrid_count=0
hybrid_missed_dense_count=0
hybrid_missed_lexical_count=0
hybrid_missed_both_count=0
all_fail_count=2
```

### Production interpretation
- Hybrid preserved dense success on the realignment-summary case where lexical alone failed top-1.
- Hybrid did **not** falsely pass the B-01 attachment-marker case; it remained `all_fail`, which is correct under the current text/table-only ingestion scope.
- Hybrid did not create new wins on this small eval set (`hybrid_only_count=0`), so we should not overclaim improvement. It mainly preserved dense wins while incorporating lexical evidence when useful.
- The unanswerable mobile-login case remained `all_fail`, so hybrid did not force a false answer.

### Next decision implied
The next disciplined step is to inspect hybrid failure and non-improvement cases, then decide whether to tune weights, filter lexical candidates, or expand evaluation before adopting hybrid as default retrieval.

## Step 60 — Hybrid weight experiment runner

### Description
Added a hybrid weight experiment runner that evaluates multiple dense/lexical weight settings over the same retrieval evaluation cases. The runner retrieves dense and lexical candidates once per case, then reuses those candidates across all weight settings to avoid repeated embedding calls for every setting.

### Code snippets used

#### Default weight pairs
```python
DEFAULT_WEIGHT_PAIRS = [
    (0.8, 0.2),
    (0.6, 0.4),
    (0.5, 0.5),
    (0.4, 0.6),
    (0.2, 0.8),
]
```

#### Weight setting schema
```python
@dataclass(frozen=True)
class HybridWeightSettingReport:
    setting: HybridWeightSetting
    total_cases: int
    hybrid_passed_count: int
    expected_outcome_count: int
    unsafe_expected_failure_pass_count: int
    unexpected_failure_count: int
    outcome_counts: dict[str, int]
```

#### Runnable experiment command
```bash
python scripts/run_hybrid_weight_experiments.py \
  --limit 5 \
  --candidate-limit 10 \
  --weights 0.8:0.2,0.6:0.4,0.5:0.5,0.4:0.6,0.2:0.8 \
  --output-file data/eval/generated/hybrid_weight_experiment_report.json
```

### Why this step matters
Hybrid retrieval should not become default just because one 50/50 fusion run looked safe. This step tests whether retrieval quality is sensitive to dense-vs-lexical weighting and whether lexical-heavy fusion promotes noise or unsafe expected-failure passes.

### Validation and generated artifact
- Added `app/retrieval/hybrid_weight_experiment.py`.
- Added `scripts/run_hybrid_weight_experiments.py`.
- Added `tests/test_hybrid_weight_experiment.py` and `tests/test_hybrid_weight_experiments_script.py`.
- Targeted Step 60 tests passed: `6 passed`.
- Full regression suite passed: `135 passed`.
- Generated `data/eval/generated/hybrid_weight_experiment_report.json`.

### Real experiment summary
Best settings by safety-first ranking:
```text
dense_0.4_lexical_0.6
dense_0.5_lexical_0.5
dense_0.6_lexical_0.4
dense_0.8_lexical_0.2
```

Settings with expected outcomes `7/7`:
```text
dense=0.8 lexical=0.2
dense=0.6 lexical=0.4
dense=0.5 lexical=0.5
dense=0.4 lexical=0.6
```

Lexical-heavy setting degraded:
```text
dense=0.2 lexical=0.8
expected_outcomes=6/7
hybrid_passed=4
unexpected_failure=1
outcome={'all_fail': 2, 'all_pass': 4, 'hybrid_missed_dense': 1}
```

### Production interpretation
- Moderate dense/lexical mixes were stable on the current small eval set.
- Heavily lexical weighting caused hybrid to miss one dense-supported case, confirming the risk that lexical-heavy fusion can promote exact-term noise.
- No setting produced unsafe expected-failure passes, so the tightened B-01 and mobile-login safety labels held.
- Because several settings tied on the current small eval set, the system still needs a larger evaluation set before choosing a default hybrid weight.

### Next decision implied
The next disciplined step is to expand retrieval evaluation cases before locking a default hybrid weight. At minimum, add more exact-identifier, semantic paraphrase, boilerplate-risk, table, and unsupported-content cases.

## Step 61 — Config-driven provisional hybrid retrieval defaults

### Description
Moved the provisional hybrid retrieval decision into centralized configuration so the retrieval mode and hybrid weights are explicit, reversible, and environment-controlled. The selected provisional default is `hybrid` mode with `0.6` dense weight and `0.4` lexical weight.

### Code snippets used

#### Settings fields
```python
retrieval_mode: str = Field(default="hybrid", alias="RETRIEVAL_MODE")
hybrid_dense_weight: float = Field(default=0.60, alias="HYBRID_DENSE_WEIGHT")
hybrid_lexical_weight: float = Field(default=0.40, alias="HYBRID_LEXICAL_WEIGHT")
hybrid_candidate_limit: int = Field(default=10, alias="HYBRID_CANDIDATE_LIMIT")
```

#### Runtime config schema
```python
@dataclass(frozen=True)
class RetrievalRuntimeConfig:
    retrieval_mode: str
    hybrid_dense_weight: float
    hybrid_lexical_weight: float
    hybrid_candidate_limit: int
```

#### Validation rules
```python
SUPPORTED_RETRIEVAL_MODES = {"dense", "lexical", "hybrid"}

if retrieval_mode not in SUPPORTED_RETRIEVAL_MODES:
    raise ValueError(...)
if hybrid_dense_weight == 0 and hybrid_lexical_weight == 0:
    raise ValueError(...)
if hybrid_candidate_limit <= 0:
    raise ValueError(...)
```

### Why this step matters
The `0.6 dense / 0.4 lexical` choice is not proven globally optimal. It is a provisional engineering default based on a small evaluation set. Keeping it config-driven prevents hardcoding premature assumptions and makes it easy to rerun experiments or switch back to dense retrieval as the corpus and evaluation set grow.

### Validation
- Updated `app/core/config.py`.
- Updated `.env.example`.
- Added `app/retrieval/retrieval_config.py`.
- Added `tests/test_retrieval_config.py`.
- Targeted Step 61 tests passed: `5 passed`.
- Full regression suite passed: `140 passed`.

### Important limitation
This step does **not** wire hybrid retrieval into answer generation yet. It only establishes validated runtime defaults. The query/answer path should be wired to respect `RETRIEVAL_MODE` in a later step.

### Next decision implied
The next step should add a retrieval service/router that uses `RETRIEVAL_MODE` to choose dense, lexical, or hybrid retrieval for a query while preserving the existing evidence sufficiency and grounded-answer safeguards.

## Step 62 — Retrieval mode service/router

### Description
Added a retrieval router that uses a validated `RetrievalRuntimeConfig` to route a query to dense, lexical, or hybrid retrieval. The router is dependency-injected: callers provide dense and lexical search callables, which keeps it testable and independent from OpenAI/Qdrant setup details.

### Code snippets used

#### Routed result schema
```python
@dataclass(frozen=True)
class RoutedRetrievalResult:
    retrieval_mode: str
    results: list[QdrantSearchResult]
```

#### Router entrypoint
```python
def route_retrieval(
    config: RetrievalRuntimeConfig,
    dense_search: SearchCallable,
    lexical_search: SearchCallable,
    limit: int = 5,
) -> RoutedRetrievalResult:
```

#### Hybrid routing behavior
```python
candidate_limit = max(limit, config.hybrid_candidate_limit)
dense_results = dense_search(candidate_limit)
lexical_results = lexical_search(candidate_limit)
hybrid_results = fuse_dense_and_lexical_results(
    dense_results=dense_results,
    lexical_results=lexical_results,
    limit=limit,
    dense_weight=config.hybrid_dense_weight,
    lexical_weight=config.hybrid_lexical_weight,
)
```

### Why this step matters
This creates the runtime selection point for retrieval mode without yet changing answer generation. It isolates routing behavior and lets us test dense-only, lexical-only, and hybrid routing before wiring it into the answer path.

### Validation
- Added `app/retrieval/retrieval_router.py`.
- Added `tests/test_retrieval_router.py`.
- Targeted Step 62 tests passed: `6 passed`.
- Full regression suite passed: `146 passed`.

### Important limitation
This router is not yet used by answer generation or CLI query scripts. That is intentional. The next step should wire a query-level retrieval service around this router and then update answer generation only after that service is tested.

### Next decision implied
The next step should create a query retrieval service that builds the dense and lexical callables from real Qdrant, embedding, and artifact dependencies, then calls `route_retrieval` using `RETRIEVAL_MODE`.

## Step 63 — Query retrieval service using the router

### Description
Added a query retrieval service that integrates real dense retrieval, lexical artifact retrieval, and the retrieval router. The service builds dense and lexical search callables using the same query, filters, Qdrant client, embedding model/client, and retrieval-ready artifact directory, then delegates retrieval-mode selection to `route_retrieval`.

### Code snippets used

#### Query retrieval service entrypoint
```python
def retrieve_query_evidence(
    qdrant_client: QdrantClient,
    collection_name: str,
    query_text: str,
    embedding_model: str,
    retrieval_config: RetrievalRuntimeConfig,
    lexical_artifact_directory: str | Path,
    embedding_client: Any | None = None,
    limit: int = 5,
    document_family: str | None = None,
    release_label: str | None = None,
    source_kind: str | None = None,
) -> RoutedRetrievalResult:
```

#### Dense callable construction
```python
def dense_search(search_limit: int):
    return search_query_text(
        qdrant_client=qdrant_client,
        collection_name=collection_name,
        query_text=query_text,
        embedding_model=embedding_model,
        embedding_client=embedding_client,
        limit=search_limit,
        document_family=document_family,
        release_label=release_label,
        source_kind=source_kind,
    )
```

#### Lexical callable construction
```python
def lexical_search(search_limit: int):
    return search_lexical_artifacts(
        artifact_directory=lexical_artifact_directory,
        query_text=query_text,
        limit=search_limit,
        document_family=document_family,
        release_label=release_label,
        source_kind=source_kind,
    )
```

### Why this step matters
This is the first real integration layer between configuration-driven routing and concrete retrieval dependencies. It still does not generate answers. It only returns routed evidence, keeping retrieval integration testable before changing answer-generation behavior.

### Validation
- Added `app/services/query_retrieval.py`.
- Added `tests/test_query_retrieval_service.py`.
- Targeted Step 63 tests passed: `5 passed`.
- Full regression suite passed: `151 passed`.

### Behaviors tested
- Dense mode uses Qdrant/query embedding retrieval.
- Lexical mode uses persisted `.retrieval_ready.json` artifacts.
- Hybrid mode fuses dense and lexical outputs through the router.
- Metadata filters are respected.
- Invalid retrieval limit fails explicitly.

### Important limitation
Answer generation still does not consume this service yet. That should be a separate step because it changes runtime answer behavior, evidence sufficiency inputs, citations, and trace outputs.

### Next decision implied
The next step should update the manual query/search or answer smoke-test path to use this retrieval service, then verify evidence sufficiency, citations, refusals, and traces still behave correctly.

## Step 64 — Manual query search script uses retrieval service

### Description
Updated the manual query search script so it uses `retrieve_query_evidence(...)` instead of directly calling dense-only Qdrant search. The script now respects `RETRIEVAL_MODE`, hybrid weights, and hybrid candidate limit from centralized settings while preserving evidence sufficiency checks.

### Code snippets used

#### Runtime config loading
```python
retrieval_config = build_retrieval_runtime_config(settings)
```

#### Routed retrieval call
```python
routed = retrieve_query_evidence(
    qdrant_client=client,
    collection_name=settings.qdrant_collection_name,
    query_text=args.query,
    embedding_model=settings.openai_embedding_model,
    retrieval_config=retrieval_config,
    lexical_artifact_directory=settings.processed_dir,
    limit=args.limit,
    document_family=args.document_family,
    release_label=args.release_label,
    source_kind=args.source_kind,
)
results = routed.results
```

#### Mode-aware Qdrant collection requirement
```python
def _requires_qdrant_collection(retrieval_mode: str) -> bool:
    return retrieval_mode in {"dense", "hybrid"}
```

### Why this step matters
Manual retrieval inspection is the safest place to verify the config-driven retrieval service before answer generation consumes it. The CLI now reports retrieval mode, weights, candidate limit, filters, sufficiency status, and per-rank evidence metadata.

### Validation
- Updated `scripts/run_qdrant_query_search.py`.
- Updated `tests/test_qdrant_query_search_script.py`.
- Targeted Step 64 tests passed: `2 passed`.
- Full regression suite passed: `152 passed`.

### Important limitation
This still does not wire answer generation to the retrieval service. It only updates the manual retrieval inspection path. That is intentional because answer-generation wiring affects citations, refusals, traces, token usage, and cost.

### Next decision implied
The next step should update the answer smoke-test path to use `retrieve_query_evidence(...)`, then verify grounded answer generation, sufficiency checks, citations, refusals, usage/cost, and trace artifacts still behave correctly with routed retrieval.

## Step 65 — Answer smoke-test script uses retrieval service

### Description
Updated the smallest end-to-end answer path, `scripts/run_answer_smoke_test.py`, so it now uses the config-driven query retrieval service instead of direct dense-only Qdrant search. The smoke test now respects `RETRIEVAL_MODE`, hybrid weights, and hybrid candidate limit before running evidence sufficiency, grounded answer generation, citation handling, usage/cost logging, and answer trace export.

### Code snippets used

#### Runtime retrieval config loading
```python
retrieval_config = build_retrieval_runtime_config(settings)
```

#### Mode-aware Qdrant collection requirement
```python
def _requires_qdrant_collection(retrieval_mode: str) -> bool:
    """Return whether the selected retrieval mode requires a Qdrant collection."""

    return retrieval_mode in {"dense", "hybrid"}
```

#### Routed retrieval in the answer smoke test
```python
routed = retrieve_query_evidence(
    qdrant_client=client,
    collection_name=settings.qdrant_collection_name,
    query_text=args.query,
    embedding_model=settings.openai_embedding_model,
    retrieval_config=retrieval_config,
    lexical_artifact_directory=settings.processed_dir,
    limit=args.limit,
    document_family=args.document_family,
    release_label=args.release_label,
    source_kind=args.source_kind,
)
retrieved_results = routed.results
```

#### Existing safety gate preserved
```python
sufficiency = assess_evidence_sufficiency(
    retrieved_results,
    min_results=1,
    min_top_score=min_top_score,
)
```

### Why this step matters
This is the first answer-generation path that consumes routed retrieval. It proves the retrieval service output can flow into the grounded answer contract without bypassing the safety gate. This is production-important because answer generation is where retrieval mistakes become user-visible hallucination, weak citation, refusal, cost, and trace problems.

### Production interpretation
- Dense and hybrid modes still require Qdrant because they use vector retrieval.
- Lexical-only mode can run from local retrieval-ready artifacts without requiring a Qdrant collection.
- Evidence sufficiency remains after retrieval because changing retrieval mode does not guarantee answerability.
- The script logs retrieval mode and hybrid parameters so answer runs can be debugged and reproduced.
- Client cleanup is handled in a `finally` block so local Qdrant resources are closed even if retrieval fails.

### Failure mode intentionally tested
The first targeted test run failed because the help-text assertion expected the exact phrase `configured retrieval mode`, but argparse wrapped the longer description and split the phrase across lines. This was a useful robustness lesson: tests for CLI help should avoid brittle assumptions caused by terminal wrapping. The script description was shortened to keep the assertion stable.

### Validation
- Updated `scripts/run_answer_smoke_test.py`.
- Updated `tests/test_answer_smoke_script.py`.
- Targeted Step 65 tests passed: `3 passed`.
- Full regression suite passed: `154 passed`.

### Behaviors tested
- The answer smoke-test help still runs.
- Dense and hybrid modes require a Qdrant collection.
- Lexical-only mode does not require a Qdrant collection.
- The smoke script calls `retrieve_query_evidence(...)` with query, filters, embedding model, retrieval config, artifact directory, and limit.
- The retrieved routed results are passed into evidence sufficiency and grounded answer generation.
- The Qdrant client is closed after the run.

### Important limitation
This does not yet wire routed retrieval into FastAPI or Streamlit. It only updates the smallest answer-generation smoke path so retrieval-to-answer behavior can be tested before changing the user-facing application layer.

### Next decision implied
The next step should inspect whether the API/backend query path exists and, if it does, wire it to the same retrieval service. If no API path is ready, the next step should add a small answer orchestration service that wraps routed retrieval, sufficiency, generation, and trace creation behind one testable function before exposing it through FastAPI/UI.

## Step 66 — Reusable answer orchestration service

### Description
Added a reusable answer orchestration service that wraps the core query flow behind one tested service function. Since no FastAPI route or Streamlit UI query path exists yet, this step deliberately stays in the service layer instead of jumping to user-facing API/UI wiring.

The new service runs:
1. routed retrieval through `retrieve_query_evidence(...)`
2. evidence sufficiency through `assess_evidence_sufficiency(...)`
3. grounded answer generation through `generate_grounded_answer(...)`
4. answer trace creation and local trace persistence

### Code snippets used

#### Orchestration result schema
```python
@dataclass(frozen=True)
class AnswerOrchestrationResult:
    retrieval_mode: str
    retrieval_results: list[QdrantSearchResult]
    sufficiency: EvidenceSufficiencyDecision
    answer_response: GroundedAnswerResponse
    trace: AnswerTrace
    trace_output_path: Path
```

#### Service entrypoint
```python
def run_grounded_answer_query(
    qdrant_client: QdrantClient,
    collection_name: str,
    query_text: str,
    embedding_model: str,
    retrieval_config: RetrievalRuntimeConfig,
    lexical_artifact_directory: str | Path,
    trace_output_directory: str | Path,
    embedding_client: Any | None = None,
    llm_client: Any | None = None,
    llm_model: str | None = None,
    limit: int = 5,
    min_results: int = 1,
    min_top_score: float = 0.30,
    document_family: str | None = None,
    release_label: str | None = None,
    source_kind: str | None = None,
    request_id: str | None = None,
) -> AnswerOrchestrationResult:
```

#### Retrieval metadata added to traces
```python
retrieval_metadata={
    "retrieval_mode": routed.retrieval_mode,
    "hybrid_dense_weight": retrieval_config.hybrid_dense_weight,
    "hybrid_lexical_weight": retrieval_config.hybrid_lexical_weight,
    "hybrid_candidate_limit": retrieval_config.hybrid_candidate_limit,
    "limit": limit,
    "min_results": min_results,
    "min_top_score": min_top_score,
}
```

### Why this step matters
Before adding FastAPI or Streamlit, the backend needs one reusable and testable query contract. Without this orchestration service, the API layer would have to duplicate retrieval, sufficiency, generation, and trace logic. That would make the system harder to test and easier to break.

### Production interpretation
- API, CLI, and future UI layers can call the same tested answer path.
- The caller still owns infrastructure lifecycle such as creating and closing the Qdrant client.
- Evidence sufficiency remains the safety gate between retrieval and generation.
- Trace artifacts now include retrieval metadata so answer runs can be reproduced and debugged.
- The service supports dependency injection for embedding and LLM clients, making it testable without real API calls.

### Failure mode intentionally protected
The orchestration service test covers insufficient evidence. Even if retrieval returns a chunk, the service must refuse when the top score is below threshold and still write a trace. This protects against the common production failure where nearest-neighbor retrieval returns something weak and the answer layer over-trusts it.

### Validation
- Added `app/services/answer_orchestration.py`.
- Extended `app/services/answer_trace.py` with optional `retrieval_metadata`.
- Added `tests/test_answer_orchestration_service.py`.
- Targeted Step 66 tests passed: `3 passed`.
- Full regression suite passed: `156 passed`.

### Behaviors tested
- The orchestration service passes query, filters, retrieval config, clients, artifact paths, and limit to `retrieve_query_evidence(...)`.
- Retrieved routed results are passed into evidence sufficiency.
- Sufficiency decisions are passed into grounded answer generation.
- Trace artifacts are written locally.
- Trace artifacts include retrieval mode, hybrid weights, candidate limit, retrieval limit, and sufficiency thresholds.
- Insufficient evidence triggers a refusal response and still writes a trace.

### Important limitation
The manual answer smoke-test script still has its own orchestration logic. The next step should refactor `scripts/run_answer_smoke_test.py` to call `run_grounded_answer_query(...)` so CLI and future API behavior share the same service path.

### Next decision implied
The next step should update `scripts/run_answer_smoke_test.py` to use `run_grounded_answer_query(...)`, while preserving mode-aware Qdrant collection checks, logging, usage/cost output, citations, and trace export.

## Step 67 — Answer smoke-test script uses answer orchestration service

### Description
Refactored `scripts/run_answer_smoke_test.py` so the CLI answer smoke path now calls the reusable `run_grounded_answer_query(...)` orchestration service instead of duplicating retrieval, evidence sufficiency, answer generation, and trace-writing logic inside the script.

### Code snippets used

#### Orchestration call from the smoke script
```python
orchestration_result = run_grounded_answer_query(
    qdrant_client=client,
    collection_name=settings.qdrant_collection_name,
    query_text=args.query,
    embedding_model=settings.openai_embedding_model,
    retrieval_config=retrieval_config,
    lexical_artifact_directory=settings.processed_dir,
    trace_output_directory=settings.exports_dir / "answer_runs",
    limit=args.limit,
    min_results=1,
    min_top_score=min_top_score,
    document_family=args.document_family,
    release_label=args.release_label,
    source_kind=args.source_kind,
)
```

#### Script now consumes orchestration output
```python
response = orchestration_result.answer_response
sufficiency = orchestration_result.sufficiency
trace = orchestration_result.trace
trace_output = orchestration_result.trace_output_path
```

#### Existing mode-aware Qdrant requirement preserved
```python
if _requires_qdrant_collection(retrieval_config.retrieval_mode) and not client.collection_exists(
    settings.qdrant_collection_name
):
    raise RuntimeError("Qdrant collection does not exist. Run scripts/run_qdrant_indexing.py first.")
```

### Why this step matters
The manual answer smoke-test script is still important for debugging, but it should not own core RAG orchestration logic. By calling the shared service, the CLI now exercises the same retrieval-to-answer contract that future FastAPI and UI layers should use.

### Production interpretation
- CLI, API, and future UI can share one answer orchestration path.
- Evidence sufficiency and refusal behavior stay consistent across entrypoints.
- Trace creation and retrieval metadata stay centralized in the service layer.
- The script remains responsible for CLI concerns: argument parsing, Qdrant lifecycle, collection checks, and human-readable logging.

### Failure mode intentionally protected
The refactor preserves mode-aware Qdrant checks before calling the orchestration service. Lexical-only mode still does not require a Qdrant collection, while dense/hybrid modes still fail early if the collection is missing.

### Validation
- Updated `scripts/run_answer_smoke_test.py`.
- Updated `tests/test_answer_smoke_script.py`.
- Targeted Step 67 tests passed: `5 passed`.
- Full regression suite passed: `156 passed`.

### Behaviors tested
- CLI help still works.
- Qdrant collection requirement still depends on retrieval mode.
- The answer smoke script calls `run_grounded_answer_query(...)` with query, filters, retrieval config, artifact directory, trace directory, limit, and sufficiency threshold.
- The Qdrant client is closed after the run.
- Existing answer orchestration service tests continue to pass.

### Important limitation
There is still no FastAPI or Streamlit query path. The backend orchestration is now ready for API integration, but API request/response schemas and route behavior have not been created yet.

### Next decision implied
The next step should introduce a minimal FastAPI query contract or schema layer that calls `run_grounded_answer_query(...)`, while keeping request validation, response formatting, and error handling separate from retrieval/generation internals.

## Step 68 — Minimal FastAPI query contract

### Description
Added the first minimal FastAPI backend query contract. The new `/query` endpoint validates request input, calls the shared `run_grounded_answer_query(...)` orchestration service, formats a grounded answer response, and handles errors safely without duplicating retrieval/generation internals.

### Code snippets used

#### FastAPI app factory
```python
def create_app() -> FastAPI:
    settings = get_settings()
    app = FastAPI(title=settings.app_name)
    app.include_router(query_router)
    return app
```

#### Query request schema
```python
class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1)
    limit: int = Field(default=5, gt=0)
    document_family: str | None = None
    release_label: str | None = None
    source_kind: str | None = None
    min_top_score: float | None = Field(default=None, ge=0)
```

#### API route calls orchestration service
```python
orchestration_result = run_grounded_answer_query(
    qdrant_client=client,
    collection_name=settings.qdrant_collection_name,
    query_text=request.query,
    embedding_model=settings.openai_embedding_model,
    retrieval_config=retrieval_config,
    lexical_artifact_directory=settings.processed_dir,
    trace_output_directory=settings.exports_dir / "answer_runs",
    limit=request.limit,
    min_results=1,
    min_top_score=min_top_score,
    document_family=request.document_family,
    release_label=request.release_label,
    source_kind=request.source_kind,
)
```

#### Safe unexpected-error response
```python
except Exception as exc:
    logger.exception("Unexpected query API failure")
    raise HTTPException(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        detail="Internal query processing error.",
    ) from exc
```

### Why this step matters
This creates the first backend product boundary for the RAG system. FastAPI is now the controlled interface between external callers and the core RAG engine. The API handles request validation and response formatting, while the orchestration service remains responsible for retrieval, sufficiency, grounded generation, and trace writing.

### Production interpretation
- The API does not duplicate retrieval/generation logic.
- Request validation rejects blank queries, invalid limits, invalid source kinds, and invalid thresholds.
- Dense/hybrid modes return `503` if the required Qdrant collection is missing.
- Lexical-only mode skips the Qdrant collection requirement because it uses local retrieval-ready artifacts.
- Unexpected internal errors return a safe generic message instead of leaking implementation details or secrets.
- The response includes answer, citations, sufficiency, retrieval mode, trace ID, trace output path, retrieval metadata, usage, and cost when available.

### Failure mode intentionally protected
The API test simulates an internal exception containing a fake secret string. The endpoint returns a generic `Internal query processing error.` response and verifies the secret text is not leaked to the client.

### Validation
- Added `app/api/main.py`.
- Added `app/api/routes/query.py`.
- Added `app/schemas/query_api.py`.
- Added `tests/test_query_api.py`.
- Updated `requirements.txt` with `fastapi` and `httpx`.
- Targeted Step 68 tests passed: `5 passed`.
- Full regression suite passed: `161 passed`.

### Behaviors tested
- `/query` calls `run_grounded_answer_query(...)` with request values and config-derived dependencies.
- API response is formatted with answer, citations, sufficiency, retrieval mode, trace ID, trace path, and retrieval metadata.
- Blank query validation returns `422`.
- Lexical-only mode skips Qdrant collection checks.
- Dense mode returns `503` when the required Qdrant collection is missing.
- Unexpected orchestration failure returns a safe `500` error without leaking sensitive details.
- Qdrant client is closed after each request.

### Important limitation
This is a minimal API contract only. It does not yet include a health endpoint, authentication, streaming responses, request IDs supplied by clients, rate limiting, async execution, or Streamlit UI integration.

### Next decision implied
The next step should add a minimal FastAPI health endpoint so clients and future UI can check whether the backend is alive and what retrieval mode/config is active without running a full query.

## Step 69 — Minimal FastAPI health endpoint

### Description
Added a lightweight FastAPI `/health` endpoint that reports backend liveness and active retrieval configuration without running retrieval, embedding, LLM generation, or Qdrant collection checks.

### Code snippets used

#### Health response schema
```python
class HealthResponse(BaseModel):
    status: str
    app_name: str
    environment: str
    retrieval_mode: str
    hybrid_dense_weight: float
    hybrid_lexical_weight: float
    hybrid_candidate_limit: int
    retrieval_min_top_score: float
    qdrant_collection_name: str
    qdrant_required_for_current_mode: bool
```

#### Health route behavior
```python
@router.get("/health", response_model=HealthResponse)
def health_check() -> HealthResponse:
    settings = get_settings()
    retrieval_config = build_retrieval_runtime_config(settings)
    return HealthResponse(...)
```

#### FastAPI app registers health before query
```python
app.include_router(health_router)
app.include_router(query_router)
```

### Why this step matters
Clients and future UI need a cheap way to confirm the backend is alive and inspect active retrieval configuration without triggering a full RAG query. This separates liveness/config visibility from expensive query execution.

### Production interpretation
- `/health` is lightweight and avoids embedding, LLM, retrieval, and vector-store calls.
- It exposes active retrieval mode and hybrid parameters for debugging.
- It reports whether the current retrieval mode requires Qdrant.
- It returns a safe generic error if retrieval config is invalid.
- This endpoint prepares the backend for future UI and operational checks.

### Failure mode intentionally protected
The health tests simulate invalid retrieval configuration containing a fake internal value. The endpoint returns `Invalid retrieval runtime configuration.` and does not leak the raw invalid mode value to the client.

### Validation
- Added `app/schemas/health_api.py`.
- Added `app/api/routes/health.py`.
- Updated `app/api/main.py` to register the health router.
- Added `tests/test_health_api.py`.
- Targeted Step 69 tests passed: `8 passed` with query API tests.
- Full regression suite passed: `164 passed`.

### Behaviors tested
- `/health` returns status, app name, environment, retrieval mode, hybrid weights, candidate limit, sufficiency threshold, Qdrant collection name, and Qdrant requirement flag.
- Hybrid mode reports `qdrant_required_for_current_mode=True`.
- Lexical mode reports `qdrant_required_for_current_mode=False`.
- Invalid retrieval config returns safe `500` error.

### Important limitation
This is not a full readiness check. It does not confirm Qdrant collection existence, embedding API availability, LLM API availability, or indexed corpus health. It is intentionally a lightweight liveness/config endpoint.

### Next decision implied
The next step should add a small API smoke-test script or documentation command showing how to run the FastAPI backend and call `/health` and `/query` locally before building a Streamlit UI.

## Step 70 — API smoke-test client script

### Description
Added a small HTTP-based API smoke-test client script for a running FastAPI backend. The script always calls `GET /health` and only calls `POST /query` when a query is explicitly provided. This avoids accidental LLM/retrieval cost while still giving a repeatable local API verification path before building a UI.

### Code snippets used

#### Script command shape
```bash
python scripts/run_api_smoke_test.py --base-url http://127.0.0.1:8000
python scripts/run_api_smoke_test.py --base-url http://127.0.0.1:8000 --query "What changed in branch reports?"
```

#### Health-first smoke flow
```python
health_response = client.get(f"{cleaned_base_url}/health")
health_payload = _extract_success_json(health_response, label="GET /health")

query_payload = None
if query is not None:
    query_response = client.post(f"{cleaned_base_url}/query", json=...)
    query_payload = _extract_success_json(query_response, label="POST /query")
```

#### Safe API error extraction
```python
def _extract_success_json(response: httpx.Response, label: str) -> dict[str, Any]:
    if response.status_code >= 400:
        raise RuntimeError(f"{label} failed with HTTP {response.status_code}")
```

### Why this step matters
Before building Streamlit, we need a simple way to verify the backend over the same HTTP boundary that a UI will use. This script validates the API contract externally rather than calling Python services directly.

### Production interpretation
- `/health` can be checked without triggering query cost.
- `/query` is opt-in to avoid accidental LLM and retrieval calls.
- The script logs retrieval mode, trace ID, sufficiency status, answer status, and citations when a query is run.
- HTTP error handling avoids leaking server response bodies or secrets in client-side exceptions.

### Failure mode intentionally protected
The test simulates a `500` response containing a fake secret in the response body. The script raises a generic `GET /health failed with HTTP 500` error and verifies the fake secret is not leaked through the exception text.

### Validation
- Added `scripts/run_api_smoke_test.py`.
- Added `tests/test_api_smoke_script.py`.
- Targeted Step 70 tests passed: `5 passed`.
- Full regression suite passed: `169 passed`.

### Behaviors tested
- CLI help works.
- Health-only smoke test calls only `/health`.
- Query smoke test calls `/health` then `/query`.
- Optional filters with `None` values are omitted from the JSON request body.
- HTTP error responses raise safe client-side errors without leaking response details.

### Important limitation
This script expects the FastAPI backend to already be running. It does not start the server itself. It is a client-side smoke test, not a server process manager.

### Next decision implied
The next step should document the local API run/smoke commands in the README before adding Streamlit, so the backend can be operated and verified independently of the UI.

## Step 71 — Backend-first API verification documentation

### Description
Hardened the README API run and smoke-test documentation so the FastAPI backend can be operated, verified, and debugged independently before adding any Streamlit UI. This step clarifies expected smoke-test outcomes, cost boundaries, failure interpretation, safe error handling, and where to inspect local answer trace artifacts.

### Code snippets used

#### Health-only smoke command
```bash
python scripts/run_api_smoke_test.py --base-url http://127.0.0.1:8000
```

#### Query smoke command
```bash
python scripts/run_api_smoke_test.py --base-url http://127.0.0.1:8000 --query "What changed in branch reports?" --limit 5
```

#### README regression test for operational interpretation
```python
def test_readme_documents_smoke_test_operational_interpretation() -> None:
    readme = Path("README.md").read_text(encoding="utf-8")

    assert "No query supplied. Skipped POST /query." in readme
    assert "does **not** run retrieval, embeddings, LLM generation" in readme
    assert "may trigger retrieval, embedding calls, LLM generation" in readme
    assert "data/exports/answer_runs/" in readme
```

#### README regression test for safe failure interpretation
```python
def test_readme_documents_safe_failure_interpretation() -> None:
    readme = Path("README.md").read_text(encoding="utf-8")

    assert "If `/query` returns `503` in dense or hybrid mode" in readme
    assert "python scripts/run_qdrant_indexing.py" in readme
    assert "avoids printing raw server response bodies" in readme
```

### Why this step matters
Before adding a UI, the backend needs to be independently reproducible. A Streamlit screen should not become the first way to discover whether the API starts, whether `/health` is cheap, whether `/query` may incur cost, or whether failures are safe and debuggable. Documentation is part of the operational contract.

### Production interpretation
- Health-only smoke testing is the cheap first check and avoids retrieval, embeddings, generation, Qdrant collection checks, and trace writing.
- Query smoke testing is opt-in because it can trigger retrieval, embeddings, LLM generation, trace writing, latency, and cost.
- `503` in dense/hybrid mode points to an unavailable Qdrant collection and should lead to indexing verification.
- Insufficient evidence is treated as a safe refusal behavior, not as a backend crash.
- Raw server error bodies are intentionally not printed because they may contain secrets, stack traces, file paths, or internal configuration values.
- Local answer traces under `data/exports/answer_runs/` are the reproducibility artifact for debugging answered queries.

### Failure mode intentionally protected
The README regression tests now assert that documentation explains safe failure behavior: Qdrant-dependent `503` interpretation, insufficient-evidence handling, and why raw server response bodies should not be exposed by the smoke-test client.

### Validation
- Updated `README.md`.
- Updated `tests/test_readme_api_docs.py`.
- Targeted Step 71 tests passed: `3 passed`.
- Full regression suite passed: `172 passed`.

### Behaviors documented and tested
- How to start the FastAPI backend.
- How to run health-only smoke testing without retrieval or LLM cost.
- How to run opt-in query smoke testing.
- How to interpret `503`, insufficient evidence, and safe HTTP client errors.
- Where to inspect local answer trace artifacts.

### Important limitation
This step only improves the backend operation contract and documentation tests. It does not add Streamlit, authentication, readiness checks, request IDs supplied by clients, rate limiting, or streaming responses.

### Next decision implied
The next step should decide whether to add a minimal UI shell or first add a dedicated readiness endpoint. Given the current production-minded sequence, a readiness endpoint is likely more valuable than Streamlit because it separates cheap liveness from dependency readiness before user-facing UI complexity.

## Step 72 — Minimal FastAPI readiness endpoint

### Description
Added a dedicated FastAPI `/ready` endpoint that checks whether the backend dependencies and local artifacts required for the active retrieval mode are available. This keeps `/health` cheap while giving operators and future UI code a separate readiness contract before running `POST /query`.

The readiness endpoint checks:
- valid retrieval runtime configuration
- required model configuration values
- local `.retrieval_ready.json` artifacts for lexical/hybrid retrieval
- Qdrant collection existence for dense/hybrid retrieval

It intentionally does **not** run retrieval, embedding API calls, LLM generation, answer trace writing, or user query execution.

### Code snippets used

#### Readiness response schema
```python
class ReadinessCheck(BaseModel):
    name: str
    required: bool
    is_ready: bool
    detail: str


class ReadinessResponse(BaseModel):
    status: str
    is_ready: bool
    retrieval_mode: str
    qdrant_required_for_current_mode: bool
    lexical_artifacts_required_for_current_mode: bool
    checks: list[ReadinessCheck]
```

#### Readiness route registration
```python
app.include_router(health_router)
app.include_router(readiness_router)
app.include_router(query_router)
```

#### Mode-aware readiness checks
```python
qdrant_required = _requires_qdrant_collection(retrieval_mode)
lexical_required = _requires_lexical_artifacts(retrieval_mode)

if qdrant_required:
    client = create_persistent_qdrant_client(settings.qdrant_local_path)
    collection_exists = client.collection_exists(settings.qdrant_collection_name)
```

#### Structured 503 readiness response
```python
if not response.is_ready:
    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content=response.model_dump(),
    )
```

### Why this step matters
`/health` should answer “is the backend process alive and minimally configured?” `/ready` should answer “can this backend serve real traffic for the active retrieval mode?” Separating these endpoints avoids making liveness slow or noisy while still giving a reliable pre-query check for Qdrant, local artifacts, and model configuration.

### Production interpretation
- Dense and hybrid readiness require Qdrant collection availability because they use vector search.
- Lexical and hybrid readiness require local `.retrieval_ready.json` artifacts because lexical retrieval searches local processed artifacts.
- Lexical-only readiness skips Qdrant collection checks because Qdrant is not required for lexical retrieval.
- Readiness checks model configuration presence but does not call model APIs, so it avoids token/cost side effects.
- A `503` from `/ready` means dependency/artifact readiness failed, not that the corpus lacks answer evidence.
- Missing corpus evidence during a real query should still produce insufficient-evidence/refusal behavior, not readiness failure.

### Failure mode intentionally protected
The readiness tests distinguish dependency readiness from evidence sufficiency. A missing Qdrant collection produces `503 Service Unavailable` for dense/hybrid readiness, while lexical-only mode does not create a Qdrant client. Missing retrieval-ready artifacts produce `503` for lexical/hybrid readiness. Missing model configuration is reported without leaking actual secrets.

### Validation
- Added `app/schemas/readiness_api.py`.
- Added `app/api/routes/readiness.py`.
- Updated `app/api/main.py` to register the readiness router.
- Added `tests/test_readiness_api.py`.
- Updated `README.md`.
- Updated `tests/test_readme_api_docs.py`.
- Targeted Step 72 tests passed: `10 passed`.
- Full regression suite passed: `179 passed`.

### Behaviors tested
- `/ready` returns `200` with `status=ready` when hybrid dependencies exist.
- `/ready` returns structured `503` when a required Qdrant collection is missing.
- Lexical-only readiness skips Qdrant client creation but requires retrieval-ready artifacts.
- Lexical readiness returns `503` when required retrieval-ready artifacts are missing.
- Invalid retrieval configuration returns a safe generic `500` without leaking raw config values.
- Missing model configuration returns structured `503` without leaking secrets.

### Important limitation
This is a readiness check, not a full end-to-end query dry run. It does not validate model API reachability, embedding dimensions, Qdrant point count, retrieval quality, answer sufficiency, citation correctness, or LLM generation behavior. Those remain separate smoke-test and evaluation concerns.

### Next decision implied
The next step should update the API smoke-test client to optionally call `/ready`, or add API documentation/tests that make the health-vs-ready-vs-query operational sequence explicit. We should still defer Streamlit until backend operational checks are stable.

## Step 73 — API smoke-test client optional readiness check

### Description
Updated the API smoke-test client so it can optionally call `GET /ready` after `GET /health` and before any optional `POST /query`. The new `--check-ready` flag makes dependency/artifact readiness verification explicit while keeping `/query` opt-in to avoid accidental retrieval, embedding, LLM, trace, latency, and cost.

### Code snippets used

#### Smoke result now includes readiness payload
```python
@dataclass(frozen=True)
class ApiSmokeResult:
    health_payload: dict[str, Any]
    readiness_payload: dict[str, Any] | None = None
    query_payload: dict[str, Any] | None = None
```

#### Optional readiness CLI flag
```python
parser.add_argument(
    "--check-ready",
    action="store_true",
    help="Optionally call GET /ready after /health and before /query.",
)
```

#### Health → readiness → query sequence
```python
health_response = client.get(f"{cleaned_base_url}/health")
health_payload = _extract_success_json(health_response, label="GET /health")

readiness_payload = None
if check_ready:
    readiness_response = client.get(f"{cleaned_base_url}/ready")
    readiness_payload = _extract_success_json(readiness_response, label="GET /ready")

query_payload = None
if query is not None:
    query_response = client.post(f"{cleaned_base_url}/query", json=...)
```

#### Readiness failure blocks query
```python
try:
    run_api_smoke_test(..., query="...", check_ready=True)
except RuntimeError as exc:
    assert str(exc) == "GET /ready failed with HTTP 503"

assert client.post_calls == []
```

### Why this step matters
The API smoke-test client now mirrors a production-minded operational sequence: first verify liveness/config, then optionally verify dependency/artifact readiness, and only then run an explicit query. This prevents a missing Qdrant collection or missing retrieval-ready artifacts from being discovered only after attempting a cost-bearing RAG query.

### Production interpretation
- Health remains the mandatory cheap first check.
- Readiness is opt-in with `--check-ready` because it may inspect local dependencies and artifacts.
- Query execution remains opt-in with `--query` because it may trigger retrieval, embedding API calls, LLM generation, trace writing, latency, and cost.
- If `/ready` fails, the smoke-test client raises a safe status-only error and does not call `/query`.
- The client still avoids leaking raw server response bodies, including readiness check details that may contain internal paths or config hints.

### Failure mode intentionally protected
The Step 73 tests simulate a `GET /ready` `503` response containing a fake secret-like readiness detail. The smoke-test client raises `GET /ready failed with HTTP 503`, does not leak the response body, and confirms `POST /query` was never called.

### Validation
- Updated `scripts/run_api_smoke_test.py`.
- Updated `tests/test_api_smoke_script.py`.
- Updated `README.md`.
- Updated `tests/test_readme_api_docs.py`.
- Targeted Step 73 tests passed: `11 passed`.
- Full regression suite passed: `181 passed`.

### Behaviors tested
- CLI help includes `--check-ready`.
- Health-only smoke testing still calls only `/health`.
- `--check-ready` calls `/health` then `/ready`.
- `--check-ready --query ...` calls `/health`, `/ready`, then `/query` only when readiness succeeds.
- A failed readiness check blocks query execution.
- HTTP error handling remains safe and does not expose raw readiness response details.

### Important limitation
The smoke-test client still expects the FastAPI backend to already be running. It does not start the server, run ingestion, run indexing, or repair failed readiness dependencies.

### Next decision implied
The next step can either add a small API operational sequence test around the live app contract, or move toward a minimal UI only after confirming the backend health/readiness/query workflow is stable enough for demo use.


## Step 71 — Backend-first API verification documentation

### Description
Hardened the README API run and smoke-test documentation so the FastAPI backend can be operated, verified, and debugged independently before adding any Streamlit UI. This step clarifies expected smoke-test outcomes, cost boundaries, failure interpretation, safe error handling, and where to inspect local answer trace artifacts.

### Code snippets used

#### Health-only smoke command
```bash
python scripts/run_api_smoke_test.py --base-url http://127.0.0.1:8000
```

#### Query smoke command
```bash
python scripts/run_api_smoke_test.py --base-url http://127.0.0.1:8000 --query "What changed in branch reports?" --limit 5
```

#### README regression test for operational interpretation
```python
def test_readme_documents_smoke_test_operational_interpretation() -> None:
    readme = Path("README.md").read_text(encoding="utf-8")

    assert "No query supplied. Skipped POST /query." in readme
    assert "does **not** run retrieval, embeddings, LLM generation" in readme
    assert "may trigger retrieval, embedding calls, LLM generation" in readme
    assert "data/exports/answer_runs/" in readme
```

#### README regression test for safe failure interpretation
```python
def test_readme_documents_safe_failure_interpretation() -> None:
    readme = Path("README.md").read_text(encoding="utf-8")

    assert "If `/query` returns `503` in dense or hybrid mode" in readme
    assert "python scripts/run_qdrant_indexing.py" in readme
    assert "avoids printing raw server response bodies" in readme
```

### Why this step matters
Before adding a UI, the backend needs to be independently reproducible. A Streamlit screen should not become the first way to discover whether the API starts, whether `/health` is cheap, whether `/query` may incur cost, or whether failures are safe and debuggable. Documentation is part of the operational contract.

### Production interpretation
- Health-only smoke testing is the cheap first check and avoids retrieval, embeddings, generation, Qdrant collection checks, and trace writing.
- Query smoke testing is opt-in because it can trigger retrieval, embeddings, LLM generation, trace writing, latency, and cost.
- `503` in dense/hybrid mode points to an unavailable Qdrant collection and should lead to indexing verification.
- Insufficient evidence is treated as a safe refusal behavior, not as a backend crash.
- Raw server error bodies are intentionally not printed because they may contain secrets, stack traces, file paths, or internal configuration values.
- Local answer traces under `data/exports/answer_runs/` are the reproducibility artifact for debugging answered queries.

### Failure mode intentionally protected
The README regression tests now assert that documentation explains safe failure behavior: Qdrant-dependent `503` interpretation, insufficient-evidence handling, and why raw server response bodies should not be exposed by the smoke-test client.

### Validation
- Updated `README.md`.
- Updated `tests/test_readme_api_docs.py`.
- Targeted Step 71 tests passed: `3 passed`.
- Full regression suite passed: `172 passed`.

### Behaviors documented and tested
- How to start the FastAPI backend.
- How to run health-only smoke testing without retrieval or LLM cost.
- How to run opt-in query smoke testing.
- How to interpret `503`, insufficient evidence, and safe HTTP client errors.
- Where to inspect local answer trace artifacts.

### Important limitation
This step only improves the backend operation contract and documentation tests. It does not add Streamlit, authentication, readiness checks, request IDs supplied by clients, rate limiting, or streaming responses.

### Next decision implied
The next step should decide whether to add a minimal UI shell or first add a dedicated readiness endpoint. Given the current production-minded sequence, a readiness endpoint is likely more valuable than Streamlit because it separates cheap liveness from dependency readiness before user-facing UI complexity.

## Step 72 — Minimal FastAPI readiness endpoint

### Description
Added a dedicated FastAPI `/ready` endpoint that checks whether the backend dependencies and local artifacts required for the active retrieval mode are available. This keeps `/health` cheap while giving operators and future UI code a separate readiness contract before running `POST /query`.

The readiness endpoint checks:
- valid retrieval runtime configuration
- required model configuration values
- local `.retrieval_ready.json` artifacts for lexical/hybrid retrieval
- Qdrant collection existence for dense/hybrid retrieval

It intentionally does **not** run retrieval, embedding API calls, LLM generation, answer trace writing, or user query execution.

### Code snippets used

#### Readiness response schema
```python
class ReadinessCheck(BaseModel):
    name: str
    required: bool
    is_ready: bool
    detail: str


class ReadinessResponse(BaseModel):
    status: str
    is_ready: bool
    retrieval_mode: str
    qdrant_required_for_current_mode: bool
    lexical_artifacts_required_for_current_mode: bool
    checks: list[ReadinessCheck]
```

#### Readiness route registration
```python
app.include_router(health_router)
app.include_router(readiness_router)
app.include_router(query_router)
```

#### Mode-aware readiness checks
```python
qdrant_required = _requires_qdrant_collection(retrieval_mode)
lexical_required = _requires_lexical_artifacts(retrieval_mode)

if qdrant_required:
    client = create_persistent_qdrant_client(settings.qdrant_local_path)
    collection_exists = client.collection_exists(settings.qdrant_collection_name)
```

#### Structured 503 readiness response
```python
if not response.is_ready:
    return JSONResponse(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        content=response.model_dump(),
    )
```

### Why this step matters
`/health` should answer “is the backend process alive and minimally configured?” `/ready` should answer “can this backend serve real traffic for the active retrieval mode?” Separating these endpoints avoids making liveness slow or noisy while still giving a reliable pre-query check for Qdrant, local artifacts, and model configuration.

### Production interpretation
- Dense and hybrid readiness require Qdrant collection availability because they use vector search.
- Lexical and hybrid readiness require local `.retrieval_ready.json` artifacts because lexical retrieval searches local processed artifacts.
- Lexical-only readiness skips Qdrant collection checks because Qdrant is not required for lexical retrieval.
- Readiness checks model configuration presence but does not call model APIs, so it avoids token/cost side effects.
- A `503` from `/ready` means dependency/artifact readiness failed, not that the corpus lacks answer evidence.
- Missing corpus evidence during a real query should still produce insufficient-evidence/refusal behavior, not readiness failure.

### Failure mode intentionally protected
The readiness tests distinguish dependency readiness from evidence sufficiency. A missing Qdrant collection produces `503 Service Unavailable` for dense/hybrid readiness, while lexical-only mode does not create a Qdrant client. Missing retrieval-ready artifacts produce `503` for lexical/hybrid readiness. Missing model configuration is reported without leaking actual secrets.

### Validation
- Added `app/schemas/readiness_api.py`.
- Added `app/api/routes/readiness.py`.
- Updated `app/api/main.py` to register the readiness router.
- Added `tests/test_readiness_api.py`.
- Updated `README.md`.
- Updated `tests/test_readme_api_docs.py`.
- Targeted Step 72 tests passed: `10 passed`.
- Full regression suite passed: `179 passed`.

### Behaviors tested
- `/ready` returns `200` with `status=ready` when hybrid dependencies exist.
- `/ready` returns structured `503` when a required Qdrant collection is missing.
- Lexical-only readiness skips Qdrant client creation but requires retrieval-ready artifacts.
- Lexical readiness returns `503` when required retrieval-ready artifacts are missing.
- Invalid retrieval configuration returns a safe generic `500` without leaking raw config values.
- Missing model configuration returns structured `503` without leaking secrets.

### Important limitation
This is a readiness check, not a full end-to-end query dry run. It does not validate model API reachability, embedding dimensions, Qdrant point count, retrieval quality, answer sufficiency, citation correctness, or LLM generation behavior. Those remain separate smoke-test and evaluation concerns.

### Next decision implied
The next step should update the API smoke-test client to optionally call `/ready`, or add API documentation/tests that make the health-vs-ready-vs-query operational sequence explicit. We should still defer Streamlit until backend operational checks are stable.

## Step 73 — API smoke-test client optional readiness check

### Description
Updated the API smoke-test client so it can optionally call `GET /ready` after `GET /health` and before any optional `POST /query`. The new `--check-ready` flag makes dependency/artifact readiness verification explicit while keeping `/query` opt-in to avoid accidental retrieval, embedding, LLM, trace, latency, and cost.

### Code snippets used

#### Smoke result now includes readiness payload
```python
@dataclass(frozen=True)
class ApiSmokeResult:
    health_payload: dict[str, Any]
    readiness_payload: dict[str, Any] | None = None
    query_payload: dict[str, Any] | None = None
```

#### Optional readiness CLI flag
```python
parser.add_argument(
    "--check-ready",
    action="store_true",
    help="Optionally call GET /ready after /health and before /query.",
)
```

#### Health → readiness → query sequence
```python
health_response = client.get(f"{cleaned_base_url}/health")
health_payload = _extract_success_json(health_response, label="GET /health")

readiness_payload = None
if check_ready:
    readiness_response = client.get(f"{cleaned_base_url}/ready")
    readiness_payload = _extract_success_json(readiness_response, label="GET /ready")

query_payload = None
if query is not None:
    query_response = client.post(f"{cleaned_base_url}/query", json=...)
```

#### Readiness failure blocks query
```python
try:
    run_api_smoke_test(..., query="...", check_ready=True)
except RuntimeError as exc:
    assert str(exc) == "GET /ready failed with HTTP 503"

assert client.post_calls == []
```

### Why this step matters
The API smoke-test client now mirrors a production-minded operational sequence: first verify liveness/config, then optionally verify dependency/artifact readiness, and only then run an explicit query. This prevents a missing Qdrant collection or missing retrieval-ready artifacts from being discovered only after attempting a cost-bearing RAG query.

### Production interpretation
- Health remains the mandatory cheap first check.
- Readiness is opt-in with `--check-ready` because it may inspect local dependencies and artifacts.
- Query execution remains opt-in with `--query` because it may trigger retrieval, embedding API calls, LLM generation, trace writing, latency, and cost.
- If `/ready` fails, the smoke-test client raises a safe status-only error and does not call `/query`.
- The client still avoids leaking raw server response bodies, including readiness check details that may contain internal paths or config hints.

### Failure mode intentionally protected
The Step 73 tests simulate a `GET /ready` `503` response containing a fake secret-like readiness detail. The smoke-test client raises `GET /ready failed with HTTP 503`, does not leak the response body, and confirms `POST /query` was never called.

### Validation
- Updated `scripts/run_api_smoke_test.py`.
- Updated `tests/test_api_smoke_script.py`.
- Updated `README.md`.
- Updated `tests/test_readme_api_docs.py`.
- Targeted Step 73 tests passed: `11 passed`.
- Full regression suite passed: `181 passed`.

### Behaviors tested
- CLI help includes `--check-ready`.
- Health-only smoke testing still calls only `/health`.
- `--check-ready` calls `/health` then `/ready`.
- `--check-ready --query ...` calls `/health`, `/ready`, then `/query` only when readiness succeeds.
- A failed readiness check blocks query execution.
- HTTP error handling remains safe and does not expose raw readiness response details.

### Important limitation
The smoke-test client still expects the FastAPI backend to already be running. It does not start the server, run ingestion, run indexing, or repair failed readiness dependencies.

### Next decision implied
The next step can either add a small API operational sequence test around the live app contract, or move toward a minimal UI only after confirming the backend health/readiness/query workflow is stable enough for demo use.

## Step 74 ? Lexical /query Qdrant dependency boundary

### Description
Hardened the FastAPI query path so lexical-only retrieval no longer creates or requires a Qdrant client. Dense and hybrid modes still require Qdrant and still fail fast when the required collection is missing.

### Code snippets used

#### Query route now creates Qdrant only when the active mode requires it
`python
retrieval_config = build_retrieval_runtime_config(settings)
qdrant_required = _requires_qdrant_collection(retrieval_config.retrieval_mode)

client = None
try:
    if qdrant_required:
        client = create_persistent_qdrant_client(settings.qdrant_local_path)
        if not client.collection_exists(settings.qdrant_collection_name):
            raise HTTPException(status_code=503, detail= Qdrant collection does not exist. Run indexing before querying.)

    orchestration_result = run_grounded_answer_query(qdrant_client=client, ...)
finally:
    if client is not None:
        client.close()
`

#### Shared orchestration now permits no Qdrant client for lexical-only mode
`python
def run_grounded_answer_query(
    qdrant_client: QdrantClient | None,
    collection_name: str,
    ...,
) -> AnswerOrchestrationResult:
    ...
`

#### Retrieval service guards dense/hybrid modes
`python
if retrieval_config.retrieval_mode in {dense, hybrid} and qdrant_client is None:
    raise ValueError(qdrant_client is required for dense or hybrid retrieval)
`

#### Lexical regression test intentionally breaks unwanted dependencies
`python
def fail_if_qdrant_created(path):
    raise AssertionError(Lexical query should not create a Qdrant client)

class FailingEmbeddingsAPI:
    def create(self, model: str, input: list[str]):
        raise AssertionError(Lexical retrieval should not call embedding APIs)
`

### Why this step matters
Readiness already treated lexical retrieval as independent from Qdrant. The query path needed to match that operational contract. If lexical mode can answer from local retrieval-ready artifacts, a missing or corrupted Qdrant store should not make lexical /query unavailable.

### Production interpretation
- Lexical mode now has a smaller dependency surface: local retrieval-ready artifacts are required, but Qdrant is not.
- Dense and hybrid modes remain protected because they still require Qdrant and fail before answer orchestration if the collection is missing.
- This improves local/offline demo resilience and avoids unnecessary vector-store startup or file locking in lexical-only runs.
- It also avoids accidental embedding calls in lexical-only retrieval tests, keeping cost-bearing behavior mode-specific.

### Failure mode intentionally protected
The tests deliberately make Qdrant creation fail in lexical /query and make embedding creation fail in lexical retrieval. Both paths still pass, proving lexical mode does not accidentally touch vector-store or embedding dependencies. A separate dense-mode test passes qdrant_client=None and confirms the service raises a clear error instead of failing later with an ambiguous attribute error.

### Validation
- Updated pp/api/routes/query.py.
- Updated pp/services/answer_orchestration.py.
- Updated pp/services/query_retrieval.py.
- Updated 	ests/test_query_api.py.
- Updated 	ests/test_query_retrieval_service.py.
- Targeted Step 74 tests passed: 20 passed.
- Full regression suite passed: 183 passed.

### Behaviors tested
- Lexical /query does not create a Qdrant client.
- Lexical query orchestration receives qdrant_client=None.
- Lexical retrieval can run without Qdrant and without embedding API calls.
- Dense retrieval rejects a missing Qdrant client with a clear ValueError.
- Existing dense/hybrid query and readiness behavior remains green.

### Important limitation
This step does not add a full lexical API integration test with real artifact files through POST /query; it hardens the route/service dependency boundary with focused unit-style tests. Retrieval quality, citation quality, and corpus evidence sufficiency remain separate evaluation concerns.

### Next decision implied
The next step can either add an end-to-end lexical-mode API smoke/evaluation path with real local artifacts, or start a minimal UI only after preserving the health/readiness/query dependency contracts.

## Step 74 - Lexical /query Qdrant dependency boundary

### Goal
Make lexical-only `/query` execution independent of Qdrant while preserving fail-fast Qdrant requirements for dense and hybrid retrieval modes.

### Files touched
- `tests/test_query_api.py`
- `tests/test_query_retrieval_service.py`

### What changed
- Confirmed `app/api/routes/query.py` already creates a Qdrant client only when retrieval mode is dense or hybrid.
- Confirmed lexical mode passes `qdrant_client=None` into orchestration and relies on local retrieval-ready artifacts.
- Hardened API tests so missing Qdrant collection returns HTTP 503 for both dense and hybrid modes.
- Hardened service tests so `retrieve_query_evidence` rejects `qdrant_client=None` for both dense and hybrid modes.
- No production code change was required because the existing route and service already respected this dependency boundary.

### Code pattern used
```python
@pytest.mark.parametrize(...)
def test_required_qdrant_boundary(...):
    # dense and hybrid require Qdrant
    # lexical does not create or require Qdrant
    ...
```

### Validation
- Targeted: `python -m pytest tests/test_query_api.py tests/test_query_retrieval_service.py -q` -> 14 passed.
- Full suite: `python -m pytest -q` -> 185 passed.

### Production interpretation
Lexical mode can remain a cheap local fallback when Qdrant is unavailable, while dense and hybrid modes fail early with a service-unavailable signal instead of entering answer generation with missing vector-store state. This reduces unnecessary embedding or LLM spend and makes operational failures easier to diagnose.

### Failure-mode thinking
Stress case: run `/query` in lexical mode while Qdrant is absent or broken. Expected behavior is no Qdrant client creation, no collection check, and no embedding call. Dense and hybrid must still fail before orchestration if the required collection is missing.

## Step 75 - Retrieval-mode dependency matrix documentation

### Goal
Document the operational dependency boundaries for `lexical`, `dense`, and `hybrid` retrieval modes so API users understand what `/ready` checks, what `/query` may call, and where failures should occur.

### Files touched
- `README.md`
- `tests/test_readme_api_docs.py`

### What changed
- Added a retrieval-mode dependency matrix under README API operational notes.
- Documented that lexical mode uses local `.retrieval_ready.json` artifacts only for retrieval and should not instantiate Qdrant or call embedding APIs for retrieval.
- Documented that dense mode requires Qdrant collection existence and an embedding call for vector search.
- Documented that hybrid mode requires both Qdrant dense search and local lexical artifacts.
- Documented failure boundaries: readiness `503`, query `503` for missing Qdrant in dense/hybrid, and safe refusal for insufficient evidence.
- Added README regression assertions so the dependency matrix stays visible.

### Python/test code used
```python
def test_readme_documents_retrieval_mode_dependency_matrix() -> None:
    readme = Path("README.md").read_text(encoding="utf-8")
    assert "### Retrieval-mode dependency matrix" in readme
    assert "| `lexical` |" in readme
    assert "local lexical artifacts only; no Qdrant client or collection check" in readme
    assert "must fail fast when vector-store state is unavailable" in readme
```

### Validation
- Targeted README docs: `python -m pytest tests/test_readme_api_docs.py -q` -> 5 passed.
- Broader API/docs: `python -m pytest tests/test_health_api.py tests/test_readiness_api.py tests/test_query_api.py tests/test_api_smoke_script.py tests/test_readme_api_docs.py -q` -> 27 passed.
- Full suite: `python -m pytest -q` -> 186 passed.

### Production interpretation
This matrix turns implicit retrieval dependencies into an explicit API operations contract. It reduces accidental coupling, helps operators triage `/ready` versus `/query` failures, and prevents engineers from accidentally making lexical degraded mode depend on vector-store infrastructure or embedding spend.

### Failure-mode thinking
Stress case: Qdrant is down but lexical artifacts exist. `/ready` in lexical mode should still pass, and lexical `/query` should not create Qdrant or call embeddings. Dense/hybrid should fail fast with readiness/query `503` instead of wasting model calls or returning misleading answers.

## Step 76 - Readiness retrieval-mode dependency boundary tests

### Goal
Verify and lock down that `GET /ready` follows the retrieval-mode dependency matrix: lexical readiness must not touch Qdrant, dense readiness must not require lexical artifacts, and hybrid readiness must require both Qdrant and lexical artifacts.

### Files touched
- `tests/test_readiness_api.py`

### What changed
- Added `pytest` parametrization for dense and hybrid missing-Qdrant readiness failures.
- Added a dense-mode readiness regression test proving dense mode can be ready without `.retrieval_ready.json` artifacts when Qdrant collection exists.
- Added parametrization proving lexical and hybrid readiness both fail when required retrieval-ready artifacts are missing.
- Confirmed lexical readiness does not create a Qdrant client.
- Confirmed hybrid readiness checks Qdrant even when lexical artifacts are missing, so operators see both dependency states in the readiness payload.
- No production code change was required because `app/api/routes/readiness.py` already followed the dependency matrix.

### Python/test code used
```python
@pytest.mark.parametrize("retrieval_mode", ["dense", "hybrid"])
def test_ready_endpoint_returns_503_when_required_qdrant_collection_is_missing(...):
    ...

@pytest.mark.parametrize("retrieval_mode", ["lexical", "hybrid"])
def test_ready_endpoint_returns_503_when_required_retrieval_ready_artifacts_are_missing(...):
    ...
```

### Validation
- Targeted readiness tests: `python -m pytest tests/test_readiness_api.py -q` -> 9 passed.
- Broader API/docs tests: `python -m pytest tests/test_health_api.py tests/test_readiness_api.py tests/test_query_api.py tests/test_api_smoke_script.py tests/test_readme_api_docs.py -q` -> 30 passed.
- Full suite: `python -m pytest -q` -> 189 passed.

### Production interpretation
The readiness endpoint now has stronger regression coverage for degraded-mode behavior. Operators can trust that lexical readiness is independent of Qdrant, dense readiness is independent of lexical artifacts, and hybrid readiness fails if either side of its fused retrieval path is unavailable.

### Failure-mode thinking
Stress cases covered: Qdrant missing in dense/hybrid, lexical artifacts missing in lexical/hybrid, dense mode with no lexical artifacts, and lexical mode with a Qdrant constructor that would fail if called.


<!-- Active history archived through Step 93 on 2026-07-29 -->

## Step 77 - Query Qdrant dependency failure returns safe 503

### Goal
Harden `POST /query` so dense and hybrid modes classify Qdrant dependency failures as safe `503 Service Unavailable` responses instead of generic `500` errors, while preserving lexical independence from Qdrant.

### Files touched
- `app/api/routes/query.py`
- `tests/test_query_api.py`

### What changed
- Wrapped dense/hybrid Qdrant client creation and collection-existence check in a dependency-specific exception boundary.
- Preserved the existing explicit `503` when the required Qdrant collection does not exist.
- Added safe `503` handling when Qdrant client creation fails.
- Added safe `503` handling when the Qdrant collection check raises unexpectedly.
- Ensured raw exception details such as local paths, secrets, or backend internals do not leak to API clients.
- Ensured query orchestration does not run when the required Qdrant dependency check fails.
- Ensured opened Qdrant clients are still closed in failure paths.

### Python/code pattern used
```python
try:
    client = create_persistent_qdrant_client(settings.qdrant_local_path)
    if not client.collection_exists(settings.qdrant_collection_name):
        raise HTTPException(status_code=503, detail="Qdrant collection does not exist...")
except HTTPException:
    raise
except Exception as exc:
    logger.exception("Qdrant dependency check failed during query")
    raise HTTPException(
        status_code=503,
        detail="Qdrant dependency check failed. Verify vector-store availability before querying.",
    ) from exc
```

### Validation
- Targeted query API tests: `python -m pytest tests/test_query_api.py -q` -> 10 passed.
- Broader API/docs tests: `python -m pytest tests/test_health_api.py tests/test_readiness_api.py tests/test_query_api.py tests/test_api_smoke_script.py tests/test_readme_api_docs.py -q` -> 34 passed.
- Full suite: `python -m pytest -q` -> 193 passed.

### Production interpretation
Dense and hybrid `/query` now fail fast and safely when vector-store availability is broken. This prevents expensive retrieval/embedding/LLM work from starting after a known dependency failure and gives operators a clear service-unavailable signal without leaking implementation details.

### Failure-mode thinking
Stress cases covered: Qdrant client creation raises, Qdrant collection check raises, required collection is missing, orchestration must not run on dependency failure, and error responses must not expose raw exception content.

## Step 78 - Answer smoke script lexical Qdrant independence

### Goal
Harden `scripts/run_answer_smoke_test.py` so lexical-only smoke tests do not instantiate Qdrant, matching the API/readiness retrieval-mode dependency matrix.

### Files touched
- `scripts/run_answer_smoke_test.py`
- `tests/test_answer_smoke_script.py`

### What changed
- Moved Qdrant client creation inside the dense/hybrid-only dependency branch.
- Preserved the existing collection-existence check for dense and hybrid modes.
- Passed `qdrant_client=None` into answer orchestration for lexical mode.
- Guarded Qdrant client cleanup so `close()` is called only when a client was actually created.
- Added a lexical-mode regression test that makes Qdrant client creation raise if touched.
- Confirmed hybrid smoke-script behavior still checks Qdrant and closes the client.

### Python/code pattern used
```python
client = None
try:
    if _requires_qdrant_collection(retrieval_config.retrieval_mode):
        client = create_persistent_qdrant_client(settings.qdrant_local_path)
        if not client.collection_exists(settings.qdrant_collection_name):
            raise RuntimeError("Qdrant collection does not exist. Run scripts/run_qdrant_indexing.py first.")

    orchestration_result = run_grounded_answer_query(
        qdrant_client=client,
        ...
    )
finally:
    if client is not None:
        client.close()
```

### Validation
- Targeted answer smoke script tests: `python -m pytest tests/test_answer_smoke_script.py -q` -> 4 passed.
- Broader dependency-boundary tests: `python -m pytest tests/test_answer_smoke_script.py tests/test_answer_orchestration_service.py tests/test_query_retrieval_service.py tests/test_query_api.py tests/test_readiness_api.py -q` -> 33 passed.
- Full suite: `python -m pytest -q` -> 194 passed.

### Production interpretation
The CLI smoke path now matches the service contract used by `/ready` and `/query`: lexical retrieval depends on local retrieval-ready artifacts, not Qdrant. Dense and hybrid modes still fail fast if their required Qdrant collection is missing. This keeps local degraded-mode testing reliable when the vector store is absent, locked, corrupted, or intentionally not started.

### Failure-mode thinking
The new test intentionally makes Qdrant construction fail in lexical mode. The script still succeeds because it never touches Qdrant and passes `None` to orchestration. If a future change accidentally reintroduces Qdrant coupling, this regression test will fail immediately.

## Step 79 - Query-search script lexical Qdrant independence

### Goal
Harden `scripts/run_qdrant_query_search.py` so lexical-only query-search smoke tests do not instantiate Qdrant, matching the API/readiness dependency matrix and the Step 78 answer-smoke behavior.

### Files touched
- `scripts/run_qdrant_query_search.py`
- `tests/test_qdrant_query_search_script.py`

### What changed
- Moved Qdrant client creation inside the dense/hybrid-only dependency branch.
- Preserved the existing Qdrant collection-existence check for dense and hybrid modes.
- Passed `qdrant_client=None` into `retrieve_query_evidence(...)` for lexical mode.
- Guarded Qdrant client cleanup so `close()` is called only when a client was actually created.
- Added a hybrid-mode regression test proving the script checks the Qdrant collection and closes the client.
- Added a lexical-mode regression test that makes Qdrant client creation raise if touched.

### Python/code pattern used
```python
client = None
try:
    if _requires_qdrant_collection(retrieval_config.retrieval_mode):
        client = create_persistent_qdrant_client(settings.qdrant_local_path)
        if not client.collection_exists(settings.qdrant_collection_name):
            raise RuntimeError("Qdrant collection does not exist. Run scripts/run_qdrant_indexing.py first.")

    routed = retrieve_query_evidence(
        qdrant_client=client,
        ...
    )
finally:
    if client is not None:
        client.close()
```

### Validation
- Targeted query-search script tests: `python -m pytest tests/test_qdrant_query_search_script.py -q` -> 4 passed.
- Broader dependency-boundary tests: `python -m pytest tests/test_qdrant_query_search_script.py tests/test_query_retrieval_service.py tests/test_answer_smoke_script.py tests/test_query_api.py tests/test_readiness_api.py -q` -> 35 passed.
- Full suite: `python -m pytest -q` -> 196 passed.

### Production interpretation
The query-search CLI now has the same retrieval-mode dependency behavior as the API and answer-smoke script. Lexical mode can search local retrieval-ready artifacts without Qdrant. Dense and hybrid modes still fail before retrieval if their required vector-store collection is missing, avoiding confusing retrieval failures and wasted embedding work.

### Failure-mode thinking
The lexical regression test intentionally makes Qdrant client creation fail. The script still succeeds because lexical mode never touches Qdrant. This protects local/offline query-search debugging when Qdrant is missing, locked, corrupted, or intentionally unavailable.

## Step 80 - Retrieval comparison script Qdrant client cleanup

### Goal
Harden `scripts/run_retrieval_comparison.py` so its persistent Qdrant client is closed on both successful runs and failure paths.

### Files touched
- `scripts/run_retrieval_comparison.py`
- `tests/test_retrieval_comparison_script.py`

### What changed
- Wrapped the Qdrant-dependent retrieval comparison flow in `try/finally`.
- Removed the special-case manual `client.close()` before the missing-collection error because `finally` now handles it.
- Preserved the existing fail-fast error when the Qdrant collection is missing.
- Preserved the dense-vs-lexical comparison behavior and report-writing behavior.
- Added a successful-path test proving the script checks collection existence, runs dense and lexical retrieval, writes a report, and closes the client.
- Added a failure-path test proving the client is still closed when dense retrieval raises.

### Python/code pattern used
```python
client = create_persistent_qdrant_client(settings.qdrant_local_path)
try:
    if not client.collection_exists(settings.qdrant_collection_name):
        raise RuntimeError("Qdrant collection does not exist. Run scripts/run_qdrant_indexing.py first.")

    # Run dense retrieval, lexical retrieval, comparison, and report writing.
finally:
    client.close()
```

### Validation
- Targeted retrieval comparison script tests: `python -m pytest tests/test_retrieval_comparison_script.py -q` -> 3 passed.
- Broader retrieval-script tests: `python -m pytest tests/test_retrieval_comparison_script.py tests/test_retrieval_comparison.py tests/test_retrieval_error_analysis_script.py tests/test_hybrid_retrieval_eval_script.py tests/test_hybrid_weight_experiments_script.py -q` -> 10 passed.
- Full suite: `python -m pytest -q` -> 198 passed.

### Production interpretation
The retrieval comparison script is intentionally Qdrant-dependent because it compares dense vector retrieval with lexical artifact retrieval. The important boundary in this step is lifecycle safety: if dense retrieval, lexical retrieval, comparison building, or report writing fails, the local persistent Qdrant client is still closed. This reduces file-lock and stale-handle risk during repeated local eval runs.

### Failure-mode thinking
The new failure-path test makes dense retrieval raise after the Qdrant collection check. The exception still propagates, but the fake client records `closed=True`, proving cleanup happens even when the script fails mid-run.

## Step 81 - Hybrid retrieval evaluation Qdrant client cleanup

### Goal
Harden `scripts/run_hybrid_retrieval_eval.py` so its persistent Qdrant client is closed after successful evaluation and after failures anywhere in the Qdrant-dependent evaluation flow.

### Files touched
- `scripts/run_hybrid_retrieval_eval.py`
- `tests/test_hybrid_retrieval_eval_script.py`

### What changed
- Wrapped the collection check, dense retrieval, lexical retrieval, fusion, evaluation, and report-writing flow in `try/finally`.
- Removed the special-case manual close before the missing-collection error because `finally` now owns client cleanup.
- Preserved fail-fast behavior when the required Qdrant collection is missing.
- Preserved propagation of retrieval and evaluation errors after cleanup.
- Added a success-path regression test proving collection checking, evaluation, report construction, and client cleanup.
- Added a failure-path regression test proving the client closes when dense retrieval raises.

### Python/code pattern used
```python
client = create_persistent_qdrant_client(settings.qdrant_local_path)
try:
    if not client.collection_exists(settings.qdrant_collection_name):
        raise RuntimeError(
            "Qdrant collection does not exist. Run scripts/run_qdrant_indexing.py first."
        )

    # Run dense and lexical retrieval, hybrid fusion, evaluation, and report writing.
finally:
    client.close()
```

### Validation
- Targeted hybrid evaluation script tests: `python -m pytest tests/test_hybrid_retrieval_eval_script.py -q` -> 3 passed.
- Broader retrieval/evaluation tests: `python -m pytest tests/test_hybrid_retrieval_eval_script.py tests/test_hybrid_evaluation.py tests/test_retrieval_comparison_script.py tests/test_retrieval_comparison.py tests/test_retrieval_error_analysis_script.py tests/test_hybrid_weight_experiments_script.py -q` -> 15 passed.
- Full suite: `python -m pytest -q` -> 200 passed.
- Diff whitespace validation: `git diff --check` -> passed; only the existing Git line-ending conversion warnings were reported.

### Production interpretation
Hybrid evaluation legitimately depends on Qdrant because its dense branch performs vector search. The new lifecycle boundary prevents failed evaluation runs from leaving persistent local Qdrant handles or locks behind, while keeping the original failure visible for diagnosis.

### Failure-mode thinking
The failure-path test intentionally raises during dense retrieval after the client and collection check succeed. The original exception still propagates, while `closed=True` proves that later local indexing, querying, or evaluation runs are not left with the leaked client from this run.

## Step 82 - Hybrid weight experiment Qdrant client cleanup

### Goal
Harden `scripts/run_hybrid_weight_experiments.py` so its persistent Qdrant client is closed after successful experiments and after failures during candidate retrieval, weight evaluation, or report persistence.

### Files touched
- `scripts/run_hybrid_weight_experiments.py`
- `tests/test_hybrid_weight_experiments_script.py`

### What changed
- Wrapped the Qdrant collection check and the complete hybrid weight experiment flow in `try/finally`.
- Removed the branch-specific manual close before the missing-collection error because `finally` now owns cleanup.
- Kept dense and lexical candidate retrieval inside the lifecycle boundary.
- Kept weight-setting evaluation, ranking, logging, and report writing inside the lifecycle boundary.
- Preserved the original exception after cleanup instead of converting a failed experiment into apparent success.
- Added success- and failure-path regression tests for client cleanup.

### Python/code pattern used
```python
client = create_persistent_qdrant_client(settings.qdrant_local_path)
try:
    if not client.collection_exists(settings.qdrant_collection_name):
        raise RuntimeError(
            "Qdrant collection does not exist. Run scripts/run_qdrant_indexing.py first."
        )

    # Retrieve candidates, evaluate weight settings, rank results, and write the report.
finally:
    client.close()
```

### Validation
- Targeted hybrid weight script tests: `python -m pytest tests/test_hybrid_weight_experiments_script.py -q` -> 3 passed.
- Broader hybrid/retrieval tests: `python -m pytest tests/test_hybrid_weight_experiments_script.py tests/test_hybrid_weight_experiment.py tests/test_hybrid_retrieval_eval_script.py tests/test_hybrid_evaluation.py tests/test_retrieval_comparison_script.py tests/test_retrieval_comparison.py -q` -> 21 passed.
- Full suite: `python -m pytest -q` -> 202 passed.
- Diff whitespace validation: `git diff --check` -> passed; only Git line-ending conversion warnings were reported.

### Production interpretation
Weight experiments may perform many dense searches and evaluate several fusion settings in one run. A mid-run exception should invalidate the experiment but must not leak the persistent local Qdrant client. Centralized cleanup makes repeated tuning runs safer without hiding retrieval, evaluation, or report-writing failures.

### Failure-mode thinking
The failure-path test raises during dense candidate retrieval after the Qdrant collection check. The test verifies two independent guarantees: the original retrieval exception reaches the caller, and the Qdrant client is still closed so the failed experiment is less likely to leave locks or handles that break the next run.

## Step 83 - Dense retrieval evaluation Qdrant client cleanup

### Goal
Harden `scripts/run_retrieval_eval.py` so its persistent Qdrant client closes after both successful dense evaluation and failures during retrieval, evaluation, result serialization, or report writing.

### Files touched
- `scripts/run_retrieval_eval.py`
- `tests/test_retrieval_eval_script.py`

### What changed
- Wrapped the complete post-client-creation evaluation flow in `try/finally`.
- Kept per-case dense retrieval, expectation evaluation, result serialization, logging, aggregate report building, and report writing inside the protected lifecycle.
- Preserved propagation of the original exception after cleanup.
- Added a dedicated script-test module.
- Added a help-path test plus success- and failure-path client cleanup tests.
- Kept this step focused on lifecycle safety; explicit collection-existence validation remains a separate dependency-boundary concern.

### Python/code pattern used
```python
client = create_persistent_qdrant_client(settings.qdrant_local_path)
try:
    # Retrieve and evaluate every case, build the aggregate report, and write JSON.
finally:
    client.close()
```

### Validation
- Targeted retrieval evaluation script tests: `python -m pytest tests/test_retrieval_eval_script.py -q` -> 3 passed.
- Related retrieval-script tests: `python -m pytest tests/test_retrieval_eval_script.py tests/test_retrieval_evaluation.py tests/test_retrieval_comparison_script.py tests/test_hybrid_retrieval_eval_script.py tests/test_hybrid_weight_experiments_script.py -q` -> 19 passed.
- Full suite: `python -m pytest -q` -> 205 passed.
- Tracked diff whitespace validation: `git diff --check` -> passed; only Git line-ending conversion warnings were reported.

### Production interpretation
The baseline evaluator may execute many embedding-backed vector searches in one run. If any case fails, the report should fail visibly, but the persistent local Qdrant client must still release its handles. This makes repeated evaluation runs safer and prevents cleanup from masking the diagnostic error.

### Failure-mode thinking
The failure-path test intentionally raises during dense search. It verifies that the same retrieval exception reaches the caller and that the client is closed. Without the `finally` boundary, a single bad case could terminate the run while leaving local vector-store resources open for the next indexing or evaluation command.

## Step 84 - Consolidated persistent Qdrant client lifecycle sweep

### Goal
Complete the remaining `try/finally` cleanup work across all production scripts that create persistent Qdrant clients, as one consolidated change rather than continuing file by file.

### Files touched
- `scripts/run_qdrant_indexing.py`
- `scripts/check_qdrant_index.py`
- `tests/test_qdrant_script_client_cleanup.py`

### Inventory result
- The two remaining unprotected production scripts were `run_qdrant_indexing.py` and `check_qdrant_index.py`.
- API routes `query.py` and `readiness.py` already had guarded `finally` cleanup.
- Answer smoke, query search, retrieval comparison, dense evaluation, hybrid evaluation, and hybrid weight experiment scripts already had guarded `finally` cleanup after Steps 78-83.
- Direct client use in unit tests is test-scoped and was not treated as a production call-site gap.

### What changed
- Wrapped Qdrant indexing work and summary logging in `try/finally`.
- Wrapped collection inspection, count, scroll, and sample logging in `try/finally`.
- Removed the check script's branch-specific manual close before its early return.
- Preserved successful indexing and inspection behavior.
- Preserved early return when the collection is absent.
- Preserved original indexing and inspection exceptions after cleanup.
- Added five consolidated regression tests covering indexing success/failure, missing-collection early return, inspection success, and inspection failure.

### Python/code pattern used
```python
client = create_persistent_qdrant_client(settings.qdrant_local_path)
try:
    # Perform indexing or collection inspection.
finally:
    client.close()
```

### Validation
- Consolidated targeted tests: `python -m pytest tests/test_qdrant_script_client_cleanup.py -q` -> 5 passed.
- Broader Qdrant lifecycle and API tests -> 49 passed.
- Full suite: `python -m pytest -q` -> 210 passed.
- Production call-site audit confirmed centralized cleanup in both Qdrant-using API routes and all eight persistent-client scripts.
- Tracked diff whitespace validation passed; only Git line-ending conversion warnings were reported.

### Production interpretation
Indexing and inspection are operational scripts that are likely to be rerun immediately after a failure. Leaked local Qdrant handles can create file locks or stale state that makes recovery harder. The consolidated sweep establishes one consistent invariant: after a persistent client is created, every normal return and protected exception path closes it.

### Failure-mode thinking
The tests intentionally fail both an indexing upsert operation and a collection metadata inspection. They also exercise the missing-collection early return. In every case, cleanup runs while the original control flow remains intact: errors propagate, and the expected early return stays non-error behavior.

## Step 85 - Dense retrieval evaluation fail-fast collection check

### Goal
Make `scripts/run_retrieval_eval.py` verify that its required Qdrant collection exists before starting embedding-backed dense searches.

### Files touched
- `scripts/run_retrieval_eval.py`
- `tests/test_retrieval_eval_script.py`

### What changed
- Added a Qdrant collection-existence check immediately inside the existing `try/finally` lifecycle boundary.
- Added an actionable missing-collection error directing operators to run the indexing script.
- Ensured no dense search begins when the collection is missing.
- Preserved client cleanup on missing-collection, successful evaluation, and search-failure paths.
- Extended existing success and search-failure tests to assert the collection check.
- Added a missing-collection regression test that fails if search is touched.

### Python/code pattern used
```python
client = create_persistent_qdrant_client(settings.qdrant_local_path)
try:
    if not client.collection_exists(settings.qdrant_collection_name):
        raise RuntimeError(
            "Qdrant collection does not exist. Run scripts/run_qdrant_indexing.py first."
        )

    # Begin embedding-backed dense evaluation only after the dependency passes.
finally:
    client.close()
```

### Validation
- Targeted retrieval evaluation script tests: `python -m pytest tests/test_retrieval_eval_script.py -q` -> 4 passed.
- Broader retrieval/Qdrant tests -> 25 passed.
- Full suite: `python -m pytest -q` -> 211 passed.
- Diff whitespace validation passed; only Git line-ending conversion warnings were reported.

### Production interpretation
The dense evaluator has a hard dependency on an indexed Qdrant collection. Failing before search gives operators a clear recovery action and avoids starting embedding calls or producing backend-specific search errors after the prerequisite is already known to be absent.

### Failure-mode thinking
The missing-collection test uses a search fake that raises if called. The expected result is the actionable collection error, zero search calls, and a closed client. This proves both fail-fast behavior and lifecycle cleanup rather than merely checking the final exception text.

## Step 86 - Evaluation script missing-collection regression coverage

### Goal
Lock down the existing fail-fast Qdrant collection contract in the remaining dense/hybrid evaluation scripts without changing already-correct production code.

### Files touched
- `tests/test_retrieval_comparison_script.py`
- `tests/test_hybrid_retrieval_eval_script.py`
- `tests/test_hybrid_weight_experiments_script.py`

### What changed
- Added a missing-collection regression test for dense-vs-lexical retrieval comparison.
- Added a missing-collection regression test for hybrid retrieval evaluation.
- Added a missing-collection regression test for hybrid weight experiments.
- Configured fake clients to report the required collection as absent.
- Used raising dense-search fakes to prove no embedding-backed retrieval was touched.
- Verified the actionable indexing prerequisite error and client cleanup in every script.
- Left production scripts unchanged because they already implemented the correct behavior.

### Python/code pattern used
```python
fake_client = FakeQdrantClient(collection_exists=False)

def unexpected_search(**kwargs):
    raise AssertionError("search should not run without the required collection")

with pytest.raises(
    RuntimeError,
    match="Qdrant collection does not exist.*run_qdrant_indexing.py",
):
    script.main()

assert fake_client.closed is True
```

### Validation
- Targeted evaluation script tests -> 12 passed.
- Broader retrieval evaluation tests -> 35 passed.
- Full suite: `python -m pytest -q` -> 214 passed.
- Diff whitespace validation passed; only Git line-ending conversion warnings were reported.

### Production interpretation
The tests now protect a consistent pre-flight contract across dense baseline, comparison, hybrid evaluation, and weight-tuning CLIs. If indexing is missing, evaluation fails before embedding cost and returns the same operator recovery action while releasing the persistent client.

### Failure-mode thinking
Final exception assertions alone could pass even if a script performed an embedding call first. The raising search fakes deliberately break on any retrieval touch, proving the dependency check truly occurs before cost-bearing work.

## Step 87 - Typed UI API client boundary

### Goal
Create a reusable, testable HTTP boundary for a future Streamlit UI without coupling presentation code directly to raw HTTP responses.

### Files touched
- `app/ui/__init__.py`
- `app/ui/api_client.py`
- `tests/test_ui_api_client.py`
- `README.md`

### What changed
- Added `RagApiClient` with typed methods for `GET /health`, `GET /ready`, and `POST /query`.
- Reused the existing Pydantic API request/response contracts.
- Added base-URL normalization and positive timeout validation.
- Added an injectable `httpx.Client` for deterministic tests.
- Added safe `UiApiError` categories for timeout, unavailable backend, `503` not-ready, other HTTP errors, and invalid responses.
- Prevented raw backend bodies and low-level network exception details from reaching the presentation layer.
- Added owned-client context-manager cleanup while leaving injected-client ownership with the caller.
- Updated the README to mark the UI client boundary complete and the visual Streamlit page as the next step.

### Python/code pattern used
```python
with RagApiClient("http://127.0.0.1:8000", timeout=10.0) as api:
    readiness = api.get_readiness()
    result = api.query(QueryRequest(query="What changed?", limit=5))
```

### Validation
- Targeted UI API client tests: `python -m pytest tests/test_ui_api_client.py -q` -> 8 passed.
- Broader API/schema tests -> 40 passed.
- Full suite: `python -m pytest -q` -> 222 passed.
- Diff whitespace validation passed; only Git line-ending conversion warnings were reported.

### Production interpretation
The future UI can render typed domain responses and one safe error contract instead of interpreting status codes, arbitrary JSON, or low-level network exceptions itself. This keeps the Streamlit layer thin, makes failure behavior reusable, and catches backend contract drift through Pydantic validation.

### Failure-mode thinking
Mock transports deliberately return `503`, raise timeout and connection errors, emit malformed JSON, and return schema-incomplete JSON. Each case becomes a stable UI-safe error without leaking secret response-body or exception details. These tests make the client fail closed when backend contracts drift.

## Step 88 - Streamlit grounded-query interface

### Goal
Build the first visual UI over the typed Step 87 API client while keeping readiness, validation, and failure behavior production-minded and testable.

### Files touched
- `app/ui/streamlit_app.py`
- `tests/test_streamlit_ui.py`
- `requirements.txt`
- `README.md`

### What changed
- Added a Streamlit page with configurable backend URL and timeout.
- Added a backend health/readiness check in the sidebar.
- Added a grounded-query form with evidence limit, document family, release label, source kind, and optional minimum-score override.
- Added Pydantic request validation before HTTP calls.
- Added readiness gating before every cost-bearing query.
- Added rendering for grounded answers, safe refusals, evidence sufficiency, citations, trace IDs, and optional model usage/cost.
- Deliberately omitted the local trace output path from the UI.
- Added `streamlit` to project dependencies and documented the two-terminal run workflow.
- Added pure boundary tests for payload construction, blank-query rejection, readiness-before-query ordering, and not-ready query blocking.

### Python/code pattern used
```python
with RagApiClient(api_base_url, timeout=timeout) as api:
    readiness = api.get_readiness()
    if not readiness.is_ready:
        raise UiApiError(code="not_ready", message="The RAG API is not ready...")
    response = api.query(request)
```

### Validation
- Streamlit/UI client/README tests -> 18 passed.
- Broader UI/API contract tests -> 47 passed.
- Full suite: `python -m pytest -q` -> 227 passed.
- Runtime dependency: Streamlit 1.52.2 was installed and callable.
- Local browser visual QA confirmed the full form layout, backend-unavailable safe message, and blank-query validation.
- The temporary Streamlit server used for QA was stopped afterward.

### Production interpretation
The UI remains a presentation layer: it builds a validated request, checks readiness, delegates HTTP policy to `RagApiClient`, and renders typed responses. A missing backend or dependency blocks retrieval/generation, while insufficient evidence remains a valid refusal response rather than a UI error.

### Failure-mode thinking
Tests and live QA intentionally exercised blank input, backend unavailability, and not-ready state. These cases never reach `POST /query`. The UI also avoids displaying raw exception bodies or the server's local trace path, reducing accidental leakage of implementation details.

## Step 89 - Reproducible uv project migration

### Goal
Replace the unpinned, manually managed `requirements.txt` workflow with a reproducible `uv` project suitable for local development, CI, packaging, and deployment.

### Files touched
- `pyproject.toml`
- `uv.lock`
- `.python-version`
- `requirements.txt` (removed)
- `README.md`
- `tests/test_readme_api_docs.py`
- `tests/test_streamlit_ui.py`
- `tests/test_uv_project_config.py`

### What changed
- Added standardized project metadata and Python policy in `pyproject.toml`.
- Declared Python `>=3.12,<3.13` and added `.python-version` with `3.12`.
- Moved runtime dependencies into `[project].dependencies`.
- Moved `pytest` into the `dev` dependency group.
- Added tested lower bounds and major-version ceilings for every direct runtime and development dependency.
- Removed unused `pandas` as a direct dependency; it remains a transitive Streamlit dependency in the lock.
- Generated the universal `uv.lock` with exact transitive versions.
- Removed handwritten `requirements.txt` as a competing source of truth.
- Converted backend, UI, smoke-test, indexing, and test commands to `uv run --locked`.
- Documented locked sync, lock freshness checks, production-only sync, and generated pip export for platforms that require it.
- Added repository tests protecting the uv metadata, lockfile, Python version, single-source dependency policy, and README workflow.

### Python/code pattern used
```toml
[project]
requires-python = ">=3.12,<3.13"
dependencies = [
    "fastapi",
    "httpx",
    "openai",
    "pydantic>=2",
    "pydantic-settings",
    "python-docx",
    "qdrant-client",
    "streamlit",
    "uvicorn",
]

[dependency-groups]
dev = ["pytest"]
```

### Validation
- `uv lock` -> resolved 72 packages.
- `uv sync --locked` -> created `.venv` and installed the locked development environment.
- Full suite: `uv run --locked pytest -q` -> 231 passed.
- `uv lock --check` -> passed.
- `uv export --locked --no-dev --format requirements-txt` -> generated a hashed pip-compatible export successfully.
- `uv sync --locked --no-dev` -> production-only environment synchronized successfully.
- Runtime-only import check -> FastAPI, HTTPX, OpenAI, Pydantic, pydantic-settings, python-docx, Qdrant, Streamlit, and Uvicorn imported successfully.
- Development dependencies were restored afterward with `uv sync --locked`.
- Refreshed the lock after adding compatibility bounds; all 72 packages and all direct locked versions remained unchanged.
- `uv tree --locked --depth 1` confirmed the exact direct versions satisfy the declared ranges.
- Bounded-dependency packaging/documentation tests -> 14 passed after correcting one stale test assertion that still expected an unversioned Streamlit dependency.
- Final bounded-dependency full suite: `uv run --locked pytest -q` -> 231 passed with the same single upstream warning.

### Warning observed
The locked environment reports one upstream Starlette deprecation warning stating that its current `TestClient` use of `httpx` is deprecated in favor of `httpx2`. It does not fail tests. The lockfile makes the exact FastAPI/Starlette/HTTPX state reproducible so this can be upgraded deliberately rather than appearing unexpectedly on another machine.

### Production interpretation
The `.venv` remains disposable and Git-ignored. Deployments reproduce dependencies from `pyproject.toml` plus `uv.lock`, while `--locked` prevents silent dependency resolution changes. Development tooling is excluded with `--no-dev`, and pip requirements are generated only as a deployment adapter rather than manually maintained.

The direct dependency ranges now encode the tested baseline and prevent accidental
major-version upgrades during a future deliberate relock. Exact installed versions
remain the responsibility of `uv.lock`.

### Failure-mode thinking
The migration was tested against newly resolved major dependency versions, not merely the preexisting global environment. Production-only sync deliberately removed pytest and its transitive tools, then verified every runtime import. Lock freshness and export checks protect CI from stale metadata and hosting platforms that still require pip-compatible inputs.

## Step 90 - Locked GitHub Actions CI

### Goal
Make every push and pull request independently verify the locked Python 3.12
environment and full regression suite.

### Files touched
- `.github/workflows/ci.yml`
- `tests/test_ci_workflow.py`
- `README.md`
- `interview-questions.md`
- `Steps_followed.md`

### What changed
- Added a least-privilege GitHub Actions workflow for pushes and pull requests.
- Pinned the official Astral uv setup action to a commit SHA and pinned uv
  `0.11.32`.
- Configured Python 3.12 and lockfile-keyed dependency caching.
- Added an explicit `uv lock --check` gate before installation.
- Installed development dependencies with `uv sync --locked --dev`.
- Ran the complete suite with `uv run --locked pytest -q`.
- Kept CI independent of application secrets and live Qdrant/model services.
- Added regression tests that protect the CI triggers, permissions, pins, and
  locked commands.

### Python/code pattern used
```python
def test_ci_rejects_lock_drift_and_runs_only_locked_project_commands() -> None:
    workflow = CI_WORKFLOW_PATH.read_text(encoding="utf-8")

    assert "run: uv lock --check" in workflow
    assert "run: uv sync --locked --dev" in workflow
    assert "run: uv run --locked pytest -q" in workflow
```

### Production interpretation
CI now rebuilds the project from committed metadata instead of trusting a
developer's local `.venv`. Lock drift fails visibly, and tests run in a clean
Linux environment before changes are merged.

### Validation
- CI/uv/README contract tests: 12 passed.
- Full regression suite: 234 passed.
- The existing upstream Starlette `TestClient`/HTTPX deprecation warning remains
  non-failing.
- Pytest's local cache was disabled for the full local run because this managed
  workspace denied writes to `.pytest_cache`; test execution was unaffected.

### Failure-mode thinking
The workflow has a 15-minute timeout, read-only repository permissions, no
secret dependency, and tests that fail if a future edit removes lock checking
or replaces locked commands with auto-resolving commands.

## Step 91 - Conversation domain and local persistence

### Goal
Create durable, conversation-scoped memory contracts without coupling the chat
application to SQLite or prematurely implementing summarization.

### Files touched
- `app/conversation/__init__.py`
- `app/conversation/models.py`
- `app/conversation/store.py`
- `app/core/config.py`
- `tests/test_conversation_models.py`
- `tests/test_conversation_store.py`
- `README.md`
- `docs/project_plan.md`
- `interview-questions.md`
- `Steps_followed.md`

### What changed
- Added immutable conversation, message, and summary-checkpoint records.
- Added user and assistant message roles with deterministic per-conversation
  sequence numbers.
- Added a runtime-checkable `ConversationStore` protocol.
- Added a durable local `SqliteConversationStore` using only the Python standard
  library.
- Added configurable storage through `CONVERSATION_DB_PATH`.
- Added forward-only, versioned summary checkpoints without implementing summary
  generation yet.
- Added readable, write-protected conversation archiving.
- Added idempotent cleanup and a safe use-after-close error.
- Kept persistence independent from FastAPI, Streamlit, retrieval, and
  generation.

### Python/code pattern used
```python
@runtime_checkable
class ConversationStore(Protocol):
    def create_conversation(self, title: str = "New conversation") -> Conversation: ...
    def add_message(
        self,
        conversation_id: str,
        role: MessageRole,
        content: str,
        *,
        trace_id: str | None = None,
    ) -> ConversationMessage: ...
    def save_summary(
        self,
        conversation_id: str,
        summary_text: str,
        *,
        summarized_through_sequence: int,
    ) -> ConversationSummary: ...
```

### Production interpretation
SQLite provides local durability now, but application code depends on the store
contract rather than SQL. A future Oracle-backed implementation can preserve the
same conversation behavior while changing connection pooling, schema, and query
details behind the adapter.

### Validation
- Focused conversation and documentation checks: 18 covered tests.
- Full regression suite: 247 passed with the existing non-failing upstream
  Starlette `TestClient`/HTTPX deprecation warning.
- The focused suite covers configurable database location, durable reopen,
  conversation isolation, ordered messages, summary versioning, stale and
  future checkpoint rejection, archive behavior, missing conversations, and
  cleanup.

### Failure-mode thinking
Tests intentionally attempt cross-conversation access patterns, backward and
future summary checkpoints, writes to archived conversations, missing IDs, and
store use after cleanup. Summary generation and token budgeting remain outside
this step so persistence failures are isolated from model behavior.

## Step 92 - Token-aware context budgeting and rolling summarization

### Goal
Bound conversation memory before prompt construction, preserve recent messages
verbatim, and compact only older history into a durable rolling summary without
treating chat memory as authoritative RAG evidence.

### Files touched
- `app/conversation/context.py`
- `app/conversation/summarizer.py`
- `app/conversation/__init__.py`
- `app/core/config.py`
- `tests/test_conversation_context.py`
- `tests/test_conversation_summarizer.py`
- `README.md`
- `docs/project_plan.md`
- `interview-questions.md`
- `Steps_followed.md`

### What changed
- Added explicit token allocations for the complete model context, system
  instructions, retrieved evidence, answer output, and rolling summaries.
- Added an injectable `TokenCounter` contract and a dependency-free,
  UTF-8-aware approximate counter for deterministic preflight estimates.
- Added a `RollingConversationContextBuilder` that keeps context unchanged
  below the threshold and summarizes only the older prefix above it.
- Retained a configurable minimum recent-message suffix verbatim.
- Consolidated each previous summary with newly compacted messages and advanced
  the durable checkpoint only after validating the new summary.
- Added an OpenAI-compatible conversation summarizer with instructions to
  preserve intent and constraints, resist instructions embedded in history,
  avoid invented facts and citations, and respect the output-token cap.
- Kept summary memory explicitly separate from newly retrieved documentary
  evidence.

### Python/code pattern used
```python
budget = ContextBudget(
    max_context_tokens=32_000,
    reserved_system_tokens=2_000,
    reserved_evidence_tokens=12_000,
    reserved_answer_tokens=4_000,
    summary_target_tokens=1_000,
)

context = RollingConversationContextBuilder(
    store=store,
    summarizer=summarizer,
    token_counter=ApproximateTokenCounter(),
    budget=budget,
).build(conversation_id)
```

### Why it is implemented this way
The complete model window is not available to chat history: system rules,
freshly retrieved evidence, and output generation need protected capacity.
Summarization is triggered by estimated tokens instead of message count because
message lengths vary substantially. A recent suffix remains verbatim to retain
exact follow-up details, while the versioned checkpoint identifies precisely
which older messages the rolling summary replaces.

Both token counting and summary generation are replaceable adapters. The local
estimator avoids a new tokenizer dependency, while a future model-specific
counter can provide exact accounting without changing the rolling policy.

### Production interpretation
The builder prevents unbounded chat growth from crowding out grounding
evidence or exceeding the model window. Summary generation is cost-bearing only
when the threshold is crossed. Durable checkpoints allow later requests to
reuse the compacted state instead of repeatedly summarizing the full history.
Conversation summaries help resolve follow-ups but cannot support
functional-spec claims without fresh retrieved evidence and citations.

### Validation
- Focused conversation, configuration, summarizer, and documentation suite:
  32 passed.
- Full regression suite: 261 passed.
- The existing upstream Starlette `TestClient`/HTTPX deprecation warning remains
  non-failing.
- `git diff --check` passed; Git reported only the existing Windows line-ending
  normalization notices.

### Failure-mode thinking
Tests force token overflow, cross-conversation access, repeated rolling
checkpoints, prompt-like instructions embedded in message history, blank model
output, oversized summary output, and mandatory recent messages that cannot fit.
Invalid summaries never advance the durable checkpoint. Impossible budgets fail
explicitly rather than silently dropping messages or consuming evidence and
answer reserves.

## Step 93 - Conversation API and multi-turn message submission

### Goal
Expose durable conversation lifecycle and grounded multi-turn submission through
FastAPI without duplicating the existing retrieval-answer orchestration or
weakening the evidence boundary.

### Files touched
- `app/api/main.py`
- `app/api/routes/query.py`
- `app/api/routes/conversations.py`
- `app/schemas/conversation_api.py`
- `app/conversation/context.py`
- `app/conversation/summarizer.py`
- `app/conversation/__init__.py`
- `app/llm/answer_contract.py`
- `app/llm/prompt_template.py`
- `app/services/answer_generation.py`
- `app/services/answer_orchestration.py`
- `tests/test_conversation_api.py`
- `tests/test_prompt_template.py`
- `tests/test_readme_api_docs.py`
- `README.md`
- `docs/project_plan.md`
- `interview-questions.md`
- `Steps_followed.md`

### What changed
- Added endpoints to create and list conversations, retrieve messages and the
  current summary, archive conversations, and submit grounded messages.
- Added typed request/response contracts with trimmed non-blank titles and
  content, bounded input length, supported source-kind validation, and
  structured conversation/turn metadata.
- Added a per-request `ConversationStore` dependency so local connections close
  safely and tests can replace persistence deterministically.
- Reused the same query execution and grounded orchestration path as
  `POST /query`.
- Rendered bounded conversation history as explicitly marked prompt data.
- Extended grounded prompting so memory may resolve conversational intent but
  cannot support functional-spec claims without retrieved evidence.
- Made the conversation summarizer client lazy so short histories do not create
  a model client or require credentials before compaction is actually needed.
- Persisted user messages before cost-bearing processing and assistant messages
  only after a completed grounded result.

### Python/code pattern used
```python
@router.post(
    "/{conversation_id}/messages",
    response_model=ConversationTurnResponse,
)
def submit_message(
    conversation_id: str,
    request: ConversationMessageRequest,
    store: ConversationStore = Depends(get_conversation_store),
) -> ConversationTurnResponse:
    user_message = store.add_message(
        conversation_id,
        MessageRole.USER,
        request.content,
    )
    context = build_conversation_context(store, conversation_id)
    answer = execute_query_request(
        QueryRequest(query=request.content, limit=request.limit),
        conversation_context=render_conversation_context(context),
    )
    assistant_message = store.add_message(
        conversation_id,
        MessageRole.ASSISTANT,
        answer.answer,
        trace_id=answer.trace_id,
    )
```

### Why it is implemented this way
FastAPI handles transport validation, dependency lifecycle, status mapping, and
response formatting while the store and grounded-query services retain domain
and orchestration responsibilities. This prevents the chat API from becoming a
second retrieval pipeline with different sufficiency, citation, or trace
behavior.

Persisting the user message first creates an honest durable record when a
dependency fails. The assistant message is not created until a grounded answer
or safe refusal exists. This allows a visible partial turn rather than
fabricating success or silently losing the user's submitted question.

### Production interpretation
Clients can now reconstruct conversation history from durable storage instead
of trusting Streamlit session state. Archive is a read-only lifecycle action,
and assistant messages retain trace IDs for debugging. Memory is passed to the
answer prompt as context only; retrieval evidence and citation validation remain
the factual authority.

The partial-turn contract requires future UI retry behavior and idempotency
design. List endpoints also need pagination before a high-volume or multi-user
deployment. Those concerns are explicit rather than hidden by the local API.

### Validation
- Focused conversation API, legacy query, prompt, orchestration, and README
  contract suite: 32 passed.
- Full regression suite: 270 passed.
- The existing upstream Starlette `TestClient`/HTTPX deprecation warning remains
  non-failing.
- `git diff --check` passed with only Windows line-ending normalization notices.

### Failure-mode thinking
Tests cover unknown conversation IDs, cross-conversation history isolation,
archived writes, invalid inputs, context overflow, downstream retrieval failure,
safe error bodies, and persistence after partial failure. Overflow stops before
the grounded query, archived chats cannot incur query cost, and backend failure
leaves a user-only partial turn instead of inventing an assistant response.

## Step 94 - Streamlit multi-turn chat UI

### Goal
Replace the single-query Streamlit form with a durable, conversation-scoped chat
experience that preserves the grounded RAG boundary and exposes safe operational
debugging information.

### Files touched
- `app/ui/api_client.py`
- `app/ui/streamlit_app.py`
- `tests/test_ui_api_client.py`
- `tests/test_streamlit_ui.py`
- `README.md`
- `docs/project_plan.md`
- `Steps_followed.md`
- `interview-questions.md`

### What changed
- Extended the typed UI client with create, list, detail, archive, and
  conversation-message methods.
- Added safe UI error codes for missing conversations, archived writes, context
  overflow, unavailable dependencies, and generic HTTP failures.
- Rebuilt Streamlit around `st.chat_message` and `st.chat_input`.
- Added New Chat, active-conversation selection, and Archive Active Chat
  controls.
- Reloaded conversation messages from the backend so durable storage remains the
  history source of truth.
- Added readiness gating before cost-bearing message submission.
- Added optional evidence/debug details for citations, sufficiency, trace ID,
  context budget, summary checkpoint, usage, and estimated cost.
- Added explicit user-only partial-turn warnings for failed prior attempts.
- Kept per-turn debug responses in Streamlit session state only as a transient
  cache because the current history endpoint stores messages and trace IDs but
  not the full citation/debug response.

### Python/code pattern used
```python
prompt = st.chat_input(
    "Ask a grounded question about the functional specifications",
    disabled=detail.conversation.is_archived,
)
if prompt:
    request = _build_message_request(content=prompt, ...)
    turn = _run_ready_turn(api, active_id, request)
    _cache_turn(st.session_state, turn)
    st.rerun()
```

### Why it is implemented this way
The UI is a presentation layer, not a second source of conversation truth.
Every rerun reloads conversations and messages from FastAPI, while session state
stores only replaceable display details. Readiness is checked before message
submission to avoid preventable retrieval/model cost. Typed response validation
and safe error mapping keep backend bodies, paths, and exception details out of
the page.

Partial user-only turns are rendered rather than hidden because Step 93
deliberately persists accepted user input before grounded processing. This makes
failures auditable and gives the future retry/idempotency flow a visible state.

### Production interpretation
The page is now a usable local multi-turn product surface with durable chat
lifecycle and debugging visibility. It remains intentionally local and
single-user: authentication, authorization, pagination, idempotent retries,
streaming responses, and persistent citation/debug history remain future
production concerns.

Debug visibility improves trust and incident diagnosis, but it is optional to
avoid overwhelming ordinary users. Conversation memory still cannot prove
functional-spec claims; the displayed citations and evidence sufficiency remain
the factual grounding signals.

### Validation
- Focused UI/client/conversation/documentation suite: 35 passed.
- Full regression suite: 279 passed.
- Existing upstream Starlette `TestClient`/HTTPX deprecation warning remains
  non-failing.
- `git diff --check` passed with only Windows line-ending notices.
- Live FastAPI `/health`: `ok`, hybrid retrieval.
- Live Streamlit HTTP check: 200.
- Browser QA verified initial render, New Chat creation, active selection,
  enabled chat input, readiness details, archive removal from the active list,
  and an empty browser warning/error console.
- No model query was submitted during QA.
- Temporary FastAPI and Streamlit listeners were stopped after verification.

### Failure-mode thinking
Tests cover malformed responses, safe mappings for `404`/`409`/`413`/`503`/500,
readiness blocking, partial-turn detection, blank input, filter normalization,
and stale active-conversation fallback. Browser QA also exposed that a fixed
startup delay can race a real server; service verification must poll an
authoritative endpoint rather than infer readiness from process launch.

## Step 95 - Complete, token-bounded prompt evidence

### Goal
Repair an answer-quality defect where retrieval found the correct BOR evidence
but answer generation received only each citation's 240-character UI preview.

### Files touched
- `app/llm/prompt_template.py`
- `app/services/answer_generation.py`
- `tests/test_prompt_template.py`
- `tests/test_answer_generation_service.py`
- `README.md`
- `docs/project_plan.md`
- `Steps_followed.md`
- `interview-questions.md`
- `Culling Blade Lineage- Gen AI RAG System_Prompt_for_GPT.txt`

### What changed
- Introduced a separate `PromptEvidence` model containing the complete text of
  each selected retrieval unit.
- Kept `Citation.text_preview` as a 240-character API/UI display field only.
- Selected evidence in retrieval-rank order using the configured reserved
  evidence-token budget.
- Admitted only whole evidence units; no unit is silently cut in the middle.
- Returned a deliberate safe refusal without calling the LLM when the
  highest-ranked complete unit cannot fit.
- Returned only citations corresponding to evidence actually sent to the LLM.
- Added BOR and B-04 regressions with decisive facts placed beyond character
  240, plus budget and no-LLM-on-overflow tests.

### Python/code pattern used
```python
prompt_evidence = select_prompt_evidence(
    request.retrieved_results,
    max_evidence_units=max_citations,
    max_evidence_tokens=max_evidence_tokens,
)
evidence_block = build_evidence_block(prompt_evidence)
citations = build_citations_from_results(
    request.retrieved_results[: len(prompt_evidence)],
    max_citations=len(prompt_evidence),
)
```

### Why it is implemented this way
Citation previews and model evidence have different jobs. A short preview keeps
API responses and UI panels compact, but using it as evidence can remove a
decisive row late in a table. Complete ranked units preserve semantic integrity.
The lightweight UTF-8 token estimate is replaceable with exact model
tokenization. Whole-unit admission makes the boundary observable and avoids
answers based on incomplete table fragments.

### Production interpretation
The first BOR failure is now addressable because the acronym row can reach the
model even when it occurs after character 240. This step does not yet guarantee
the second answer, “17 reduced to 4,” because hybrid top-5 retrieval omitted the
decisive R24 realignment tables for that natural query. Retrieval fusion,
alias-aware query handling, table-aware chunking, and answer-level evaluation
remain the next quality layer.

### Validation
- Focused prompt/answer/API/conversation suite: 36 passed.
- Full regression suite: 283 passed.
- Existing upstream Starlette `TestClient`/HTTPX warning remains non-failing.

### Failure-mode thinking
Tests prove that late BOR/B-04 facts survive prompt construction, citation
previews remain bounded, lower-ranked evidence is omitted only at whole-unit
boundaries, oversized top evidence produces a safe refusal, and the LLM is not
called on that refusal path. They do not prove that hybrid retrieval ranks every
decisive R24 table inside top-k or that the model always derives the correct
temporal count.

## Step 96 - Weighted Reciprocal Rank Fusion

### Goal
Repair the demonstrated hybrid-ranking defect where excellent lexical table
results were capped below mediocre dense-only results because incompatible
score scales were normalized and added.

### Files touched
- `app/retrieval/hybrid_search.py`
- `app/retrieval/evaluation.py`
- `app/core/config.py`
- `app/services/answer_orchestration.py`
- `.env.example`
- `data/eval/retrieval_eval.json`
- `tests/test_hybrid_search.py`
- `tests/test_retrieval_evaluation.py`
- `tests/test_retrieval_config.py`
- `README.md`
- `docs/project_plan.md`
- `Steps_followed.md`
- `interview-questions.md`

### What changed
- Replaced max-normalized score addition with weighted Reciprocal Rank Fusion.
- Selected lexical-heavy defaults of `0.40` dense and `0.60` lexical for this
  identifier- and table-heavy functional-spec corpus.
- Used an RRF rank constant of `1.0` for the short ten-result candidate lists.
- Normalized only the final RRF score to `[0,1]`, preserving the existing
  sufficiency-threshold scale without comparing raw dense and lexical scores.
- Added raw scores, ranks, per-retriever RRF contributions, fusion method, and
  rank constant to result/trace diagnostics.
- Extended retrieval evaluation with `expected_text_contains_all` so a test can
  require baseline, teller, and branch evidence together.
- Added an R24 regression reproducing the observed dense and lexical ranking
  pattern and requiring both realignment tables inside final top-5.

### Python/code pattern used
```python
rrf_contribution = weight / (rrf_rank_constant + rank)
raw_rrf_score += rrf_contribution
hybrid_score = raw_rrf_score / maximum_possible_rrf_score
```

### Why it is implemented this way
Cosine similarity and lexical relevance are different measurement scales. Rank
fusion uses only their ordering, so one retriever's arbitrary score magnitude
cannot dominate another. A lexical-heavy weight is appropriate here because
report identifiers such as `B-17`, action terms, and table headers carry exact
business meaning. The final normalization is solely a stable API/sufficiency
scale; it does not reintroduce input-score comparison.

### Production interpretation
For the user's original wording, real hybrid retrieval moved the R24 branch
realignment table to rank 2 and teller table to rank 4. The previous irrelevant
traceability result was removed from final top-5. No reindex was required.

RRF did not solve the complete answer. A controlled live variant retrieved the
detailed R24 business-requirements table but `gpt-5-mini` still answered 6/17,
treating the document's pre-change “Existing Functionality” as current. This
isolates the next defect to production-deployed latest-release temporal
semantics, query/release scoping, and answer-level validation.

### Validation
- The new RRF regression failed under the old normalized-score algorithm.
- Focused retrieval/evaluation suite: 42 passed.
- Full regression suite: 285 passed.
- Real retrieval check returned the branch and teller tables inside top-5 for
  the original natural-language query.
- Live answer trace:
  `c49b8880-be04-4216-80c3-20e86d9797c5`.
- The live check made one embedding call and one answer-generation call.
- Existing upstream Starlette `TestClient`/HTTPX warning remains non-failing.

### Failure-mode thinking
The implementation rejects invalid weights, limits, and RRF constants. Tests
cover overlap, rank-only evidence, final limits, bounded score output, and
multi-fact coverage failure. The live answer intentionally demonstrates that
retrieval recall is necessary but not sufficient: correct evidence can still be
synthesized under the wrong temporal policy.

## Step 97 - Current-state temporal synthesis and release scoping

### Goal
Make “current” mean the resulting production state after the latest relevant
deployed release, resolve conversational release references safely, and prevent
the R24 pre-change 6/17 baseline from being returned as the current answer.

### Files touched
- `app/retrieval/temporal_query.py`
- `app/services/answer_orchestration.py`
- `app/llm/answer_contract.py`
- `app/llm/prompt_template.py`
- `app/services/answer_generation.py`
- `app/llm/answer_evaluation.py`
- `data/eval/answer_eval.json`
- `scripts/evaluate_answer_trace.py`
- `tests/test_temporal_query.py`
- `tests/test_prompt_template.py`
- `tests/test_answer_orchestration_service.py`
- `tests/test_answer_evaluation.py`
- `README.md`
- `docs/project_plan.md`
- `Steps_followed.md`
- `interview-questions.md`

### What changed
- Added deterministic current-state intent detection and numeric release
  parsing so `R24` sorts after `R2`.
- Added a guard so domain phrases such as “current application date” do not
  incorrectly trigger latest-release behavior.
- Allowed referential queries such as “summarize it” to inherit an explicit
  release label from bounded conversation memory.
- Expanded current-state retrieval to the configured hybrid candidate depth in
  one search, then scoped the final results to the highest relevant deployed
  release before answer generation.
- Added retrieval expansion terms for retained, consolidated, renamed, and
  removed items while explicitly marking Existing Functionality as baseline.
- Added prompt rules stating that all indexed releases are deployed and that
  current/latest answers must apply the effective release changes.
- Added a reusable answer-trace evaluator requiring the 6/17 baseline, current
  2/4 result, T/B report identifiers, R24-only citations, and the baseline and
  teller/branch table evidence units.

### Python/code pattern used
```python
plan = build_temporal_query_plan(
    query_text,
    requested_release_label=release_label,
    conversation_context=conversation_context,
)
routed = retrieve_query_evidence(
    query_text=plan.retrieval_query,
    limit=max(limit, hybrid_candidate_limit),
    ...
)
results, plan = scope_results_to_temporal_plan(
    routed.results,
    plan,
    limit=limit,
)
```

### Why it is implemented this way
Conversation memory may resolve what “it” refers to, but it is not proof. The
resolved release is therefore used only as a retrieval filter; the answer still
requires fresh indexed R24 evidence. Candidate expansion and in-memory release
scoping avoid a second embedding/API call while giving change tables room to
enter the final evidence set.

Release numbers are parsed numerically rather than sorted as strings. The prompt
separates the latest release's baseline section from its resulting state so the
model must count retained/renamed outputs and exclude removed reports.

### Production interpretation
The exact reported question now returns:
- Pre-R24 baseline: 6 teller and 17 branch reports.
- Current deployed R24 state: 2 teller reports (`T-1`, `T-2`) and 4 branch
  reports (`B-01` through `B-04`).

All five live citations were R24. The older R2 result was removed, and the
evidence contained the baseline paragraph plus teller, branch, and detailed
requirements tables. The retrieval trace records original and expanded queries,
current-state intent, candidate depth, effective release, and how release scope
was resolved.

### Validation
- Focused temporal/answer/API/conversation suite: 41 passed.
- Full regression suite: 296 passed.
- Exact live query answered 2 teller and 4 branch reports.
- Live trace: `b286e5c6-c4ed-4338-af10-9b95927650a2`.
- Persisted live trace passed `r24_current_teller_and_branch_report_state`.
- Live validation made one embedding request and one `gpt-5-mini` generation
  request; current-state candidate expansion did not make a second embedding
  request.
- Existing upstream Starlette `TestClient`/HTTPX warning remains non-failing.

### Failure-mode thinking
Tests cover numeric release ordering, explicit-filter precedence,
conversation-derived release scope, older-release removal, missing or malformed
release metadata, non-temporal “current application date,” incomplete answers,
baseline-as-current answers, R2 citation leakage, and missing table citations.

The current resolver is deterministic and deliberately narrow. If future
conversations compare multiple releases or the corpus includes draft/unreleased
documents, explicit lifecycle metadata and a richer query planner will be
required instead of assuming every indexed release is deployed.

## Step 98 - Conversation/RAG reliability evaluation

### Goal
Turn the main multi-turn failure modes into repeatable automated checks across
the real API, durable store, context, grounding, and answer-evaluation
boundaries without making paid or nondeterministic external calls.

### Files touched
- `app/conversation/reliability_evaluation.py`
- `app/llm/answer_evaluation.py`
- `data/eval/answer_eval.json`
- `scripts/evaluate_answer_trace.py`
- `tests/test_conversation_reliability_evaluation.py`
- `tests/test_conversation_rag_reliability_e2e.py`
- `tests/test_answer_evaluation.py`
- `README.md`
- `docs/project_plan.md`
- `Steps_followed.md`
- `interview-questions.md`

### What changed
- Added an identifier-level rolling-summary drift evaluator that detects
  invented release/report IDs and loss of explicit release scope.
- Extended persisted answer-trace evaluation with `expected_is_answered` and a
  required refusal-reason pattern.
- Added an unsupported mobile-login abstention case alongside the R24
  current-state answer case.
- Added end-to-end FastAPI/SQLite tests proving durable follow-up context,
  cross-conversation isolation, safe abstention persistence, and invalid
  citation suppression.
- Reused the existing route/context tests for safe `413` overflow behavior,
  partial-turn persistence, and forward-only summary checkpoints.

### Python/code pattern used
```python
result = evaluate_summary_drift(
    previous_summary=prior,
    messages=older_messages,
    candidate_summary=generated_summary,
)
assert not result.invented_anchors
assert not result.missing_release_anchors
```

### Why it is implemented this way
Deterministic dependency doubles remove network, latency, model variability,
and cost from the regression gate while the real application boundaries remain
under test. Summary checks focus on high-signal functional-spec identifiers:
inventing `R25` or losing `R24` can redirect later retrieval even when the
summary sounds fluent.

The answer evaluator checks both human-readable refusal text and the
machine-readable `is_answered`/`refusal_reason` contract. A polite refusal with
`is_answered=true` would otherwise be misclassified by clients and metrics.

### Production interpretation
The suite demonstrates correctness for deterministic single-process flows. It
does not prove high-concurrency SQLite behavior, browser rendering,
accessibility, retries, streaming quality, live-model stability, or semantic
citation entailment. Those require separate load, browser, and model-based
evaluation layers.

### Validation
- Focused reliability suite: 28 passed.
- Full regression suite: 300 passed.
- Existing non-failing upstream Starlette `TestClient`/HTTPX deprecation
  warning remains.

### Failure-mode thinking
The tests deliberately inject an unrelated question, an invalid `[C99]`
citation, invented summary anchors (`R25`, `B-09`), missing release scope,
cross-chat secrets, oversized context, and failed processing. Unsafe generated
content is not persisted as an answer, overflow stops before retrieval, and
failed turns retain only accepted user input for retry/audit.

## Step 99 - Secure request correlation and privacy-safe API auditing

### Goal
Create one correlation path across HTTP responses, structured audit events, and
persisted answer traces without logging request content, credentials, concrete
conversation identifiers, or internal errors.

### Files touched
- `app/core/request_observability.py`
- `app/api/main.py`
- `app/api/routes/query.py`
- `app/services/answer_orchestration.py`
- `app/services/answer_trace.py`
- `tests/test_request_observability.py`
- `tests/test_query_api.py`
- `tests/test_answer_trace.py`
- `README.md`
- `docs/project_plan.md`
- `Steps_followed.md`
- `interview-questions.md`

### What changed
- Added an HTTP middleware that accepts only bounded, injection-safe request IDs
  and generates a UUID when the supplied value is absent or invalid.
- Added one structured JSON completion event with method, route template,
  status, and elapsed milliseconds.
- Added defensive no-store, MIME-sniffing, framing, referrer, and browser
  permissions response headers.
- Passed the API request ID to answer traces as `correlation_id`.
- Kept correlation IDs separate from unique trace IDs and filenames so a
  repeated or attacker-controlled request ID cannot overwrite trace artifacts.

### Python/code pattern used
```python
request_id = _resolve_request_id(request.headers.get("X-Request-ID"))
token = request_context.set(request_id)
response = await call_next(request)
response.headers["X-Request-ID"] = request_id
audit_logger.info(json.dumps(safe_request_metadata))
```

### Why it is implemented this way
Correlation is useful only when the same safe identifier crosses API, logs, and
traces. A client-supplied identifier is untrusted input, so it is validated
before becoming a header or log value. Audit events use route templates such as
`/conversations/{conversation_id}` rather than resource IDs and exclude bodies,
queries, titles, credentials, IP addresses, and exception details.

The trace ID remains server-generated and unique. Treating a caller's request
ID as the trace filename would allow collisions and overwrite prior artifacts.

### Production interpretation
An operator can join an API response to its safe request event and answer trace
without searching user content. Status and latency provide a minimal base for
error-rate and latency aggregation. Security headers reduce caching and common
browser misuse but do not establish user identity or access control.

### Validation
- Focused request/trace/query/conversation suite: 25 passed.
- Full regression suite: 304 passed.
- Existing non-failing upstream Starlette `TestClient`/HTTPX deprecation
  warning remains.

### Failure-mode thinking
Tests inject newline/log-forging content and a secret-like token into
`X-Request-ID`; the value is replaced before logging. They verify bodies,
private titles, concrete conversation IDs, and unsafe IDs do not enter audit
events. They also prove correlation and trace identifiers remain different.

This step does not claim authentication, authorization, rate limiting, TLS,
centralized retention, tamper evidence, or Oracle deployment packaging. Those
depend on the selected production identity and runtime boundary.

## Step 100 - Deterministic native deployment bundle and preflight

### Goal
Package the application for an Oracle-approved native runtime without Docker,
without embedding secrets or mutable business data, and without assuming an OS
service manager before the target environment is confirmed.

### Files touched
- `app/deployment/native_package.py`
- `app/deployment/preflight.py`
- `app/deployment/__init__.py`
- `deployment/native_runtime.json`
- `scripts/build_native_deployment.py`
- `scripts/check_deployment_preflight.py`
- `tests/test_native_deployment_package.py`
- `tests/test_deployment_preflight.py`
- `README.md`
- `docs/project_plan.md`
- `Steps_followed.md`
- `interview-questions.md`

### What changed
- Added a deterministic ZIP builder with sorted entries, fixed timestamps and
  permissions, and SHA-256 hashes for every included file.
- Allowlisted runtime source, scripts, lock/configuration templates, and the
  native process contract.
- Excluded secrets, certificates, virtual environments, raw documents,
  processed artifacts, Qdrant state, SQLite history, traces, logs, tests, and
  generated exports.
- Added an offline deployment preflight for Python 3.12, lockfiles,
  environment labeling, model configuration presence, mode-specific retrieval
  state, and writable conversation/trace paths.
- Added an explicit native runtime contract for locked installation, preflight,
  FastAPI, and Streamlit processes while leaving supervision target-specific.

### Python/code pattern used
```python
result = build_native_package(
    project_root=ROOT_DIR,
    output_path="data/exports/deployment/lineage-rag-native.zip",
)
report = run_deployment_preflight(
    settings,
    project_root=ROOT_DIR,
)
```

### Why it is implemented this way
Source and immutable dependency metadata have a different lifecycle from
indexes, documents, conversations, and traces. Bundling mutable state would
make deployments large, leak data, and blur rollback/backup ownership.

Deterministic archives let release engineering compare hashes and reproduce
the exact bundle from the same source. The caller must still verify provenance
and sign or attest the artifact in the approved release pipeline.

Preflight checks local prerequisites without spending tokens or depending on
network availability. Service-manager configuration is deferred because
systemd, Windows Service, and Oracle-managed supervisors have materially
different identity, restart, logging, and secret-injection contracts.

### Production interpretation
The bundle can be installed with `uv sync --locked --no-dev`, validated before
traffic, and launched using the recorded process commands. Mutable retrieval
and conversation state must be mounted/provisioned separately and protected by
environment-specific backup, permission, and recovery controls.

### Validation
- Focused native deployment suite: 11 passed.
- Full regression suite: 308 passed.
- Two consecutive real builds produced the identical SHA-256:
  `b032289e194d36259154b199ffe066bdd09f557ed274ac89be2ccb8b857684d6`.
- Final archive contains 97 files.
- Local offline preflight passed with `--allow-development`.
- The lockfile was not modified. A separate `uv lock --check` could not run in
  this shell because the `uv` executable/module is not installed on its PATH;
  CI already enforces the committed lock with pinned `uv`.
- Existing non-failing upstream Starlette `TestClient`/HTTPX warning remains.

### Failure-mode thinking
Tests prove that `.env` and data files do not enter the archive, builds are
byte-for-byte deterministic, and missing Python/lock/model/retrieval/writable
requirements fail with actionable messages that do not expose secret values.
The `--allow-development` result explicitly states that it is local validation,
not production safety.

The bundle is not a signed supply-chain artifact and does not configure TLS,
authentication, process restart, resource limits, centralized logging,
backups, or zero-downtime rollout. Those controls belong to the confirmed
deployment platform and release pipeline.

## Step 101 - Tamper-evident local audit journal and verification boundary

### Goal
Persist the existing privacy-safe API request events in a durable integrity
chain that can be verified locally and exported to a future approved central
audit platform, without claiming that local mutable storage is immutable.

### Files touched
- `app/core/audit_journal.py`
- `app/core/config.py`
- `app/core/request_observability.py`
- `app/api/main.py`
- `app/deployment/preflight.py`
- `scripts/verify_audit_journal.py`
- `tests/test_audit_journal.py`
- `tests/test_request_observability.py`
- `tests/test_deployment_preflight.py`
- `.env.example`
- `deployment/native_runtime.json`
- `README.md`
- `docs/project_plan.md`
- `Steps_followed.md`
- `interview-questions.md`

### Python/code pattern used
```python
unsigned = {
    "sequence": sequence,
    "event": asdict(safe_event),
    "previous_hmac": previous_hmac,
}
record_hmac = hmac.new(
    secret_key,
    canonical_json(unsigned),
    hashlib.sha256,
).hexdigest()
journal.write(canonical_json({**unsigned, "hmac_sha256": record_hmac}))
journal.flush()
os.fsync(journal.fileno())
```

### What the code does
- Persists only the Step 99 fixed safe event schema, never request bodies,
  prompts, titles, credentials, concrete resource IDs, IPs, or raw errors.
- Chains each canonical JSONL record to the prior HMAC using a secret containing
  at least 32 UTF-8 bytes.
- Verifies schema, contiguous sequence, chain links, and every record HMAC on
  startup or through an offline CLI.
- Accepts an externally trusted final HMAC and count so verification can detect
  deletion of an otherwise valid suffix.
- Flushes and `fsync`s each record for a durability-first audit policy.
- Rejects a file changed by another writer rather than silently forking the
  chain.
- Keeps API responses available on journal write failure and emits a minimal
  critical event that must be alerted.
- Requires enabled, strong, writable audit configuration in production
  preflight while allowing explicit local development validation without it.

### Why it is implemented this way
A plain SHA-256 chain can be recomputed by anyone who can edit the journal.
HMAC makes undetected rewriting depend on access to a separately held secret.
Canonical JSON makes signing and verification deterministic. Sequence and
previous-HMAC fields expose insertion, edits, and reordering.

`fsync` favors audit durability over peak throughput. Fail-open request behavior
avoids turning a log-volume failure into a total RAG outage, but it creates an
audit gap; therefore the critical failure event is an operational page, not a
harmless warning.

### Production interpretation
The final HMAC and record count are checkpoint material, not business evidence.
They should be shipped to a separately controlled platform along with the
journal. The HMAC key belongs in the approved secret store with access control
and rotation procedures. Current retrieval evidence and validated citations
remain the authority for functional answers; audit integrity does not make an
answer grounded.

The journal is a local integrity/export boundary, not centralized retention or
WORM storage. The present native API command uses one process. Multi-worker or
multi-host ordering should be delegated to the selected central audit service
rather than approximated with one shared local file.

### Failure-mode testing
- Modified event content invalidates the HMAC and blocks writer startup.
- Malformed UTF-8 returns a safe verification failure without echoing bytes.
- Valid suffix deletion passes internal chain validation but fails when checked
  against an externally trusted final count/HMAC.
- A stale second writer is rejected instead of creating a silent fork.
- An attacker-controlled unmatched URL path is recorded as `<unmatched>`.
- A weak HMAC key fails production preflight without printing the key.
- Simulated journal I/O failure leaves the API response available, emits only a
  safe critical event, and does not expose the internal exception.

### Validation
- Focused audit/observability/deployment/package/docs suite: 20 passed.
- Final core audit/observability/deployment suite after verifier hardening:
  14 passed.
- A first full run exposed 12 health/readiness failures because older settings
  doubles did not define the new optional flag. App creation now uses a
  backward-compatible disabled default; the affected suite then passed 25/25.
- Final full regression suite: 316 passed.
- One existing non-failing upstream Starlette `TestClient`/HTTPX deprecation
  warning remains.
- Two consecutive deployment builds were identical: 99 files, SHA-256
  `8891058ce848c1ed2c66e2a61064c39a2b317dec90c676311893f4f323f78439`.
- The bundle manifest includes `app/core/audit_journal.py` and
  `scripts/verify_audit_journal.py`.

## Step 102 - Measure local audit durability cost

### Goal
Measure the actual local cost of the Step 101 per-request HMAC, append, flush,
and `fsync` policy before deciding whether production should use synchronous
durability, grouped commits, or a central durable collector.

### Files touched
- `app/core/audit_benchmark.py`
- `scripts/benchmark_audit_journal.py`
- `tests/test_audit_benchmark.py`
- `README.md`
- `docs/project_plan.md`
- `Steps_followed.md`
- `interview-questions.md`

### Python/code pattern used
```python
started = perf_counter_ns()
journal.append(synthetic_event)
latencies_ms.append((perf_counter_ns() - started) / 1_000_000)

verification_started = perf_counter_ns()
verification = verify_audit_journal(journal_path, ephemeral_key)
verification_ms = (perf_counter_ns() - verification_started) / 1_000_000
```

### What the code does
- Generates only synthetic request IDs and the fixed `/benchmark` route.
- Creates a random ephemeral benchmark HMAC key in memory rather than reading or
  exposing the production key.
- Writes warm-up events, then measures individual durable append latency for a
  configurable number of events.
- Reports linearly interpolated p50, p95, p99, maximum latency, measured
  single-writer throughput, journal size, average bytes per record, and
  full-chain verification time.
- Verifies the generated HMAC chain before accepting the measurement.
- Removes the temporary journal and persists only an aggregate local JSON
  report under `data/exports/audit_benchmarks/`.

### Why it is implemented this way
Durability policy should be based on measured storage behavior and the
business's acceptable audit-loss window. Averages hide tail latency, so the
report emphasizes percentiles and maximum latency. Warm-up operations reduce
first-write noise. The temporary journal prevents synthetic events from
contaminating the real audit trail.

The benchmark does not disable `fsync` for comparison because the current
production candidate is the durability-first writer. Grouped commit would be a
separate implementation with a defined loss window and must not be simulated by
silently weakening the existing writer.

### Local result
For 200 measured events after 10 warm-up events on the current Windows
filesystem and Python 3.12.13:

- p50 append latency: `3.392300 ms`
- p95 append latency: `6.351130 ms`
- p99 append latency: `7.638621 ms`
- maximum append latency: `28.502300 ms`
- measured single-writer throughput: `256.203 events/second`
- average storage: `406.390 bytes/record`
- verification time for 210 records: `10.213200 ms`

Report:
`data/exports/audit_benchmarks/audit-journal-benchmark.json`

### Production interpretation
On this one local run, synchronous audit durability adds several milliseconds
to the request path and shows a materially higher maximum. That may be small
relative to LLM latency, but health checks, refusals, cached operations, and
future low-latency endpoints may feel the overhead more strongly.

The `256 events/second` figure is not API capacity. It is a single-writer local
journal result without concurrent requests, model calls, retrieval, central
shipping, antivirus variation, production disks, or repeated-trial confidence
intervals. Production selection still requires representative end-to-end load,
p95/p99 SLOs, event volume, storage-growth projections, and an explicit maximum
acceptable loss window.

### Failure-mode testing
- Empty or negative latency samples are rejected.
- Zero measured events and negative warm-up counts fail before creating work.
- The real benchmark verifies its chain before returning metrics.
- Temporary synthetic journals are removed after the run.
- The persisted report is checked not to contain HMAC key fields or query
  content.

### Validation
- Focused audit benchmark/journal suite: 13 passed.
- Full regression suite: 323 passed.
- Existing non-failing upstream Starlette `TestClient`/HTTPX warning remains.
- Two consecutive native bundle builds were identical: 101 files, SHA-256
  `5d8fd5c25585bb1db2a8e34b0e20e69b75e494b47bf4d90d5bd6a8931fbc31c8`.
- The bundle includes `app/core/audit_benchmark.py` and
  `scripts/benchmark_audit_journal.py`.

## Step 103 - Extract a storage-neutral audit sink boundary

### Goal
Decouple FastAPI request auditing from the local JSONL implementation so the
storage target can later become a mounted/network filesystem, database, central
collector, or grouped-commit writer without changing request middleware or
copying privacy logic.

### Files touched
- `app/core/audit_sink.py`
- `app/core/config.py`
- `app/core/request_observability.py`
- `app/api/main.py`
- `app/deployment/preflight.py`
- `tests/test_audit_sink.py`
- `tests/test_deployment_preflight.py`
- `.env.example`
- `README.md`
- `docs/project_plan.md`
- `Steps_followed.md`
- `interview-questions.md`

### Python/code pattern used
```python
class AuditSink(Protocol):
    backend: str
    durability: Literal["durable_on_return", "accepted_not_durable"]

    def append(self, event: ApiAuditEvent) -> AuditAppendResult: ...

audit_sink = build_audit_sink(settings)
install_request_observability(app, audit_sink)
```

### What the code does
- Introduces an `AuditSink` protocol containing only the privacy-safe event
  append contract and explicit durability semantics.
- Wraps the existing `AuditJournal` in `HmacJsonlAuditSink` without weakening
  its HMAC, flush, `fsync`, verification, or failure behavior.
- Declares the current adapter `durable_on_return` and returns a checkpoint in
  a storage-neutral result object.
- Moves backend construction into one factory, leaving FastAPI unaware of file
  paths, JSONL serialization, HMAC keys, or future database clients.
- Adds `AUDIT_SINK_BACKEND=hmac_jsonl`; disabled audit still builds no sink.
- Rejects unsupported production backends before service startup/preflight.

### Why it is implemented this way
Changing the existing path already supports another local directory, mounted
volume, or syntactically valid network path. That does not make a network share
equivalent to a local durable disk: remote caching, acknowledgements, mount
options, disconnects, locking, and server failure change what `fsync` means.

A database adapter should own its transaction, schema, integrity, retry, and
idempotency behavior. A grouped adapter should own its queue, batch commit,
backpressure, shutdown, and loss-window contract. Keeping those policies behind
one protocol prevents storage-specific code from entering request middleware.

### Production interpretation
The boundary improves maintainability; it does not prove that every future
adapter is safe. `durable_on_return` means the current call has completed local
flush/`fsync` under the filesystem's contract. `accepted_not_durable` is
reserved for a future buffered adapter whose acknowledged events can still be
lost before batch commit.

The next grouped-commit experiment must compare performance and failure loss
against the Step 102 synchronous baseline. It must not silently change the
production default or claim remote durability without deployment-specific
evidence.

### Failure-mode testing
- Disabled auditing produces no sink and performs no storage initialization.
- The HMAC JSONL adapter persists and verifies a valid chain through the new
  boundary.
- Unknown backend configuration fails without echoing the attacker-controlled
  backend value.
- Production preflight rejects unsupported backends with a safe message.
- Existing middleware, audit failure, health, readiness, and deployment tests
  confirm the refactor preserves current behavior.

### Validation
- Focused sink/journal/observability/preflight/health/readiness suite: 30 passed.
- Full regression suite: 327 passed.
- One existing non-failing upstream Starlette `TestClient`/HTTPX deprecation
  warning remains.
- Two consecutive native bundles were identical: 102 files, SHA-256
  `3ffa4df318a23c164570bc65c41f0bc03bc5a427cc343c7452754ea6cfc07059`.
- The bundle contains `app/core/audit_sink.py`.

## Step 104 – Portable master FDD ingestion and verified archival

### Objective

Provide one portable command that processes all reviewed FDD DOCX files in
`data/raw_specs/` through the existing ingestion, embedding, Qdrant indexing,
and verification stages, then archives only verified source files.

### Files added or changed

- `scripts/master_ingestion_embedding_docs.py`
- `scripts/run_embedding_smoke_test.py`
- `scripts/check_qdrant_index.py`
- `app/embeddings/client.py`
- `app/core/config.py`
- `docs/Steps_for_FDD_Ingestion.md`
- `.env.example`
- `tests/test_master_ingestion_embedding_docs.py`
- `tests/test_qdrant_index_verification.py`
- `tests/test_embedding_client.py`

### Python/code pattern used

```python
commands = build_pipeline_commands(
    documents=documents,
    cache_directory=settings.cache_dir / "embeddings",
    request_batch_size=args.request_batch_size,
)
for command in commands:
    subprocess.run(command, cwd=ROOT_DIR, check=True)

verify_embedding_artifacts(
    client=client,
    collection_name=collection_name,
    artifact_paths=artifact_paths,
)
archive_documents(documents, settings.embedded_docs_dir)
```

### What the code does

- Adds the portable master command:
  `uv run --locked python scripts/master_ingestion_embedding_docs.py`.
- Reuses existing child scripts rather than duplicating DOCX extraction,
  OpenAI embedding, Qdrant indexing, or Qdrant inspection logic.
- Adds `--all-units` to the existing per-document embedding script and splits
  uncached retrieval units into bounded OpenAI embedding requests (64 by
  default, configurable with `--request-batch-size`).
- Extends Qdrant inspection to verify every deterministic point ID and its
  identifying payload metadata from the specific embedding artifacts.
- Adds `EMBEDDED_DOCS_DIR`, defaulting to `data/docs_embedded/`.
- Supports `--dry-run`, which lists selected documents and exact child commands
  without API, Qdrant, or file-move actions.
- Archives only after every child command, including exact Qdrant verification,
  has succeeded. Archive-destination conflicts stop the batch before child
  stages begin.

### Why it is implemented this way

A `.bat` file would be Windows-only and difficult to test; the Python command
works with the pinned project interpreter on Windows, Linux, CI, and future
cloud environments. The master contains orchestration only. Existing scripts
continue to own their individual stage behavior.

A collection-level count is not proof that a new FDD was indexed. Exact
deterministic IDs plus `unit_id`, release, document-family, content-hash, and
cache-key payload checks prevent an unsupported archival claim.

### Production interpretation

The master processes documents sequentially but bounds one embedding API call
by retrieval units, not by documents. Smaller request batches reduce the blast
radius of a transient API failure but increase requests, latency, and possibly
cost. Repeated runs reuse unchanged cached vectors; altered text, chunking,
artifact version, or embedding model requires re-embedding.

The workflow is not an all-or-nothing transaction across OpenAI, Qdrant, and
filesystem archival. A failure before archival leaves source DOCX files in
`data/raw_specs/`; Qdrant/cache artifacts may be partially written but are
safe to reconcile through deterministic IDs and exact verification before any
archive move. A filesystem failure during a multi-file archive can leave an
already-verified batch partly archived and requires operator reconciliation.

### Failure-mode testing

- A missing `data/raw_specs/` batch fails before external actions.
- Dry run invokes no child process and moves no source document.
- A simulated Qdrant-stage subprocess failure leaves the DOCX in
  `data/raw_specs/`.
- An existing archive destination blocks all stages and avoids overwriting a
  prior source document.
- A missing Qdrant point fails exact verification.
- Non-positive embedding API batch size is rejected.

### Validation

- Focused embedding/Qdrant/master tests: 18 passed.
- Actual `--dry-run` with the current empty `data/raw_specs/` directory failed
  safely before API, Qdrant, or archive actions.
- Full regression suite: 336 passed; one existing non-failing Starlette/HTTPX
  deprecation warning remains.
- No live OpenAI embedding request, Qdrant write, or source-document move was
  performed for this implementation step.

## Step 105 – Duplicate-content embedding safety and explicit Qdrant rebuild

### Objective

Correct the runtime failure found during the first full R21 ingestion: identical
chunk content created one embedding cache key but persisted different vectors,
which also exposed that Qdrant point IDs were not citeable-unit unique.

### Files added or changed

- `app/embeddings/client.py`
- `app/vectorstore/qdrant_upsert.py`
- `scripts/run_embedding_smoke_test.py`
- `scripts/run_qdrant_indexing.py`
- `scripts/master_ingestion_embedding_docs.py`
- `docs/Steps_for_FDD_Ingestion.md`
- `tests/test_embedding_client.py`
- `tests/test_embedding_artifact_quarantine.py`
- `tests/test_qdrant_upsert.py`
- `tests/test_qdrant_script_client_cleanup.py`
- `tests/test_master_ingestion_embedding_docs.py`

### Python/code pattern used

```python
grouped_records = _group_uncached_records_by_cache_key(uncached_records)
for _, representative, matching_records in unique_request_records:
    vector = embed(representative.text)
    for index, record in matching_records:
        updated_records[index] = replace(record, vector=vector)

point_id = uuid5(
    NAMESPACE_URL,
    json.dumps({"cache_key": record.cache_key, "unit_id": record.unit_id}),
)
```

### What the code does

- Deduplicates identical uncached content before calling the embedding API and
  copies the one resulting vector to every matching retrieval unit.
- Retains a content-based cache key for cost-efficient reuse, but builds each
  Qdrant point ID from both cache identity and `unit_id` so duplicate text in
  separate chunks/releases remains separately citeable.
- Keeps persisted cache-conflict detection strict; it does not silently choose
  between conflicting stored vectors.
- Adds `--replace-existing-artifact`, which quarantines a specifically selected
  prior artifact outside the active cache glob rather than deleting it.
- Adds destructive Qdrant rebuild support only behind both `--rebuild` and
  `--confirm-rebuild`; the master exposes this only as `--rebuild-qdrant`.
- Documents the explicit, reviewable recovery command with dry-run first.

### Why it is implemented this way

Embedding-vector reuse and vector-store identity serve different purposes.
Identical content can share one vector to reduce cost, but each document/release
occurrence needs independent payload metadata, retrieval filtering, and
citations. A point ID based only on content can overwrite another occurrence
and create a false lineage claim.

The original R21 artifact contains three records with one cache key and three
different vector fingerprints. Selecting one silently would hide an integrity
incident. Explicit quarantine plus regeneration makes the recovery visible and
preserves the artifact for investigation.

### Production interpretation

Changing the deterministic point-ID scheme requires a Qdrant rebuild; otherwise
old point IDs remain and may produce duplicate or stale retrieval evidence. The
rebuild is deliberately opt-in because it deletes only the configured local
collection. It never deletes FDD source, processed artifacts, active embedding
cache, or quarantined conflicting artifacts.

Before executing recovery, back up or otherwise retain the local Qdrant state
if operational policy requires it, review the dry-run command, and confirm that
the selected raw FDD batch is the intended scope. The actual recovery will make
new OpenAI embedding requests and therefore has cost.

### Failure-mode testing

- Three duplicate-content units make one API request and receive the same
  resulting vector in deterministic tests.
- Duplicate-content units create distinct Qdrant point IDs and persist as two
  points rather than overwriting one payload.
- The artifact quarantine preserves the original file outside the active cache.
- `--rebuild` without `--confirm-rebuild` exits before collection access.
- The real R21 recovery dry run lists artifact replacement and Qdrant rebuild
  commands without API calls, Qdrant writes, or source moves.

### Validation

- Focused duplicate-content/cache/Qdrant/master suite: 35 passed.
- Real rebuild-guard and master recovery dry-run checks: passed without
  mutation.
- Full regression suite: 343 passed; one existing non-failing Starlette/HTTPX
  deprecation warning remains.
- The live R21 recovery command was not run and awaits explicit user approval.

## Step 106 – Preserve duplicate evidence units and version the Qdrant collection

### Objective

Correct the second R21 runtime failure: the Qdrant indexer reused the
content-cache dictionary and therefore discarded five repeated-content evidence
units before upsert. Correct the unsafe embedded-Qdrant rebuild assumption and
separate full document identity from lineage family/release metadata.

### Evidence from the failed live run

- The R21 artifact had 149 records, 149 unique `unit_id` values, and 149 unique
  point IDs, so filename/release parsing did not cause the missing IDs.
- R21 had 145 unique content cache keys. Exactly five units were absent from
  Qdrant because the indexer loaded one record per cache key.
- A disposable persistent-Qdrant probe inserted one old point, deleted and
  recreated the collection, then inserted one new point; it still counted two
  points. Embedded local Qdrant delete-and-recreate cannot support a truthful
  in-place rebuild guarantee.

### Files added or changed

- `app/embeddings/embedding_cache.py`
- `app/embeddings/embedding_contract.py`
- `app/ingestion/filename_parser.py`
- `app/ingestion/retrieval_ready_artifact.py`
- `app/vectorstore/qdrant_indexer.py`
- `app/vectorstore/qdrant_upsert.py`
- `app/llm/answer_contract.py`
- `scripts/check_qdrant_index.py`
- `scripts/run_qdrant_indexing.py`
- `scripts/master_ingestion_embedding_docs.py`
- `docs/Steps_for_FDD_Ingestion.md`
- parser, artifact, embedding-cache, Qdrant-indexer, Qdrant-script, and master
  regression tests.

### Python/code pattern used

```python
# Cache lookup: one canonical vector per exact content key.
cache = {record.cache_key: record for record in load_embedding_records(path)}

# Indexing: retain every citeable occurrence, including repeated text.
records = load_embedding_records(path)

document_id = full_filename_without_docx
document_family = cross_release_lineage_family
release_label = "R21"
```

### What the code does

- Adds `load_embedding_records`, which validates vector consistency but returns
  every usable record. Cache lookup remains deduplicated; Qdrant indexing no
  longer is.
- Adds `document_id` as the full filename without `.docx` to new ingestion,
  embedding, Qdrant payload, and citation metadata while preserving
  `document_family` and `release_label`.
- Makes Qdrant verification include the document identifier for newly generated
  artifacts.
- Disables `--rebuild` and `--rebuild-qdrant` with a safe error before any
  collection access.
- Documents a versioned-collection migration: set a new
  `QDRANT_COLLECTION_NAME`, build/verify it, then point the API/UI to it.

### Why it is implemented this way

Content-vector reuse and evidence occurrence indexing have different identity
requirements. A dictionary keyed by content is appropriate for cache lookup but
wrong for a citation-bearing vector index. Each occurrence must survive upsert.

Multiple FDDs can share one R21 release. Replacing `document_family` with the
full filename would prevent valid cross-release lineage grouping; a distinct
`document_id` retains full-source identity without collapsing the other axes.

### Production interpretation

The old `functional_specs` collection is now considered legacy/stale because it
contains mixed point-ID generations. Preserve it for investigation and rollback
but do not use it for new grounded answers. Create a new versioned collection,
such as `functional_specs_v2`, through the `.env` configuration and validate it
before switching application traffic.

Existing old embedding artifacts do not yet contain the new optional
`document_id`; their `unit_id` still retains the full filename. Reprocessing a
source document upgrades its artifact and payload without requiring semantic
guesses. Do not claim complete metadata migration until all intended source
documents have been reprocessed.

### Failure-mode testing

- Duplicate-content records remain distinct in the Qdrant indexing batch and
  become distinct points.
- Legacy rebuild CLI flags fail before collection access.
- A disposable persistent-Qdrant probe demonstrates old-point retention after
  delete-and-recreate, proving why versioned collections are required.
- Full filename identity is present separately from family and release in new
  retrieval-ready artifacts.

### Validation

- Focused identity/cache/indexing/master suite: 40 passed.
- Full regression suite: 345 passed; one existing non-failing Starlette/HTTPX
  deprecation warning remains.
- No additional live OpenAI call, Qdrant upsert, archive action, or `.env`
  collection-name change was performed after the failed recovery run.

### Live recovery result

After setting a new local `QDRANT_COLLECTION_NAME` and reviewing the dry run,
the user ran `scripts/master_ingestion_embedding_docs.py` successfully. The
master workflow completed as expected on the new versioned collection. This
establishes the collection cutover/build path; it does not itself replace the
separate SME-reviewed retrieval and citation evaluation required before wider
FDD expansion.

## Step 107 — Four-FDD batch preflight and safe rejection

### Objective

Validate the newly staged four-FDD R1 batch before any paid embedding request,
Qdrant write, source archival move, or API/UI collection change.

### Staged batch

- `FS_FCIS_14.4.0.0.0$ASNB_R1_Cheque_Processing_v1.1.docx`
- `FS_FCIS_14.4.0.0.0$ASNB_R1_FinancialPlan_v1.1.docx`
- `FS_FCIS_14.4.0.0.0$ASNB_R1_Fund_Rule_v1.2.docx`
- `FS_FCIS_14.4.0.0.0$ASNB_R1_NFL_Enhancement_v1.1.docx`

### Python/code used

```python
from pathlib import Path
from app.ingestion.filename_parser import parse_document_filename

documents = sorted(Path("data/raw_specs").glob("*.docx"))
assert len(documents) == 4

seen_document_ids: set[str] = set()
for path in documents:
    parsed = parse_document_filename(path)
    assert parsed.document_id not in seen_document_ids
    seen_document_ids.add(parsed.document_id)
    print(parsed.document_id, parsed.document_family, parsed.release_label)
```

The read-only master plan was also executed with:

```powershell
& .\.venv\Scripts\python.exe scripts/master_ingestion_embedding_docs.py --dry-run
```

### What the preflight proved

- Four DOCX files are staged and each has a unique full-source `document_id`.
- All four map to `FS_FCIS_14.4.0.0.0$ASNB` and release `R1`; same-release,
  same-family FDDs are valid when `document_id` remains distinct.
- The master will ingest the batch, embed all units per document, index the
  resulting active artifacts, exact-verify all four artifacts, and archive a
  source only after those checks succeed.
- The dry run made no OpenAI request, Qdrant write, or source move.

### Failure-mode test

```python
parse_document_filename("FS_FCIS_14.4.0.0.0$ASNB_RX_Invalid.docx")
```

This raised the expected `ValueError` before ingestion. It proves a malformed
release label cannot silently enter a release-aware lineage index.

### Production interpretation

The four FDDs will be separate citeable document occurrences even though they
share a family and release. This prevents one R1 module from overwriting or
being cited as another. Dry-run review contains no cost or storage guarantee;
the live run will call the embedding provider for cache misses and may archive
only fully verified source files.

### Gate

Preflight accepted. Await the user's interview answers before authorizing the
live master ingestion command.

## Step 108 — Clean versioned-collection reconstruction from cached embeddings

### Objective

Correct the discovered collection-routing error without repeating paid
embeddings: build a clean `functional_specs_v2` from all active local embedding
artifacts and prove exact structural coverage.

### Configuration correction

The live four-FDD run showed `functional_specs` because the local `.env`
explicitly configured that legacy name. The prior versioned collection did not
exist. The local setting was changed to:

```env
QDRANT_COLLECTION_NAME=functional_specs_v2
```

The effective runtime setting was then printed from `get_settings()` and
confirmed as `functional_specs_v2` before indexing.

### Python/code used

```powershell
# Reuse existing artifacts: no DOCX parsing and no embedding API call.
& .\.venv\Scripts\python.exe scripts/run_qdrant_indexing.py

# Verify every active artifact, not only the four newest ones.
$artifactPaths = Get-ChildItem data\cache\embeddings -Filter '*.embeddings.json'
$verifyArgs = foreach ($artifactPath in $artifactPaths) {
    '--embedding-artifact'; $artifactPath.FullName
}
& .\.venv\Scripts\python.exe scripts/check_qdrant_index.py @verifyArgs
```

### Result

- `functional_specs_v2` was created/updated with 579 attempted and 579
  upserted evidence occurrences.
- Exact verification passed for all 9 active embedding artifacts and all 579
  expected records.
- The new collection has 579 points, 3072-dimensional vectors, and cosine
  distance.
- No OpenAI request, DOCX re-ingestion, or source archive move was needed for
  this reconstruction.
- The legacy `functional_specs` collection remains preserved with 591 points
  and is not the configured target.

### Failure-mode test

```powershell
$env:QDRANT_COLLECTION_NAME = 'functional_specs_negative_verification_test'
& .\.venv\Scripts\python.exe scripts/check_qdrant_index.py `
  --embedding-artifact data\cache\embeddings\FS_FCIS_14.4.0.0.0$ASNB_R1_Cheque_Processing_v1.1.embeddings.json
```

The verifier returned exit code 1 with `Qdrant collection does not exist;
cannot verify the requested embedding artifacts.` No replacement collection was
created. This demonstrates fail-closed verification.

### Production interpretation

The point count alone is insufficient; exact artifact-to-point verification
proves all intended source occurrences are present in the selected collection.
The application processes must be restarted before they pick up the changed
environment configuration. Structural verification does not prove answer
correctness, citation entailment, or current-state synthesis; those require the
next retrieval and SME-reviewed evaluation step.

### Gate

Step 108 implementation is complete. Await the user's interview answers before
restarting clients or running answer-quality evaluation.

## Step 109 — Duplicate raw-versus-archive source guard

### Objective

Prevent a reviewed FDD that has already been successfully archived from being
accidentally copied back into `data/raw_specs/` and triggering duplicate
ingestion, embedding cost, or duplicate evidence occurrences.

### Python/code change

```python
archived_files_by_casefolded_name = {
    path.name.casefold(): path
    for path in archive_directory.iterdir()
    if path.is_file()
}
conflicts = [
    archived_files_by_casefolded_name[document.file_name.casefold()]
    for document in documents
    if document.file_name.casefold() in archived_files_by_casefolded_name
]
if conflicts:
    raise FileExistsError("Refusing to ingest duplicate FDD filename(s) ...")
```

The pre-existing master preflight was strengthened to compare raw and archive
filenames case-insensitively. It runs before command construction and before
the `--dry-run` return, so no child process can ingest, embed, index, verify,
or move a colliding source.

### Failure-mode testing

```python
master_ingestion_embedding_docs.main(["--dry-run"])
```

The regression creates a raw DOCX and an archive DOCX with the same filename
but different casing. It expects `FileExistsError`, asserts the raw source
remains in place, and makes any attempted child process fail the test.

### Validation

- Focused master-ingestion regression suite: 7 passed.
- `git diff --check`: passed; only existing LF-to-CRLF working-tree warnings
  were emitted.
- The runbook now tells operators to remove an accidental raw duplicate or
  investigate the archive; it never instructs overwriting archived source.

### Production interpretation

Filename identity is a low-cost idempotency boundary for this manual-drop
workflow. It prevents a common operator error, but it is not a content-hash or
SVN-revision identity system: a genuinely revised FDD needs a distinct reviewed
filename/release and will be handled by the planned manifest-based FDD/code
work.

### Next bounded plan

1. Restart API and UI processes so they load `functional_specs_v2`.
2. Run one known retrieval/citation smoke query per newly added R1 FDD plus one
   unsupported query to verify refusal behavior.
3. Create a small SME-reviewed, document-specific evaluation set that includes
   cross-document confusion and citation checks; do not rely on generic R1
   questions.
4. Run retrieval and answer-trace evaluation, analyse failures, and only then
   stage the next 3–5 deployed delta FDDs across additional releases.

No interview questions were requested for this validation-only step.

## Step 110 — Versioned FDD grounded-evaluation runner and draft gate

### Objective

Turn the 30-case `data/evaluations/fdd_grounded_eval_v1.jsonl` asset into a
repeatable evaluation run that uses the real retrieval-to-grounded-answer path,
records local answer traces, deterministically checks response/citation
contracts, and leaves semantic claim correctness to SME review.

### Manifest preflight

```python
records = [json.loads(line) for line in eval_file.read_text().splitlines() if line]
assert len(records) == 30
assert len({record["case_id"] for record in records}) == len(records)
```

- The file contains 30 consistent JSONL cases and 6 abstention cases.
- It covers R1, R2, R18, R21, and R24 evidence expectations.
- All cases currently have `sme_reviewed=false` and
  `review_status=pending_sme_approval`; it is a draft baseline, not a release
  quality gate.

### Files added

- `app/llm/fdd_grounded_evaluation.py`
- `scripts/run_fdd_grounded_eval.py`
- `tests/test_fdd_grounded_evaluation.py`
- `tests/test_fdd_grounded_eval_script.py`

### Python/code pattern

```python
require_reviewed_cases(cases, allow_unreviewed=args.allow_unreviewed)
orchestration = run_grounded_answer_query(..., limit=10)
result = evaluate_fdd_grounded_response(case, orchestration.answer_response)
```

The runner validates JSONL fields and abstention shape, refuses unreviewed
cases by default, executes the same grounded-answer orchestration used by the
application only when authorized, writes isolated answer traces and a report,
and checks answer/refusal state plus required `document_id` and release
citations. It reports expected claims for an SME to review but never calls an
LLM judge or treats string matching as factual entailment.

### Failure-mode testing

Running an unreviewed case without `--allow-unreviewed` exits with the expected
rejection. Running with `--allow-unreviewed --dry-run --max-cases 3` prints the
three planned cases against `functional_specs_v2` without any OpenAI or Qdrant
call.

### Validation

- Focused grounded-evaluation tests: 6 passed.
- `git diff --check`: passed; only existing LF-to-CRLF working-tree warnings
  were emitted.

### Production interpretation

An explicit draft flag prevents an operator from presenting unreviewed expected
claims as an acceptance metric. The runner can measure citation/abstention
contract regressions deterministically; only SME review can confirm that the
answer entails all intended claims and omits unsupported ones.

### Gate

Await the user's decision: mark all cases SME-approved and run the quality gate,
or authorize `--allow-unreviewed` for a clearly labelled, non-gating draft
baseline. No live 30-case model evaluation has been executed yet.

### Draft baseline execution and result

The user ran:

```powershell
& .\.venv\Scripts\python.exe scripts/run_fdd_grounded_eval.py --allow-unreviewed
```

The resulting local report is
`data/exports/evaluations/fdd-grounded-eval-20260803T020554Z.json`.

- Target collection: `functional_specs_v2`; retrieval mode: hybrid; limit: 10.
- This remains a draft baseline because all 30 cases are pending SME approval.
- Structural result: 14/30 passed; 16/30 failed.
- 29 cases answered and only 1 abstained, although 6 cases expected abstention.
- Eleven answered cases failed required document-ID and/or release-citation
  checks; five abstention cases answered instead of refusing.
- The report's estimated LLM cost is `0.0` because configured price fields are
  zero; that is not proof that the live provider calls were free.

### Evidence-led failure classification

- R24 and R2 citations often carried an empty or null `document_id` while
  retaining the release label. This is consistent with legacy artifacts that
  predate document-ID payload backfill and prevents exact-source validation.
- Other confusion cases retrieved the wrong release (for example R1/R18/R21
  evidence where R18, R24, or R1 was expected). That is a retrieval/current
  release-selection defect, not merely a missing payload field.
- Five unsupported cases set `is_answered=true` without a refusal reason. This
  is a grounded-RAG safety defect and must be fixed before expansion.

### Decision

The draft run is useful diagnostic evidence but fails every planned acceptance
gate. Do not mark the evaluation cases SME-approved or ingest the next FDD batch
until identity metadata, release selection, and abstention failures are
investigated and measured again.

## Step 111 — Direct-support decision and six-case abstention rerun

### Objective

Prevent high-scoring but merely related evidence from being presented as an
answer, while allowing a clearly labelled redirecting abstention that helps the
user ask a more evidence-aligned question.

### Evidence-led diagnosis

The first draft-run traces showed that five unsupported cases passed the numeric
sufficiency threshold because they retrieved related terms: investment limits
for an interest-rate question, a document date for an implementation-date
question, and a reference to attached layouts for exact field-position
questions. The answer-generation code set `is_answered=true` whenever numeric
sufficiency passed, so the model had no machine-readable way to decline.

### Python/code change

```python
decision, answer = _parse_grounded_decision(completion.content)
if decision is None:
    return safe_refusal("Grounded answerability decision was missing or invalid.")

GroundedAnswerResponse(
    answer=answer,
    is_answered=decision,
    refusal_reason=None if decision else "No direct evidence supports every material part ...",
    citations=prompt.citations,
)
```

The prompt now requires a first-line `DECISION: ANSWER` only when evidence
directly supports every material part of the question, or `DECISION: REFUSE`
when evidence is only related, a requested value/date is absent, or attachment
content is not extracted. Invalid/missing decisions fail closed. A repeatable
`--case-id` CLI option was added so a bounded rerun cannot accidentally execute
the whole paid manifest.

### Validation and live rerun

- Focused decision/prompt/evaluation tests: 19 passed.
- Focused selector/answer-generation tests: 7 passed.
- A dry run selected exactly `abstain-001` through `abstain-006`.
- The six-case live draft rerun wrote
  `data/exports/evaluations/fdd-grounded-eval-20260803T071504Z.json` and passed
  structural abstention checks: 6/6, all with `is_answered=false`.
- Five cases used the new model-level redirect path; one used the existing
  below-threshold refusal without a chat completion.

### Remaining usability finding

The six responses contain safe refusals and most include labelled related
evidence, but none contains an explicit suggested next question. The prompt
instruction alone did not reliably deliver that UX. The next repair should add
a deterministic, clearly labelled follow-up-question fallback for both
model-level and score-threshold refusals, then rerun the six cases only after
that change is tested.

### Production interpretation

Direct support is a different decision from semantic retrieval score. This
change makes unsupported answers fail closed even when retrieval is related.
It does not prove factual claim entailment for answered cases, nor does it
repair missing legacy `document_id` metadata or release-selection failures.

## Step 112 — Enforced helpful recovery guidance for refusals

### Objective

Complete the redirecting-abstention experience: every safe refusal should give
the user a clearly labelled next question, even if the model omits that section
or the response was produced by the score-threshold path without a model call.

### Python/code change

```python
def _ensure_refusal_follow_up(answer: str) -> str:
    if "suggested next question:" in answer.casefold():
        return answer
    return f"{answer.rstrip()}\n\nSuggested next question: Ask about a named " \
        "function, report, field, or release explicitly described in the cited evidence."
```

- The model prompt now requires every `DECISION: REFUSE` to end with
  `Suggested next question:`.
- A deterministic fallback appends safe generic guidance when the model omits
  it, while preserving a model-provided suggestion exactly once.
- The existing below-score refusal now includes the same guidance.

### Validation and live rerun

- Focused answer-generation/prompt/evaluation suite: 20 passed.
- The exact six abstention cases ran again under `--allow-unreviewed` and wrote
  `data/exports/evaluations/fdd-grounded-eval-20260803T083217Z.json`.
- Structural abstention result: 6/6 passed with `is_answered=false`.
- Persisted-output verification confirmed `Suggested next question:` for every
  one of the six cases, including the below-score path.
- The bounded live run made six embedding calls and five chat-completion calls;
  configured estimated cost remains zero only because price configuration is
  not populated.

### Production interpretation

This is a safer user-assistance pattern: the response never treats related
evidence as a direct answer, yet provides a recovery path. The generic fallback
is intentionally conservative; it should not fabricate a specific question
from unsupported details. It does not solve unrelated positive-answer citation
identity or release-selection failures from the full draft run.

## Step 113 — R21 table retrieval linkage diagnosis

### Reported grounded-answer failure

For the R21 CIF Data Correction question, the application refused because its
selected evidence included the paragraph that introduces a following list but
not the list itself. The source DOCX visibly contains a table listing Race,
Religion, residential address fields, PEP Status, and mailing address fields.

### Read-only artifact evidence

```python
for unit in r21_retrieval_ready_artifact["units"]:
    if unit["unit_id"].endswith("table_chunk_10"):
        print(unit["source_kind"], unit["text"])
```

The R21 retrieval-ready artifact contains `table_chunk_10` with the complete
eleven-field list as a `source_kind=table` unit. Its preceding `chunk_33`
contains the shared query context: `System will allow user to perform CIF data
correction for following Data types or fields ...`.

### Failure-mode measurement

A no-cost lexical query probe ranked the preceding R21 paragraph first. The
matching R21 table ranked 244th of 628 candidates because its standalone text
lacks CIF, bulk-patching, and unit-holder context. It cannot reliably reach a
top-10 hybrid candidate set despite correct extraction and embedding.

### Design conclusion

This is not missing table ingestion, broken embedding linkage, or proof that
weighted-RRF must be replaced. It is an ingestion-modeling gap: standalone
table content lacks the parent section/preceding semantic anchor required for
retrieval.

The next bounded repair should preserve original table text for citations but
add structured parent/section context for retrieval and an explicit
paragraph-to-following-table relationship. It must be tested on this R21 case,
re-embedded from versioned artifacts, indexed into a new collection generation,
and evaluated without weakening citation provenance.

## Step 114 — Deterministic parent-table retrieval relationship model

### Objective

Implement the R21 diagnosis as a general ingestion model without changing
weighted-RRF: retain original table evidence for citations, use parent context
only for retrieval representation, and record a stable parent paragraph chunk.

### Python/code pattern

```python
# DOCX extraction preserves the nearest preceding top-level paragraph.
ExtractedTable(
    preceding_paragraph_index=preceding_index,
    preceding_paragraph_text=preceding_text,
)

# Retrieval-ready table keeps source text separate from search representation.
retrieval_text = f"Parent context: {context}\n\nTable:\n{table.text}"
parent_unit_id = _find_parent_unit_id(paragraph_chunks, preceding_paragraph_index)
```

- Raw DOCX paragraph/table order is captured deterministically at extraction.
- Normalized paragraph chunks retain original paragraph-index ranges.
- Each table can now reference its preceding parent chunk through
  `parent_unit_id`.
- `text` remains the original citeable table text; `retrieval_text` is the
  context-enriched embedding/lexical representation.
- Embedding records hash/embed `retrieval_text`, while Qdrant and lexical
  result payloads preserve original `text` for citations and expose the
  relationship metadata for inspection.
- Existing artifacts remain backward compatible: absent `retrieval_text` falls
  back to original text and absent `parent_unit_id` is `None`.

### Failure-mode testing

- Initial focused suite caught a backward-compatibility failure in manually
  constructed lexical test documents; default values and fallback scoring fixed
  it before any artifact migration.
- Final focused ingestion/embedding/lexical/Qdrant contract suite: 35 passed.
- `git diff --check`: passed; only existing LF-to-CRLF warnings were emitted.

### Real R21 validation without mutation

The archived R21 DOCX rebuilt in memory produces:

```text
table_chunk_10 -> parent_unit_id ...::chunk_33
retrieval_context_present=True
citation_text=<original eleven-field table only>
```

For the reported CIF Data Correction query, the original lexical table rank was
244/628. The context-linked in-memory rebuild ranks the same table 2nd in the
top-10 candidate set. No OpenAI call, artifact write, Qdrant write, source move,
or collection change occurred in this validation.

### Production interpretation

The repair models document structure rather than gaming a score. It makes the
table retrievable through the semantics that introduce it while retaining a
precise table citation. The current `functional_specs_v2` artifacts/vectors do
not yet contain this representation; activation requires an explicit archived-
source reprocessing workflow, a new versioned collection, exact verification,
and the R21 positive/negative evaluation gate.

### Gate

Step 114 implementation is complete. Await the user's interview answers before
designing the controlled artifact migration and collection activation.

## Step 115 — Isolated all-FDD staged rebuild workflow

### Objective

Create a controlled migration path for all eight archived FDDs so the
parent-table retrieval representation can be materialized without overwriting
the live `functional_specs_v2` collection, active artifacts, or API/UI
configuration.

### Python/code pattern

```python
# The active cache is read-only seed input. Only changed retrieval text becomes
# an OpenAI embedding request; unchanged units reuse its validated v1 vector.
embedded_batch = embed_batch(
    batch,
    cache_directory=settings.cache_dir / "embeddings",
    request_batch_size=64,
)
write_embedding_batch_to_json(embedded_batch, stage_directory / "cache" / "embeddings")

# A fresh collection is required and is verified point-by-point before any
# later cutover decision.
index_embedding_cache_directory(client, QdrantCollectionConfig("functional_specs_v3", 3072), staged_cache)
verify_embedding_artifacts(client=client, collection_name="functional_specs_v3", artifact_paths=paths)
```

Added `scripts/stage_archived_fdd_rebuild.py` and
`tests/test_stage_archived_fdd_rebuild.py`.

- Inputs are the eight immutable DOCX files in `data/docs_embedded/`.
- The script hashes every source and writes a source/operation manifest to the
  new `data/staging/table_context_v1/` directory only on a real run.
- It rebuilds processed/retrieval-ready artifacts there, with the Step 114
  parent-table relationship representation.
- It reads the active embedding cache as a seed but writes newly produced
  embedding artifacts only under the staged directory. Because `artifact_version`
  and embedding model remain `v1`/`text-embedding-3-large`, unchanged retrieval
  text can reuse its existing vector; context-enriched tables get a different
  content hash and are embedded anew.
- It refuses the active collection name, any existing target collection, and
  any existing stage directory. It never deletes or reuses a collection.
- It validates every vector's dimension before Qdrant indexing, then verifies
  every deterministic point ID and payload from each staged artifact.
- On a real failure it records a failed manifest and leaves the partial stage
  and any partial new collection for investigation; the operator must choose a
  new generation rather than silently overwrite them.

### Failure-mode testing

- Dry-run test: hashes archived source inputs without creating a stage.
- Active collection test: fails before stage writes.
- Existing target collection test: fails without deleting/reusing it.
- Vector-dimension test: fails before Qdrant indexing.
- Focused suite: `14 passed` covering the staged workflow, embedding cache,
  exact Qdrant verification, and retrieval-ready contracts.
- `git diff --check` passed with no whitespace errors.

### Real dry-run evidence

```text
sources=8
stage=data/staging/table_context_v1
target_collection=functional_specs_v3
cache_seed=data/cache/embeddings
DRY RUN complete: no artifacts, OpenAI calls, Qdrant writes, or configuration changes.
```

All eight archived source SHA-256 values and sizes were printed by the dry run.

### Production interpretation

This is a generation-builder, not a cutover. It contains re-embedding cost to
the exact retrieval units whose text changed, captures reproducible source
identity, and protects the current service from a partial rebuild. The next
operation, only after review of this step, is a paid isolated staging run;
after that, retrieval/evaluation and explicit configuration activation remain
separate gates.

### Gate

Step 115 implementation and no-cost dry run are complete. Do not run the paid
staging command or change `QDRANT_COLLECTION_NAME` until the interview check
is accepted.

## Step 116 — Separate embedding-input compatibility from index generation

### Objective

Remove the ambiguous meaning of “version” from the staged rebuild before any
paid embedding/indexing operation. An index generation must not be mistaken for
an attempt to transform old vectors into a new embedding space.

### Python/code pattern

```python
batch = build_embedding_batch_contract(
    retrieval_ready_artifact,
    embedding_model=embedding_model,
    # This legacy field now explicitly represents input compatibility,
    # not the new Qdrant generation.
    artifact_version=embedding_input_version,
)

manifest = {
    "embedding_input_version": "v1",
    "index_generation": "table_context_v1",
    "collection_name": "functional_specs_v3",
}
```

`stage_archived_fdd_rebuild.py` now exposes `--embedding-input-version` for
cache/input compatibility and `--index-generation` for the retrieval/index
generation. The staged manifest records both independently.

### Failure-mode testing

- Added a manifest test proving both version values persist independently.
- Focused staged-rebuild/cache/Qdrant verification suite: `14 passed`.
- The eight-FDD dry run reports `index_generation=table_context_v1` and
  `embedding_input_version=v1`, with no artifact writes, OpenAI calls, Qdrant
  point writes, or configuration change.
- `git diff --check` passed.

### Production interpretation

Cached-vector reuse is exact reuse, never conversion. It is permitted only for
identical retrieval text under the same model and input contract. The
parent-enriched table text differs and is embedded anew. Any changed model,
preprocessing contract, chunking, or retrieval text requires re-embedding the
affected corpus; incompatible embedding spaces must not be mixed.

### Gate

Step 116 is complete. The user explicitly skipped the interview check for this
naming-hardening step. The next bounded operation is the paid isolated all-FDD
staging rebuild, still with no API/UI cutover.

## Step 117 — Paid staged rebuild integrity failure

### Objective

Run the approved isolated all-FDD rebuild and stop safely if cache/index
integrity cannot be proven.

### Python/code and observed result

```powershell
& .\.venv\Scripts\python.exe scripts\stage_archived_fdd_rebuild.py
```

The run processed eight immutable sources and made successful OpenAI embedding
requests, but failed before Qdrant upsert because staged artifacts contained
conflicting vectors for an identical `cache_key`.

```text
records=937  cached=452  embedded=485
failure=Conflicting cached embeddings found for cache_key=...
```

### Failure-mode interpretation

The fail-closed cache contract prevented an ambiguous vector from entering the
index. The failed manifest is retained at
`data/staging/table_context_v1/stage_manifest.json`; the new empty
`functional_specs_v3` collection and all staged artifacts are preserved for
diagnosis. They must not be deleted, reused, or activated.

### Production interpretation

The attempted run incurred embedding cost but produced no usable index. This
is the correct outcome: a vector that could correspond to a different input
unit would silently corrupt grounded retrieval more severely than a hard stop.

## Step 118 — Cross-document cache and response-order hardening

### Objective

Prevent duplicate retrieval text across different FDDs from receiving multiple
independent vectors, and protect against a provider response arriving in a
different order from its request inputs.

### Python/code pattern

```python
# Map provider items by their declared input index, never response position.
response_items = _response_items_in_input_order(
    response.data,
    expected_count=len(request_records),
)

# Earlier staged outputs are read-only cache input for later source documents.
embedded_batch = embed_batch(
    batch,
    cache_directory=active_cache,
    additional_cache_directories=[staged_cache],
)
```

- Added response-index validation for missing, invalid, or duplicate indexes.
- Added compatible multi-directory cache loading with conflict rejection.
- Each written staged artifact is now a cache source for subsequent documents,
  so duplicate text receives one canonical vector across the generation.
- Added deterministic tests for out-of-order provider items and cross-document
  staged-cache reuse.

### Failure-mode testing

Focused embedding/cache/staging/Qdrant suite: `21 passed`. The tests prove
that out-of-order response data is mapped back to its request input and that a
later document reuses the first staged vector rather than making a duplicate
embedding call.

### Production interpretation

This is a correctness and cost repair. It does not assume embedding API output
is safely positional or repeatably identical across independent calls. It
builds one canonical vector per compatible cache key while retaining separate
document/unit Qdrant point identities for citations.

## Step 119 — Clean retry generation preflight

### Objective

Plan a retry without touching failed v3 state or the live v2 service.

### Python/code pattern

```powershell
& .\.venv\Scripts\python.exe scripts\stage_archived_fdd_rebuild.py `
  --dry-run `
  --stage-directory data\staging\table_context_v1_retry1 `
  --collection-name functional_specs_v4 `
  --index-generation table_context_v1_retry1
```

### Failure-mode testing

The dry run verified all eight archived source SHA-256 values, an absent retry
stage directory, and an absent target collection. It made no OpenAI calls,
Qdrant point writes, artifact writes, or API/UI configuration changes.

### Production interpretation

`functional_specs_v4` is a new generation, not a replacement of v3. A retry
requires a new explicit paid-operation approval because it will issue OpenAI
embedding requests again. v2 remains the live rollback target.

### Batch gate

Steps 117–119 are complete. Await the nine-question batch interview and a
separate explicit approval before the paid v4 retry.

### Batch interview evaluation

Pass. All nine answers meet the production rubric. The user correctly explained
why conflicting vectors fail closed, separated operational counters from
grounding proof, retained incident evidence, validated provider response order,
preserved distinct citation identities, rejected cache-source precedence,
protected failed-generation isolation, bounded dry-run evidence, and required
fresh approval for a new paid retry.

### Gate

Steps 117–119 are accepted. The next paid boundary remains the isolated v4
retry; do not activate or alter the current v2 API/UI configuration.

## Step 120 — Paid isolated v4 FDD rebuild

### Python/code

```powershell
& .\.venv\Scripts\python.exe scripts\stage_archived_fdd_rebuild.py `
  --stage-directory data\staging\table_context_v1_retry1 `
  --collection-name functional_specs_v4 `
  --index-generation table_context_v1_retry1
```

The approved rebuild completed successfully against all eight archived FDDs.
It wrote only the retry stage and `functional_specs_v4`; `.env` remains on
`functional_specs_v2`.

### Production interpretation

The retry is a separate immutable generation. It applies cross-document vector
reuse and response-index validation, but does not activate the new collection
or make any claim about answer quality.

## Step 121 — Staged v4 integrity and cost verification

### Python/code

```python
manifest = json.loads((stage / "stage_manifest.json").read_text())
assert manifest["status"] == "verified"
assert manifest["qdrant"]["verified_records"] == 937
assert client.count("functional_specs_v4").count == 937
```

### Verified evidence

```text
manifest_status=verified
records=937
cached=473
embedded=464
qdrant_verified_records=937
functional_specs_v4_points=937
staged_artifacts=8
conflicting_cache_keys=0
```

### Failure-mode testing

The earlier v3 failure had 485 newly embedded records and conflicting cache
keys. v4 has zero conflicts and 464 new embeddings, avoiding 21 duplicate API
embeddings while retaining 937 separate citeable evidence records.

### Production interpretation

Exact verification proves the intended staged artifact records and payloads
exist in v4. It does not yet prove hybrid ranking, answer grounding, citation
entailment, abstention, or production readiness.

## Step 122 — R21 parent-table staged lexical evidence check

### Python/code

```python
r21 = [unit for unit in documents if unit.release_label == "R21"]
results = search_lexical_documents(r21, positive_question, limit=len(r21))
rank = next(i for i, result in enumerate(results, 1) if result.point_id == table_id)
assert rank <= 10
assert "marital status" not in table_citation_text.lower()
```

### Verified evidence

For the reported CIF Data Correction question, the R21 table
`table_chunk_10` ranks `2` and is inside the top-10 bounded evidence set. Its
original citation text contains all eleven fields: Race, Religion, Residential
address Zip code/City/State/Country, PEP Status, and Mailing address Zip
code/City/State/Country.

For the deliberately unsupported `marital status` query, nearby R21 CIF
evidence still ranks, but `marital status` is absent from the citeable table
text. This verifies retrieval context does not establish unsupported fields;
the grounded answer layer must refuse that claim.

### Production interpretation

The parent-table repair solves the controlled lexical candidate gap without
altering the source citation. This is not yet a hybrid/LLM evaluation and does
not authorize API/UI activation.

### Batch gate

Steps 120–122 are complete. Keep `.env` on `functional_specs_v2`. Await the
nine-question interview before the next no-cost evaluation batch.

## Step 123 — Generation-coherent staged evaluation target

### Python/code

```python
target = resolve_evaluation_target(args=args, settings=settings)
if bool(args.collection_name) != bool(args.lexical_artifact_directory):
    raise ValueError("--collection-name and --lexical-artifact-directory must be supplied together")
```

`scripts/run_fdd_grounded_eval.py` now accepts paired
`--collection-name` and `--lexical-artifact-directory` overrides. This allows
v4 hybrid evaluation without changing live `.env`, and rejects a dangerous mix
of v4 dense vectors with v2 lexical artifacts. Focused staged-target tests:
`19 passed`.

## Step 124 — Full v4 automated evaluation preflight

### Python/code

```powershell
& .\.venv\Scripts\python.exe scripts\run_fdd_grounded_eval.py `
  --allow-unreviewed --dry-run `
  --collection-name functional_specs_v4 `
  --lexical-artifact-directory data\staging\table_context_v1_retry1\processed
```

The no-cost preflight validated all 30 JSONL cases against the paired v4 target.
All cases are `sme_reviewed=false`, so any following run is a draft baseline,
not a release-quality acceptance gate.

## Step 125 — Resumable v4 automated draft baseline

### Python/code

```python
# Reuse a durable interrupted trace only when it names the same case and query.
resumed_results = load_resumed_evaluation_results(cases, trace_directory)
cases_to_run = [case for case in cases if case.case_id not in resumed_results]
```

The first 30-case run reached the local command-time limit after eight durable
traces. Added `--resume-trace-directory`, which validates one trace per case
and reuses it only when the question and answer contract match. Focused resume
tests initially caught a missing `json` import before any retry call; after the
fix, `10 passed`. The resumed dry run scheduled only 22 new calls.

The consolidated v4 report is:
`data/exports/evaluations/fdd-grounded-v4-draft-20260805.json`.

```text
total_cases=30
structural_passed_count=23
claim_review_required_count=24
resumed_case_count=8
retrieval_mode=hybrid
estimated_llm_cost=0.0 (configuration has no price values; this is not billing proof)
```

All six expected abstentions structurally passed. Seven answered/cross-release
cases failed the deterministic contract:

- `lineage-r2-r18-002`
- `lineage-r24-006`
- `confusion-release-001`, `003`, `004`, `005`, `006`

Failures are direct evidence for targeted analysis: some safe abstentions
occurred where the draft expected an answer; others omitted required historical
or current-release citations. No retrieval weights, thresholds, prompts, or
live configuration were changed.

### Production interpretation

Automation provides repeatable structural evidence and preserves answer traces,
but it does not prove semantic entailment. The 24 answered cases require manual
SME review; the seven failures must be classified as incorrect expectations,
retrieval/release-selection gaps, citation-contract gaps, or valid safe
refusals before any correction is attempted.

### Batch gate

Steps 123–125 are complete. Keep v2 live. Perform a targeted manual review and
failure classification next; do not tune retrieval globally or activate v4 from
this draft baseline.

## Step 126 — Deterministic SME review packet for v4 failures

### Python/code

```powershell
& .\.venv\Scripts\python.exe scripts\export_fdd_manual_review_packet.py `
  --report-file data\exports\evaluations\fdd-grounded-v4-draft-20260805.json `
  --output-file data\exports\evaluations\fdd-grounded-v4-draft-20260805-manual-review.md
```

Added `scripts/export_fdd_manual_review_packet.py`. It joins each failed report
case to its source JSONL expectations and validates exactly one durable trace
from either the resumed or original run. The generated packet includes the
question, expected claims/releases/documents, actual answer/refusal, returned
citations, deterministic failures, trace path, and blank SME verdict fields.

### Failure-mode testing

- The exporter fails if a failed case is absent from the manifest or does not
  have exactly one trace, preventing ambiguous manual review evidence.
- Focused packet/evaluation suite: `11 passed`.
- Generated local packet covers exactly the seven structural failures.

### Production interpretation

The packet makes manual review repeatable and auditable without allowing an
automation result to masquerade as a semantic decision. SME verdicts must be
one of `expected_case_incorrect`, `retrieval_or_release_gap`,
`citation_contract_gap`, `correct_safe_refusal`, or `other`, with source-based
rationale.

### Gate

Step 126 is complete. The batch pauses for SME/manual review of the seven local
packet entries; do not change v4 retrieval or activation state before those
verdicts are recorded.

## Step 127 — Recorded normalized SME decisions for v4 draft failures

### Local artifact

Created `data/evaluations/fdd_grounded_eval_v4_sme_review_20260805.json` with
the seven normalized, source-based decisions:

- `expected_case_incorrect`: `lineage-r2-r18-002`,
  `confusion-release-005`
- `retrieval_or_release_gap`: `lineage-r24-006`,
  `confusion-release-001`, `confusion-release-003`,
  `confusion-release-004`, `confusion-release-006`

Each decision records rationale and a bounded required follow-up. The ledger
uses only the permitted review taxonomy and validates seven decisions.

### Production interpretation

The review distinguishes incorrect benchmark requirements from actual
retrieval/release-selection failures. This prevents changing weighted-RRF or
prompting to compensate for a bad expectation.

## Step 128 — SME-reviewed R18 reinvestment-consumption regression

### Python/code data contract

```json
{
  "case_id": "r18-minor-program-reinvestment-consumption-001",
  "required_citation_document_ids": ["...R18_Minor_Program_v1.3"],
  "expected_release_labels": ["R18"],
  "sme_reviewed": true,
  "review_status": "approved_by_sme_2026-08-05"
}
```

Appended the user-approved functionality-first question to
`data/evaluations/fdd_grounded_eval_v1.jsonl`. It tests R18 reinvestment
consumption behavior across Non-ADAM50 minors, ADAM50 Block/Non-Block cases,
and the Fund Rule MP-bucket restriction, requiring the R18 source and table
aware evidence.

### Production interpretation

This adds a realistic user-facing functional query rather than relying only on
release-labelled questions. The case is SME-reviewed, but its runtime answer
still needs structural and semantic evaluation.

## Step 129 — No-cost validation of the new targeted regression

### Python/code

```powershell
& .\.venv\Scripts\python.exe scripts\run_fdd_grounded_eval.py `
  --dry-run --case-id r18-minor-program-reinvestment-consumption-001 `
  --collection-name functional_specs_v4 `
  --lexical-artifact-directory data\staging\table_context_v1_retry1\processed
```

### Failure-mode testing

The review ledger validated exactly seven permitted verdicts. The evaluator
accepted the new case with `reviewed=1` and `draft=False`, confirming its
schema, review status, paired v4 target, and explicit case selection. No
embedding, LLM, Qdrant write, trace, or configuration change occurred.

### Batch gate

Steps 127–129 are complete. Keep v2 live. The next bounded action is one paid
targeted v4 run for the approved R18 reinvestment regression, followed by SME
review of its answer/trace before any broader correction.

### Batch interview evaluation

Pass. All nine answers meet the production rubric. The user correctly justified
durable decision evidence, benchmark correction before tuning, primary verdict
ownership, functionality-first/table-aware evaluation, the boundary between
SME benchmark review and runtime behavior, paired-generation testing, and
targeted cost-bounded diagnosis.

### Gate

Steps 127–129 are accepted. The next action is an explicit paid v4 runtime run
for `r18-minor-program-reinvestment-consumption-001`; retain v2 as live and do
not activate v4.

## Step 130 — Targeted v4 R18 reinvestment runtime evaluation

### Python/code

```powershell
& .\.venv\Scripts\python.exe scripts\run_fdd_grounded_eval.py `
  --case-id r18-minor-program-reinvestment-consumption-001 `
  --collection-name functional_specs_v4 `
  --lexical-artifact-directory data\staging\table_context_v1_retry1\processed `
  --output-file data\exports\evaluations\fdd-grounded-v4-r18-reinvestment-20260805.json
```

### Structural result

The one-case reviewed v4 run completed with `structural_passed=True`,
`is_answered=True`, and five R18 citations. The report and trace are stored in
`data/exports/evaluations/fdd-grounded-v4-r18-reinvestment-20260805.json` and
`data/exports/evaluations/fdd-grounded-eval-20260805T134140Z/answer_traces/`.

### Evidence review required

The answer correctly described the main Non-ADAM50 and ADAM50
Block/Non-Block reinvestment cases, but omitted the SME-required Fund Rule
condition: when Minor Program=Yes and Restrict Consumption to MP Bucket=Yes,
the system always consumes from the Minor Program bucket. It also added an
unrelated unit-holder-merge note.

This is not evidence absence: the trace retrieved R18 `chunk_18` and
`table_chunk_11`, which contain the missing condition. The current structural
evaluator checks answer state and document/release citations, not expected-claim
completeness, so an SME semantic verdict is required.

### Gate

Step 130 is structurally complete. Await SME classification before treating the
structural pass as semantic acceptance or considering v4 activation.

### SME semantic review

Accepted on 2026-08-05. The SME confirmed that the answer contained the
required information. The result is recorded in
`data/evaluations/fdd_grounded_eval_v4_sme_review_20260805.json` with the
report and trace paths. No retrieval, prompt, index, or configuration change is
justified by this accepted result.

### Production interpretation

Structural evaluation caught only answer-state and source/release citation
contracts; the SME review is the authority for whether the functional answer
actually meets the intended business requirement. An accepted single case is
useful regression evidence but does not establish full v4 readiness.

### Failure-mode testing

The ledger now ties the acceptance to the exact case, v4 report, and answer
trace. It prevents a later reviewer from treating a structurally passing trace
from another question or generation as this SME approval.

### Gate

Step 130 is accepted. Keep v2 live. Do not activate v4 until the remaining
reviewed lineage/release gaps have targeted regressions and the staged
generation passes the agreed evaluation gate.

## Step 131 — Evidence-based v4 release-gap diagnosis

### Python/code

```powershell
$env:PYTHONIOENCODING='utf-8'
@'
from app.retrieval.lexical_search import search_lexical_artifacts

results = search_lexical_artifacts(
    'data/staging/table_context_v1_retry1/processed',
    'Which current release contains the Teller and Branch Reports Re-alignment change, rather than the original R2 report specifications?',
    limit=10,
)
for rank, result in enumerate(results, start=1):
    print(rank, result.payload['release_label'], result.payload['unit_id'])
'@ | & .\.venv\Scripts\python.exe -
```

### Result and production interpretation

The saved v4 traces and a local lexical replay were classified in
`data/evaluations/fdd_v4_release_gap_diagnosis_20260805.json`. Two failures
are confirmed temporal-planning defects: a historical R2 mention inside a
current-state question was incorrectly turned into a hard R2 filter, excluding
R24 before fusion. The remaining three cannot be diagnosed reliably from old
traces because they stored only the final fused results, not raw dense and
lexical candidate lanes.

### Failure-mode testing

The diagnosis keeps confirmed findings distinct from unresolved ones. It does
not infer a global ranking problem from a fused trace that cannot show which
lane supplied or excluded a candidate. No OpenAI call, Qdrant write, embedding,
or configuration change occurred.

## Step 132 — Privacy-safe retrieval candidate-lane diagnostics

### Python/code

```python
"candidate_lanes": {
    "dense": _summarize_retrieval_candidates(routed.dense_candidates),
    "lexical": _summarize_retrieval_candidates(routed.lexical_candidates),
}
```

`RoutedRetrievalResult` now preserves the dense and lexical candidate lists.
New answer traces record only rank, point ID, unit ID, document ID, release,
source kind, and score for each lane; they intentionally do not duplicate
source text.

### Production interpretation

A future failed answer can now be classified as candidate generation, fusion,
temporal filtering, evidence packing, or generation behavior. The trace stays
auditable without creating an extra copy of potentially sensitive FDD text.

### Failure-mode testing

Router tests cover dense-only, lexical-only, and hybrid modes, including
candidate limits and lane membership. The absence of raw candidates in old
traces remains explicit; old artifacts are not reinterpreted as having this
new diagnostic evidence.

## Step 133 — Correct current-state historical-release scoping

### Python/code

```python
elif not is_current_state:
    query_releases = extract_release_labels(original_query)
    if len(query_releases) == 1:
        effective_release_label = query_releases[0]
        release_source = "query"
```

For current/latest queries, a release name written in the question no longer
becomes a hard retrieval filter. The service retrieves broad candidates and
then selects the highest relevant retrieved release. An explicit API
`release_label` parameter remains a deliberate filter.

### Production interpretation

This fixes the R24 lineage pattern without changing embeddings, source text,
weighted RRF weights, or unrelated historical-release questions. It addresses
candidate eligibility before ranking, which is safer than tuning global scores.

### Failure-mode testing

```powershell
& .\.venv\Scripts\python.exe -m pytest `
  tests/test_temporal_query.py `
  tests/test_retrieval_router.py `
  tests/test_answer_orchestration_service.py `
  --basetemp C:\tmp\fdd-v4-current-filter -p no:cacheprovider
```

Result: `17 passed`. New tests prove that an R2 baseline mention in a
current-state question leaves retrieval unfiltered and scopes the returned R2
and R24 candidates to R24. Existing tests retain explicit request-filter and
conversation-reference behavior.

### Batch gate

Steps 131–133 are complete. Keep v2 live. The next action is a bounded paid
v4 replay of the two repaired current-state cases, using the new candidate-lane
trace contract, followed by SME review. Do not rerun the unresolved
multi-release cases until those trace diagnostics establish their first broken
layer.

### Batch interview evaluation

Pass. All nine answers distinguish historical baseline from deployed state,
candidate eligibility from global ranking, lane-level observability from final
fusion, and deterministic regression tests from semantic/production proof. One
precision refinement: the runtime release-pinning authority is the explicit
API/request `release_label` filter; an SME or benchmark may define its value,
but a release mentioned only in prose is not that filter.

### Gate

Steps 131–133 are accepted. Await explicit approval for one paid, two-case v4
replay of `lineage-r24-006` and `confusion-release-004`. It must use paired
`functional_specs_v4` and `data/staging/table_context_v1_retry1/processed`,
and its new candidate-lane traces must be reviewed before any broader replay or
activation.

## Step 134 — Preserve historical evidence for multi-part current-state questions

### Python/code

```python
permitted_releases = {effective_release, *plan.referenced_release_labels}
scoped = [
    result
    for result in results
    if _normalized_payload_release(result) in permitted_releases
]
```

For a current-state query, textual release mentions are now retained as
historical evidence references, not retrieval filters. After broad retrieval,
the planner scopes evidence to the latest retrieved release plus any explicit
historical releases named in the question. `effective_release_label` still
states the latest deployed state; `referenced_release_labels` makes the
additional historical scope visible in the trace.

### Production interpretation

This supports questions such as “R2 PDF format and current T-1 name” with R2
and R24 evidence in one grounded answer. It does not interpret every prose
release name as an API filter; callers that need a single release retain the
explicit request `release_label` authority.

### Failure-mode testing

The regression includes R1, R2, and R24 candidates. It proves R1 is excluded,
R2 baseline evidence is preserved, and R24 remains the effective current
release. This prevents repairing stale current answers by accidentally losing
the historical sub-question.

## Step 135 — Deterministic multi-scope and trace-contract verification

### Python/code

```powershell
& .\.venv\Scripts\python.exe -m pytest `
  tests/test_temporal_query.py `
  tests/test_retrieval_router.py `
  tests/test_answer_orchestration_service.py `
  --basetemp C:\tmp\fdd-v4-multiscope -p no:cacheprovider
```

### Result and production interpretation

Result: `18 passed`. Tests cover current-state historical references,
multi-part R2/R24 scope preservation, explicit request filters, dense/lexical
candidate lanes, and trace metadata. They prove the deterministic temporal and
observability contracts, not live retrieval relevance or LLM answer quality.

### Failure-mode testing

The test uses an unrelated R1 candidate to confirm that preserving named R2
evidence does not disable release bounds. A malformed or missing raw lane is
still represented as an empty lane summary rather than silently inferred.

## Step 136 — Reviewed two-case v4 replay manifest and no-cost preflight

### Python/code

```powershell
& .\.venv\Scripts\python.exe scripts\run_fdd_grounded_eval.py `
  --dry-run `
  --eval-file data\evaluations\fdd_v4_current_state_replay_20260805.jsonl `
  --collection-name functional_specs_v4 `
  --lexical-artifact-directory data\staging\table_context_v1_retry1\processed
```

### Result and production interpretation

The dedicated manifest contains only the two SME-approved current-state
regressions and preserves the 30-case source manifest as a draft baseline. The
preflight reported `cases=2`, `reviewed=2`, and `draft=False`. This makes the
next paid run eligible to produce release-gate evidence rather than a draft
result.

### Failure-mode testing

An earlier dry run of the source manifest reported `reviewed=0` and
`draft=True`; it was not used as a quality gate. The dedicated manifest fixes
review governance without silently changing the original draft cases. The dry
run made no OpenAI, Qdrant, trace, or configuration changes.

### Batch gate

Steps 134–136 are complete. Keep v2 live. The next action is a single paid v4
run of the reviewed two-case manifest, followed by trace inspection and SME
review. Do not activate v4 from a structural pass alone.
