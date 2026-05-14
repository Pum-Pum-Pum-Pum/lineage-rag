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