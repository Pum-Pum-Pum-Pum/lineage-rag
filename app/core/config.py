from __future__ import annotations

from pathlib import Path

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


ROOT_DIR = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    """Centralized application settings for the RAG system."""

    model_config = SettingsConfigDict(
        env_file=str(ROOT_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "Culling Blade Lineage GenAI RAG System"
    environment: str = "dev"
    log_level: str = "INFO"

    root_dir: Path = ROOT_DIR
    data_dir: Path = ROOT_DIR / "data"
    raw_specs_dir: Path = ROOT_DIR / "data" / "raw_specs"
    raw_code_dir: Path = Field(
        default=ROOT_DIR / "data" / "raw_code",
        alias="RAW_CODE_DIR",
    )
    code_snapshots_dir: Path = Field(
        default=ROOT_DIR / "data" / "code_snapshots",
        alias="CODE_SNAPSHOTS_DIR",
    )
    code_staging_dir: Path = Field(
        default=ROOT_DIR / "data" / "staging" / "code",
        alias="CODE_STAGING_DIR",
    )
    code_indexes_dir: Path = Field(
        default=ROOT_DIR / "data" / "indexes" / "code",
        alias="CODE_INDEXES_DIR",
    )
    code_modes_enabled: bool = Field(default=False, alias="CODE_MODES_ENABLED")
    code_qdrant_local_path: Path = Field(
        default=ROOT_DIR / "data" / "qdrant_code_local",
        alias="CODE_QDRANT_LOCAL_PATH",
    )
    code_qdrant_collection_name: str = Field(
        default="code_custom_r1_v2", alias="CODE_QDRANT_COLLECTION_NAME"
    )
    code_index_artifact_path: Path = Field(
        default=(
            ROOT_DIR
            / "data/staging/code_embeddings/fci-custom-r1-b1c79c6dc2c5/"
            "code_index_text_embedding_3_large_v1/code_index_artifact.json"
        ),
        alias="CODE_INDEX_ARTIFACT_PATH",
    )
    code_analysis_directory: Path = Field(
        default=(
            ROOT_DIR
            / "data/staging/code/fci-custom-r1-b1c79c6dc2c5/"
            "plsql_antlr_4_13_2_analysis_v12"
        ),
        alias="CODE_ANALYSIS_DIRECTORY",
    )
    fdd_code_lineage_artifact_path: Path = Field(
        default=ROOT_DIR / "data/staging/fdd_code_lineage/neo_aml_v1/reviewed_lineage_artifact.json",
        alias="FDD_CODE_LINEAGE_ARTIFACT_PATH",
    )
    fdd_generation: str = Field(default="functional_specs_v5", alias="FDD_GENERATION")
    code_parse_timeout_seconds: float = Field(
        default=120.0,
        alias="CODE_PARSE_TIMEOUT_SECONDS",
        gt=0,
    )
    code_parse_memory_limit_mib: int = Field(
        default=1024,
        alias="CODE_PARSE_MEMORY_LIMIT_MIB",
        gt=0,
    )
    code_parse_max_segment_characters: int = Field(
        default=500,
        alias="CODE_PARSE_MAX_SEGMENT_CHARACTERS",
        gt=0,
    )
    code_retrieval_max_unit_characters: int = Field(
        default=6_000,
        alias="CODE_RETRIEVAL_MAX_UNIT_CHARACTERS",
        gt=0,
    )
    code_retrieval_overlap_characters: int = Field(
        default=400,
        alias="CODE_RETRIEVAL_OVERLAP_CHARACTERS",
        ge=0,
    )
    ingestion_source_policy_path: Path = Field(
        default=ROOT_DIR / "config" / "ingestion_sources.toml",
        alias="INGESTION_SOURCE_POLICY_PATH",
    )
    code_analysis_policy_path: Path = Field(
        default=ROOT_DIR / "config" / "code_analysis.toml",
        alias="CODE_ANALYSIS_POLICY_PATH",
    )
    embedded_docs_dir: Path = Field(
        default=ROOT_DIR / "data" / "docs_embedded",
        alias="EMBEDDED_DOCS_DIR",
    )
    processed_dir: Path = ROOT_DIR / "data" / "processed"
    ingestion_output_dir: Path = Field(
        default=ROOT_DIR / "data" / "processed",
        alias="INGESTION_OUTPUT_DIR",
    )
    cache_dir: Path = ROOT_DIR / "data" / "cache"
    eval_dir: Path = ROOT_DIR / "data" / "eval"
    exports_dir: Path = ROOT_DIR / "data" / "exports"
    audit_journal_enabled: bool = Field(
        default=False,
        alias="AUDIT_JOURNAL_ENABLED",
    )
    audit_sink_backend: str = Field(
        default="hmac_jsonl",
        alias="AUDIT_SINK_BACKEND",
    )
    audit_journal_path: Path = Field(
        default=ROOT_DIR / "data" / "exports" / "audit" / "api_audit.jsonl",
        alias="AUDIT_JOURNAL_PATH",
    )
    audit_hmac_key: SecretStr = Field(
        default=SecretStr(""),
        alias="AUDIT_HMAC_KEY",
    )
    conversation_db_path: Path = Field(
        default=ROOT_DIR / "data" / "conversations.sqlite3",
        alias="CONVERSATION_DB_PATH",
    )
    conversation_max_context_tokens: int = Field(
        default=32_000,
        alias="CONVERSATION_MAX_CONTEXT_TOKENS",
        gt=0,
    )
    conversation_reserved_system_tokens: int = Field(
        default=2_000,
        alias="CONVERSATION_RESERVED_SYSTEM_TOKENS",
        ge=0,
    )
    conversation_reserved_evidence_tokens: int = Field(
        default=12_000,
        alias="CONVERSATION_RESERVED_EVIDENCE_TOKENS",
        ge=0,
    )
    conversation_reserved_answer_tokens: int = Field(
        default=4_000,
        alias="CONVERSATION_RESERVED_ANSWER_TOKENS",
        ge=0,
    )
    conversation_summary_target_tokens: int = Field(
        default=1_000,
        alias="CONVERSATION_SUMMARY_TARGET_TOKENS",
        gt=4,
    )

    artifact_version: str = "v1"
    index_version: str = "fsrag_v1"

    openai_api_key: str = Field(default="", alias="OPENAI_API_KEY")
    openai_base_url: str = Field(default="", alias="OPENAI_BASE_URL")
    openai_embedding_model: str = Field(default="text-embedding-3-large", alias="OPENAI_EMBEDDING_MODEL")
    openai_chat_model: str = Field(default="gpt-4.1-mini", alias="OPENAI_CHAT_MODEL")

    qdrant_host: str = Field(default="localhost", alias="QDRANT_HOST")
    qdrant_port: int = Field(default=6333, alias="QDRANT_PORT")
    qdrant_collection_name: str = Field(default="functional_specs", alias="QDRANT_COLLECTION_NAME")
    qdrant_vector_size: int = Field(default=3072, alias="QDRANT_VECTOR_SIZE")
    qdrant_local_path: Path = Field(default=ROOT_DIR / "data" / "qdrant_local", alias="QDRANT_LOCAL_PATH")

    retrieval_mode: str = Field(default="hybrid", alias="RETRIEVAL_MODE")
    retrieval_min_top_score: float = Field(default=0.30, alias="RETRIEVAL_MIN_TOP_SCORE")
    hybrid_dense_weight: float = Field(default=0.40, alias="HYBRID_DENSE_WEIGHT")
    hybrid_lexical_weight: float = Field(default=0.60, alias="HYBRID_LEXICAL_WEIGHT")
    hybrid_candidate_limit: int = Field(default=10, alias="HYBRID_CANDIDATE_LIMIT")

    llm_input_cost_per_1k_tokens: float = Field(default=0.0, alias="LLM_INPUT_COST_PER_1K_TOKENS")
    llm_output_cost_per_1k_tokens: float = Field(default=0.0, alias="LLM_OUTPUT_COST_PER_1K_TOKENS")


def get_settings() -> Settings:
    return Settings()
