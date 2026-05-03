from __future__ import annotations

from pathlib import Path

from pydantic import Field
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
    processed_dir: Path = ROOT_DIR / "data" / "processed"
    cache_dir: Path = ROOT_DIR / "data" / "cache"
    eval_dir: Path = ROOT_DIR / "data" / "eval"
    exports_dir: Path = ROOT_DIR / "data" / "exports"

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

    retrieval_min_top_score: float = Field(default=0.30, alias="RETRIEVAL_MIN_TOP_SCORE")


def get_settings() -> Settings:
    return Settings()
