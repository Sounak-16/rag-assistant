from functools import lru_cache
from typing import List, Optional, Union
from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Application
    app_env: str = Field(default="development", description="App environment")
    host: str = Field(default="0.0.0.0", description="Server host")
    port: int = Field(default=8000, description="Server port")
    log_level: str = Field(default="INFO", description="Logging level")

    # Vector store (ChromaDB)
    chroma_persist_dir: str = Field(
        default="./data/chroma_db",
        description="Path to persistent ChromaDB storage directory",
    )
    chroma_collection_name: str = Field(
        default="rag_documents",
        description="Collection name in ChromaDB",
    )

    # Embeddings
    embedding_provider: str = Field(
        default="sentence-transformers",
        description="Embedding provider: sentence-transformers | openai",
    )
    embedding_model: str = Field(
        default="all-MiniLM-L6-v2",
        description="Model name or path for embeddings",
    )

    # LLM
    llm_provider: str = Field(
        default="mock",
        description="LLM provider: gemini | openai | anthropic | ollama | mock",
    )
    llm_model: str = Field(
        default="gemini-2.5-flash",
        description="Model identifier for the LLM",
    )
    llm_api_key: Optional[str] = Field(
        default=None,
        description="API key for cloud LLM provider",
    )
    llm_base_url: Optional[str] = Field(
        default=None,
        description="Base URL for custom/Ollama endpoint",
    )
    llm_temperature: float = Field(
        default=0.0,
        description="Sampling temperature (0.0 for deterministic RAG answers)",
    )
    llm_timeout_seconds: float = Field(
        default=30.0,
        description="Request timeout in seconds",
    )
    llm_max_retries: int = Field(
        default=2,
        description="Max retry attempts for failed LLM calls",
    )

    # RAG / Retrieval
    chunk_size: int = Field(
        default=500,
        description="Target character size for text chunks",
    )
    chunk_overlap: int = Field(
        default=100,
        description="Character overlap between consecutive chunks",
    )
    top_k: int = Field(
        default=4,
        description="Default number of chunks to retrieve",
    )
    similarity_threshold: float = Field(
        default=0.38,
        description="Minimum cosine similarity score required for retrieval inclusion",
    )

    # Ingestion constraints
    max_file_size_mb: int = Field(
        default=10,
        description="Maximum file upload size in megabytes",
    )
    allowed_extensions: Union[List[str], str] = Field(
        default=[".pdf", ".txt", ".md"],
        description="Allowed file extensions for upload",
    )

    # Session Lifecycle & Limits
    session_ttl_minutes: int = Field(
        default=30,
        description="Session idle timeout in minutes before automatic cleanup",
    )
    max_files_per_session: int = Field(
        default=10,
        description="Maximum number of files allowed per session",
    )
    max_total_upload_mb: int = Field(
        default=25,
        description="Maximum total upload size in megabytes per session",
    )

    @field_validator("allowed_extensions", mode="after")
    @classmethod
    def parse_allowed_extensions(cls, v):
        if isinstance(v, str):
            return [ext.strip().lower() for ext in v.split(",") if ext.strip()]
        return [ext.strip().lower() for ext in v]


@lru_cache()
def get_settings() -> Settings:
    return Settings()
