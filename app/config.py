"""Configuration settings for the Contracter application."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application configuration settings loaded from environment or defaults."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    UPLOAD_DIR: Path = Field(
        default=Path("data/uploads"),
        description="Directory where uploaded legal documents are stored",
    )
    CHROMA_DIR: Path = Field(
        default=Path("data/chroma_db"),
        description="Directory for local persistent ChromaDB vector storage",
    )
    OLLAMA_BASE_URL: str = Field(
        default="http://localhost:11434",
        description="Base URL for local Ollama inference service",
    )
    OLLAMA_MODEL: str = Field(
        default="gemma3n:e4b",
        description="Name of Ollama model to use for legal analysis",
    )
    EMBEDDING_MODEL: str = Field(
        default="all-MiniLM-L6-v2",
        description="SentenceTransformer model name for document chunk embeddings",
    )
    CHUNK_SIZE: int = Field(
        default=1000,
        description="Target character size for text chunks",
        ge=100,
    )
    CHUNK_OVERLAP: int = Field(
        default=200,
        description="Character overlap between consecutive chunks",
        ge=0,
    )
    TOP_K_RESULTS: int = Field(
        default=5,
        description="Number of top relevant chunks to retrieve from vector store",
        ge=1,
    )
    MAX_FILE_SIZE_MB: int = Field(
        default=50,
        description="Maximum allowed file size for upload in megabytes",
        ge=1,
    )

    def model_post_init(self, __context: object) -> None:
        """Ensure required data directories exist upon initialization."""
        self.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        self.CHROMA_DIR.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    """Return a cached singleton instance of application settings."""
    return Settings()


settings = get_settings()
