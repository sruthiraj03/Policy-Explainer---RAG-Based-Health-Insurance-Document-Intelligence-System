"""
Centralized configuration from environment variables.

Import and use::

    from backend.config import get_settings

    settings = get_settings()
    settings.openai_api_key  # never log or expose
    settings.embedding_model
    settings.llm_model
    settings.vector_db_path

No secrets are hardcoded. All values come from the environment or safe defaults.
"""

from functools import lru_cache
from pathlib import Path

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """
    Application settings loaded from environment.

    Required: OPENAI_API_KEY (must be non-empty).
    Optional: EMBEDDING_MODEL, LLM_MODEL, VECTOR_DB_PATH have safe defaults.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # Required: no default; validation ensures non-empty when set
    openai_api_key: str = Field(
        ...,
        min_length=1,
        description="OpenAI API key; set via OPENAI_API_KEY. Never committed.",
    )

    # Safe defaults for model names (no secrets)
    embedding_model: str = Field(
        default="text-embedding-3-small",
        description="Model used for embedding policy chunks (EMBEDDING_MODEL).",
    )
    llm_model: str = Field(
        default="gpt-4o-mini",
        description="Model used for summarization and Q&A (LLM_MODEL).",
    )

    # Vector DB persistence path; default is local directory
    vector_db_path: str = Field(
        default="./chroma_data",
        description="Directory for Chroma persistence (VECTOR_DB_PATH). Use empty for in-memory.",
    )

    @field_validator("openai_api_key", mode="before")
    @classmethod
    def strip_api_key(cls, v: str | None) -> str:
        if v is None or (isinstance(v, str) and not v.strip()):
            raise ValueError("OPENAI_API_KEY must be set and non-empty")
        return v.strip() if isinstance(v, str) else v

    @field_validator("vector_db_path", mode="before")
    @classmethod
    def normalize_vector_db_path(cls, v: str | None) -> str:
        if v is None or (isinstance(v, str) and not v.strip()):
            return "./chroma_data"
        return v.strip()

    def get_vector_db_path_resolved(self) -> Path:
        """Return vector_db_path as a resolved Path for use by Chroma."""
        return Path(self.vector_db_path).resolve()


@lru_cache
def get_settings() -> Settings:
    """
    Return cached settings instance. Reusable across modules.

    Usage::

        from backend.config import get_settings
        settings = get_settings()
    """
    return Settings()
