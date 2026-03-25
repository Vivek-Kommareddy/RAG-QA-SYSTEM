"""Configuration module for the RAG Q&A system.

This module defines a `Settings` class derived from `BaseSettings` in the
pydantic‑settings package.  It reads configuration from environment variables
and provides sensible defaults for all parameters.  Using a strongly typed
settings object reduces the likelihood of misconfiguration at runtime.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Optional

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application configuration loaded from environment variables.

    Attributes mirror the keys in `.env.example`.  When new configuration
    parameters are added they should be defined here with appropriate types
    and default values.
    """

    openai_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    llm_provider: str = "openai"
    llm_model: str = "gpt-3.5-turbo"
    embedding_model: str = "text-embedding-3-small"
    chroma_persist_dir: str = "chroma_data"
    chunk_size: int = 512
    chunk_overlap: int = 50
    top_k: int = 5
    rerank_enabled: bool = False

    model_config = SettingsConfigDict(env_file=".env", env_prefix="", extra="ignore")


@lru_cache
def get_settings() -> Settings:
    """Return a cached settings instance.

    This helper uses an LRU cache to ensure that the settings are only read
    from the environment once.  Downstream components should call this
    function rather than instantiating `Settings` directly.
    """

    return Settings()  # type: ignore[arg-type]
