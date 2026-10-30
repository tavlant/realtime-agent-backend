"""Application settings, loaded from environment variables (and an optional .env file)."""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    # --- service -------------------------------------------------------------
    app_name: str = "realtime-agent-backend"
    environment: Literal["dev", "test", "prod"] = "dev"
    log_level: str = "INFO"
    log_json: bool = True

    # --- storage -------------------------------------------------------------
    # Postgres is the source of truth. SQLite works for local dev and tests.
    database_url: str = "sqlite+aiosqlite:///./dev.db"
    db_echo: bool = False
    # "memory://" uses an in-process fake Redis (needs the dev extra) for offline dev.
    redis_url: str = "redis://localhost:6379/0"

    # --- LLM -----------------------------------------------------------------
    # "auto" picks anthropic/openai when a key is configured, else the offline FakeLLM.
    llm_provider: Literal["auto", "fake", "openai", "anthropic"] = "auto"
    openai_api_key: str | None = None
    openai_model: str = "gpt-4.1-mini"
    anthropic_api_key: str | None = None
    anthropic_model: str = "claude-opus-5-5"
    anthropic_effort: Literal["low", "medium", "high", "xhigh", "max"] = "medium"
    # Server-side refusal fallback (Anthropic beta): retries a declined request on a
    # fallback model inside the same call.
    anthropic_refusal_fallbacks: bool = True
    llm_max_output_tokens: int = 16000
    fake_llm_token_delay_ms: int = 0

    system_prompt: str = (
        "You are a concise, helpful assistant. Use the available tools when they help: "
        "the calculator for arithmetic, current_time for dates and times, and "
        "search_knowledge_base for questions about ingested documents."
    )

    # --- reliability ---------------------------------------------------------
    llm_first_token_timeout_s: float = 30.0
    llm_idle_timeout_s: float = 30.0
    llm_max_attempts: int = 3
    retry_base_delay_s: float = 0.25
    retry_max_delay_s: float = 4.0
    tool_timeout_s: float = 10.0
    max_tool_rounds: int = 5

    # --- memory --------------------------------------------------------------
    hot_state_ttl_s: int = 3600
    context_token_budget: int = 3000
    summary_target_ratio: float = 0.5
    min_recent_messages: int = 4
    lock_ttl_s: float = 30.0
    lock_wait_timeout_s: float = 20.0
    shutdown_drain_timeout_s: float = 20.0

    # --- RAG -----------------------------------------------------------------
    vector_store: Literal["auto", "memory", "pgvector"] = "auto"
    embedding_provider: Literal["hashing", "openai"] = "hashing"
    embedding_model: str = "text-embedding-3-small"
    embedding_dim: int = Field(default=256, ge=8, le=4096)
    chunk_max_chars: int = 800
    chunk_overlap_chars: int = 120

    # --- telemetry -----------------------------------------------------------
    otel_enabled: bool = False
    otel_exporter_otlp_endpoint: str | None = None

    @property
    def is_postgres(self) -> bool:
        return self.database_url.startswith("postgresql")

    def resolved_llm_provider(self) -> str:
        if self.llm_provider != "auto":
            return self.llm_provider
        if self.anthropic_api_key:
            return "anthropic"
        if self.openai_api_key:
            return "openai"
        return "fake"

    def resolved_vector_store(self) -> str:
        if self.vector_store != "auto":
            return self.vector_store
        return "pgvector" if self.is_postgres else "memory"


@lru_cache
def get_settings() -> Settings:
    return Settings()
