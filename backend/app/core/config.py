"""
Core Configuration Settings for the FastAPI Concurrency Gateway & Session Store.
"""

from typing import List
from pydantic_settings import BaseSettings, SettingsConfigDict


class GatewaySettings(BaseSettings):
    """Configuration settings for high-concurrency gateway and session storage."""

    app_name: str = "Unified RAG Enterprise Science Gateway"
    api_v1_prefix: str = "/api/v1"
    debug: bool = False

    # Concurrency & Rate Throttling
    concurrency_limit: int = 50
    semaphore_timeout_seconds: float = 10.0
    client_pool_limit: int = 100
    client_pool_keepalive: int = 50

    # Session Management
    session_ttl_seconds: int = 7200  # 2 hours
    max_history_turns: int = 4        # Sliding window (last 4 dialogue turns)
    redis_url: str = "redis://localhost:6379/0"
    use_redis: bool = True

    # Vector & Core Settings
    qdrant_collection_name: str = "science_textbook_dense"
    cors_origins: List[str] = ["*"]

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore"
    )


settings = GatewaySettings()
