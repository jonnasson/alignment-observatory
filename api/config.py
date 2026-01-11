"""
Application configuration using Pydantic Settings.
"""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
    )

    # Application
    app_name: str = "Alignment Observatory API"
    debug: bool = False
    environment: Literal["development", "staging", "production"] = "development"

    # Server
    host: str = "0.0.0.0"
    port: int = 8000

    # CORS
    cors_origins: list[str] = ["http://localhost:5173", "http://127.0.0.1:5173"]

    # Paths
    model_cache_dir: Path = Path("./models")
    trace_cache_dir: Path = Path("./traces")

    # Model defaults
    default_model: str = "gpt2"
    max_sequence_length: int = 1024

    # Cache settings
    trace_cache_max_size: int = 100  # Max traces to keep in memory
    tensor_compression: bool = True

    # WebSocket
    ws_heartbeat_interval: int = 30  # seconds


@lru_cache
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
