"""Centralized configuration via pydantic-settings."""

from __future__ import annotations

from enum import Enum

from pydantic_settings import BaseSettings


class BackendType(str, Enum):
    VLLM = "vllm"
    SGLANG = "sglang"
    MINDIE = "mindie"


class Settings(BaseSettings):
    """All KVBridge configurable options, loaded from env / .env file."""

    # Server
    host: str = "0.0.0.0"
    port: int = 8001
    log_level: str = "info"

    # Backend
    backend: BackendType = BackendType.VLLM
    backend_url: str = "http://localhost:8000"
    model_name: str = "qwen3-coder"

    # Redis (empty string -> in-memory fallback)
    redis_url: str = ""

    # Session
    session_ttl: int = 3600  # seconds
    max_context_rounds: int = 3

    # Delta compression
    enable_delta: bool = True

    # Prometheus
    metrics_enabled: bool = True

    model_config = {
        "env_prefix": "KVBRIDGE_",
        "env_file": ".env",
        "env_file_encoding": "utf-8",
    }


# Singleton — importable everywhere
settings = Settings()
