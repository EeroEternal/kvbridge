"""Shared utilities."""

from __future__ import annotations

import hashlib
import time
from typing import Any

import orjson


def now_iso() -> str:
    """Return current UTC time in ISO 8601 format."""
    from datetime import datetime, timezone

    return datetime.now(timezone.utc).isoformat()


def hash_messages(messages: list[dict[str, Any]]) -> str:
    """Produce a deterministic SHA-256 hex digest for a message list."""
    raw = orjson.dumps(messages, option=orjson.OPT_SORT_KEYS)
    return hashlib.sha256(raw).hexdigest()


def prefix_id(session_id: str, digest: str) -> str:
    """Build a namespaced prefix id."""
    return f"kvbridge:{session_id}:{digest[:16]}"


def monotonic_ms() -> float:
    return time.monotonic() * 1000
