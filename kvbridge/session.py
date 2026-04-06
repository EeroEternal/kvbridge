"""Session tracker and delta compressor.

The core logic: given a full message list from an agent, identify what's new
(the delta) relative to the cached prefix, then forward only the minimal
payload to the inference backend.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from kvbridge.config import settings
from kvbridge.metrics import (
    ACTIVE_SESSIONS,
    DELTA_RATIO,
    PREFIX_HIT_TOTAL,
    PREFIX_MISS_TOTAL,
)
from kvbridge.utils import hash_messages, now_iso, prefix_id

logger = logging.getLogger("kvbridge.session")


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class SessionState:
    """In-memory representation of a cached session."""

    session_id: str
    prefix_messages: list[dict[str, Any]] = field(default_factory=list)
    prefix_hash: str = ""
    prefix_token_estimate: int = 0
    hit_count: int = 0
    miss_count: int = 0
    last_updated: str = ""

    @property
    def hit_rate(self) -> float:
        total = self.hit_count + self.miss_count
        return self.hit_count / total if total else 0.0


@dataclass
class DeltaResult:
    """Output of delta compression."""

    full_messages: list[dict[str, Any]]
    delta_messages: list[dict[str, Any]]
    prefix_reused: bool
    prefix_id: str
    delta_ratio: float  # len(delta) / len(full)


# ---------------------------------------------------------------------------
# Session Tracker
# ---------------------------------------------------------------------------

class SessionTracker:
    """Manages per-session prefix caching and delta computation."""

    def __init__(self, redis_client=None):
        self._local: dict[str, SessionState] = {}
        self._redis = redis_client

    # ----- public API -----

    async def compute_delta(
        self,
        session_id: str,
        messages: list[dict[str, Any]],
    ) -> DeltaResult:
        """Compare *messages* against cached prefix and return the delta."""
        if not settings.enable_delta:
            return self._no_compression(session_id, messages)

        state = await self._get_state(session_id)
        if state is None:
            return await self._first_request(session_id, messages)

        # Find the longest common prefix
        common_len = self._common_prefix_length(state.prefix_messages, messages)

        if common_len == 0:
            # Completely new conversation or context reset
            return await self._first_request(session_id, messages)

        # Delta = everything after the common prefix
        delta = messages[common_len:]

        # Apply context-window trimming: keep system + last N rounds
        trimmed_full = self._trim_context(messages)
        trimmed_delta = self._trim_context(delta) if delta else delta

        # Update state
        state.prefix_messages = trimmed_full
        state.prefix_hash = hash_messages(trimmed_full)
        state.hit_count += 1
        state.last_updated = now_iso()
        await self._save_state(state)

        pid = prefix_id(session_id, state.prefix_hash)
        ratio = len(trimmed_delta) / max(len(trimmed_full), 1)

        PREFIX_HIT_TOTAL.inc()
        DELTA_RATIO.observe(ratio)

        logger.info(
            "Session %s: prefix hit, delta %d/%d messages (ratio=%.2f)",
            session_id,
            len(trimmed_delta),
            len(trimmed_full),
            ratio,
        )

        return DeltaResult(
            full_messages=trimmed_full,
            delta_messages=trimmed_delta,
            prefix_reused=True,
            prefix_id=pid,
            delta_ratio=ratio,
        )

    async def get_status(self, session_id: str) -> dict[str, Any] | None:
        """Return cache status for /cache/status endpoint."""
        state = await self._get_state(session_id)
        if state is None:
            return None
        return {
            "session_id": session_id,
            "prefix_id": prefix_id(session_id, state.prefix_hash),
            "hit_rate": round(state.hit_rate, 4),
            "prefix_messages_count": len(state.prefix_messages),
            "prefix_token_estimate": state.prefix_token_estimate,
            "max_delta_tokens": 8192,
            "last_updated": state.last_updated,
        }

    async def append_delta(
        self,
        session_id: str,
        delta_messages: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Append delta messages to an existing session."""
        state = await self._get_state(session_id)
        if state is None:
            state = SessionState(session_id=session_id)

        state.prefix_messages.extend(delta_messages)
        state.prefix_messages = self._trim_context(state.prefix_messages)
        state.prefix_hash = hash_messages(state.prefix_messages)
        state.last_updated = now_iso()
        await self._save_state(state)

        pid = prefix_id(session_id, state.prefix_hash)
        return {
            "session_id": session_id,
            "prefix_id": pid,
            "total_messages": len(state.prefix_messages),
            "last_updated": state.last_updated,
        }

    # ----- internal helpers -----

    def _no_compression(
        self, session_id: str, messages: list[dict[str, Any]]
    ) -> DeltaResult:
        """Pass-through when delta compression is disabled."""
        return DeltaResult(
            full_messages=messages,
            delta_messages=messages,
            prefix_reused=False,
            prefix_id="",
            delta_ratio=1.0,
        )

    async def _first_request(
        self, session_id: str, messages: list[dict[str, Any]]
    ) -> DeltaResult:
        """Handle the very first request for a session."""
        trimmed = self._trim_context(messages)
        state = SessionState(
            session_id=session_id,
            prefix_messages=trimmed,
            prefix_hash=hash_messages(trimmed),
            prefix_token_estimate=self._estimate_tokens(trimmed),
            miss_count=1,
            last_updated=now_iso(),
        )
        await self._save_state(state)

        PREFIX_MISS_TOTAL.inc()
        ACTIVE_SESSIONS.set(len(self._local))

        pid = prefix_id(session_id, state.prefix_hash)
        return DeltaResult(
            full_messages=trimmed,
            delta_messages=trimmed,
            prefix_reused=False,
            prefix_id=pid,
            delta_ratio=1.0,
        )

    @staticmethod
    def _common_prefix_length(
        cached: list[dict[str, Any]],
        incoming: list[dict[str, Any]],
    ) -> int:
        """Return the number of leading messages that are identical."""
        length = 0
        for a, b in zip(cached, incoming):
            if a.get("role") == b.get("role") and a.get("content") == b.get("content"):
                length += 1
            else:
                break
        return length

    def _trim_context(self, messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Keep system messages + last N rounds (user+assistant pairs).

        This is the MVP fixed-rule compression strategy.
        """
        system_msgs = [m for m in messages if m.get("role") == "system"]
        non_system = [m for m in messages if m.get("role") != "system"]

        max_rounds = settings.max_context_rounds
        # Each round = 1 user + 1 assistant = 2 messages
        max_msgs = max_rounds * 2
        if len(non_system) > max_msgs:
            non_system = non_system[-max_msgs:]

        return system_msgs + non_system

    @staticmethod
    def _estimate_tokens(messages: list[dict[str, Any]]) -> int:
        """Rough token estimation: ~4 chars per token."""
        total_chars = sum(len(str(m.get("content", ""))) for m in messages)
        return total_chars // 4

    # ----- storage layer -----

    async def _get_state(self, session_id: str) -> SessionState | None:
        """Retrieve session state from Redis or local cache."""
        # Always check local first
        if session_id in self._local:
            return self._local[session_id]

        # Try Redis
        if self._redis is not None:
            try:
                import orjson

                raw = await self._redis.get(f"kvbridge:session:{session_id}")
                if raw:
                    data = orjson.loads(raw)
                    state = SessionState(**data)
                    self._local[session_id] = state
                    return state
            except Exception:
                logger.warning("Redis read failed for session %s", session_id, exc_info=True)

        return None

    async def _save_state(self, state: SessionState) -> None:
        """Persist session state to local cache and optionally Redis."""
        self._local[state.session_id] = state

        if self._redis is not None:
            try:
                import orjson

                data = orjson.dumps({
                    "session_id": state.session_id,
                    "prefix_messages": state.prefix_messages,
                    "prefix_hash": state.prefix_hash,
                    "prefix_token_estimate": state.prefix_token_estimate,
                    "hit_count": state.hit_count,
                    "miss_count": state.miss_count,
                    "last_updated": state.last_updated,
                })
                await self._redis.set(
                    f"kvbridge:session:{state.session_id}",
                    data,
                    ex=settings.session_ttl,
                )
            except Exception:
                logger.warning(
                    "Redis write failed for session %s",
                    state.session_id,
                    exc_info=True,
                )

        ACTIVE_SESSIONS.set(len(self._local))
