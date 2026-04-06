"""KVBridge proprietary Cache-Aware protocol endpoints.

These endpoints allow cache-aware agents (e.g. OpenCode, OpenHands) to
explicitly query and manipulate the prefix cache, enabling cooperative
cache management between the agent and the inference backend.

Endpoints:
    GET  /cache/status?session_id=xxx
    POST /cache/append
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

logger = logging.getLogger("kvbridge.cache_protocol")

router = APIRouter(prefix="/cache", tags=["Cache Protocol"])


# ---------------------------------------------------------------------------
# Request / Response schemas
# ---------------------------------------------------------------------------

class AppendRequest(BaseModel):
    """Body for POST /cache/append."""

    session_id: str
    prefix_id: str = ""
    delta_messages: list[dict]


class AppendResponse(BaseModel):
    session_id: str
    prefix_id: str
    total_messages: int
    last_updated: str


class StatusResponse(BaseModel):
    session_id: str
    prefix_id: str
    hit_rate: float
    prefix_messages_count: int
    prefix_token_estimate: int
    max_delta_tokens: int
    last_updated: str


# ---------------------------------------------------------------------------
# Dependency — injected at app startup via `router.state`
# ---------------------------------------------------------------------------

_tracker = None  # Set by main.py at startup


def set_tracker(tracker) -> None:
    """Called by main.py to inject the SessionTracker instance."""
    global _tracker
    _tracker = tracker


def _get_tracker():
    if _tracker is None:
        raise HTTPException(status_code=503, detail="Session tracker not initialized")
    return _tracker


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.get("/status", response_model=StatusResponse)
async def cache_status(session_id: str = Query(..., description="Session ID")):
    """Query the current cache status for a session.

    Returns prefix ID, hit rate, token estimates, and last-updated timestamp.
    Agents can use this to decide whether to send full context or delta-only.
    """
    tracker = _get_tracker()
    status = await tracker.get_status(session_id)
    if status is None:
        raise HTTPException(
            status_code=404,
            detail=f"Session '{session_id}' not found in cache",
        )
    return StatusResponse(**status)


@router.post("/append", response_model=AppendResponse)
async def cache_append(req: AppendRequest):
    """Append delta messages to an existing session's prefix cache.

    This allows agents to proactively push context updates without
    triggering a full inference call, pre-warming the cache for the
    next actual completion request.
    """
    tracker = _get_tracker()
    if not req.delta_messages:
        raise HTTPException(status_code=400, detail="delta_messages must not be empty")

    result = await tracker.append_delta(req.session_id, req.delta_messages)
    return AppendResponse(**result)
