"""KVBridge — FastAPI application entry point.

Transparent, cache-aware OpenAI-compatible proxy that maximizes
prefix cache hit rates on vLLM / SGLang / MindIE backends.
"""

from __future__ import annotations

import logging
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any

import httpx
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from prometheus_client import make_asgi_app as prometheus_asgi_app

from kvbridge import __version__
from kvbridge.backend import get_backend
from kvbridge.backend.base import BaseBackend
from kvbridge.cache_protocol import router as cache_router, set_tracker
from kvbridge.config import settings
from kvbridge.metrics import (
    BACKEND_LATENCY,
    PREFIX_HIT_RATE,
    PROXY_LATENCY,
    REQUEST_TOTAL,
)
from kvbridge.session import SessionTracker
from kvbridge.utils import monotonic_ms

logger = logging.getLogger("kvbridge")

# ---------------------------------------------------------------------------
# Global references (populated during lifespan)
# ---------------------------------------------------------------------------
_http_client: httpx.AsyncClient | None = None
_backend: BaseBackend | None = None
_tracker: SessionTracker | None = None


# ---------------------------------------------------------------------------
# Lifespan
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup / shutdown lifecycle."""
    global _http_client, _backend, _tracker

    # ---- Startup ----
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)-8s [%(name)s] %(message)s",
    )

    logger.info("KVBridge %s starting — backend=%s url=%s", __version__, settings.backend.value, settings.backend_url)

    # HTTP client (shared connection pool)
    _http_client = httpx.AsyncClient(
        timeout=httpx.Timeout(connect=10.0, read=120.0, write=30.0, pool=10.0),
        limits=httpx.Limits(max_connections=200, max_keepalive_connections=50),
    )

    # Backend
    _backend = get_backend(_http_client)

    # Redis (optional)
    redis_client = None
    if settings.redis_url:
        try:
            import redis.asyncio as aioredis

            redis_client = aioredis.from_url(
                settings.redis_url,
                decode_responses=False,
            )
            await redis_client.ping()
            logger.info("Redis connected: %s", settings.redis_url)
        except Exception:
            logger.warning("Redis unavailable, falling back to in-memory storage", exc_info=True)
            redis_client = None

    # Session tracker
    _tracker = SessionTracker(redis_client=redis_client)
    set_tracker(_tracker)

    logger.info("KVBridge ready on :%d", settings.port)

    yield

    # ---- Shutdown ----
    if _http_client:
        await _http_client.aclose()
    if redis_client:
        await redis_client.aclose()
    logger.info("KVBridge shut down")


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

app = FastAPI(
    title="KVBridge",
    description="Cache-Aware OpenAI Compatible Proxy",
    version=__version__,
    lifespan=lifespan,
)

# CORS (agents may call from browsers)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Prometheus metrics (mounted at /metrics)
if settings.metrics_enabled:
    metrics_app = prometheus_asgi_app()
    app.mount("/metrics", metrics_app)

# Cache protocol routes
app.include_router(cache_router)


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------

@app.get("/health", tags=["Health"])
async def health():
    return {"status": "ok", "version": __version__, "backend": settings.backend.value}


# ---------------------------------------------------------------------------
# /v1/models — model listing (proxied from backend)
# ---------------------------------------------------------------------------

@app.get("/v1/models", tags=["OpenAI Compatible"])
async def list_models():
    """Proxy the model list from the backend, or return the configured model."""
    try:
        return await _backend.list_models()
    except Exception:
        # Fallback: return the configured model name
        return {
            "object": "list",
            "data": [
                {
                    "id": settings.model_name,
                    "object": "model",
                    "created": 0,
                    "owned_by": "kvbridge",
                }
            ],
        }


# ---------------------------------------------------------------------------
# /v1/chat/completions — the core proxy endpoint
# ---------------------------------------------------------------------------

@app.post("/v1/chat/completions", tags=["OpenAI Compatible"])
async def chat_completions(request: Request):
    """OpenAI-compatible chat completion proxy with transparent delta compression.

    Flow:
        1. Extract or generate a session_id from the request.
        2. Compute delta via SessionTracker.
        3. Forward the optimized payload to the inference backend.
        4. Return the response (streaming or non-streaming).
    """
    proxy_start = monotonic_ms()

    try:
        body: dict[str, Any] = await request.json()
    except Exception:
        REQUEST_TOTAL.labels(method="POST", endpoint="/v1/chat/completions", status="400").inc()
        return JSONResponse(
            status_code=400,
            content={"error": {"message": "Invalid JSON body", "type": "invalid_request_error"}},
        )

    messages = body.get("messages", [])
    stream = body.get("stream", False)

    if not messages:
        REQUEST_TOTAL.labels(method="POST", endpoint="/v1/chat/completions", status="400").inc()
        return JSONResponse(
            status_code=400,
            content={"error": {"message": "'messages' field is required and must not be empty", "type": "invalid_request_error"}},
        )

    # ---- Session identification ----
    # Use x-session-id header, or a hash-based session id, or generate one
    session_id = (
        request.headers.get("x-session-id")
        or request.headers.get("x-request-id")
        or _derive_session_id(messages)
    )

    # ---- Delta compression ----
    delta_result = await _tracker.compute_delta(session_id, messages)

    # Build the forwarded payload
    forward_payload = {**body}
    # Use full messages (after trimming) — the backend sees a stable prefix
    forward_payload["messages"] = delta_result.full_messages
    # Override model name if configured
    if settings.model_name:
        forward_payload["model"] = settings.model_name

    # Inject prefix hint for backends that support it
    if delta_result.prefix_reused and delta_result.prefix_id:
        forward_payload["kvbridge_prefix_id"] = delta_result.prefix_id

    # Update rolling hit rate
    if _tracker:
        state = await _tracker.get_status(session_id)
        if state:
            PREFIX_HIT_RATE.set(state["hit_rate"])

    proxy_elapsed = (monotonic_ms() - proxy_start) / 1000
    PROXY_LATENCY.observe(proxy_elapsed)

    # ---- Forward to backend ----
    backend_start = time.monotonic()

    try:
        if stream:
            response_stream = await _backend.chat_completions(forward_payload, stream=True)

            async def _stream_wrapper():
                try:
                    async for chunk in response_stream:
                        yield chunk
                finally:
                    elapsed = time.monotonic() - backend_start
                    BACKEND_LATENCY.labels(backend=settings.backend.value).observe(elapsed)
                    REQUEST_TOTAL.labels(method="POST", endpoint="/v1/chat/completions", status="200").inc()

            return StreamingResponse(
                _stream_wrapper(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "Connection": "keep-alive",
                    "X-KVBridge-Session": session_id,
                    "X-KVBridge-Prefix-Hit": str(delta_result.prefix_reused).lower(),
                    "X-KVBridge-Delta-Ratio": f"{delta_result.delta_ratio:.4f}",
                },
            )
        else:
            result = await _backend.chat_completions(forward_payload, stream=False)
            elapsed = time.monotonic() - backend_start
            BACKEND_LATENCY.labels(backend=settings.backend.value).observe(elapsed)
            REQUEST_TOTAL.labels(method="POST", endpoint="/v1/chat/completions", status="200").inc()

            return JSONResponse(
                content=result,
                headers={
                    "X-KVBridge-Session": session_id,
                    "X-KVBridge-Prefix-Hit": str(delta_result.prefix_reused).lower(),
                    "X-KVBridge-Delta-Ratio": f"{delta_result.delta_ratio:.4f}",
                },
            )

    except httpx.HTTPStatusError as exc:
        elapsed = time.monotonic() - backend_start
        BACKEND_LATENCY.labels(backend=settings.backend.value).observe(elapsed)
        status = exc.response.status_code
        REQUEST_TOTAL.labels(method="POST", endpoint="/v1/chat/completions", status=str(status)).inc()
        logger.error("Backend returned HTTP %d: %s", status, exc.response.text[:500])
        return JSONResponse(
            status_code=status,
            content={
                "error": {
                    "message": f"Backend error: {exc.response.text[:500]}",
                    "type": "backend_error",
                }
            },
        )
    except httpx.ConnectError:
        REQUEST_TOTAL.labels(method="POST", endpoint="/v1/chat/completions", status="502").inc()
        logger.error("Cannot connect to backend at %s", settings.backend_url)
        return JSONResponse(
            status_code=502,
            content={
                "error": {
                    "message": f"Cannot connect to backend at {settings.backend_url}",
                    "type": "backend_unavailable",
                }
            },
        )
    except Exception:
        REQUEST_TOTAL.labels(method="POST", endpoint="/v1/chat/completions", status="500").inc()
        logger.exception("Unexpected error during chat completion")
        return JSONResponse(
            status_code=500,
            content={
                "error": {
                    "message": "Internal proxy error",
                    "type": "internal_error",
                }
            },
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _derive_session_id(messages: list[dict[str, Any]]) -> str:
    """Derive a stable session ID from the system message(s).

    If the system prompt is stable (which it is for most agents),
    this ensures continuity across requests even without explicit
    session headers.
    """
    system_msgs = [m for m in messages if m.get("role") == "system"]
    if system_msgs:
        from kvbridge.utils import hash_messages
        return hash_messages(system_msgs)[:16]
    return uuid.uuid4().hex[:16]


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    """Run the server via `python -m kvbridge.main`."""
    import uvicorn

    uvicorn.run(
        "kvbridge.main:app",
        host=settings.host,
        port=settings.port,
        log_level=settings.log_level,
        reload=False,
    )


if __name__ == "__main__":
    main()
