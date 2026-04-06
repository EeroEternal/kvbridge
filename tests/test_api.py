"""Integration tests for the FastAPI application."""

import pytest
from httpx import ASGITransport, AsyncClient

from kvbridge.main import app


@pytest.fixture
async def client():
    """Create an async test client."""
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://testserver",
    ) as ac:
        yield ac


# ---------------------------------------------------------------------------
# Health endpoint
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_health(client: AsyncClient):
    resp = await client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert "version" in data


# ---------------------------------------------------------------------------
# Cache protocol
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_cache_status_not_found(client: AsyncClient):
    resp = await client.get("/cache/status", params={"session_id": "nonexistent"})
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_cache_append_empty_delta(client: AsyncClient):
    resp = await client.post("/cache/append", json={
        "session_id": "test",
        "delta_messages": [],
    })
    assert resp.status_code == 400


@pytest.mark.asyncio
async def test_cache_append_and_status(client: AsyncClient):
    # Append some messages
    resp = await client.post("/cache/append", json={
        "session_id": "int-test-1",
        "delta_messages": [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hello"},
        ],
    })
    assert resp.status_code == 200
    data = resp.json()
    assert data["session_id"] == "int-test-1"
    assert data["total_messages"] == 2

    # Now query status
    resp = await client.get("/cache/status", params={"session_id": "int-test-1"})
    assert resp.status_code == 200
    status = resp.json()
    assert status["session_id"] == "int-test-1"
    assert status["prefix_messages_count"] == 2


# ---------------------------------------------------------------------------
# /v1/chat/completions — validation
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_chat_completions_no_messages(client: AsyncClient):
    resp = await client.post("/v1/chat/completions", json={"messages": []})
    assert resp.status_code == 400


@pytest.mark.asyncio
async def test_chat_completions_invalid_json(client: AsyncClient):
    resp = await client.post(
        "/v1/chat/completions",
        content=b"not json",
        headers={"content-type": "application/json"},
    )
    assert resp.status_code == 400


# ---------------------------------------------------------------------------
# /v1/models — fallback
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_models_fallback(client: AsyncClient):
    """When backend is unavailable, should return the configured model."""
    resp = await client.get("/v1/models")
    assert resp.status_code == 200
    data = resp.json()
    assert data["object"] == "list"
    assert len(data["data"]) >= 1
