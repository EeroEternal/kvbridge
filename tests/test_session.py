"""Tests for SessionTracker and DeltaCompressor."""

import pytest
from kvbridge.session import SessionTracker, DeltaResult


@pytest.fixture
def tracker():
    """Create a tracker with no Redis (in-memory only)."""
    return SessionTracker(redis_client=None)


# ---------------------------------------------------------------------------
# Basic delta compression
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_first_request_is_full(tracker: SessionTracker):
    """First request for a session should be a cache miss (delta = full)."""
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello"},
    ]

    result = await tracker.compute_delta("sess-1", messages)

    assert isinstance(result, DeltaResult)
    assert result.prefix_reused is False
    assert result.delta_ratio == 1.0
    assert len(result.delta_messages) == len(result.full_messages)


@pytest.mark.asyncio
async def test_second_request_detects_prefix(tracker: SessionTracker):
    """Second request with same prefix should detect the common prefix."""
    messages_1 = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there!"},
    ]
    messages_2 = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "Hello"},
        {"role": "assistant", "content": "Hi there!"},
        {"role": "user", "content": "What is 1+1?"},
    ]

    await tracker.compute_delta("sess-2", messages_1)
    result = await tracker.compute_delta("sess-2", messages_2)

    assert result.prefix_reused is True
    assert result.delta_ratio < 1.0


@pytest.mark.asyncio
async def test_completely_different_context_resets(tracker: SessionTracker):
    """If the new request has zero common prefix, it should reset."""
    messages_1 = [
        {"role": "system", "content": "System prompt A"},
        {"role": "user", "content": "Hello"},
    ]
    messages_2 = [
        {"role": "system", "content": "Completely different system prompt"},
        {"role": "user", "content": "Different question"},
    ]

    await tracker.compute_delta("sess-3", messages_1)
    result = await tracker.compute_delta("sess-3", messages_2)

    assert result.prefix_reused is False
    assert result.delta_ratio == 1.0


# ---------------------------------------------------------------------------
# Context trimming
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_context_trimming(tracker: SessionTracker):
    """Messages exceeding max_context_rounds should be trimmed."""
    # Build a conversation with 10 rounds (20 non-system messages)
    messages = [{"role": "system", "content": "System prompt"}]
    for i in range(10):
        messages.append({"role": "user", "content": f"Question {i}"})
        messages.append({"role": "assistant", "content": f"Answer {i}"})

    result = await tracker.compute_delta("sess-4", messages)

    # Default max_context_rounds=3, so 1 system + 6 non-system = 7 total
    assert len(result.full_messages) == 7
    # System message should always be preserved
    assert result.full_messages[0]["role"] == "system"


# ---------------------------------------------------------------------------
# Session status
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_get_status_unknown_session(tracker: SessionTracker):
    """Unknown session should return None."""
    status = await tracker.get_status("nonexistent")
    assert status is None


@pytest.mark.asyncio
async def test_get_status_after_request(tracker: SessionTracker):
    """After a request, get_status should return valid data."""
    messages = [
        {"role": "system", "content": "System"},
        {"role": "user", "content": "Hello"},
    ]
    await tracker.compute_delta("sess-5", messages)

    status = await tracker.get_status("sess-5")
    assert status is not None
    assert status["session_id"] == "sess-5"
    assert "prefix_id" in status
    assert status["max_delta_tokens"] == 8192


# ---------------------------------------------------------------------------
# Append delta
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_append_delta(tracker: SessionTracker):
    """Appending delta should extend the session prefix."""
    messages = [
        {"role": "system", "content": "System"},
        {"role": "user", "content": "Hello"},
    ]
    await tracker.compute_delta("sess-6", messages)

    result = await tracker.append_delta("sess-6", [
        {"role": "assistant", "content": "Hi!"},
        {"role": "user", "content": "Follow up"},
    ])

    assert result["session_id"] == "sess-6"
    assert result["total_messages"] >= 3


@pytest.mark.asyncio
async def test_append_to_new_session(tracker: SessionTracker):
    """Appending to a non-existent session should create one."""
    result = await tracker.append_delta("new-sess", [
        {"role": "user", "content": "First message"},
    ])
    assert result["session_id"] == "new-sess"
    assert result["total_messages"] == 1
