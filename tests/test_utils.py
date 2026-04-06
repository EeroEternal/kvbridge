"""Tests for utility functions."""

from kvbridge.utils import hash_messages, prefix_id, now_iso


def test_hash_messages_deterministic():
    msgs = [{"role": "user", "content": "hello"}]
    h1 = hash_messages(msgs)
    h2 = hash_messages(msgs)
    assert h1 == h2
    assert len(h1) == 64  # SHA-256 hex


def test_hash_messages_different_content():
    msgs1 = [{"role": "user", "content": "hello"}]
    msgs2 = [{"role": "user", "content": "world"}]
    assert hash_messages(msgs1) != hash_messages(msgs2)


def test_prefix_id_format():
    pid = prefix_id("sess-1", "abcdef1234567890extra")
    assert pid.startswith("kvbridge:sess-1:")
    assert len(pid.split(":")) == 3


def test_now_iso_format():
    ts = now_iso()
    assert "T" in ts
    assert "+" in ts or "Z" in ts
