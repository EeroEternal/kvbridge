"""Base backend interface and factory."""

from __future__ import annotations

import abc
import logging
from typing import Any, AsyncIterator

import httpx

from kvbridge.config import BackendType, settings

logger = logging.getLogger("kvbridge.backend")


class BaseBackend(abc.ABC):
    """Abstract interface for inference backends."""

    def __init__(self, base_url: str, http_client: httpx.AsyncClient):
        self.base_url = base_url.rstrip("/")
        self.client = http_client

    @abc.abstractmethod
    async def chat_completions(
        self,
        payload: dict[str, Any],
        *,
        stream: bool = False,
    ) -> dict[str, Any] | AsyncIterator[bytes]:
        """Send a chat completion request to the backend.

        Returns:
            If stream=False: parsed JSON response dict.
            If stream=True: async iterator yielding raw SSE bytes.
        """
        ...

    @abc.abstractmethod
    async def list_models(self) -> dict[str, Any]:
        """Return the model list from the backend."""
        ...

    # ---- shared helpers ----

    async def _post_json(
        self, path: str, payload: dict[str, Any],
    ) -> dict[str, Any]:
        """POST JSON and return parsed response."""
        url = f"{self.base_url}{path}"
        resp = await self.client.post(url, json=payload, timeout=120.0)
        resp.raise_for_status()
        return resp.json()

    async def _post_stream(
        self, path: str, payload: dict[str, Any],
    ) -> AsyncIterator[bytes]:
        """POST and yield raw SSE byte chunks."""
        url = f"{self.base_url}{path}"
        req = self.client.build_request("POST", url, json=payload)
        resp = await self.client.send(req, stream=True)
        resp.raise_for_status()
        async for chunk in resp.aiter_bytes():
            yield chunk

    async def _get_json(self, path: str) -> dict[str, Any]:
        """GET and return parsed JSON."""
        url = f"{self.base_url}{path}"
        resp = await self.client.get(url, timeout=30.0)
        resp.raise_for_status()
        return resp.json()


def get_backend(http_client: httpx.AsyncClient) -> BaseBackend:
    """Factory: create the appropriate backend based on config."""
    backend_type = settings.backend
    base_url = settings.backend_url

    if backend_type == BackendType.VLLM:
        from kvbridge.backend.vllm import VLLMBackend
        return VLLMBackend(base_url, http_client)
    elif backend_type == BackendType.SGLANG:
        from kvbridge.backend.sglang import SGLangBackend
        return SGLangBackend(base_url, http_client)
    elif backend_type == BackendType.MINDIE:
        from kvbridge.backend.mindie import MindIEBackend
        return MindIEBackend(base_url, http_client)
    else:
        raise ValueError(f"Unsupported backend: {backend_type}")
