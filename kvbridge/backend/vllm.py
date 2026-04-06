"""vLLM backend — OpenAI-compatible API."""

from __future__ import annotations

import logging
from typing import Any, AsyncIterator

from kvbridge.backend.base import BaseBackend

logger = logging.getLogger("kvbridge.backend.vllm")


class VLLMBackend(BaseBackend):
    """vLLM exposes a standard OpenAI-compatible /v1/* API.

    When `enable_prefix_caching=True` is set on the vLLM server,
    repeated prefixes are automatically cached in GPU KV memory.
    KVBridge ensures the prefix stays stable across requests.

    Extra vLLM-specific fields (e.g. `extra_body.prefix_id`) can be
    injected here to give vLLM explicit hints.
    """

    async def chat_completions(
        self,
        payload: dict[str, Any],
        *,
        stream: bool = False,
    ) -> dict[str, Any] | AsyncIterator[bytes]:
        payload.setdefault("stream", stream)

        # Inject vLLM-specific prefix hint if available
        if "kvbridge_prefix_id" in payload:
            extra = payload.pop("kvbridge_prefix_id")
            payload.setdefault("extra_body", {})
            payload["extra_body"]["prefix_id"] = extra

        if stream:
            return self._post_stream("/v1/chat/completions", payload)
        return await self._post_json("/v1/chat/completions", payload)

    async def list_models(self) -> dict[str, Any]:
        return await self._get_json("/v1/models")
