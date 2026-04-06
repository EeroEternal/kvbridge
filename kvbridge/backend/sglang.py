"""SGLang backend — OpenAI-compatible API with RadixAttention hints."""

from __future__ import annotations

import logging
from typing import Any, AsyncIterator

from kvbridge.backend.base import BaseBackend

logger = logging.getLogger("kvbridge.backend.sglang")


class SGLangBackend(BaseBackend):
    """SGLang server with RadixAttention automatic prefix caching.

    SGLang's RadixAttention uses a radix tree to cache KV blocks,
    so stable prefixes naturally get high cache hit rates.
    KVBridge ensures the messages prefix stays unchanged across calls.

    The SGLang server also exposes /v1/chat/completions in OpenAI format.
    """

    async def chat_completions(
        self,
        payload: dict[str, Any],
        *,
        stream: bool = False,
    ) -> dict[str, Any] | AsyncIterator[bytes]:
        payload.setdefault("stream", stream)

        # Remove KVBridge-internal fields before forwarding
        payload.pop("kvbridge_prefix_id", None)

        if stream:
            return self._post_stream("/v1/chat/completions", payload)
        return await self._post_json("/v1/chat/completions", payload)

    async def list_models(self) -> dict[str, Any]:
        return await self._get_json("/v1/models")
