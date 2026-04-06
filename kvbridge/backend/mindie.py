"""MindIE-Ascend backend — Huawei Ascend NPU inference engine."""

from __future__ import annotations

import logging
from typing import Any, AsyncIterator

from kvbridge.backend.base import BaseBackend

logger = logging.getLogger("kvbridge.backend.mindie")


class MindIEBackend(BaseBackend):
    """MindIE (MindSpore Inference Engine) on Ascend NPU.

    MindIE provides an OpenAI-compatible endpoint and supports
    HiCache for KV-cache pooling across requests.

    KVBridge ensures prefix stability so HiCache can maximise reuse.
    Additional MindIE-specific headers/fields can be injected here.
    """

    async def chat_completions(
        self,
        payload: dict[str, Any],
        *,
        stream: bool = False,
    ) -> dict[str, Any] | AsyncIterator[bytes]:
        payload.setdefault("stream", stream)

        # Inject MindIE-specific cache hint via custom header or field
        prefix_hint = payload.pop("kvbridge_prefix_id", None)
        if prefix_hint:
            payload.setdefault("extra_body", {})
            payload["extra_body"]["cache_prefix_id"] = prefix_hint

        if stream:
            return self._post_stream("/v1/chat/completions", payload)
        return await self._post_json("/v1/chat/completions", payload)

    async def list_models(self) -> dict[str, Any]:
        return await self._get_json("/v1/models")
