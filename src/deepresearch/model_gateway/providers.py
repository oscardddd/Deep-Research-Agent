from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import httpx

from .schemas import ModelUsage


@dataclass(frozen=True)
class ProviderResponse:
    content: str
    model: str
    finish_reason: str | None
    usage: ModelUsage
    raw: dict[str, Any]


class DeepSeekAdapter:
    name = "deepseek"

    def __init__(self, *, api_key: str, timeout: float):
        self.api_key = api_key
        self.timeout = timeout

    def complete(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        max_tokens: int,
        response_format: dict[str, str] | None = None,
    ) -> ProviderResponse:
        response = httpx.post(
            "https://api.deepseek.com/chat/completions",
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": model,
                "messages": messages,
                "response_format": response_format or {"type": "json_object"},
                "thinking": {"type": "disabled"},
                "max_tokens": max_tokens,
                "stream": False,
            },
            timeout=self.timeout,
        )
        response.raise_for_status()
        body = response.json()
        choice = body["choices"][0]
        usage = body.get("usage") or {}
        prompt_details = usage.get("prompt_tokens_details") or {}
        completion_details = usage.get("completion_tokens_details") or {}
        return ProviderResponse(
            content=choice["message"].get("content") or "",
            model=body.get("model") or model,
            finish_reason=choice.get("finish_reason"),
            usage=ModelUsage(
                input_tokens=int(usage.get("prompt_tokens") or 0),
                output_tokens=int(usage.get("completion_tokens") or 0),
                cached_input_tokens=int(prompt_details.get("cached_tokens") or 0),
                reasoning_tokens=int(completion_details.get("reasoning_tokens") or 0),
            ),
            raw=body,
        )
