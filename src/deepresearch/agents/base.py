from __future__ import annotations

import json
from typing import TypeVar

import httpx
from pydantic import BaseModel, ValidationError

from ..config import Settings
from ..model_gateway import ModelCallSignature, ModelGateway
from ..model_gateway.context import current_model_call_context


SchemaT = TypeVar("SchemaT", bound=BaseModel)


class BaseDeepSeekAgent:
    """Shared DeepSeek transport and typed-JSON repair behavior.

    This class deliberately contains no research-role prompt or policy. Role
    agents inherit it only to reuse the model boundary.
    """

    def __init__(self, settings: Settings, gateway: ModelGateway | None = None):
        if not settings.deepseek_api_key:
            raise RuntimeError("DEEPSEEK_API_KEY is required")
        self.api_key = settings.deepseek_api_key
        self.timeout = settings.request_timeout_seconds
        self.gateway = gateway or ModelGateway(settings)

    def _json_completion(
        self,
        *,
        model: str,
        system_prompt: str,
        user_prompt: str,
        schema: type[SchemaT],
        max_tokens: int,
        profile: str = "standard_research",
    ) -> SchemaT:
        last_error: Exception | None = None
        original_messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        messages = original_messages
        current_max_tokens = max_tokens
        max_attempts = 4
        for attempt in range(max_attempts):
            content = ""
            finish_reason = None
            try:
                context = current_model_call_context()
                response = self.gateway.complete(
                    messages=messages,
                    signature=ModelCallSignature(
                        run_id=context.run_id,
                        task_id=context.task_id,
                        agent_id=self.__class__.__name__,
                        operation=context.operation,
                        profile=profile,
                        attempt=attempt + 1,
                    ),
                    preferred_model=model,
                    max_tokens=current_max_tokens,
                )
                finish_reason = response.finish_reason
                content = response.content
                if not content.strip():
                    raise ValueError("DeepSeek returned empty JSON content")
                return schema.model_validate_json(content)
            except ValidationError as error:
                last_error = error
                if attempt < max_attempts - 1 and content:
                    issues = error.errors(include_input=False, include_url=False)
                    truncated = finish_reason == "length" or self._is_truncated_json(
                        issues
                    )
                    if truncated:
                        current_max_tokens = max(
                            current_max_tokens + 1000,
                            current_max_tokens * 2,
                        )
                        messages = original_messages + [
                            {
                                "role": "user",
                                "content": (
                                    "The previous response was truncated. Regenerate the "
                                    "complete JSON from the beginning. Keep it concise, "
                                    "close every string and container, and include no "
                                    "commentary."
                                ),
                            }
                        ]
                    else:
                        messages = original_messages + [
                            {"role": "assistant", "content": content},
                            {
                                "role": "user",
                                "content": (
                                    "Your JSON failed schema validation. Correct the JSON "
                                    "without adding commentary. Validation issues:\n"
                                    + json.dumps(issues, ensure_ascii=False)
                                ),
                            },
                        ]
            except (httpx.HTTPError, KeyError, ValueError) as error:
                last_error = error
        raise RuntimeError(
            f"DeepSeek JSON call failed after {max_attempts} attempts: {last_error}"
        )

    @staticmethod
    def _is_truncated_json(issues: list[dict[str, object]]) -> bool:
        """Recognize parser failures that commonly mean the output ended early."""
        truncation_markers = (
            "eof while parsing",
            "unterminated string",
            "expected `,` or `}` at eof",
            "expected `,` or `]` at eof",
        )
        for issue in issues:
            if issue.get("type") != "json_invalid":
                continue
            message = str(issue.get("msg", "")).lower()
            context = str(issue.get("ctx", "")).lower()
            if any(marker in f"{message} {context}" for marker in truncation_markers):
                return True
        return False
