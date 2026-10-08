from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class ModelCallSignature:
    run_id: str
    task_id: str | None
    agent_id: str
    operation: str
    profile: str
    attempt: int = 1


@dataclass(frozen=True)
class ModelUsage:
    input_tokens: int = 0
    output_tokens: int = 0
    cached_input_tokens: int = 0
    reasoning_tokens: int = 0

    @property
    def context_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


@dataclass(frozen=True)
class GatewayResponse:
    content: str
    model: str
    provider: str
    finish_reason: str | None
    usage: ModelUsage
    call_id: str
    routing_reason: str
    raw: dict[str, Any] = field(default_factory=dict, repr=False)


@dataclass(frozen=True)
class ModelSpec:
    name: str
    provider: str
    premium: bool
    context_window: int
    input_cost_per_million: float = 0.0
    output_cost_per_million: float = 0.0
    modalities: frozenset[str] = frozenset({"text"})
    structured_output: bool = True

    def estimate_cost(self, input_tokens: int, output_tokens: int) -> float:
        return (
            input_tokens * self.input_cost_per_million
            + output_tokens * self.output_cost_per_million
        ) / 1_000_000


@dataclass(frozen=True)
class RoutingProfile:
    name: str
    preferred_models: tuple[str, ...]
    premium_allowed_until: float
    required_modalities: frozenset[str] = frozenset({"text"})
    requires_structured_output: bool = True
