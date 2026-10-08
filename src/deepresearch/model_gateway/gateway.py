from __future__ import annotations

import json
import time
import uuid

from ..config import Settings
from .providers import DeepSeekAdapter
from .router import DeterministicRouter
from .schemas import (
    GatewayResponse,
    ModelCallSignature,
    ModelSpec,
    RoutingProfile,
)
from .telemetry import ModelTelemetryStore


def _estimate_tokens(messages: list[dict[str, str]]) -> int:
    # Conservative dependency-free estimate; provider usage replaces it after completion.
    chars = len(json.dumps(messages, ensure_ascii=False))
    return max(1, (chars + 2) // 3)


class ModelGateway:
    """Single policy, routing, budget, provider, and telemetry boundary."""

    def __init__(self, settings: Settings):
        if not settings.deepseek_api_key:
            raise RuntimeError("DEEPSEEK_API_KEY is required")
        self.settings = settings
        self.telemetry = ModelTelemetryStore(settings.model_gateway_db)
        self.adapter = DeepSeekAdapter(
            api_key=settings.deepseek_api_key,
            timeout=settings.request_timeout_seconds,
        )
        self.models = self._build_models(settings)
        if settings.model_gateway_budget_usd:
            missing_prices = [
                spec.name
                for spec in self.models.values()
                if spec.input_cost_per_million <= 0
                or spec.output_cost_per_million <= 0
            ]
            if missing_prices:
                raise RuntimeError(
                    "A model-gateway budget requires positive input/output prices "
                    "for every configured model. Missing: "
                    + ", ".join(sorted(missing_prices))
                )
        self.profiles = self._build_profiles(settings)
        self.router = DeterministicRouter(
            models=self.models,
            profiles=self.profiles,
            budget_usd=settings.model_gateway_budget_usd,
        )

    @staticmethod
    def _build_models(settings: Settings) -> dict[str, ModelSpec]:
        names = {settings.planner_model, settings.fast_model, settings.audit_model}
        models: dict[str, ModelSpec] = {}
        for name in names:
            prices = settings.model_gateway_pricing.get(name, {})
            models[name] = ModelSpec(
                name=name,
                provider="deepseek",
                premium=name != settings.fast_model,
                context_window=settings.model_gateway_context_window,
                input_cost_per_million=float(prices.get("input", 0.0)),
                output_cost_per_million=float(prices.get("output", 0.0)),
            )
        return models

    @staticmethod
    def _build_profiles(settings: Settings) -> dict[str, RoutingProfile]:
        premium = settings.model_gateway_premium_threshold
        return {
            "research_planning": RoutingProfile(
                "research_planning",
                (settings.planner_model, settings.fast_model),
                max(premium, 0.95),
            ),
            "research_audit": RoutingProfile(
                "research_audit",
                (settings.audit_model, settings.fast_model),
                max(premium, 0.95),
            ),
            "audit_specialist": RoutingProfile(
                "audit_specialist",
                (settings.fast_model, settings.audit_model),
                premium,
            ),
            "deep_reasoning": RoutingProfile(
                "deep_reasoning",
                (settings.planner_model, settings.audit_model, settings.fast_model),
                max(premium, 0.90),
            ),
            "standard_research": RoutingProfile(
                "standard_research",
                (settings.fast_model, settings.planner_model),
                premium,
            ),
            "lightweight_extraction": RoutingProfile(
                "lightweight_extraction",
                (settings.fast_model,),
                min(premium, 0.60),
            ),
        }

    def complete(
        self,
        *,
        messages: list[dict[str, str]],
        signature: ModelCallSignature,
        preferred_model: str | None,
        max_tokens: int,
    ) -> GatewayResponse:
        estimated_input = _estimate_tokens(messages)
        current_spend = self.telemetry.spend(signature.run_id)
        spec, routing_reason = self.router.route(
            profile_name=signature.profile,
            preferred_model=preferred_model,
            estimated_context_tokens=estimated_input + max_tokens,
            current_spend_usd=current_spend,
        )
        estimated_cost = spec.estimate_cost(estimated_input, max_tokens)
        call_id = f"mc_{uuid.uuid4().hex}"
        self.telemetry.reserve(
            call_id=call_id,
            signature=signature,
            spec=spec,
            estimated_input_tokens=estimated_input,
            max_output_tokens=max_tokens,
            estimated_cost_usd=estimated_cost,
            routing_reason=routing_reason,
            budget_usd=self.settings.model_gateway_budget_usd,
        )
        started = time.perf_counter()
        try:
            response = self.adapter.complete(
                model=spec.name,
                messages=messages,
                max_tokens=max_tokens,
                response_format={"type": "json_object"},
            )
        except Exception as error:
            self.telemetry.fail(
                call_id=call_id,
                error=str(error),
                latency_ms=round((time.perf_counter() - started) * 1000),
            )
            raise
        usage = response.usage
        if not usage.input_tokens and not usage.output_tokens:
            from .schemas import ModelUsage

            usage = ModelUsage(
                input_tokens=estimated_input,
                output_tokens=max(1, (len(response.content) + 2) // 3),
            )
        actual_cost = spec.estimate_cost(usage.input_tokens, usage.output_tokens)
        self.telemetry.succeed(
            call_id=call_id,
            usage=usage,
            actual_cost_usd=actual_cost,
            finish_reason=response.finish_reason,
            latency_ms=round((time.perf_counter() - started) * 1000),
        )
        return GatewayResponse(
            content=response.content,
            model=response.model,
            provider=spec.provider,
            finish_reason=response.finish_reason,
            usage=usage,
            call_id=call_id,
            routing_reason=routing_reason,
            raw=response.raw,
        )
