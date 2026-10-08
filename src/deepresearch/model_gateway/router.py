from __future__ import annotations

from .schemas import ModelSpec, RoutingProfile


class NoEligibleModelError(RuntimeError):
    pass


class DeterministicRouter:
    def __init__(
        self,
        *,
        models: dict[str, ModelSpec],
        profiles: dict[str, RoutingProfile],
        budget_usd: float | None,
    ):
        self.models = models
        self.profiles = profiles
        self.budget_usd = budget_usd

    def route(
        self,
        *,
        profile_name: str,
        preferred_model: str | None,
        estimated_context_tokens: int,
        current_spend_usd: float,
    ) -> tuple[ModelSpec, str]:
        profile = self.profiles.get(profile_name) or self.profiles["standard_research"]
        ordered_names = list(profile.preferred_models)
        # Concrete model names are a compatibility hint only. A declared profile
        # owns routing; otherwise callers could bypass budget policy by naming a
        # premium model directly.
        if profile_name not in self.profiles and preferred_model in self.models:
            ordered_names = [preferred_model] + [
                name for name in ordered_names if name != preferred_model
            ]
        candidates = [self.models[name] for name in ordered_names if name in self.models]
        candidates = [
            spec
            for spec in candidates
            if estimated_context_tokens <= spec.context_window
            and profile.required_modalities.issubset(spec.modalities)
            and (not profile.requires_structured_output or spec.structured_output)
        ]
        budget_ratio = (
            current_spend_usd / self.budget_usd
            if self.budget_usd and self.budget_usd > 0
            else 0.0
        )
        premium_blocked = budget_ratio >= profile.premium_allowed_until
        if premium_blocked:
            candidates = [spec for spec in candidates if not spec.premium]
        if not candidates:
            raise NoEligibleModelError(
                f"No model satisfies profile={profile.name}, "
                f"context={estimated_context_tokens}, budget_ratio={budget_ratio:.1%}"
            )
        selected = candidates[0]
        reason = (
            f"profile={profile.name}; budget={budget_ratio:.1%}; "
            f"premium={'blocked' if premium_blocked else 'allowed'}; "
            f"first eligible={selected.name}"
        )
        return selected, reason
