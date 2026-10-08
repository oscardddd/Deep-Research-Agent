from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Iterator


@dataclass(frozen=True)
class ModelCallContext:
    run_id: str = "unscoped"
    task_id: str | None = None
    operation: str = "unknown"


_CURRENT_CONTEXT: ContextVar[ModelCallContext] = ContextVar(
    "model_call_context",
    default=ModelCallContext(),
)


def current_model_call_context() -> ModelCallContext:
    return _CURRENT_CONTEXT.get()


@contextmanager
def model_call_scope(
    *, run_id: str, task_id: str | None, operation: str
) -> Iterator[None]:
    token = _CURRENT_CONTEXT.set(
        ModelCallContext(run_id=run_id, task_id=task_id, operation=operation)
    )
    try:
        yield
    finally:
        _CURRENT_CONTEXT.reset(token)
