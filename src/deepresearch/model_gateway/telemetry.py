from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Iterator

from .schemas import ModelCallSignature, ModelSpec, ModelUsage


def _utc_now() -> str:
    return datetime.now(UTC).isoformat()


class BudgetExceededError(RuntimeError):
    pass


class ModelTelemetryStore:
    """Durable model-call ledger; reservations make parallel budget checks safe."""

    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        connection = sqlite3.connect(self.path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA busy_timeout = 30000")
        try:
            yield connection
            connection.commit()
        except Exception:
            connection.rollback()
            raise
        finally:
            connection.close()

    def _initialize(self) -> None:
        with self.connect() as connection:
            connection.execute("PRAGMA journal_mode = WAL")
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS model_calls (
                    call_id TEXT PRIMARY KEY,
                    created_at TEXT NOT NULL,
                    completed_at TEXT,
                    run_id TEXT NOT NULL,
                    task_id TEXT,
                    agent_id TEXT NOT NULL,
                    operation TEXT NOT NULL,
                    profile TEXT NOT NULL,
                    attempt INTEGER NOT NULL,
                    provider TEXT NOT NULL,
                    selected_model TEXT NOT NULL,
                    status TEXT NOT NULL,
                    routing_reason TEXT NOT NULL,
                    finish_reason TEXT,
                    estimated_input_tokens INTEGER NOT NULL,
                    input_tokens INTEGER NOT NULL DEFAULT 0,
                    output_tokens INTEGER NOT NULL DEFAULT 0,
                    cached_input_tokens INTEGER NOT NULL DEFAULT 0,
                    reasoning_tokens INTEGER NOT NULL DEFAULT 0,
                    max_output_tokens INTEGER NOT NULL,
                    context_window INTEGER NOT NULL,
                    estimated_cost_usd REAL NOT NULL,
                    actual_cost_usd REAL NOT NULL DEFAULT 0,
                    latency_ms INTEGER,
                    error TEXT
                );
                CREATE INDEX IF NOT EXISTS idx_model_calls_run
                    ON model_calls(run_id);
                CREATE INDEX IF NOT EXISTS idx_model_calls_agent
                    ON model_calls(run_id, agent_id, operation);
                """
            )

    def spend(self, run_id: str | None = None) -> float:
        where = "WHERE run_id = ?" if run_id else ""
        params = (run_id,) if run_id else ()
        with self.connect() as connection:
            row = connection.execute(
                f"""
                SELECT COALESCE(SUM(
                    CASE WHEN status = 'reserved'
                         THEN estimated_cost_usd ELSE actual_cost_usd END
                ), 0) AS spend
                FROM model_calls {where}
                """,
                params,
            ).fetchone()
        return float(row["spend"])

    def reserve(
        self,
        *,
        call_id: str,
        signature: ModelCallSignature,
        spec: ModelSpec,
        estimated_input_tokens: int,
        max_output_tokens: int,
        estimated_cost_usd: float,
        routing_reason: str,
        budget_usd: float | None,
    ) -> float:
        with self.connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            row = connection.execute(
                """
                SELECT COALESCE(SUM(
                    CASE WHEN status = 'reserved'
                         THEN estimated_cost_usd ELSE actual_cost_usd END
                ), 0) AS spend
                FROM model_calls WHERE run_id = ?
                """,
                (signature.run_id,),
            ).fetchone()
            current_spend = float(row["spend"])
            if budget_usd and current_spend + estimated_cost_usd > budget_usd:
                raise BudgetExceededError(
                    f"Model budget exceeded for {signature.run_id}: "
                    f"${current_spend + estimated_cost_usd:.4f} > ${budget_usd:.4f}"
                )
            connection.execute(
                """
                INSERT INTO model_calls (
                    call_id, created_at, run_id, task_id, agent_id, operation,
                    profile, attempt, provider, selected_model, status,
                    routing_reason, estimated_input_tokens, max_output_tokens,
                    context_window, estimated_cost_usd
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'reserved', ?, ?, ?, ?, ?)
                """,
                (
                    call_id,
                    _utc_now(),
                    signature.run_id,
                    signature.task_id,
                    signature.agent_id,
                    signature.operation,
                    signature.profile,
                    signature.attempt,
                    spec.provider,
                    spec.name,
                    routing_reason,
                    estimated_input_tokens,
                    max_output_tokens,
                    spec.context_window,
                    estimated_cost_usd,
                ),
            )
        return current_spend

    def succeed(
        self,
        *,
        call_id: str,
        usage: ModelUsage,
        actual_cost_usd: float,
        finish_reason: str | None,
        latency_ms: int,
    ) -> None:
        with self.connect() as connection:
            connection.execute(
                """
                UPDATE model_calls SET status='succeeded', completed_at=?,
                    finish_reason=?, input_tokens=?, output_tokens=?,
                    cached_input_tokens=?, reasoning_tokens=?, actual_cost_usd=?,
                    latency_ms=? WHERE call_id=?
                """,
                (
                    _utc_now(), finish_reason, usage.input_tokens,
                    usage.output_tokens, usage.cached_input_tokens,
                    usage.reasoning_tokens, actual_cost_usd, latency_ms, call_id,
                ),
            )

    def fail(self, *, call_id: str, error: str, latency_ms: int) -> None:
        with self.connect() as connection:
            connection.execute(
                """
                UPDATE model_calls SET status='failed', completed_at=?, error=?,
                    actual_cost_usd=0, latency_ms=? WHERE call_id=?
                """,
                (_utc_now(), error, latency_ms, call_id),
            )

    def summary(self, run_id: str | None = None) -> list[dict[str, object]]:
        where = "WHERE run_id = ?" if run_id else ""
        params = (run_id,) if run_id else ()
        with self.connect() as connection:
            rows = connection.execute(
                f"""
                SELECT run_id, task_id, agent_id, operation, selected_model,
                       COUNT(*) AS calls,
                       SUM(CASE WHEN status='succeeded' THEN 1 ELSE 0 END) AS succeeded,
                       SUM(input_tokens) AS input_tokens,
                       SUM(output_tokens) AS output_tokens,
                       SUM(actual_cost_usd) AS cost_usd,
                       MAX(CAST(input_tokens + output_tokens AS REAL)
                           / context_window) AS max_context_ratio
                FROM model_calls {where}
                GROUP BY run_id, task_id, agent_id, operation, selected_model
                ORDER BY run_id, task_id, agent_id, operation, selected_model
                """,
                params,
            ).fetchall()
        return [dict(row) for row in rows]

    def list_calls(self, run_id: str) -> list[dict[str, object]]:
        with self.connect() as connection:
            rows = connection.execute(
                """
                SELECT *,
                       CASE WHEN context_window > 0
                            THEN CAST(
                                CASE WHEN input_tokens + output_tokens > 0
                                     THEN input_tokens + output_tokens
                                     ELSE estimated_input_tokens END
                                AS REAL)
                                 / context_window
                            ELSE 0 END AS context_ratio,
                       CASE WHEN status = 'reserved'
                            THEN estimated_cost_usd ELSE actual_cost_usd
                       END AS effective_cost_usd
                FROM model_calls
                WHERE run_id = ?
                ORDER BY created_at, call_id
                """,
                (run_id,),
            ).fetchall()
        return [dict(row) for row in rows]
