from __future__ import annotations

import os
import json
from dataclasses import dataclass, field
from pathlib import Path


def load_env_file(path: Path) -> None:
    """Load a minimal KEY=VALUE env file without adding a dependency."""
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            os.environ.setdefault(key, value)


@dataclass(frozen=True)
class Settings:
    workspace: Path
    tavily_api_key: str | None
    deepseek_api_key: str | None
    planner_model: str = "deepseek-v4-pro"
    fast_model: str = "deepseek-v4-flash"
    audit_model: str = "deepseek-v4-pro"
    max_initial_tasks: int = 6
    max_follow_up_rounds: int = 2
    max_reconnaissance_queries: int = 2
    max_new_tasks_per_replan: int = 3
    max_query_candidates_per_worker: int = 3
    max_queries_per_worker: int = 2
    max_adaptive_steps: int = 6
    max_results_per_search: int = 5
    page_chunk_chars: int = 12000
    retrieval_chunk_tokens: int = 500
    retrieval_chunk_overlap_tokens: int = 75
    extraction_window_tokens: int = 1800
    max_relevant_chunks_per_source: int = 4
    max_extraction_windows_per_source: int = 2
    embedding_api_key: str | None = None
    embedding_base_url: str | None = None
    embedding_model: str = "qwen3-embedding:0.6b"
    embedding_dimensions: int = 512
    embedding_batch_size: int = 10
    hybrid_embedding_weight: float = 0.35
    hybrid_rrf_k: int = 60
    manage_local_embedding_service: bool = True
    request_timeout_seconds: float = 45.0
    model_gateway_budget_usd: float | None = None
    model_gateway_premium_threshold: float = 0.85
    model_gateway_context_window: int = 128000
    replan_context_token_budget: int = 24000
    single_audit_context_token_budget: int = 30000
    audit_evidence_token_budget_per_check: int = 12000
    max_audit_specialists: int = 4
    model_gateway_pricing: dict[str, dict[str, float]] = field(default_factory=dict)

    @property
    def data_dir(self) -> Path:
        return self.workspace / ".deepresearch"

    @property
    def research_db(self) -> Path:
        return self.data_dir / "research.sqlite"

    @property
    def checkpoint_db(self) -> Path:
        return self.data_dir / "checkpoints.sqlite"

    @property
    def model_gateway_db(self) -> Path:
        return self.data_dir / "model_gateway.sqlite"

    @property
    def runs_dir(self) -> Path:
        return self.workspace / "runs"

    @property
    def hybrid_retrieval_enabled(self) -> bool:
        """Dense retrieval is opt-in; BM25 remains the zero-config fallback."""
        return bool(self.embedding_base_url and self.hybrid_embedding_weight > 0)

    @classmethod
    def from_env(cls, workspace: Path | None = None) -> "Settings":
        root = (workspace or Path.cwd()).resolve()
        budget = float(os.getenv("MODEL_GATEWAY_BUDGET_USD", "0"))
        pricing_raw = os.getenv("MODEL_GATEWAY_PRICING_JSON", "{}")
        try:
            pricing = json.loads(pricing_raw)
        except json.JSONDecodeError as error:
            raise RuntimeError(
                f"MODEL_GATEWAY_PRICING_JSON must be valid JSON: {error}"
            ) from error
        if not isinstance(pricing, dict):
            raise RuntimeError("MODEL_GATEWAY_PRICING_JSON must be a JSON object")
        return cls(
            workspace=root,
            tavily_api_key=os.getenv("TAVILY_API_KEY"),
            deepseek_api_key=os.getenv("DEEPSEEK_API_KEY"),
            planner_model=os.getenv("DEEPSEEK_PLANNER_MODEL", "deepseek-v4-pro"),
            fast_model=os.getenv("DEEPSEEK_FAST_MODEL", "deepseek-v4-flash"),
            audit_model=os.getenv("DEEPSEEK_AUDIT_MODEL", "deepseek-v4-pro"),
            max_initial_tasks=int(os.getenv("MAX_INITIAL_TASKS", "6")),
            max_follow_up_rounds=int(
                os.getenv(
                    "MAX_REPLAN_ROUNDS",
                    os.getenv("MAX_FOLLOW_UP_ROUNDS", "2"),
                )
            ),
            max_reconnaissance_queries=max(
                0, int(os.getenv("MAX_RECONNAISSANCE_QUERIES", "2"))
            ),
            max_new_tasks_per_replan=max(
                1, int(os.getenv("MAX_NEW_TASKS_PER_REPLAN", "3"))
            ),
            max_query_candidates_per_worker=int(
                os.getenv("MAX_QUERY_CANDIDATES_PER_WORKER", "3")
            ),
            max_queries_per_worker=int(os.getenv("MAX_QUERIES_PER_WORKER", "2")),
            max_adaptive_steps=max(1, int(os.getenv("MAX_ADAPTIVE_STEPS", "6"))),
            max_results_per_search=int(os.getenv("MAX_RESULTS_PER_SEARCH", "5")),
            page_chunk_chars=max(2000, int(os.getenv("PAGE_CHUNK_CHARS", "12000"))),
            max_relevant_chunks_per_source=max(
                1,
                int(
                    os.getenv(
                        "MAX_RETRIEVAL_CHUNKS_PER_SOURCE",
                        os.getenv("MAX_RELEVANT_CHUNKS_PER_SOURCE", "4"),
                    )
                ),
            ),
            retrieval_chunk_tokens=max(
                100, int(os.getenv("RETRIEVAL_CHUNK_TOKENS", "500"))
            ),
            retrieval_chunk_overlap_tokens=max(
                0, int(os.getenv("RETRIEVAL_CHUNK_OVERLAP_TOKENS", "75"))
            ),
            extraction_window_tokens=max(
                500, int(os.getenv("EXTRACTION_WINDOW_TOKENS", "1800"))
            ),
            max_extraction_windows_per_source=max(
                1, int(os.getenv("MAX_EXTRACTION_WINDOWS_PER_SOURCE", "2"))
            ),
            embedding_api_key=(
                os.getenv("QWEN_EMBEDDING_API_KEY")
                or os.getenv("DASHSCOPE_API_KEY")
            ),
            embedding_base_url=(
                os.getenv("QWEN_EMBEDDING_BASE_URL")
                or os.getenv("DASHSCOPE_BASE_URL")
            ),
            embedding_model=os.getenv(
                "QWEN_EMBEDDING_MODEL", "qwen3-embedding:0.6b"
            ),
            embedding_dimensions=max(
                1, int(os.getenv("QWEN_EMBEDDING_DIMENSIONS", "512"))
            ),
            embedding_batch_size=max(
                1, int(os.getenv("QWEN_EMBEDDING_BATCH_SIZE", "10"))
            ),
            hybrid_embedding_weight=min(
                1.0,
                max(0.0, float(os.getenv("HYBRID_EMBEDDING_WEIGHT", "0.35"))),
            ),
            hybrid_rrf_k=max(1, int(os.getenv("HYBRID_RRF_K", "60"))),
            manage_local_embedding_service=os.getenv(
                "MANAGE_LOCAL_EMBEDDING_SERVICE", "true"
            ).casefold() not in {"0", "false", "no", "off"},
            request_timeout_seconds=float(
                os.getenv("REQUEST_TIMEOUT_SECONDS", "45")
            ),
            model_gateway_budget_usd=budget if budget > 0 else None,
            model_gateway_premium_threshold=min(
                1.0,
                max(0.0, float(os.getenv("MODEL_GATEWAY_PREMIUM_THRESHOLD", "0.85"))),
            ),
            model_gateway_context_window=max(
                1, int(os.getenv("MODEL_GATEWAY_CONTEXT_WINDOW", "128000"))
            ),
            replan_context_token_budget=max(
                4000, int(os.getenv("REPLAN_CONTEXT_TOKEN_BUDGET", "24000"))
            ),
            single_audit_context_token_budget=max(
                8000, int(os.getenv("SINGLE_AUDIT_CONTEXT_TOKEN_BUDGET", "30000"))
            ),
            audit_evidence_token_budget_per_check=max(
                4000,
                int(os.getenv("AUDIT_EVIDENCE_TOKEN_BUDGET_PER_CHECK", "12000")),
            ),
            max_audit_specialists=max(
                1, int(os.getenv("MAX_AUDIT_SPECIALISTS", "4"))
            ),
            model_gateway_pricing=pricing,
        )

    @property
    def initial_query_budget_per_worker(self) -> int:
        """Per-worker cap; initial workers no longer share a global call budget."""
        return self.max_queries_per_worker

    @property
    def max_replan_rounds(self) -> int:
        """Preferred name; keep max_follow_up_rounds for config compatibility."""
        return self.max_follow_up_rounds

    def ensure_directories(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.runs_dir.mkdir(parents=True, exist_ok=True)

    def require_api_keys(self) -> None:
        missing: list[str] = []
        if not self.tavily_api_key:
            missing.append("TAVILY_API_KEY")
        if not self.deepseek_api_key:
            missing.append("DEEPSEEK_API_KEY")
        if missing:
            raise RuntimeError(
                "Missing required environment variables: " + ", ".join(missing)
            )
