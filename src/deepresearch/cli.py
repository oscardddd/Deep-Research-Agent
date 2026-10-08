from __future__ import annotations

import argparse
import json
import sys
from contextlib import nullcontext
from datetime import datetime, timezone
from pathlib import Path

from .config import Settings, load_env_file
from .eventlog import configure_logging, log_step
from .graph import run_graph
from .local_services import managed_local_embedding_service
from .model_gateway import ModelTelemetryStore
from .providers import (
    DeepSeekModelProvider,
    FakeLanguageModelProvider,
    FakeSearchProvider,
    LanguageModelProvider,
    SearchProvider,
    TavilySearchProvider,
)
from .runtime import ResearchRuntime
from .store import EvidenceStore
from .source_scoring import SOURCE_RUBRIC_VERSION
from .visualization import render_run_visualization


def timestamped_run_id() -> str:
    """Return a sortable, filesystem-safe run ID based on the current UTC time."""
    return datetime.now(timezone.utc).strftime("run_%Y%m%dT%H%M%S_%fZ")


def _execute_research_run(
    *,
    settings: Settings,
    store: EvidenceStore,
    search_provider: SearchProvider,
    model_provider: LanguageModelProvider,
    run_id: str,
    question: str,
    resume: bool,
) -> int:
    runtime = ResearchRuntime(settings, store, search_provider, model_provider)

    try:
        max_graph_attempts = 3
        for graph_attempt in range(1, max_graph_attempts + 1):
            try:
                result = run_graph(
                    settings=settings,
                    store=store,
                    runtime=runtime,
                    run_id=run_id,
                    question=question,
                    resume=resume or graph_attempt > 1,
                )
                break
            except Exception as error:
                if graph_attempt == max_graph_attempts:
                    raise
                log_step(
                    "Orchestrator",
                    "run.auto_resume",
                    run_id=run_id,
                    attempt=graph_attempt + 1,
                    previous_error=str(error),
                )
    except Exception as error:
        store.set_run_status(run_id, "failed")
        visualization_path = None
        try:
            visualization_path = render_run_visualization(
                settings=settings,
                store=store,
                run_id=run_id,
                status_override="failed",
            )
        except Exception as visualization_error:
            print(
                f"Could not render failed-run visualization: {visualization_error}",
                file=sys.stderr,
            )
        print(f"Run {run_id} failed: {error}", file=sys.stderr)
        print(f"Resume with: deep-research --resume {run_id}", file=sys.stderr)
        if visualization_path:
            print(f"Visualization: {visualization_path}", file=sys.stderr)
        return 1

    print(f"Run ID: {run_id}")
    print(f"Terminal reason: {result.get('terminal_reason', 'unknown')}")
    print(f"Report: {result.get('report_path', 'not produced')}")
    print(f"Visualization: {result.get('visualization_path', 'not produced')}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="deep-research",
        description="Budget-bounded, evidence-first research MVP",
    )
    parser.add_argument("question", nargs="?", help="Research question")
    parser.add_argument("--run-id", help="Optional stable run ID")
    parser.add_argument("--resume", metavar="RUN_ID", help="Resume a checkpointed run")
    parser.add_argument(
        "--visualize",
        metavar="RUN_ID",
        help="Regenerate linked run and observability HTML without calling APIs",
    )
    parser.add_argument(
        "--gateway-stats",
        metavar="RUN_ID",
        help="Print model calls, tokens, context usage, and cost for a run",
    )
    parser.add_argument(
        "--fake",
        action="store_true",
        help="Use deterministic offline providers; no API keys required",
    )
    parser.add_argument(
        "--workspace",
        type=Path,
        default=Path.cwd(),
        help="Directory for SQLite state and report artifacts",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress step-by-step terminal logs",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    configure_logging(quiet=args.quiet)
    load_env_file(args.workspace.resolve() / ".env")
    settings = Settings.from_env(args.workspace)
    settings.ensure_directories()
    store = EvidenceStore(settings.research_db)

    if args.gateway_stats:
        if args.resume or args.visualize or args.question:
            print(
                "--gateway-stats cannot be combined with a question, --resume, "
                "or --visualize.",
                file=sys.stderr,
            )
            return 2
        rows = ModelTelemetryStore(settings.model_gateway_db).summary(
            args.gateway_stats
        )
        print(json.dumps(rows, ensure_ascii=False, indent=2))
        return 0

    if args.visualize:
        if args.resume or args.question:
            print("--visualize cannot be combined with a question or --resume.", file=sys.stderr)
            return 2
        if not store.get_run(args.visualize):
            print(f"Unknown run ID: {args.visualize}", file=sys.stderr)
            return 2
        path = render_run_visualization(
            settings=settings,
            store=store,
            run_id=args.visualize,
        )
        print(f"Visualization: {path}")
        return 0

    if args.resume:
        run_id = args.resume
        existing = store.get_run(run_id)
        if not existing:
            print(f"Unknown run ID: {run_id}", file=sys.stderr)
            return 2
        question = existing["question"]
    else:
        if not args.question:
            print("A research question is required unless --resume is used.", file=sys.stderr)
            return 2
        run_id = args.run_id or timestamped_run_id()
        question = args.question
        store.create_run(
            run_id,
            question,
            {
                "planner_model": settings.planner_model,
                "fast_model": settings.fast_model,
                "audit_model": settings.audit_model,
                "source_rubric_version": SOURCE_RUBRIC_VERSION,
                "max_initial_tasks": settings.max_initial_tasks,
                "max_reconnaissance_queries": (
                    settings.max_reconnaissance_queries
                ),
                "max_replan_rounds": settings.max_replan_rounds,
                "max_new_tasks_per_replan": (
                    settings.max_new_tasks_per_replan
                ),
                "search_budget": "unbounded",
                "max_query_candidates_per_worker": (
                    settings.max_query_candidates_per_worker
                ),
                "max_queries_per_worker": settings.max_queries_per_worker,
                "max_adaptive_steps": settings.max_adaptive_steps,
                "page_chunk_chars": settings.page_chunk_chars,
                "retrieval_chunk_tokens": settings.retrieval_chunk_tokens,
                "retrieval_chunk_overlap_tokens": (
                    settings.retrieval_chunk_overlap_tokens
                ),
                "extraction_window_tokens": settings.extraction_window_tokens,
                "max_retrieval_chunks_per_source": (
                    settings.max_relevant_chunks_per_source
                ),
                "max_extraction_windows_per_source": (
                    settings.max_extraction_windows_per_source
                ),
                "hybrid_retrieval_enabled": settings.hybrid_retrieval_enabled,
                "embedding_model": settings.embedding_model,
                "embedding_dimensions": settings.embedding_dimensions,
                "hybrid_embedding_weight": settings.hybrid_embedding_weight,
                "hybrid_rrf_k": settings.hybrid_rrf_k,
                "single_audit_context_token_budget": (
                    settings.single_audit_context_token_budget
                ),
                "audit_evidence_token_budget_per_check": (
                    settings.audit_evidence_token_budget_per_check
                ),
                "max_audit_specialists": settings.max_audit_specialists,
                "model_gateway_budget_usd": settings.model_gateway_budget_usd,
                "model_gateway_premium_threshold": (
                    settings.model_gateway_premium_threshold
                ),
                "fake": args.fake,
            },
        )

    if args.fake:
        search_provider = FakeSearchProvider()
        model_provider = FakeLanguageModelProvider()
    else:
        try:
            settings.require_api_keys()
        except RuntimeError as error:
            print(str(error), file=sys.stderr)
            print("Copy .env.example, export the keys, or run with --fake.", file=sys.stderr)
            return 2
        search_provider = TavilySearchProvider(settings)
        model_provider = DeepSeekModelProvider(settings)

    service_context = (
        nullcontext() if args.fake else managed_local_embedding_service(settings)
    )
    with service_context:
        return _execute_research_run(
            settings=settings,
            store=store,
            search_provider=search_provider,
            model_provider=model_provider,
            run_id=run_id,
            question=question,
            resume=bool(args.resume),
        )


if __name__ == "__main__":
    raise SystemExit(main())
