from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

from .config import Settings, load_env_file
from .store import EvidenceStore


OFFICIAL_REPOSITORY = "https://github.com/Ayanami0730/deep_research_bench.git"
OFFICIAL_COMMIT = "469cce54ea7f6a63c163d3d9fec879cf289ec484"
DEFAULT_BENCHMARK_DIR = Path("benchmarks/deep_research_bench")


@dataclass(frozen=True)
class BenchmarkTask:
    task_id: int | str
    topic: str
    language: str
    prompt: str

    @classmethod
    def from_payload(cls, payload: dict[str, Any], line_number: int) -> "BenchmarkTask":
        missing = [key for key in ("id", "prompt") if key not in payload]
        if missing:
            raise ValueError(
                f"Benchmark row {line_number} is missing: {', '.join(missing)}"
            )
        language = str(payload.get("language") or "").strip()
        if language not in {"en", "zh"}:
            raise ValueError(
                f"Benchmark row {line_number} has unsupported language={language!r}"
            )
        prompt = str(payload["prompt"]).strip()
        if not prompt:
            raise ValueError(f"Benchmark row {line_number} has an empty prompt")
        return cls(
            task_id=payload["id"],
            topic=str(payload.get("topic") or "unknown"),
            language=language,
            prompt=prompt,
        )

    def query_payload(self) -> dict[str, Any]:
        return {
            "id": self.task_id,
            "topic": self.topic,
            "language": self.language,
            "prompt": self.prompt,
        }


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError as error:
        raise ValueError(f"JSONL file does not exist: {path}") from error
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"Invalid JSONL at {path}:{line_number}: {error}") from error
        if not isinstance(payload, dict):
            raise ValueError(f"Expected an object at {path}:{line_number}")
        rows.append(payload)
    return rows


def load_tasks(
    dataset: Path,
    *,
    language: str = "all",
    task_ids: set[str] | None = None,
    offset: int = 0,
    limit: int | None = None,
) -> list[BenchmarkTask]:
    if offset < 0:
        raise ValueError("offset cannot be negative")
    tasks = [
        BenchmarkTask.from_payload(payload, index)
        for index, payload in enumerate(_read_jsonl(dataset), start=1)
    ]
    if language != "all":
        tasks = [task for task in tasks if task.language == language]
    if task_ids:
        tasks = [task for task in tasks if str(task.task_id) in task_ids]
    return tasks[offset:][:limit]


def _safe_name(value: str) -> str:
    normalized = re.sub(r"[^a-zA-Z0-9._-]+", "-", value.strip()).strip("-._")
    if not normalized:
        raise ValueError("model name must contain at least one safe character")
    return normalized[:80]


def _run_id(model_name: str, task_id: int | str) -> str:
    task_component = _safe_name(str(task_id))
    return f"drb_{_safe_name(model_name)}_{task_component}"


def _write_jsonl(path: Path, rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = "".join(
        json.dumps(row, ensure_ascii=False) + "\n" for row in rows
    )
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(payload, encoding="utf-8")
    temporary.replace(path)


def _start_evaluation_manifest(
    output_root: Path,
    expected_ids: set[str],
    input_path: Path,
    queries_path: Path,
) -> None:
    """Bind all phase results to the exact article and query inputs."""
    manifest = {
        "expected_ids": sorted(expected_ids),
        "input_path": str(input_path),
        "input_sha256": hashlib.sha256(input_path.read_bytes()).hexdigest(),
        "queries_path": str(queries_path),
        "queries_sha256": hashlib.sha256(queries_path.read_bytes()).hexdigest(),
        "phases": {},
    }
    (output_root / "evaluation_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def _evaluation_ids(path: Path) -> tuple[set[str], set[str]]:
    if not path.exists():
        return set(), set()
    rows = _read_jsonl(path)
    succeeded = {str(row["id"]) for row in rows if "id" in row and "error" not in row}
    failed = {str(row["id"]) for row in rows if "id" in row and "error" in row}
    return succeeded, failed


def _record_evaluation_coverage(
    output_root: Path,
    expected_ids: set[str],
    phase: str,
    result_path: Path,
) -> None:
    succeeded, failed = _evaluation_ids(result_path)
    scored_expected = succeeded & expected_ids
    failed_expected = failed & expected_ids
    unexpected = (succeeded | failed) - expected_ids
    missing = expected_ids - scored_expected - failed_expected
    manifest_path = output_root / "evaluation_manifest.json"
    manifest = (
        json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest_path.exists()
        else {"expected_ids": sorted(expected_ids), "phases": {}}
    )
    manifest["phases"][phase] = {
        "expected": len(expected_ids),
        "succeeded": len(scored_expected),
        "failed": len(failed_expected),
        "succeeded_ids": sorted(scored_expected),
        "failed_ids": sorted(failed_expected),
        "missing_ids": sorted(missing),
        "unexpected_ids": sorted(unexpected),
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    if missing or failed_expected or unexpected:
        details = []
        if missing:
            details.append("missing=" + ",".join(sorted(missing)))
        if failed_expected:
            details.append("failed=" + ",".join(sorted(failed_expected)))
        if unexpected:
            details.append("unexpected=" + ",".join(sorted(unexpected)))
        raise RuntimeError(
            f"Official {phase.upper()} evaluation did not score every input: "
            + "; ".join(details)
        )


def _completed_articles(path: Path) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    return {str(row["id"]): row for row in _read_jsonl(path) if "id" in row}


def fetch_official_benchmark(target: Path) -> None:
    target = target.resolve()
    if target.exists() and not (target / ".git").exists():
        raise ValueError(f"Target exists but is not a git checkout: {target}")
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        subprocess.run(
            ["git", "clone", "--no-checkout", OFFICIAL_REPOSITORY, str(target)],
            check=True,
        )
    subprocess.run(
        ["git", "-C", str(target), "fetch", "--depth", "1", "origin", OFFICIAL_COMMIT],
        check=True,
    )
    subprocess.run(
        ["git", "-C", str(target), "checkout", "--detach", OFFICIAL_COMMIT],
        check=True,
    )
    dataset = target / "data" / "prompt_data" / "query.jsonl"
    if not dataset.exists():
        raise RuntimeError(f"Official checkout is missing query.jsonl: {target}")


def run_tasks(args: argparse.Namespace) -> int:
    workspace = args.workspace.resolve()
    dataset = args.dataset.resolve()
    all_tasks = load_tasks(dataset)
    selected = load_tasks(
        dataset,
        language=args.language,
        task_ids={item.strip() for item in args.ids.split(",") if item.strip()}
        if args.ids else None,
        offset=args.offset,
        limit=None if args.all else args.limit,
    )
    if not selected:
        raise ValueError("No benchmark tasks match the requested selection")

    model_name = _safe_name(args.model_name)
    output = (
        args.output.resolve()
        if args.output
        else workspace
        / "benchmark_results"
        / "deepresearch_bench"
        / f"{model_name}.jsonl"
    )
    queries_output = output.with_name(f"{output.stem}.queries.jsonl")
    manifest_output = output.with_name(f"{output.stem}.manifest.jsonl")
    articles = _completed_articles(output)
    manifest = _completed_articles(manifest_output)

    if args.dry_run:
        print(json.dumps({
            "dataset": str(dataset),
            "tasks": [task.query_payload() for task in selected],
            "output": str(output),
        }, ensure_ascii=False, indent=2))
        return 0

    load_env_file(workspace / ".env")
    settings = Settings.from_env(workspace)
    settings.ensure_directories()
    store = EvidenceStore(settings.research_db)

    from .cli import main as research_main

    failures = 0
    task_by_id = {str(task.task_id): task for task in all_tasks}
    for position, task in enumerate(selected, start=1):
        task_key = str(task.task_id)
        run_id = _run_id(model_name, task.task_id)
        if task_key in articles:
            print(
                f"[{position}/{len(selected)}] task {task.task_id}: "
                "output exists, skipping"
            )
            continue

        existing = store.get_run(run_id)
        cli_args = ["--workspace", str(workspace)]
        if args.fake:
            cli_args.append("--fake")
        if args.quiet:
            cli_args.append("--quiet")
        if existing:
            cli_args.extend(["--resume", run_id])
        else:
            cli_args.extend([task.prompt, "--run-id", run_id])

        print(f"[{position}/{len(selected)}] task {task.task_id}: run_id={run_id}")
        return_code = research_main(cli_args)
        run = store.get_run(run_id)
        report_path = (
            Path(str(run.get("report_path")))
            if run and run.get("report_path")
            else None
        )
        if return_code != 0 or not report_path or not report_path.exists():
            failures += 1
            manifest[task_key] = {
                "id": task.task_id,
                "prompt": task.prompt,
                "run_id": run_id,
                "status": run.get("status") if run else "missing",
                "report_path": str(report_path) if report_path else None,
            }
            _write_jsonl(manifest_output, manifest.values())
            if not args.continue_on_error:
                break
            continue

        articles[task_key] = {
            "id": task.task_id,
            "prompt": task.prompt,
            "article": report_path.read_text(encoding="utf-8"),
        }
        manifest[task_key] = {
            "id": task.task_id,
            "prompt": task.prompt,
            "run_id": run_id,
            "status": "completed",
            "report_path": str(report_path),
        }
        _write_jsonl(output, articles.values())
        _write_jsonl(manifest_output, manifest.values())
        completed_ids = set(articles)
        _write_jsonl(
            queries_output,
            (
                task_by_id[key].query_payload()
                for key in task_by_id if key in completed_ids
            ),
        )

    completed_ids = set(articles)
    unknown_ids = completed_ids - set(task_by_id)
    if unknown_ids:
        raise ValueError(
            "Output contains task IDs absent from the dataset: "
            + ", ".join(sorted(unknown_ids))
        )
    _write_jsonl(
        queries_output,
        (
            task_by_id[key].query_payload()
            for key in task_by_id if key in completed_ids
        ),
    )

    print(f"Official-format articles: {output}")
    print(f"Matching query subset: {queries_output}")
    print(f"Run manifest: {manifest_output}")
    return 1 if failures else 0


def _require_eval_credentials(phase: str) -> None:
    backend = os.getenv("LLM_BACKEND", "openrouter").lower()
    key_name = "OPENAI_API_KEY" if backend == "openai" else "OPENROUTER_API_KEY"
    missing = [key_name] if not os.getenv(key_name) else []
    if phase in {"fact", "all"} and not os.getenv("JINA_API_KEY"):
        missing.append("JINA_API_KEY")
    if missing:
        raise ValueError("Official evaluator requires: " + ", ".join(missing))


def evaluate_official(args: argparse.Namespace) -> int:
    load_env_file(Path.cwd() / ".env")
    benchmark_dir = args.benchmark_dir.resolve()
    input_path = args.input.resolve()
    queries = (
        args.queries.resolve()
        if args.queries
        else input_path.with_name(f"{input_path.stem}.queries.jsonl")
    )
    if not (benchmark_dir / "deepresearch_bench_race.py").exists():
        raise ValueError(
            f"Not an official DeepResearch Bench checkout: {benchmark_dir}"
        )
    if not input_path.exists() or not queries.exists():
        raise ValueError(
            "Both official-format article JSONL and matching queries are required"
        )
    _require_eval_credentials(args.phase)

    model_name = _safe_name(args.model_name or input_path.stem)
    raw_dir = benchmark_dir / "data" / "test_data" / "raw_data"
    staged_input = raw_dir / f"{model_name}.jsonl"
    if input_path != staged_input:
        shutil.copyfile(input_path, staged_input)
    output_root = (
        args.output_dir.resolve()
        if args.output_dir
        else input_path.parent / "official_eval" / model_name
    )
    output_root.mkdir(parents=True, exist_ok=True)
    expected_ids = {
        str(row["id"]) for row in _read_jsonl(queries) if "id" in row
    }
    _start_evaluation_manifest(output_root, expected_ids, input_path, queries)

    def run(command: list[str]) -> None:
        subprocess.run(command, cwd=benchmark_dir, check=True)

    if args.phase in {"race", "all"}:
        race_output = output_root / "race"
        race_output.mkdir(parents=True, exist_ok=True)
        run([
            sys.executable, "-u", "deepresearch_bench_race.py", model_name,
            "--raw_data_dir", str(raw_dir),
            "--max_workers", str(args.max_workers),
            "--query_file", str(queries),
            "--output_dir", str(race_output),
            "--force",
        ])
        _record_evaluation_coverage(
            output_root,
            expected_ids,
            "race",
            race_output / "raw_results.jsonl",
        )

    if args.phase in {"fact", "all"}:
        fact_output = output_root / "fact"
        fact_output.mkdir(parents=True, exist_ok=True)
        extracted = fact_output / "extracted.jsonl"
        deduplicated = fact_output / "deduplicated.jsonl"
        scraped = fact_output / "scraped.jsonl"
        validated = fact_output / "validated.jsonl"
        fact_result = fact_output / "fact_result.txt"
        # Official FACT utilities append and skip IDs already present. Clear
        # only their known generated files so a rerun always scores this input.
        for generated in (
            extracted,
            deduplicated,
            scraped,
            validated,
            fact_result,
        ):
            generated.unlink(missing_ok=True)
        workers = str(args.max_workers)
        run([
            sys.executable, "-u", "-m", "utils.extract",
            "--raw_data_path", str(staged_input),
            "--output_path", str(extracted),
            "--query_data_path", str(queries),
            "--n_total_process", workers,
        ])
        run([
            sys.executable, "-u", "-m", "utils.deduplicate",
            "--raw_data_path", str(extracted),
            "--output_path", str(deduplicated),
            "--query_data_path", str(queries),
            "--n_total_process", workers,
        ])
        run([
            sys.executable, "-u", "-m", "utils.scrape",
            "--raw_data_path", str(deduplicated),
            "--output_path", str(scraped),
            "--n_total_process", workers,
        ])
        run([
            sys.executable, "-u", "-m", "utils.validate",
            "--raw_data_path", str(scraped),
            "--output_path", str(validated),
            "--query_data_path", str(queries),
            "--n_total_process", workers,
        ])
        run([
            sys.executable, "-u", "-m", "utils.stat",
            "--input_path", str(validated),
            "--output_path", str(fact_result),
        ])
        _record_evaluation_coverage(
            output_root,
            expected_ids,
            "fact",
            validated,
        )

    print(f"Official evaluation results: {output_root}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="deep-research-benchmark",
        description="DeepResearch Bench adapter for evidence-deep-research",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    fetch = subparsers.add_parser(
        "fetch", help="Fetch the pinned official benchmark"
    )
    fetch.add_argument("--target", type=Path, default=DEFAULT_BENCHMARK_DIR)

    run = subparsers.add_parser(
        "run", help="Run selected benchmark tasks and export JSONL"
    )
    run.add_argument("--dataset", type=Path, required=True)
    run.add_argument("--workspace", type=Path, default=Path.cwd())
    run.add_argument("--model-name", default="evidence-deep-research")
    run.add_argument("--output", type=Path)
    run.add_argument("--language", choices=("all", "en", "zh"), default="all")
    run.add_argument("--ids", help="Comma-separated official task IDs")
    run.add_argument("--offset", type=int, default=0)
    selection = run.add_mutually_exclusive_group()
    selection.add_argument("--limit", type=int)
    selection.add_argument("--all", action="store_true")
    run.add_argument("--fake", action="store_true")
    run.add_argument("--dry-run", action="store_true")
    run.add_argument("--quiet", action="store_true")
    run.add_argument("--continue-on-error", action="store_true")

    evaluate = subparsers.add_parser(
        "evaluate", help="Run the official RACE/FACT evaluator"
    )
    evaluate.add_argument("--benchmark-dir", type=Path, default=DEFAULT_BENCHMARK_DIR)
    evaluate.add_argument("--input", type=Path, required=True)
    evaluate.add_argument("--queries", type=Path)
    evaluate.add_argument("--model-name")
    evaluate.add_argument("--phase", choices=("race", "fact", "all"), default="all")
    evaluate.add_argument("--max-workers", type=int, default=4)
    evaluate.add_argument("--output-dir", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "fetch":
            fetch_official_benchmark(args.target)
            print(f"Pinned DeepResearch Bench checkout: {args.target.resolve()}")
            return 0
        if args.command == "run":
            if not args.all and args.limit is None and not args.ids:
                parser.error("run requires --limit, --ids, or explicit --all")
            if args.limit is not None and args.limit < 1:
                parser.error("--limit must be positive")
            if args.offset < 0:
                parser.error("--offset cannot be negative")
            return run_tasks(args)
        if args.max_workers < 1:
            parser.error("--max-workers must be positive")
        return evaluate_official(args)
    except (OSError, subprocess.CalledProcessError, RuntimeError, ValueError) as error:
        print(f"Benchmark error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
