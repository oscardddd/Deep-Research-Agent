#!/bin/sh

set -eu

usage() {
  cat <<'EOF'
Usage: scripts/evaluate_benchmark.sh [race|fact|all] [input.jsonl]

Runs the official DeepResearch Bench evaluator with the OpenAI backend.
Credentials are read by the benchmark CLI from the project-root .env file.

Environment overrides:
  BENCHMARK_MAX_WORKERS  Parallel evaluator workers (default: 4)
  BENCHMARK_MODEL_NAME   Evaluation result directory/model label
  BENCHMARK_DIR          Official benchmark checkout
EOF
}

script_dir=$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)
project_root=$(CDPATH= cd -- "$script_dir/.." && pwd)
cd "$project_root"

phase=${1:-all}
case "$phase" in
  race|fact|all) ;;
  -h|--help)
    usage
    exit 0
    ;;
  *)
    printf 'Invalid phase: %s\n\n' "$phase" >&2
    usage >&2
    exit 64
    ;;
esac

input=${2:-benchmark_results/deepresearch_bench/evidence-deep-research-v1.jsonl}
benchmark_dir=${BENCHMARK_DIR:-benchmarks/deep_research_bench}
max_workers=${BENCHMARK_MAX_WORKERS:-4}
model_name=${BENCHMARK_MODEL_NAME:-$(basename "$input" .jsonl)}
cli=.venv/bin/deep-research-benchmark
env_file=.env
queries=${input%.jsonl}.queries.jsonl

case "$max_workers" in
  ''|*[!0-9]*|0)
    printf 'BENCHMARK_MAX_WORKERS must be a positive integer.\n' >&2
    exit 64
    ;;
esac

if [ ! -x "$cli" ]; then
  printf 'Benchmark CLI not found: %s\n' "$cli" >&2
  printf "Install it with: .venv/bin/pip install -e '.[benchmark]'\n" >&2
  exit 2
fi

if [ ! -f "$input" ]; then
  printf 'Benchmark output not found: %s\n' "$input" >&2
  exit 2
fi

if [ ! -f "$queries" ]; then
  printf 'Matching benchmark queries not found: %s\n' "$queries" >&2
  exit 2
fi

if [ ! -f "$benchmark_dir/deepresearch_bench_race.py" ]; then
  printf 'Official benchmark checkout not found: %s\n' "$benchmark_dir" >&2
  printf 'Fetch it with: %s fetch\n' "$cli" >&2
  exit 2
fi

has_env_value() {
  key=$1
  [ -f "$env_file" ] && awk -v wanted="$key" '
    /^[[:space:]]*#/ { next }
    {
      line = $0
      sub(/^[[:space:]]*/, "", line)
      split(line, parts, "=")
      key = parts[1]
      sub(/[[:space:]]*$/, "", key)
      if (key == wanted) {
        sub(/^[^=]*=[[:space:]]*/, "", line)
        if (length(line) > 0) found = 1
      }
    }
    END { exit(found ? 0 : 1) }
  ' "$env_file"
}

if [ -z "${OPENAI_API_KEY:-}" ] && ! has_env_value OPENAI_API_KEY; then
  printf 'OPENAI_API_KEY is missing. Add it to %s.\n' "$env_file" >&2
  exit 2
fi

if [ "$phase" != race ] && [ -z "${JINA_API_KEY:-}" ] && ! has_env_value JINA_API_KEY; then
  printf 'JINA_API_KEY is required for FACT. Add it to %s.\n' "$env_file" >&2
  exit 2
fi

export LLM_BACKEND=openai

printf 'DeepResearch Bench evaluation\n'
printf '  backend: openai\n'
printf '  phase: %s\n' "$phase"
printf '  input: %s\n' "$input"
printf '  model name: %s\n' "$model_name"
printf '  workers: %s\n' "$max_workers"

exec "$cli" evaluate \
  --benchmark-dir "$benchmark_dir" \
  --input "$input" \
  --queries "$queries" \
  --model-name "$model_name" \
  --phase "$phase" \
  --max-workers "$max_workers"
