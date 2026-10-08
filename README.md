# Evidence-First Deep Research

A checkpointed multi-agent research CLI that turns a question into an auditable,
source-backed report. Agents exchange typed plans, tasks, evidence, and audit
results; the final report can only cite evidence already stored in the ledger.

Architecture diagrams live in [`docs/diagrams/`](docs/diagrams/) and the slide deck
is at [`docs/DeepResearchAgent_Presentation.pptx`](docs/DeepResearchAgent_Presentation.pptx).
An example run is in [`runs/drb_evidence-deep-research-v2_51/`](runs/drb_evidence-deep-research-v2_51/).

## How it works

```text
Question
  → reconnaissance search
  → Planner creates a coverage contract and parallel tasks
  → Workers search, read pages, assess sources, and extract verbatim evidence
  → Audit Coordinator chooses a single or hierarchical bounded audit
  → Dimension Auditors emit claims and replan-ready gaps; local code aggregates them
  → Supervisor optionally creates one bounded follow-up wave
  → Markdown report + interactive run and model-usage pages
```

Key properties:

- LangGraph checkpoints support resume after interruption.
- Search/model operations are idempotently cached.
- Extracted quotations must occur verbatim in stored source content.
- Source quality and claim fitness are scored separately.
- A deterministic model gateway handles routing, budgets, retries, and telemetry.
- Reports use paper-style numeric citations rendered from audited evidence
  without another generative call; full excerpts stay in audit artifacts.

## Quick start

Requires Python 3.11+.

```bash
python3 -m venv .venv
.venv/bin/pip install -e .
```

Run the offline deterministic demo:

```bash
.venv/bin/deep-research --fake \
  "Does the intervention work, and what are its limitations?"
```

For live research, copy the environment template and add your keys:

```bash
cp .env.example .env
```

```env
TAVILY_API_KEY=...
DEEPSEEK_API_KEY=...
```

Then run:

```bash
.venv/bin/deep-research \
  "What are the risks and benefits of synthetic data for LLM training?"
```

## CLI

```bash
# Resume from the latest checkpoint
.venv/bin/deep-research --resume <run_id>

# Rebuild both HTML pages without API calls
.venv/bin/deep-research --visualize <run_id>

# Print model usage grouped by task, agent, operation, and model
.venv/bin/deep-research --gateway-stats <run_id>

# Suppress step logs
.venv/bin/deep-research --quiet "<question>"
```

Failed graph executions are retried from their checkpoint. A run is considered
complete only when it produces both a terminal reason and a report path.

## DeepResearch Bench

The optional adapter uses the Apache-2.0
[DeepResearch Bench](https://github.com/Ayanami0730/deep_research_bench), pinned
to a known commit. It keeps the third-party evaluator outside the core graph.

Install the evaluator dependencies and fetch the official tasks:

```bash
.venv/bin/pip install -e '.[benchmark]'
.venv/bin/deep-research-benchmark fetch
```

Inspect a small English slice without making model or search calls:

```bash
.venv/bin/deep-research-benchmark run \
  --dataset benchmarks/deep_research_bench/data/prompt_data/query.jsonl \
  --language en --limit 2 --dry-run
```

Run the same slice with the configured live providers:

```bash
.venv/bin/deep-research-benchmark run \
  --dataset benchmarks/deep_research_bench/data/prompt_data/query.jsonl \
  --model-name evidence-deep-research-v1 \
  --language en --limit 2 --continue-on-error
```

The runner is sequential and checkpoint-aware. Re-running the command skips
finished reports and resumes existing failed runs. Running all 100 expensive
tasks requires the explicit `--all` flag. It writes:

```text
benchmark_results/deepresearch_bench/<model>.jsonl
benchmark_results/deepresearch_bench/<model>.queries.jsonl
benchmark_results/deepresearch_bench/<model>.manifest.jsonl
```

The first file uses the official `{id, prompt, article}` format. After setting
the evaluator credentials, run RACE and FACT with:

```bash
export LLM_BACKEND=openai
export OPENAI_API_KEY=...
export JINA_API_KEY=...

.venv/bin/deep-research-benchmark evaluate \
  --benchmark-dir benchmarks/deep_research_bench \
  --input benchmark_results/deepresearch_bench/evidence-deep-research-v1.jsonl \
  --phase all
```

Or put `OPENAI_API_KEY` and `JINA_API_KEY` in the project-root `.env` and use
the convenience script. It defaults to the two exported
`evidence-deep-research-v1` tasks:

```bash
scripts/evaluate_benchmark.sh all

# Run only one metric, or evaluate another official-format export
scripts/evaluate_benchmark.sh race
scripts/evaluate_benchmark.sh fact benchmark_results/deepresearch_bench/my-model.jsonl
```

RACE needs the selected evaluator backend key. FACT additionally needs Jina for
source-page extraction. These evaluator calls are external to this project's
model gateway, so their cost is not included in the run observability report.

## Outputs

Each run writes to `runs/<run_id>/`:

| File | Contents |
| --- | --- |
| `report.md` | Evidence-backed final report |
| `run_view.html` | Agents, execution trace, evidence, and audit |
| `model_observability.html` | Model routing, tokens, context, latency, and cost |
| `evidence.json` | Retained atomic evidence |
| `citations.json` | Numeric report citations mapped back to evidence IDs and URLs |
| `audit.json` | Final audited claims and gaps |
| `audit_plan.json` | Single/hierarchical audit routing and evidence slices |
| `source_assessments.json` | Source rubric results |
| `trace.json` | Cached operation trace |

Durable state is separated by concern:

```text
.deepresearch/checkpoints.sqlite    LangGraph checkpoints
.deepresearch/research.sqlite       runs, tasks, sources, evidence, audits
.deepresearch/model_gateway.sqlite  model calls, routing, usage, and cost
```

## Model gateway

Agents request an execution profile rather than choosing a concrete model:

```text
lightweight_extraction | standard_research | deep_reasoning
```

The gateway selects an eligible model using the profile, context size, remaining
run budget, and premium-model threshold. Every call records a signature with its
`run_id`, `task_id`, agent, operation, profile, and retry attempt.

Optional budget configuration:

```env
MODEL_GATEWAY_BUDGET_USD=5
MODEL_GATEWAY_PREMIUM_THRESHOLD=0.85
MODEL_GATEWAY_CONTEXT_WINDOW=128000
MODEL_GATEWAY_PRICING_JSON={"model-name":{"input":0.0,"output":0.0}}
```

Prices are USD per million tokens. When a budget is enabled, every configured
model must have positive input and output prices. The gateway reserves estimated
cost before concurrent calls begin and reconciles it with provider-reported usage.

## Main configuration

| Variable | Default | Purpose |
| --- | ---: | --- |
| `MAX_INITIAL_TASKS` | `6` | Maximum tasks created by the Planner |
| `MAX_RECONNAISSANCE_QUERIES` | `2` | Searches before task decomposition |
| `MAX_QUERIES_PER_WORKER` | `2` | Initial queries selected per Worker |
| `MAX_ADAPTIVE_STEPS` | `6` | Worker search/follow-up safety ceiling |
| `MAX_REPLAN_ROUNDS` | `2` | Maximum follow-up research waves |
| `MAX_NEW_TASKS_PER_REPLAN` | `3` | New tasks allowed per follow-up wave |
| `REPLAN_CONTEXT_TOKEN_BUDGET` | `24000` | Digest plus drill-down evidence allowance |
| `SINGLE_AUDIT_CONTEXT_TOKEN_BUDGET` | `30000` | Fast-path ceiling before audit fan-out |
| `AUDIT_EVIDENCE_TOKEN_BUDGET_PER_CHECK` | `12000` | Raw evidence allowance per specialist |
| `MAX_AUDIT_SPECIALISTS` | `4` | Maximum bounded dimension checks per round |
| `MAX_RESULTS_PER_SEARCH` | `5` | Tavily results per search |
| `PAGE_CHUNK_CHARS` | `12000` | Legacy/source-assessment content ceiling |
| `RETRIEVAL_CHUNK_TOKENS` | `500` | Target child size for local retrieval |
| `RETRIEVAL_CHUNK_OVERLAP_TOKENS` | `75` | Context overlap between child chunks |
| `EXTRACTION_WINDOW_TOKENS` | `1800` | Expanded context around retrieval hits |
| `MAX_RETRIEVAL_CHUNKS_PER_SOURCE` | `4` | Child hits retained before merging |
| `MAX_EXTRACTION_WINDOWS_PER_SOURCE` | `2` | Maximum model calls per source |
| `QWEN_EMBEDDING_BASE_URL` | unset | Enables an OpenAI-compatible Qwen embedding endpoint |
| `QWEN_EMBEDDING_MODEL` | `qwen3-embedding:0.6b` | Local Ollama model or hosted model name |
| `QWEN_EMBEDDING_DIMENSIONS` | `512` | Dense-vector size used by retrieval |
| `HYBRID_EMBEDDING_WEIGHT` | `0.35` | Dense rank contribution to weighted RRF; BM25 gets the remainder |
| `HYBRID_RRF_K` | `60` | RRF rank offset that controls how quickly lower ranks decay |
| `MANAGE_LOCAL_EMBEDDING_SERVICE` | `true` | Start local Ollama for a run and stop it afterward |

Page retrieval uses paragraph-aware child chunks. With an embedding endpoint
configured, BM25 and Qwen cosine similarity each produce a child-chunk ranking.
Weighted Reciprocal Rank Fusion combines those rankings with 65% lexical and 35%
semantic contribution by default (`k=60`). The system retains up to four children,
clusters nearby hits, and expands them into at most two contiguous source windows.
Without that endpoint, or if it fails during a run, retrieval falls back to local
BM25. Reducing adaptive steps or extraction windows directly controls call fan-out.

For the recommended local setup, install Ollama, run
`ollama pull qwen3-embedding:0.6b`, and set the base URL to
`http://localhost:11434/v1`. The model runs locally; no embedding API key is
required. The CLI owns the Ollama process only when it had to start it: the
process is stopped when the run succeeds or fails. An already-running Ollama
instance is reused and left untouched.

Replanning does not resend the full evidence corpus. Each round persists a
bounded per-worker research digest. The Auditor hands off structured actionable
gaps, and retrieval prepares a separate task-scoped evidence slice for each of
the three highest-priority gaps. Full excerpts stay addressable in SQLite.

Auditing also has a bounded memory path. Small evidence sets use one Auditor.
When the estimated input exceeds the configured ceiling, a deterministic
Coordinator partitions every required dimension across at most four checks,
prioritizing missing provenance, failed tasks, weak sources, and contradictions.
Each specialist retrieves a diverse slice from SQLite and returns structured
claims and actionable gaps. A deterministic local aggregator merges and dedupes
those results, so hierarchical audit adds no Reducer model call. The durable
plan is stored in `audit_plans` and exported with the run artifacts.

## Code map

```text
src/deepresearch/
├── graph.py             workflow, fan-out, audit gates, checkpoint recovery
├── benchmark.py         DeepResearch Bench fetch, batch runner, and evaluator adapter
├── runtime.py           operation cache and durable side-effect boundary
├── store.py             research evidence ledger
├── agents/              Planner, Worker, SourceEvaluator, Auditor
├── model_gateway/       routing, budget enforcement, provider adapter, telemetry
├── visualization.py     interactive run page
└── observability.py     model-usage page
```

The graph controls execution. Agents perform typed model tasks. `ResearchRuntime`
connects the two and supplies call signatures to the gateway.

## Tests

Tests use fake providers and require no API keys:

```bash
.venv/bin/python -m unittest discover -s tests -v
```

They cover end-to-end execution, checkpoint recovery, caching, source scoring,
evidence invariants, benchmark export, gateway routing/budgets, and safe HTML
generation.
