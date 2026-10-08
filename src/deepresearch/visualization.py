from __future__ import annotations

import html
import json
from datetime import datetime
from pathlib import Path
from typing import Any

from .config import Settings
from .observability import render_model_observability
from .schemas import AuditResult, EvidenceRecord
from .store import EvidenceStore


def _parse_json(value: str | None) -> dict[str, Any]:
    if not value:
        return {}
    try:
        parsed = json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _duration_ms(started_at: str | None, completed_at: str | None) -> int | None:
    if not started_at or not completed_at:
        return None
    try:
        started = datetime.fromisoformat(started_at)
        completed = datetime.fromisoformat(completed_at)
    except ValueError:
        return None
    return max(0, round((completed - started).total_seconds() * 1000))


def _agent_for_operation(operation_type: str, task_id: str | None) -> str:
    if operation_type in {"model.plan", "model.reconnaissance_queries"}:
        return "Planner"
    if operation_type.startswith("model.replan"):
        return "ResearchSupervisor"
    if operation_type == "model.assess_sources":
        return f"SourceEvaluator:{task_id or 'unknown'}"
    if operation_type.startswith("model.audit"):
        return "Auditor"
    if task_id == "T0":
        return "ReconnaissanceWorker"
    if task_id:
        return f"ResearchWorker:{task_id}"
    return "Orchestrator"


def _summarize_operation(row: dict[str, Any]) -> dict[str, Any]:
    operation_type = row["operation_type"]
    request = _parse_json(row.get("request_json"))
    response = _parse_json(row.get("response_json"))
    details: dict[str, Any] = {}

    if operation_type == "model.plan":
        details = {
            "model": request.get("model"),
            "maximum_tasks": request.get("max_tasks"),
            "tasks_created": len(response.get("tasks", [])),
        }
    elif operation_type == "model.reconnaissance_queries":
        details = {
            "model": request.get("model"),
            "query_candidates": response.get("queries", []),
        }
    elif operation_type.startswith("model.replan"):
        details = {
            "model": request.get("model"),
            "round": request.get("round"),
            "should_continue": response.get("should_continue"),
            "new_tasks": len(response.get("new_tasks", [])),
            "rationale": response.get("rationale"),
        }
    elif operation_type == "model.formulate_queries":
        details = {
            "model": request.get("model"),
            "query_candidates": response.get("queries", []),
        }
    elif operation_type == "search.tavily":
        details = {
            "query": request.get("query"),
            "intent": request.get("intent"),
            "provider": request.get("provider"),
            "results": len(response.get("results", [])),
        }
    elif operation_type == "extract.tavily":
        details = {
            "urls_requested": len(request.get("urls", [])),
            "pages_extracted": len(response.get("pages", [])),
            "failed_urls": len(response.get("failed_urls", [])),
            "credits_used": response.get("credits_used", 0),
        }
    elif operation_type == "model.extract":
        details = {
            "model": request.get("model"),
            "sources_processed": len(request.get("results", [])),
            "evidence_candidates": len(response.get("evidence", [])),
        }
    elif operation_type == "model.assess_sources":
        assessments = response.get("assessments", [])
        details = {
            "model": request.get("model"),
            "rubric": request.get("rubric_version"),
            "assessments": len(assessments),
            "final_evidence": sum(
                item.get("eligibility") == "final_evidence"
                for item in assessments
            ),
            "discovery_only": sum(
                item.get("eligibility") == "discovery_only"
                for item in assessments
            ),
        }
    elif operation_type == "model.decide_worker_next_step":
        next_query = response.get("next_query") or {}
        details = {
            "model": request.get("model"),
            "step": request.get("step_number"),
            "evidence_seen": len(request.get("evidence", [])),
            "queries_seen": len(request.get("executed_queries", [])),
            "action": response.get("action"),
            "decision_summary": response.get("decision_summary"),
            "stop_reason": response.get("stop_reason"),
            "unresolved_gaps": response.get("unresolved_gaps", []),
            "next_query": next_query.get("query"),
        }
    elif operation_type.startswith("model.audit"):
        details = {
            "model": request.get("model"),
            "round": request.get("round"),
            "evidence_seen": len(request.get("evidence", [])),
            "sufficient": response.get("sufficient"),
            "claims": len(response.get("claims", [])),
            "gaps": len(response.get("gaps", [])),
        }

    return {
        "operation_key": row["operation_key"],
        "task_id": row.get("task_id"),
        "agent": _agent_for_operation(operation_type, row.get("task_id")),
        "operation_type": operation_type,
        "status": row["status"],
        "error": row.get("error"),
        "started_at": row.get("started_at"),
        "completed_at": row.get("completed_at"),
        "duration_ms": _duration_ms(row.get("started_at"), row.get("completed_at")),
        "details": details,
    }


def _build_payload(
    *,
    store: EvidenceStore,
    run_id: str,
    audit: AuditResult | None,
    evidence: list[EvidenceRecord],
    status_override: str | None,
) -> dict[str, Any]:
    run = store.get_run(run_id)
    if run is None:
        raise ValueError(f"Unknown run ID: {run_id}")

    config = _parse_json(run.get("config_json"))
    tasks = store.list_tasks(run_id)
    sources = store.list_sources(run_id)
    source_assessments = store.list_source_assessments(run_id)
    operations = [
        _summarize_operation(row)
        for row in store.list_operations(run_id, include_response=True)
    ]

    operation_counts: dict[str, int] = {}
    for operation in operations:
        agent = operation["agent"]
        operation_counts[agent] = operation_counts.get(agent, 0) + 1

    agents: list[dict[str, Any]] = [
        {
            "id": "Planner",
            "class_name": "PlannerAgent",
            "kind": "planner",
            "role": "Coverage planner and task decomposer",
            "objective": (
                "Define the coverage contract and create complementary evidence tasks."
            ),
            "model": config.get("planner_model"),
            "operation_count": operation_counts.get("Planner", 0),
        }
    ]
    if operation_counts.get("ReconnaissanceWorker", 0):
        agents.append(
            {
                "id": "ReconnaissanceWorker",
                "class_name": "ResearchWorkerAgent",
                "kind": "worker",
                "role": "Pre-planning landscape scout",
                "objective": "Find terminology, candidate original sources, and disputes before decomposition.",
                "model": config.get("fast_model"),
                "operation_count": operation_counts.get("ReconnaissanceWorker", 0),
            }
        )
    source_evaluator_operations = sum(
        count
        for agent, count in operation_counts.items()
        if agent.startswith("SourceEvaluator:")
    )
    if source_evaluator_operations:
        agents.append(
            {
                "id": "SourceEvaluator",
                "class_name": "SourceEvaluatorAgent",
                "kind": "auditor",
                "role": "Claim-contextual source quality evaluator",
                "objective": "Apply source-rubric-v1 and gate sources before evidence extraction.",
                "model": config.get("fast_model"),
                "operation_count": source_evaluator_operations,
            }
        )
    for task in tasks:
        spec = task["spec"]
        agent_id = f"ResearchWorker:{task['task_id']}"
        agents.append(
            {
                "id": agent_id,
                "class_name": "ResearchWorkerAgent",
                "kind": "worker",
                "task_id": task["task_id"],
                "status": task["status"],
                "role": spec.get("research_role", "evidence researcher"),
                "objective": spec.get("objective") or spec.get("question"),
                "question": spec.get("question"),
                "must_find": spec.get("must_find", []),
                "avoid": spec.get("avoid", []),
                "covered_dimensions": spec.get("covered_dimensions", []),
                "search_mode": spec.get("search_mode"),
                "query_budget": spec.get("query_budget"),
                "model": config.get("fast_model"),
                "operation_count": operation_counts.get(agent_id, 0),
                "error": task.get("error"),
            }
        )
    agents.append(
        {
            "id": "ResearchSupervisor",
            "class_name": "PlannerAgent",
            "kind": "planner",
            "role": "Evidence-conditioned research supervisor",
            "objective": "Stop or revise the plan after each complete Worker wave.",
            "model": config.get("planner_model"),
            "operation_count": operation_counts.get("ResearchSupervisor", 0),
        }
    )
    agents.append(
        {
            "id": "Auditor",
            "class_name": "AuditorAgent",
            "kind": "auditor",
            "role": "Independent evidence auditor",
            "objective": (
                "Synthesize cross-worker evidence, preserve conflicts, and judge sufficiency."
            ),
            "model": config.get("audit_model"),
            "operation_count": operation_counts.get("Auditor", 0),
        }
    )

    return {
        "run": {
            "run_id": run_id,
            "question": run["question"],
            "status": status_override or run["status"],
            "created_at": run["created_at"],
            "updated_at": run["updated_at"],
            "config": config,
        },
        "agents": agents,
        "tasks": tasks,
        "operations": operations,
        "sources": sources,
        "source_assessments": source_assessments,
        "evidence": [item.model_dump(mode="json") for item in evidence],
        "audit": audit.model_dump(mode="json") if audit else None,
    }


def render_run_visualization(
    *,
    settings: Settings,
    store: EvidenceStore,
    run_id: str,
    audit: AuditResult | None = None,
    evidence: list[EvidenceRecord] | None = None,
    status_override: str | None = None,
) -> Path:
    """Render one run as a self-contained, offline, interactive HTML file."""
    final_audit = audit if audit is not None else store.get_latest_audit(run_id)
    retained_evidence = evidence if evidence is not None else store.list_evidence(run_id)
    payload = _build_payload(
        store=store,
        run_id=run_id,
        audit=final_audit,
        evidence=retained_evidence,
        status_override=status_override,
    )
    encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    encoded = encoded.replace("</", "<\\/")
    question = html.escape(payload["run"]["question"])
    run_id_label = html.escape(run_id)

    document = rf"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Research run · {run_id_label}</title>
  <style>
    :root {{
      color-scheme: light dark;
      --bg: #f5f7fb; --surface: #ffffff; --surface-2: #eef2f8;
      --text: #172033; --muted: #667085; --border: #d8deea;
      --accent: #3157d5; --accent-soft: #e8edff; --good: #18794e;
      --warn: #a15c00; --bad: #b42318; --planner: #6d4bc3;
      --worker: #176b87; --auditor: #9a4a14;
      --shadow: 0 12px 34px rgba(25, 40, 72, .08);
    }}
    @media (prefers-color-scheme: dark) {{
      :root {{
        --bg: #0e1420; --surface: #151e2d; --surface-2: #1b2638;
        --text: #edf2fa; --muted: #a8b3c5; --border: #2d3a50;
        --accent: #91a7ff; --accent-soft: #202d55; --good: #5bd6a0;
        --warn: #f0b45f; --bad: #ff8b83; --planner: #b79cff;
        --worker: #72cee5; --auditor: #f3aa78;
        --shadow: 0 14px 36px rgba(0, 0, 0, .24);
      }}
    }}
    * {{ box-sizing: border-box; }}
    body {{ margin: 0; background: var(--bg); color: var(--text); font: 14px/1.55 Inter, ui-sans-serif, system-ui, sans-serif; }}
    button, select {{ font: inherit; }}
    a {{ color: var(--accent); }}
    .shell {{ width: min(1180px, calc(100% - 32px)); margin: 28px auto 56px; }}
    .hero {{ padding: 28px; background: var(--surface); border: 1px solid var(--border); border-radius: 18px; box-shadow: var(--shadow); }}
    .eyebrow, .label {{ color: var(--muted); font-size: 12px; letter-spacing: .06em; text-transform: uppercase; }}
    h1 {{ margin: 8px 0 12px; font-size: clamp(22px, 4vw, 36px); line-height: 1.18; font-weight: 650; }}
    h2 {{ margin: 0 0 14px; font-size: 20px; }}
    h3 {{ margin: 0; font-size: 15px; }}
    p {{ margin: 6px 0; }}
    .meta, .chips, .toolbar {{ display: flex; flex-wrap: wrap; gap: 8px; align-items: center; }}
    .chip, .status {{ display: inline-flex; align-items: center; gap: 6px; padding: 4px 9px; border-radius: 999px; background: var(--surface-2); color: var(--muted); font-size: 12px; }}
    .status.succeeded, .status.completed, .status.supported {{ color: var(--good); }}
    .status.partial, .status.mixed, .status.running {{ color: var(--warn); }}
    .status.failed, .status.failed_retryable, .status.unsupported {{ color: var(--bad); }}
    .stats {{ display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 12px; margin: 16px 0; }}
    .stat {{ padding: 15px 16px; background: var(--surface); border: 1px solid var(--border); border-radius: 12px; }}
    .stat strong {{ display: block; margin-top: 4px; font-size: 24px; font-weight: 650; }}
    .nav {{ display: flex; gap: 4px; margin: 22px 0 16px; padding: 4px; border-bottom: 1px solid var(--border); overflow-x: auto; }}
    .nav button {{ border: 0; border-radius: 8px; padding: 8px 13px; color: var(--muted); background: transparent; cursor: pointer; white-space: nowrap; }}
    .nav button[aria-selected="true"] {{ color: var(--text); background: var(--surface); box-shadow: 0 1px 4px rgba(0,0,0,.08); }}
    .view[hidden] {{ display: none; }}
    .section {{ margin: 18px 0 26px; }}
    .grid {{ display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 12px; }}
    .card {{ padding: 17px; background: var(--surface); border: 1px solid var(--border); border-radius: 13px; }}
    .agent {{ border-top: 3px solid var(--border); }}
    .agent.planner {{ border-top-color: var(--planner); }}
    .agent.worker {{ border-top-color: var(--worker); }}
    .agent.auditor {{ border-top-color: var(--auditor); }}
    .agent-head {{ display: flex; justify-content: space-between; gap: 12px; align-items: flex-start; }}
    .agent-id {{ font-family: ui-monospace, SFMono-Regular, Menlo, monospace; color: var(--muted); font-size: 12px; }}
    .detail-grid {{ display: grid; grid-template-columns: repeat(2, minmax(0,1fr)); gap: 10px; margin-top: 13px; }}
    .detail {{ min-width: 0; }}
    .detail div:last-child {{ overflow-wrap: anywhere; }}
    .toolbar {{ justify-content: space-between; margin-bottom: 12px; }}
    select {{ max-width: 100%; padding: 8px 10px; color: var(--text); background: var(--surface); border: 1px solid var(--border); border-radius: 8px; }}
    .timeline {{ position: relative; margin-left: 8px; }}
    .timeline::before {{ content: ""; position: absolute; left: 8px; top: 8px; bottom: 8px; width: 1px; background: var(--border); }}
    .operation {{ position: relative; display: grid; grid-template-columns: 18px minmax(145px, .8fr) minmax(0, 2fr) auto; gap: 11px; align-items: start; padding: 10px 0; border-bottom: 1px solid var(--border); }}
    .dot {{ width: 9px; height: 9px; margin-top: 6px; border-radius: 50%; background: var(--muted); z-index: 1; }}
    .operation[data-kind="planner"] .dot {{ background: var(--planner); }}
    .operation[data-kind="worker"] .dot {{ background: var(--worker); }}
    .operation[data-kind="auditor"] .dot {{ background: var(--auditor); }}
    .op-title {{ font-weight: 600; overflow-wrap: anywhere; }}
    .op-detail {{ color: var(--muted); overflow-wrap: anywhere; }}
    .duration {{ font-variant-numeric: tabular-nums; color: var(--muted); white-space: nowrap; }}
    .evidence {{ border-left: 3px solid var(--border); }}
    .evidence.supports {{ border-left-color: var(--good); }}
    .evidence.contradicts {{ border-left-color: var(--bad); }}
    blockquote {{ margin: 10px 0 0; padding-left: 12px; border-left: 2px solid var(--border); color: var(--muted); }}
    .claim-links {{ margin-top: 10px; font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 12px; color: var(--muted); }}
    .empty {{ padding: 24px; text-align: center; color: var(--muted); border: 1px dashed var(--border); border-radius: 12px; }}
    .gate-grid {{ display: grid; grid-template-columns: repeat(4, minmax(0, 1fr)); gap: 8px; margin-bottom: 14px; }}
    .gate {{ padding: 10px; background: var(--surface-2); border-radius: 9px; text-align: center; }}
    .gate strong {{ display: block; }}
    @media (max-width: 760px) {{
      .stats, .gate-grid {{ grid-template-columns: repeat(2, minmax(0,1fr)); }}
      .grid {{ grid-template-columns: 1fr; }}
      .operation {{ grid-template-columns: 18px minmax(0, 1fr) auto; }}
      .op-detail {{ grid-column: 2 / -1; }}
    }}
    @media (max-width: 460px) {{
      .shell {{ width: min(100% - 18px, 1180px); margin-top: 10px; }}
      .hero {{ padding: 18px; }}
      .stats {{ grid-template-columns: 1fr 1fr; }}
      .detail-grid {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <main class="shell">
    <header class="hero">
      <div class="eyebrow">Deep research run · <span id="run-id"></span></div>
      <h1>{question}</h1>
      <div class="meta" id="run-meta"></div>
    </header>
    <section class="stats" id="stats" aria-label="Run summary"></section>
    <nav class="nav" aria-label="Run views">
      <button type="button" data-view="overview" aria-selected="true">Agents & roles</button>
      <button type="button" data-view="execution" aria-selected="false">Execution trace</button>
      <button type="button" data-view="evidence" aria-selected="false">Evidence</button>
      <button type="button" data-view="audit" aria-selected="false">Audit</button>
    </nav>
    <section class="view" id="view-overview"></section>
    <section class="view" id="view-execution" hidden></section>
    <section class="view" id="view-evidence" hidden></section>
    <section class="view" id="view-audit" hidden></section>
  </main>
  <script id="run-data" type="application/json">{encoded}</script>
  <script>
    const data = JSON.parse(document.getElementById('run-data').textContent);
    const esc = (value) => String(value ?? '').replace(/[&<>"']/g, char => ({{
      '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'
    }})[char]);
    const safeUrl = (value) => /^https?:\/\//i.test(String(value || '')) ? esc(value) : '#';
    const values = (items) => (items || []).filter(Boolean);
    const status = (value) => `<span class="status ${{esc(value)}}">${{esc(value || 'unknown')}}</span>`;
    const chips = (items) => values(items).map(item => `<span class="chip">${{esc(item)}}</span>`).join('');
    const formatTime = (value) => value ? new Date(value).toLocaleString() : '—';
    const formatDuration = (ms) => ms == null ? '—' : ms < 1000 ? `${{ms}} ms` : `${{(ms / 1000).toFixed(1)}} s`;
    const agentKind = (agent) => ['Planner', 'ResearchSupervisor'].includes(agent) ? 'planner' : agent === 'Auditor' ? 'auditor' : (agent.startsWith('ResearchWorker:') || agent === 'ReconnaissanceWorker') ? 'worker' : agent.startsWith('SourceEvaluator:') ? 'auditor' : 'system';
    const detailText = (operation) => {{
      const d = operation.details || {{}};
      if (operation.operation_type === 'model.plan') return `${{d.tasks_created || 0}} tasks · model ${{d.model || 'unknown'}}`;
      if (operation.operation_type === 'model.reconnaissance_queries') return `${{values(d.query_candidates).length}} reconnaissance queries`;
      if (operation.operation_type.startsWith('model.replan')) return `${{d.new_tasks || 0}} new tasks · continue=${{String(d.should_continue)}}${{d.rationale ? ' · ' + d.rationale : ''}}`;
      if (operation.operation_type === 'model.formulate_queries') return values(d.query_candidates).map(q => q.query).join(' · ');
      if (operation.operation_type === 'search.tavily') return `${{d.query || ''}} · ${{d.results || 0}} results`;
      if (operation.operation_type === 'extract.tavily') return `${{d.pages_extracted || 0}}/${{d.urls_requested || 0}} pages extracted`;
      if (operation.operation_type === 'model.extract') return `${{d.evidence_candidates || 0}} evidence candidates from ${{d.sources_processed || 0}} sources`;
      if (operation.operation_type === 'model.assess_sources') return `${{d.assessments || 0}} assessed · ${{d.final_evidence || 0}} final · ${{d.discovery_only || 0}} discovery`;
      if (operation.operation_type === 'model.decide_worker_next_step') return `${{d.action || 'decision'}}${{d.stop_reason ? ' · ' + d.stop_reason : ''}}${{d.next_query ? ' · next: ' + d.next_query : ''}}${{d.decision_summary ? ' · ' + d.decision_summary : ''}}`;
      if (operation.operation_type.startsWith('model.audit')) return `${{d.claims || 0}} claims · ${{d.gaps || 0}} gaps · sufficient=${{String(d.sufficient)}}`;
      return '';
    }};

    document.getElementById('run-id').textContent = data.run.run_id;
    document.getElementById('run-meta').innerHTML = [
      status(data.run.status),
      `<span class="chip">Created ${{esc(formatTime(data.run.created_at))}}</span>`,
      `<span class="chip">Updated ${{esc(formatTime(data.run.updated_at))}}</span>`,
      '<a class="chip" href="model_observability.html">Model observability →</a>'
    ].join('');

    const workerCount = data.agents.filter(agent => agent.kind === 'worker').length;
    document.getElementById('stats').innerHTML = [
      ['Workers', workerCount], ['Operations', data.operations.length],
      ['Sources', data.sources.length], ['Assessments', data.source_assessments.length], ['Evidence', data.evidence.length]
    ].map(([label, value]) => `<div class="stat"><span class="label">${{esc(label)}}</span><strong>${{esc(value)}}</strong></div>`).join('');

    const coverage = [...new Set(data.tasks.flatMap(task => task.spec.covered_dimensions || []))];
    document.getElementById('view-overview').innerHTML = `
      <section class="section">
        <h2>Coverage dimensions</h2>
        <div class="chips">${{chips(coverage) || '<span class="chip">No dimensions recorded</span>'}}</div>
      </section>
      <section class="section">
        <h2>Agents and assigned roles</h2>
        <div class="grid">${{data.agents.map(agent => `
          <article class="card agent ${{esc(agent.kind)}}">
            <div class="agent-head">
              <div><div class="agent-id">${{esc(agent.id)}}</div><h3>${{esc(agent.role)}}</h3></div>
              ${{agent.status ? status(agent.status) : ''}}
            </div>
            <p>${{esc(agent.objective)}}</p>
            <div class="detail-grid">
              <div class="detail"><div class="label">Class</div><div>${{esc(agent.class_name)}}</div></div>
              <div class="detail"><div class="label">Model</div><div>${{esc(agent.model || 'not recorded')}}</div></div>
              <div class="detail"><div class="label">Operations</div><div>${{esc(agent.operation_count)}}</div></div>
              ${{agent.search_mode ? `<div class="detail"><div class="label">Search mode</div><div>${{esc(agent.search_mode)}}</div></div>` : ''}}
              ${{agent.covered_dimensions?.length ? `<div class="detail"><div class="label">Dimensions</div><div>${{esc(agent.covered_dimensions.join(', '))}}</div></div>` : ''}}
              ${{agent.must_find?.length ? `<div class="detail"><div class="label">Must find</div><div>${{esc(agent.must_find.join(', '))}}</div></div>` : ''}}
            </div>
            ${{agent.error ? `<p class="status failed">${{esc(agent.error)}}</p>` : ''}}
          </article>`).join('')}}</div>
      </section>`;

    const agentOptions = ['all', ...new Set(data.operations.map(operation => operation.agent))];
    document.getElementById('view-execution').innerHTML = `
      <section class="section">
        <div class="toolbar"><h2>Operation trace</h2><label>Agent <select id="agent-filter">${{agentOptions.map(agent => `<option value="${{esc(agent)}}">${{esc(agent === 'all' ? 'All agents' : agent)}}</option>`).join('')}}</select></label></div>
        <div class="timeline" id="timeline"></div>
      </section>`;
    const renderTimeline = () => {{
      const selected = document.getElementById('agent-filter').value;
      const rows = data.operations.filter(operation => selected === 'all' || operation.agent === selected);
      document.getElementById('timeline').innerHTML = rows.length ? rows.map(operation => `
        <article class="operation" data-kind="${{agentKind(operation.agent)}}">
          <span class="dot" aria-hidden="true"></span>
          <div><div class="op-title">${{esc(operation.agent)}}</div><div class="label">${{esc(operation.operation_type)}}</div></div>
          <div class="op-detail">${{esc(detailText(operation))}}${{operation.error ? `<div class="status failed">${{esc(operation.error)}}</div>` : ''}}</div>
          <div class="duration">${{esc(formatDuration(operation.duration_ms))}}</div>
        </article>`).join('') : '<div class="empty">No operations match this agent.</div>';
    }};
    document.getElementById('agent-filter').addEventListener('change', renderTimeline);
    renderTimeline();

    const evidenceTasks = ['all', ...new Set(data.evidence.map(item => item.task_id))];
    document.getElementById('view-evidence').innerHTML = `
      <section class="section">
        <div class="toolbar"><h2>Retained atomic evidence</h2><label>Worker <select id="evidence-filter">${{evidenceTasks.map(task => `<option value="${{esc(task)}}">${{esc(task === 'all' ? 'All workers' : task)}}</option>`).join('')}}</select></label></div>
        <div class="grid" id="evidence-grid"></div>
      </section>`;
    const renderEvidence = () => {{
      const selected = document.getElementById('evidence-filter').value;
      const rows = data.evidence.filter(item => selected === 'all' || item.task_id === selected);
      document.getElementById('evidence-grid').innerHTML = rows.length ? rows.map(item => `
        <article class="card evidence ${{esc(item.stance)}}">
          <div class="agent-head"><div class="agent-id">${{esc(item.evidence_id)}} · ${{esc(item.task_id)}}</div>${{status(item.stance)}}</div>
          <h3>${{esc(item.claim_candidate)}}</h3>
          <blockquote>${{esc(item.verbatim_excerpt)}}</blockquote>
          <p><a href="${{safeUrl(item.source_url)}}" target="_blank" rel="noreferrer">${{esc(item.source_title)}}</a> · ${{esc(item.source_domain)}}</p>
          <div class="chips">${{chips([item.source_type, `tier ${{item.credibility_tier}}`, item.source_directness, item.source_eligibility, ...(item.source_hard_flags || [])])}}</div>
          <p class="label">Quality ${{esc(item.source_quality_score ?? '—')}}/15 · fitness ${{esc(item.evidence_fitness_score ?? '—')}}/12</p>
          <p class="label">Provenance: ${{esc(item.provenance_key || item.source_url)}}</p>
        </article>`).join('') : '<div class="empty">No retained evidence for this worker.</div>';
    }};
    document.getElementById('evidence-filter').addEventListener('change', renderEvidence);
    renderEvidence();

    const audit = data.audit;
    document.getElementById('view-audit').innerHTML = audit ? `
      <section class="section">
        <div class="toolbar"><h2>Auditor conclusion</h2>${{status(audit.sufficient ? 'sufficient' : 'open gaps')}}</div>
        <div class="gate-grid">
          ${{[['Coverage', audit.coverage_sufficient], ['Sources', audit.source_sufficient], ['Tasks', audit.task_execution_sufficient], ['Challenge', audit.challenge_sufficient]].map(([label, ok]) => `<div class="gate"><span class="label">${{esc(label)}}</span><strong>${{ok ? 'Pass' : 'Open'}}</strong></div>`).join('')}}
        </div>
        <div class="grid">${{audit.claims.map(claim => `
          <article class="card">
            <div class="agent-head"><div class="agent-id">${{esc(claim.claim_id)}} · ${{esc(claim.dimension || 'cross-cutting')}} · ${{esc(claim.confidence)}} confidence</div>${{status(claim.status)}}</div>
            <h3>${{esc(claim.claim)}}</h3><p>${{esc(claim.reasoning)}}</p>
            <div class="claim-links">support: ${{esc(values(claim.supporting_evidence_ids).join(', ') || '—')}}<br>counter: ${{esc(values(claim.contradicting_evidence_ids).join(', ') || '—')}}</div>
          </article>`).join('')}}</div>
      </section>
      <section class="section"><h2>Open gaps</h2>${{audit.gaps.length ? `<div class="card">${{audit.gaps.map(gap => `<p>• ${{esc(gap)}}</p>`).join('')}}</div>` : '<div class="empty">No report-changing gaps were retained.</div>'}}</section>`
      : '<div class="empty">This run has no completed audit yet.</div>';

    document.querySelectorAll('[data-view]').forEach(button => button.addEventListener('click', () => {{
      document.querySelectorAll('[data-view]').forEach(item => item.setAttribute('aria-selected', String(item === button)));
      document.querySelectorAll('.view').forEach(view => view.hidden = view.id !== `view-${{button.dataset.view}}`);
    }}));
  </script>
</body>
</html>
"""
    run_dir = settings.runs_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / "run_view.html"
    path.write_text(document, encoding="utf-8")
    render_model_observability(settings=settings, store=store, run_id=run_id)
    return path
