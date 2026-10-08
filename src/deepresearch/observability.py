from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

from .config import Settings
from .model_gateway import ModelTelemetryStore
from .store import EvidenceStore


def _group_calls(
    calls: list[dict[str, object]], keys: tuple[str, ...]
) -> list[dict[str, object]]:
    groups: dict[tuple[object, ...], dict[str, Any]] = {}
    for call in calls:
        identity = tuple(call.get(key) for key in keys)
        row = groups.setdefault(
            identity,
            {
                **dict(zip(keys, identity)),
                "calls": 0,
                "succeeded": 0,
                "failed": 0,
                "input_tokens": 0,
                "output_tokens": 0,
                "cached_input_tokens": 0,
                "reasoning_tokens": 0,
                "cost_usd": 0.0,
                "latency_ms": 0,
                "max_context_ratio": 0.0,
            },
        )
        row["calls"] += 1
        row["succeeded"] += call.get("status") == "succeeded"
        row["failed"] += call.get("status") == "failed"
        for field in (
            "input_tokens",
            "output_tokens",
            "cached_input_tokens",
            "reasoning_tokens",
            "latency_ms",
        ):
            row[field] += int(call.get(field) or 0)
        row["cost_usd"] += float(call.get("effective_cost_usd") or 0)
        row["max_context_ratio"] = max(
            row["max_context_ratio"], float(call.get("context_ratio") or 0)
        )
    return list(groups.values())


def _build_observability_payload(
    *, settings: Settings, store: EvidenceStore, run_id: str
) -> dict[str, Any]:
    run = store.get_run(run_id)
    if run is None:
        raise ValueError(f"Unknown run ID: {run_id}")
    calls = ModelTelemetryStore(settings.model_gateway_db).list_calls(run_id)
    config = json.loads(run["config_json"])
    totals = {
        "calls": len(calls),
        "succeeded": sum(call["status"] == "succeeded" for call in calls),
        "failed": sum(call["status"] == "failed" for call in calls),
        "input_tokens": sum(int(call.get("input_tokens") or 0) for call in calls),
        "output_tokens": sum(int(call.get("output_tokens") or 0) for call in calls),
        "cached_input_tokens": sum(
            int(call.get("cached_input_tokens") or 0) for call in calls
        ),
        "reasoning_tokens": sum(
            int(call.get("reasoning_tokens") or 0) for call in calls
        ),
        "cost_usd": sum(
            float(call.get("effective_cost_usd") or 0) for call in calls
        ),
        "latency_ms": sum(int(call.get("latency_ms") or 0) for call in calls),
        "max_context_ratio": max(
            (float(call.get("context_ratio") or 0) for call in calls), default=0.0
        ),
    }
    budget = config.get("model_gateway_budget_usd")
    if budget is None:
        budget = settings.model_gateway_budget_usd
    totals["budget_usd"] = budget
    totals["budget_ratio"] = (
        totals["cost_usd"] / float(budget) if budget else None
    )
    return {
        "run": {
            "run_id": run_id,
            "question": run["question"],
            "status": run["status"],
            "created_at": run["created_at"],
            "updated_at": run["updated_at"],
        },
        "totals": totals,
        "by_agent": _group_calls(calls, ("task_id", "agent_id", "operation")),
        "by_model": _group_calls(calls, ("provider", "selected_model", "profile")),
        "calls": calls,
    }


def render_model_observability(
    *, settings: Settings, store: EvidenceStore, run_id: str
) -> Path:
    """Render an offline second page for model routing, cost, and context usage."""
    payload = _build_observability_payload(
        settings=settings,
        store=store,
        run_id=run_id,
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
  <title>Model observability · {run_id_label}</title>
  <style>
    :root {{ color-scheme: light dark; --bg:#f5f7fb; --surface:#fff; --surface2:#eef2f8; --text:#172033; --muted:#667085; --border:#d8deea; --accent:#3157d5; --good:#18794e; --warn:#a15c00; --bad:#b42318; --shadow:0 12px 34px rgba(25,40,72,.08); }}
    @media (prefers-color-scheme: dark) {{ :root {{ --bg:#0e1420; --surface:#151e2d; --surface2:#1b2638; --text:#edf2fa; --muted:#a8b3c5; --border:#2d3a50; --accent:#91a7ff; --good:#5bd6a0; --warn:#f0b45f; --bad:#ff8b83; --shadow:0 14px 36px rgba(0,0,0,.24); }} }}
    * {{ box-sizing:border-box; }} body {{ margin:0; background:var(--bg); color:var(--text); font:14px/1.5 Inter,ui-sans-serif,system-ui,sans-serif; }}
    .shell {{ width:min(1240px,calc(100% - 32px)); margin:28px auto 56px; }} .hero,.card {{ background:var(--surface); border:1px solid var(--border); box-shadow:var(--shadow); }}
    .hero {{ padding:26px; border-radius:18px; }} .card {{ padding:16px; border-radius:13px; box-shadow:none; }}
    h1 {{ margin:7px 0 10px; font-size:clamp(22px,4vw,34px); line-height:1.2; }} h2 {{ margin:26px 0 12px; font-size:20px; }}
    .eyebrow,.label {{ color:var(--muted); font-size:12px; letter-spacing:.06em; text-transform:uppercase; }} .meta,.chips {{ display:flex; flex-wrap:wrap; gap:8px; align-items:center; }}
    .chip,.status {{ display:inline-flex; padding:4px 9px; border-radius:999px; background:var(--surface2); color:var(--muted); font-size:12px; }} a {{ color:var(--accent); }}
    .stats {{ display:grid; grid-template-columns:repeat(5,minmax(0,1fr)); gap:10px; margin:16px 0; }} .stat strong {{ display:block; margin-top:5px; font-size:23px; }}
    .grid {{ display:grid; grid-template-columns:repeat(2,minmax(0,1fr)); gap:11px; }} .bar {{ height:7px; margin-top:10px; overflow:hidden; background:var(--surface2); border-radius:999px; }} .bar span {{ display:block; height:100%; background:var(--accent); border-radius:inherit; }}
    .table-wrap {{ overflow:auto; border:1px solid var(--border); border-radius:13px; background:var(--surface); }} table {{ width:100%; border-collapse:collapse; white-space:nowrap; }} th,td {{ padding:10px 12px; border-bottom:1px solid var(--border); text-align:left; }} th {{ position:sticky; top:0; background:var(--surface2); color:var(--muted); font-size:12px; }} td.num {{ text-align:right; font-variant-numeric:tabular-nums; }}
    .status.succeeded {{ color:var(--good); }} .status.failed {{ color:var(--bad); }} .status.reserved {{ color:var(--warn); }} .empty {{ padding:28px; text-align:center; color:var(--muted); }} .error {{ max-width:360px; overflow:hidden; text-overflow:ellipsis; }}
    @media(max-width:820px) {{ .stats {{ grid-template-columns:repeat(2,minmax(0,1fr)); }} .grid {{ grid-template-columns:1fr; }} }}
  </style>
</head>
<body>
  <main class="shell">
    <header class="hero">
      <div class="eyebrow">Model observability · {run_id_label}</div>
      <h1>{question}</h1>
      <div class="meta"><a class="chip" href="run_view.html">← Research run</a><span class="chip" id="run-status"></span><span class="chip" id="run-time"></span></div>
    </header>
    <section class="stats" id="stats"></section>
    <h2>Model and profile allocation</h2><section class="grid" id="models"></section>
    <h2>Sub-agent usage</h2><div class="table-wrap"><table><thead><tr><th>Task / Agent</th><th>Operation</th><th>Calls</th><th>Tokens</th><th>Cost</th><th>Peak context</th><th>Latency</th></tr></thead><tbody id="agents"></tbody></table></div>
    <h2>Call trace</h2><div class="table-wrap"><table><thead><tr><th>Time</th><th>Signature</th><th>Profile → model</th><th>Status</th><th>Input</th><th>Output</th><th>Context</th><th>Cost</th><th>Latency</th><th>Routing / error</th></tr></thead><tbody id="calls"></tbody></table></div>
  </main>
  <script id="observability-data" type="application/json">{encoded}</script>
  <script>
    const data=JSON.parse(document.getElementById('observability-data').textContent);
    const esc=v=>String(v??'').replace(/[&<>"']/g,c=>({{'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}})[c]);
    const num=v=>Number(v||0).toLocaleString(); const usd=v=>'$'+Number(v||0).toFixed(4); const pct=v=>(Number(v||0)*100).toFixed(1)+'%'; const duration=v=>v>=1000?(v/1000).toFixed(1)+' s':num(v)+' ms';
    document.getElementById('run-status').textContent=data.run.status; document.getElementById('run-time').textContent=new Date(data.run.updated_at).toLocaleString();
    const t=data.totals; const budget=t.budget_usd?`${{usd(t.cost_usd)}} / ${{usd(t.budget_usd)}} (${{pct(t.budget_ratio)}})`:usd(t.cost_usd);
    document.getElementById('stats').innerHTML=[['Model calls',t.calls],['Input tokens',num(t.input_tokens)],['Output tokens',num(t.output_tokens)],['Cost / budget',budget],['Peak context',pct(t.max_context_ratio)]].map(([k,v])=>`<article class="card stat"><span class="label">${{esc(k)}}</span><strong>${{esc(v)}}</strong></article>`).join('');
    const maxModelCost=Math.max(...data.by_model.map(x=>x.cost_usd),0.000001);
    document.getElementById('models').innerHTML=data.by_model.length?data.by_model.map(row=>`<article class="card"><div class="label">${{esc(row.provider)}} · ${{esc(row.profile)}}</div><h3>${{esc(row.selected_model)}}</h3><div class="chips"><span class="chip">${{num(row.calls)}} calls</span><span class="chip">${{num(row.input_tokens+row.output_tokens)}} tokens</span><span class="chip">${{usd(row.cost_usd)}}</span></div><div class="bar"><span style="width:${{Math.min(100,row.cost_usd/maxModelCost*100)}}%"></span></div></article>`).join(''):'<div class="empty card">No model calls were recorded for this run.</div>';
    document.getElementById('agents').innerHTML=data.by_agent.length?data.by_agent.map(row=>`<tr><td>${{esc(row.task_id||'run')}} · ${{esc(row.agent_id)}}</td><td>${{esc(row.operation)}}</td><td class="num">${{num(row.calls)}}</td><td class="num">${{num(row.input_tokens+row.output_tokens)}}</td><td class="num">${{usd(row.cost_usd)}}</td><td class="num">${{pct(row.max_context_ratio)}}</td><td class="num">${{duration(row.latency_ms)}}</td></tr>`).join(''):'<tr><td colspan="7" class="empty">No sub-agent calls recorded.</td></tr>';
    document.getElementById('calls').innerHTML=data.calls.length?data.calls.map(row=>`<tr><td>${{esc(new Date(row.created_at).toLocaleTimeString())}}</td><td>${{esc(row.task_id||'run')}} · ${{esc(row.agent_id)}} · attempt ${{esc(row.attempt)}}</td><td>${{esc(row.profile)}} → ${{esc(row.selected_model)}}</td><td><span class="status ${{esc(row.status)}}">${{esc(row.status)}}</span></td><td class="num">${{num(row.input_tokens||row.estimated_input_tokens)}}</td><td class="num">${{num(row.output_tokens)}}</td><td class="num">${{pct(row.context_ratio)}}</td><td class="num">${{usd(row.effective_cost_usd)}}</td><td class="num">${{duration(row.latency_ms||0)}}</td><td class="error" title="${{esc(row.error||row.routing_reason)}}">${{esc(row.error||row.routing_reason)}}</td></tr>`).join(''):'<tr><td colspan="10" class="empty">No model calls recorded.</td></tr>';
  </script>
</body>
</html>"""
    run_dir = settings.runs_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / "model_observability.html"
    path.write_text(document, encoding="utf-8")
    (run_dir / "model_calls.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return path
