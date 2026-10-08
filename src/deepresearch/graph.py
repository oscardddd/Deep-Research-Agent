from __future__ import annotations

import json
import operator
from pathlib import Path
from typing import Annotated, Any, TypedDict

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import Send

from .config import Settings
from .audit_policy import (
    aggregate_audits,
    build_audit_plan,
    estimate_single_audit_tokens,
    scoped_research_context,
    select_audit_evidence,
)
from .evidence_policy import (
    build_research_digest,
    curate_evidence,
    select_drilldown_evidence,
    source_risk_summary,
)
from .report_policy import fallback_report_draft, select_report_evidence
from .eventlog import log_step
from .runtime import ResearchRuntime
from .schemas import (
    AuditGap,
    AuditResult,
    ClaimAssessment,
    ClaimStatus,
    Confidence,
    CredibilityTier,
    EvidenceBatch,
    EvidenceRecord,
    QueryIntent,
    QuerySpec,
    ReportDraft,
    ResearchPlan,
    ResearchTask,
    SearchMode,
    SearchResult,
    TaskStatus,
    WorkerDecisionAction,
    WorkerResult,
)
from .store import EvidenceStore, canonicalize_url
from .visualization import render_run_visualization


class GraphState(TypedDict, total=False):
    run_id: str
    question: str
    plan: dict[str, Any]
    tasks: list[dict[str, Any]]
    active_task: dict[str, Any]
    worker_results: Annotated[list[dict[str, Any]], operator.add]
    task_statuses: dict[str, str]
    evidence_ids: list[str]
    preliminary_evidence_ids: list[str]
    active_replan_task_ids: list[str]
    search_calls_used: Annotated[int, operator.add]
    follow_up_round: int
    audit: dict[str, Any]
    report_draft: dict[str, Any]
    report_path: str
    visualization_path: str
    terminal_reason: str


def _bounded_query(value: str | None, max_length: int = 400) -> str | None:
    """Normalize generated search text without violating typed query limits."""
    if value is None:
        return None
    normalized = " ".join(value.split())
    if len(normalized) <= max_length:
        return normalized
    clipped = normalized[:max_length]
    if " " in clipped:
        clipped = clipped.rsplit(" ", 1)[0]
    return clipped or normalized[:max_length]


def _normalize_audit(
    audit: AuditResult, evidence: list[EvidenceRecord]
) -> AuditResult:
    """Enforce claim/evidence invariants after the semantic audit."""
    evidence_by_id = {item.evidence_id: item for item in evidence}
    normalized_claims: list[ClaimAssessment] = []
    used_evidence_ids: set[str] = set()

    for claim in audit.claims:
        claim_provenance: set[str] = set()

        def retain(ids: list[str], limit: int) -> list[str]:
            retained: list[str] = []
            for evidence_id in ids:
                item = evidence_by_id.get(evidence_id)
                if (
                    item is None
                    or item.discovery_only
                    or evidence_id in used_evidence_ids
                ):
                    continue
                provenance = item.provenance_key or item.source_url
                if provenance in claim_provenance:
                    continue
                claim_provenance.add(provenance)
                used_evidence_ids.add(evidence_id)
                retained.append(evidence_id)
                if len(retained) >= limit:
                    break
            return retained

        supporting = retain(claim.supporting_evidence_ids, 4)
        contradicting = retain(claim.contradicting_evidence_ids, 2)

        support_provenance = {
            evidence_by_id[item].provenance_key or evidence_by_id[item].source_url
            for item in supporting
        }
        has_credible_support = any(
            evidence_by_id[item].credibility_tier
            in {CredibilityTier.A, CredibilityTier.B}
            for item in supporting
        )
        status = claim.status
        confidence = claim.confidence

        if not supporting and not contradicting:
            status = ClaimStatus.UNRESOLVED
            confidence = Confidence.LOW
        elif supporting and contradicting:
            status = ClaimStatus.MIXED
            confidence = Confidence.LOW if len(supporting) == 1 else Confidence.MEDIUM
        elif not supporting and contradicting:
            status = ClaimStatus.UNSUPPORTED
            confidence = Confidence.LOW
        elif len(support_provenance) < 2 or not has_credible_support:
            # One origin, or only low-authority origins, is a lead rather than a claim.
            status = ClaimStatus.UNRESOLVED
            confidence = Confidence.LOW

        normalized_claims.append(
            claim.model_copy(
                update={
                    "supporting_evidence_ids": supporting,
                    "contradicting_evidence_ids": contradicting,
                    "status": status,
                    "confidence": confidence,
                }
            )
        )

    return audit.model_copy(update={"claims": normalized_claims})


def _apply_sufficiency_gates(
    audit: AuditResult,
    evidence: list[EvidenceRecord],
    plan: ResearchPlan,
    task_statuses: dict[str, str],
    question: str,
) -> AuditResult:
    """Apply deterministic coverage, execution, challenge, and source gates."""
    evidence_by_id = {item.evidence_id: item for item in evidence}
    succeeded = {
        task_id for task_id, status in task_statuses.items()
        if status == TaskStatus.SUCCEEDED.value
    }
    covered = {
        dimension
        for task in plan.tasks
        if task.task_id in succeeded
        for dimension in task.covered_dimensions
    }
    high_priority_ok = all(
        task.importance != "high"
        or task.task_id in succeeded
        or (task.covered_dimensions and set(task.covered_dimensions) <= covered)
        for task in plan.tasks
    )
    missing_dimensions = [
        dimension for dimension in plan.coverage_contract.required_dimensions
        if dimension not in covered
    ]
    claim_covered_dimensions = {
        claim.dimension
        for claim in audit.claims
        if claim.dimension in plan.coverage_contract.required_dimensions
        and claim.status != ClaimStatus.UNRESOLVED
    }
    missing_claim_dimensions = [
        dimension
        for dimension in plan.coverage_contract.required_dimensions
        if dimension not in claim_covered_dimensions
    ]
    challenge_tasks = [
        task for task in plan.tasks if task.search_mode == SearchMode.CHALLENGE
    ]
    challenge_ok = not challenge_tasks or any(
        task.task_id in succeeded
        or (task.covered_dimensions and set(task.covered_dimensions) <= covered)
        for task in challenge_tasks
    )
    robust_claims = 0
    source_ok = True
    for claim in audit.claims:
        if claim.status not in {ClaimStatus.SUPPORTED, ClaimStatus.MIXED}:
            continue
        robust_claims += 1
        cited = claim.supporting_evidence_ids
        if claim.status == ClaimStatus.MIXED:
            cited = [*cited, *claim.contradicting_evidence_ids]
        provenance = {
            evidence_by_id[item].provenance_key
            or evidence_by_id[item].source_url
            for item in cited if item in evidence_by_id
        }
        has_credible = any(
            evidence_by_id[item].credibility_tier
            in {CredibilityTier.A, CredibilityTier.B}
            for item in cited if item in evidence_by_id
        )
        if len(provenance) < 2 or not has_credible:
            source_ok = False
    source_ok = source_ok and robust_claims > 0

    gaps = list(dict.fromkeys([
        *audit.gaps,
        *(gap.description for gap in audit.actionable_gaps),
    ]))
    actionable = list(audit.actionable_gaps)

    def add_gap(
        description: str,
        *,
        gap_type: str,
        dimension: str | None = None,
        task_ids: list[str] | None = None,
        priority: str = "high",
    ) -> None:
        gaps.append(description)
        if any(gap.description == description for gap in actionable):
            return
        actionable.append(AuditGap(
            gap_id=f"GATE-G{len(actionable) + 1}",
            dimension=dimension,
            task_ids=task_ids or [],
            gap_type=gap_type,
            priority=priority,
            description=description,
            missing_evidence=description,
            suggested_query=_bounded_query(
                f"{question} {description} primary independent evidence"
            ),
        ))

    # Backfill structured handoff for old checkpoints and model fixtures.
    for description in audit.gaps:
        if not any(gap.description == description for gap in actionable):
            actionable.append(AuditGap(
                gap_id=f"LEGACY-G{len(actionable) + 1}",
                gap_type="other",
                priority="medium",
                description=description,
                suggested_query=audit.follow_up_query,
            ))
    if not high_priority_ok:
        add_gap(
            "One or more high-priority research tasks are incomplete.",
            gap_type="task_execution",
            task_ids=[
                task.task_id for task in plan.tasks
                if task.importance == "high" and task.task_id not in succeeded
            ],
        )
    if missing_dimensions:
        add_gap(
            "Required dimensions lack completed research: " + ", ".join(missing_dimensions),
            gap_type="coverage",
            task_ids=[
                task.task_id for task in plan.tasks
                if set(task.covered_dimensions) & set(missing_dimensions)
            ],
        )
    if missing_claim_dimensions:
        add_gap(
            "Required dimensions lack an evidence-backed audited conclusion: "
            + ", ".join(missing_claim_dimensions),
            gap_type="verification",
            task_ids=[
                task.task_id for task in plan.tasks
                if set(task.covered_dimensions) & set(missing_claim_dimensions)
            ],
        )
    if not challenge_ok:
        add_gap(
            "The planned counterevidence search is incomplete.",
            gap_type="counterevidence",
            task_ids=[task.task_id for task in challenge_tasks],
        )
    if not source_ok:
        add_gap(
            "One or more core claims lack two independent origins including an A/B-tier source.",
            gap_type="source_quality",
        )
    gaps = list(dict.fromkeys(gaps))[:6]
    visible_gaps = set(gaps)
    actionable = [
        gap for gap in actionable if gap.description in visible_gaps
    ][:6]
    overall = bool(
        audit.sufficient and high_priority_ok and not missing_dimensions
        and not missing_claim_dimensions
        and challenge_ok and source_ok and not gaps
    )
    follow_up = audit.follow_up_query
    if not overall and not follow_up and gaps:
        follow_up = f"{question} {gaps[0]} primary independent evidence"
    updated = {
        **audit.model_dump(mode="json"),
        "sufficient": overall,
        "coverage_sufficient": not missing_dimensions and not missing_claim_dimensions,
        "source_sufficient": source_ok,
        "task_execution_sufficient": high_priority_ok,
        "challenge_sufficient": challenge_ok,
        "gaps": gaps,
        "actionable_gaps": [
            gap.model_dump(mode="json") for gap in actionable
        ],
        "follow_up_query": _bounded_query(follow_up),
    }
    return AuditResult.model_validate(updated)


def build_graph(
    settings: Settings,
    store: EvidenceStore,
    runtime: ResearchRuntime,
) -> StateGraph:
    def reconnaissance_node(state: GraphState) -> dict[str, Any]:
        if settings.max_reconnaissance_queries <= 0:
            return {"preliminary_evidence_ids": []}

        run_id = state["run_id"]
        task = ResearchTask(
            task_id="T0",
            question=f"Map the research landscape for: {state['question']}",
            research_role="reconnaissance researcher",
            objective=(
                "Identify terminology, major dimensions, candidate primary sources, "
                "and likely disputes before decomposing the full research plan."
            ),
            must_find=["major dimensions", "candidate original sources"],
            avoid=["treating a preliminary synthesis as established evidence"],
            query_budget=min(3, max(1, settings.max_reconnaissance_queries)),
            importance="medium",
        )
        evidence_ids: list[str] = []
        search_calls = 0
        errors: list[str] = []
        try:
            query_plan = runtime.reconnaissance_queries(
                run_id, state["question"]
            )
            for query in sorted(
                query_plan.queries,
                key=lambda item: item.priority,
                reverse=True,
            )[: settings.max_reconnaissance_queries]:
                search_calls += 1
                try:
                    results = runtime.search(run_id, task, query)
                    results = runtime.extract_pages(run_id, task, results)
                    store.persist_worker_artifacts(
                        run_id, task, results, EvidenceBatch()
                    )
                    assessments = runtime.assess_sources(
                        run_id, task, results
                    )
                    store.persist_source_assessments(
                        run_id, task, results, assessments
                    )
                    batch = runtime.extract(run_id, task, results)
                    _, new_ids = store.persist_worker_artifacts(
                        run_id, task, results, batch
                    )
                    evidence_ids.extend(new_ids)
                except Exception as error:
                    errors.append(f"{query.query}: {error}")
        except Exception as error:
            errors.append(str(error))

        log_step(
            "Planner",
            "reconnaissance.completed",
            run_id=run_id,
            searches=search_calls,
            evidence=len(set(evidence_ids)),
            errors=errors,
        )
        return {
            "preliminary_evidence_ids": list(dict.fromkeys(evidence_ids)),
            "search_calls_used": search_calls,
        }

    def plan_node(state: GraphState) -> dict[str, Any]:
        preliminary_ids = set(state.get("preliminary_evidence_ids", []))
        preliminary = curate_evidence(
            [
                item
                for item in store.list_evidence(state["run_id"])
                if item.evidence_id in preliminary_ids
            ],
            include_discovery=True,
        )
        plan = runtime.plan(
            state["run_id"], state["question"], preliminary
        )
        for task in plan.tasks:
            store.upsert_task(state["run_id"], task)
            log_step(
                "Planner",
                "task.created",
                run_id=state["run_id"],
                task_id=task.task_id,
                mode=task.search_mode.value,
                role=task.research_role,
                objective=task.objective or task.question,
                query_budget=task.query_budget,
            )
        return {
            "plan": plan.model_dump(mode="json"),
            "tasks": [task.model_dump(mode="json") for task in plan.tasks],
            "task_statuses": {
                task.task_id: TaskStatus.PENDING.value for task in plan.tasks
            },
        }

    def dispatch_initial_workers(state: GraphState) -> list[Send]:
        plan = ResearchPlan.model_validate(state["plan"])
        return [
            Send(
                "research_worker",
                {
                    "run_id": state["run_id"],
                    "question": state["question"],
                    "active_task": task.model_dump(mode="json"),
                },
            )
            for task in plan.tasks
        ]

    def research_worker(state: GraphState) -> dict[str, Any]:
        run_id = state["run_id"]
        task = ResearchTask.model_validate(state["active_task"])
        store.set_task_status(run_id, task.task_id, TaskStatus.RUNNING)
        agent = f"ResearchWorker:{task.task_id}"
        log_step(
            agent,
            "worker.started",
            run_id=run_id,
            mode=task.search_mode.value,
            role=task.research_role,
            objective=task.objective or task.question,
        )
        search_calls_used = 0
        try:
            query_plan = runtime.formulate_queries(run_id, task)
            initial_query_count = (
                1 if task.search_mode == SearchMode.VERIFY
                else min(task.query_budget, settings.max_queries_per_worker)
            )
            seed_queries = sorted(
                query_plan.queries,
                key=lambda item: item.priority,
                reverse=True,
            )[:initial_query_count]
            log_step(
                agent,
                "queries.formulated",
                run_id=run_id,
                candidates=len(query_plan.queries),
                selected=len(seed_queries),
                queries=[
                    {
                        "intent": item.intent.value,
                        "query": item.query,
                        "priority": item.priority,
                        "provider": item.provider,
                    }
                    for item in seed_queries
                ],
            )

            if not seed_queries:
                raise RuntimeError("Worker did not formulate an initial query")

            def query_key(query: str) -> str:
                return " ".join(query.casefold().split())

            current_query = seed_queries[0]
            current_source_url: str | None = None
            unused_seed_queries = list(seed_queries[1:])
            executed_queries = []
            executed_query_keys: set[str] = set()
            visited_source_urls: set[str] = set()
            source_ids: list[str] = []
            evidence_ids: list[str] = []
            search_errors: list[str] = []
            unresolved_gaps: list[str] = []
            stop_reason: str | None = None
            adaptive_steps = 0

            for step_number in range(1, settings.max_adaptive_steps + 1):
                adaptive_steps = step_number
                following_source = current_source_url is not None
                if following_source:
                    step_label = current_source_url or "unknown source"
                else:
                    executed_queries.append(current_query)
                    executed_query_keys.add(query_key(current_query.query))
                    search_calls_used += 1
                    step_label = current_query.query
                log_step(
                    agent,
                    "adaptive.step.started",
                    run_id=run_id,
                    step=step_number,
                    max_steps=settings.max_adaptive_steps,
                    intent=(
                        "follow_source"
                        if following_source
                        else current_query.intent.value
                    ),
                    query=(None if following_source else current_query.query),
                    target_source_url=current_source_url,
                )

                step_evidence_ids: list[str] = []
                try:
                    if following_source:
                        search_results = [
                            SearchResult(
                                title=current_source_url or "Attributed source",
                                url=current_source_url or "",
                                content="",
                                rank=1,
                                query=f"follow attributed source from {task.task_id}",
                            )
                        ]
                    else:
                        search_results = runtime.search(
                            run_id, task, current_query
                        )
                except Exception as error:
                    search_results = []
                    search_errors.append(
                        f"step {step_number} retrieve {step_label}: {error}"
                    )

                if not search_results:
                    search_errors.append(
                        f"step {step_number} retrieve {step_label}: no results"
                    )
                else:
                    try:
                        search_results = runtime.extract_pages(
                            run_id, task, search_results
                        )
                        snippet_fallbacks = sum(
                            item.content_source != "full_page"
                            for item in search_results
                        )
                        if snippet_fallbacks:
                            search_errors.append(
                                f"step {step_number} full-page extraction unavailable "
                                f"for {snippet_fallbacks} source(s); used snippets"
                            )
                    except Exception as error:
                        search_errors.append(
                            f"step {step_number} full-page extraction: {error}; "
                            "used search snippets"
                        )

                    persisted_source_ids, _ = store.persist_worker_artifacts(
                        run_id, task, search_results, EvidenceBatch()
                    )
                    source_ids.extend(persisted_source_ids)
                    try:
                        assessments = runtime.assess_sources(
                            run_id, task, search_results
                        )
                        store.persist_source_assessments(
                            run_id, task, search_results, assessments
                        )
                        batch = runtime.extract(run_id, task, search_results)
                        new_source_ids, step_evidence_ids = (
                            store.persist_worker_artifacts(
                                run_id, task, search_results, batch
                            )
                        )
                        source_ids.extend(new_source_ids)
                        evidence_ids.extend(step_evidence_ids)
                    except Exception as error:
                        search_errors.append(
                            f"step {step_number} extract {step_label}: {error}"
                        )

                if following_source and current_source_url:
                    visited_source_urls.add(current_source_url)
                    current_source_url = None

                evidence_by_id = {
                    item.evidence_id: item for item in store.list_evidence(run_id)
                }
                accepted_evidence = [
                    evidence_by_id[evidence_id]
                    for evidence_id in dict.fromkeys(evidence_ids)
                    if evidence_id in evidence_by_id
                ]
                log_step(
                    agent,
                    "adaptive.progress",
                    run_id=run_id,
                    step=step_number,
                    new_evidence=len(step_evidence_ids),
                    total_evidence=len(accepted_evidence),
                    source_domains=len(
                        {item.source_domain for item in accepted_evidence}
                    ),
                )

                if step_number >= settings.max_adaptive_steps:
                    stop_reason = "safety_limit"
                    log_step(
                        agent,
                        "adaptive.limit_reached",
                        run_id=run_id,
                        step=step_number,
                        max_steps=settings.max_adaptive_steps,
                    )
                    break

                try:
                    decision = runtime.decide_worker_next_step(
                        run_id,
                        task,
                        executed_queries,
                        accepted_evidence,
                        step_number,
                        search_errors,
                    )
                except Exception as error:
                    search_errors.append(
                        f"step {step_number} adaptive decision: {error}"
                    )
                    fallback = next(
                        (
                            item for item in unused_seed_queries
                            if query_key(item.query) not in executed_query_keys
                        ),
                        None,
                    )
                    if fallback is None:
                        stop_reason = "blocked"
                        unresolved_gaps = ["Adaptive decision failed"]
                        break
                    current_query = fallback
                    continue

                unresolved_gaps = decision.unresolved_gaps
                log_step(
                    agent,
                    "adaptive.decision",
                    run_id=run_id,
                    step=step_number,
                    action=decision.action.value,
                    stop_reason=(
                        decision.stop_reason.value if decision.stop_reason else None
                    ),
                    next_query=(
                        decision.next_query.query if decision.next_query else None
                    ),
                    target_source_url=decision.target_source_url,
                    unresolved_gaps=decision.unresolved_gaps,
                    summary=decision.decision_summary,
                )
                if decision.action == WorkerDecisionAction.STOP:
                    stop_reason = decision.stop_reason.value
                    break

                if decision.action == WorkerDecisionAction.FOLLOW_SOURCE:
                    target_url = decision.target_source_url
                    allowed_source_urls = {
                        canonicalize_url(item.attributed_source_url):
                        item.attributed_source_url
                        for item in accepted_evidence
                        if item.attributed_source_url
                    }
                    canonical_target = (
                        canonicalize_url(target_url) if target_url else ""
                    )
                    if canonical_target not in allowed_source_urls:
                        follow_up = next(
                            (
                                item.follow_up_query
                                for item in accepted_evidence
                                if item.follow_up_query
                                and query_key(item.follow_up_query)
                                not in executed_query_keys
                            ),
                            None,
                        )
                        fallback = (
                            QuerySpec(
                                intent=QueryIntent.VERIFICATION,
                                query=follow_up,
                                expected_evidence=(
                                    "Locate and verify the attributed original source"
                                ),
                                priority=3,
                            )
                            if follow_up
                            else next(
                                (
                                    item for item in unused_seed_queries
                                    if query_key(item.query)
                                    not in executed_query_keys
                                ),
                                None,
                            )
                        )
                        if fallback is None:
                            stop_reason = "blocked"
                            unresolved_gaps = [
                                "Worker proposed an original-source URL absent from accepted evidence"
                            ]
                            break
                        current_query = fallback
                        continue
                    target_url = allowed_source_urls[canonical_target]
                    if target_url in visited_source_urls:
                        stop_reason = "saturated"
                        unresolved_gaps = list(
                            dict.fromkeys(
                                [
                                    *unresolved_gaps,
                                    "Attributed source URL was already inspected",
                                ]
                            )
                        )
                        break
                    current_source_url = target_url
                    continue

                next_query = decision.next_query
                if next_query is None:
                    stop_reason = "blocked"
                    unresolved_gaps = ["Adaptive search decision omitted next_query"]
                    break
                if query_key(next_query.query) in executed_query_keys:
                    fallback = next(
                        (
                            item for item in unused_seed_queries
                            if query_key(item.query) not in executed_query_keys
                        ),
                        None,
                    )
                    if fallback is None:
                        stop_reason = "saturated"
                        unresolved_gaps = list(
                            dict.fromkeys(
                                [
                                    *unresolved_gaps,
                                    "No materially new query was proposed",
                                ]
                            )
                        )
                        log_step(
                            agent,
                            "adaptive.duplicate_query_stopped",
                            run_id=run_id,
                            query=next_query.query,
                        )
                        break
                    next_query = fallback
                current_query = next_query

            unique_source_ids = list(dict.fromkeys(source_ids))
            unique_evidence_ids = list(dict.fromkeys(evidence_ids))
            if not unique_evidence_ids:
                outcome = "failed_retryable"
                if not search_errors:
                    search_errors.append("No valid evidence was extracted")
            elif search_errors:
                outcome = "partial"
            else:
                outcome = "succeeded"
            result = WorkerResult(
                task_id=task.task_id,
                execution_outcome=outcome,
                evidence_ids=unique_evidence_ids,
                source_ids=unique_source_ids,
                search_calls_used=search_calls_used,
                new_evidence_count=len(unique_evidence_ids),
                executed_queries=executed_queries,
                adaptive_steps=adaptive_steps,
                stop_reason=stop_reason,
                unresolved_gaps=unresolved_gaps,
                error="; ".join(search_errors) or None,
            )
        except Exception as error:
            result = WorkerResult(
                task_id=task.task_id,
                execution_outcome="failed_retryable",
                search_calls_used=search_calls_used,
                adaptive_steps=search_calls_used,
                stop_reason="blocked",
                error=str(error),
            )

        log_step(
            agent,
            "worker.completed",
            run_id=run_id,
            outcome=result.execution_outcome,
            sources=len(result.source_ids),
            evidence=result.new_evidence_count,
            search_calls=result.search_calls_used,
            adaptive_steps=result.adaptive_steps,
            stop_reason=result.stop_reason,
            unresolved_gaps=result.unresolved_gaps,
            error=result.error,
        )

        return {
            "worker_results": [result.model_dump(mode="json")],
            "search_calls_used": result.search_calls_used,
        }

    def aggregate_node(state: GraphState) -> dict[str, Any]:
        statuses = dict(state.get("task_statuses", {}))
        evidence_ids: list[str] = []
        task_by_id = {
            task.task_id: task
            for task in (
                ResearchTask.model_validate(payload)
                for payload in state.get("tasks", [])
            )
        }
        persisted_ids = {
            item.evidence_id for item in store.list_evidence(state["run_id"])
        }

        for payload in state.get("worker_results", []):
            result = WorkerResult.model_validate(payload)
            valid_evidence_ids = [
                item for item in result.evidence_ids if item in persisted_ids
            ]
            evidence_ids.extend(valid_evidence_ids)
            task = task_by_id.get(result.task_id)
            task_evidence = curate_evidence(
                [
                    item
                    for item in store.list_evidence(state["run_id"])
                    if item.evidence_id in valid_evidence_ids
                ]
            )
            task_origins = {
                item.provenance_key or item.source_url
                for item in task_evidence
            }
            criteria_satisfied = bool(task) and (
                len(task_evidence)
                >= task.success_criteria.min_relevant_evidence
                and len(task_origins)
                >= task.success_criteria.min_independent_sources
            )

            if result.execution_outcome == "succeeded" and criteria_satisfied:
                status = TaskStatus.SUCCEEDED
            elif result.execution_outcome in {"succeeded", "partial"}:
                status = TaskStatus.PARTIAL
            else:
                status = TaskStatus.FAILED_RETRYABLE
            statuses[result.task_id] = status.value
            store.set_task_status(
                state["run_id"], result.task_id, status, result.error
            )

        log_step(
            "Orchestrator",
            "workers.aggregated",
            run_id=state["run_id"],
            task_statuses=statuses,
            evidence=len(set(evidence_ids)),
        )

        return {
            "task_statuses": statuses,
            "evidence_ids": list(dict.fromkeys(evidence_ids)),
        }

    def audit_node(state: GraphState) -> dict[str, Any]:
        round_number = state.get("follow_up_round", 0)
        raw_evidence = store.list_evidence(state["run_id"])
        evidence = curate_evidence(raw_evidence)

        if evidence:
            plan = ResearchPlan.model_validate(state["plan"])
            task_statuses = state.get("task_statuses", {})
            risks = source_risk_summary(raw_evidence)
            research_context = {
                "plan": state.get("plan", {}),
                "task_statuses": task_statuses,
                "search_calls_used": state.get("search_calls_used", 0),
                "search_budget": None,
                "source_risks": risks,
            }
            estimated_tokens = estimate_single_audit_tokens(
                evidence, research_context
            )
            audit_plan = build_audit_plan(
                plan,
                evidence,
                task_statuses,
                round_number=round_number,
                estimated_single_tokens=estimated_tokens,
                single_context_budget=(
                    settings.single_audit_context_token_budget
                ),
                max_specialists=settings.max_audit_specialists,
            )
            if audit_plan.mode == "single":
                store.save_audit_plan(
                    state["run_id"],
                    round_number,
                    audit_plan.model_dump(mode="json"),
                )
                log_step(
                    "AuditCoordinator",
                    "audit.planned",
                    run_id=state["run_id"],
                    round=round_number,
                    mode="single",
                    estimated_single_audit_tokens=estimated_tokens,
                    checks=0,
                )
                audit = runtime.audit(
                    state["run_id"],
                    state["question"],
                    evidence,
                    round_number,
                    research_context=research_context,
                )
            else:
                populated_checks = []
                selected_by_check: dict[str, list[EvidenceRecord]] = {}
                for check in audit_plan.checks:
                    selected, selected_tokens = select_audit_evidence(
                        evidence,
                        check,
                        token_budget=(
                            settings.audit_evidence_token_budget_per_check
                        ),
                    )
                    populated_check = check.model_copy(update={
                        "evidence_ids": [item.evidence_id for item in selected],
                        "estimated_evidence_tokens": selected_tokens,
                    })
                    populated_checks.append(populated_check)
                    selected_by_check[check.check_id] = selected
                audit_plan = audit_plan.model_copy(
                    update={"checks": populated_checks}
                )
                store.save_audit_plan(
                    state["run_id"],
                    round_number,
                    audit_plan.model_dump(mode="json"),
                )
                log_step(
                    "AuditCoordinator",
                    "audit.planned",
                    run_id=state["run_id"],
                    round=round_number,
                    mode="hierarchical",
                    estimated_single_audit_tokens=estimated_tokens,
                    checks=len(populated_checks),
                    evidence_per_check=[
                        len(check.evidence_ids) for check in populated_checks
                    ],
                    tokens_per_check=[
                        check.estimated_evidence_tokens
                        for check in populated_checks
                    ],
                )
                partial_audits: list[AuditResult] = []
                for check in populated_checks:
                    partial_audits.append(runtime.audit(
                        state["run_id"],
                        state["question"],
                        selected_by_check[check.check_id],
                        round_number,
                        research_context=scoped_research_context(
                            plan, task_statuses, risks, check
                        ),
                        audit_scope=check.check_id,
                    ))
                audit = aggregate_audits(audit_plan, partial_audits)
                log_step(
                    "AuditCoordinator",
                    "audit.aggregated",
                    run_id=state["run_id"],
                    round=round_number,
                    specialists=len(partial_audits),
                    claims=len(audit.claims),
                    actionable_gaps=len(audit.actionable_gaps),
                    sufficient=audit.sufficient,
                )
            audit = _normalize_audit(audit, evidence)
            audit = _apply_sufficiency_gates(
                audit,
                evidence,
                ResearchPlan.model_validate(state["plan"]),
                state.get("task_statuses", {}),
                state["question"],
            )
        else:
            audit = AuditResult(
                sufficient=False,
                coverage_sufficient=False,
                source_sufficient=False,
                task_execution_sufficient=False,
                claims=[],
                gaps=["No valid verbatim evidence was extracted."],
                follow_up_query=_bounded_query(
                    f"{state['question']} primary evidence"
                ),
            )

        store.save_audit(state["run_id"], round_number, audit)
        return {"audit": audit.model_dump(mode="json")}

    def route_after_audit(state: GraphState) -> str:
        audit = AuditResult.model_validate(state["audit"])
        can_replan = (
            not audit.sufficient
            and state.get("follow_up_round", 0) < settings.max_replan_rounds
        )
        route = "replan" if can_replan else "synthesize"
        log_step(
            "Orchestrator",
            "audit.routed",
            run_id=state["run_id"],
            route=route,
            sufficient=audit.sufficient,
            searches_used=state.get("search_calls_used", 0),
            search_budget="unbounded",
        )
        return route

    def replan_node(state: GraphState) -> dict[str, Any]:
        audit = AuditResult.model_validate(state["audit"])
        next_round = state.get("follow_up_round", 0) + 1
        plan = ResearchPlan.model_validate(state["plan"])
        raw_evidence = store.list_evidence(state["run_id"])
        evidence = curate_evidence(
            raw_evidence,
            include_discovery=True,
        )
        risks = source_risk_summary(raw_evidence)
        digest = build_research_digest(
            plan,
            audit,
            evidence,
            state.get("task_statuses", {}),
            state.get("worker_results", []),
            risks,
            next_round,
        )
        # Reserve most of the supervisor budget for the stable digest and output;
        # raw excerpts are a bounded, query-relevant drill-down only.
        digest_tokens = max(1, len(digest.model_dump_json()) // 3)
        drilldown_budget = max(
            1200,
            settings.replan_context_token_budget - digest_tokens - 4000,
        )
        evidence = select_drilldown_evidence(
            evidence,
            audit,
            state["question"],
            token_budget=drilldown_budget,
        )
        digest = digest.model_copy(
            update={"drilldown_evidence_ids": [item.evidence_id for item in evidence]}
        )
        store.save_research_digest(
            state["run_id"], next_round, digest.model_dump(mode="json")
        )
        log_step(
            "ResearchSupervisor",
            "replan.context_prepared",
            run_id=state["run_id"],
            round=next_round,
            corpus_evidence=len(raw_evidence),
            worker_digests=len(digest.workers),
            drilldown_evidence=len(evidence),
            estimated_digest_tokens=digest_tokens,
            drilldown_token_budget=drilldown_budget,
        )
        existing_ids = {task.task_id for task in plan.tasks}
        next_task_index = max(
            [int(task_id[1:]) for task_id in existing_ids] or [0]
        ) + 1
        decision = runtime.replan(
            state["run_id"],
            state["question"],
            plan,
            audit,
            digest,
            evidence,
            state.get("task_statuses", {}),
            risks,
            next_round,
            next_task_index,
        )
        required_dimensions = set(
            plan.coverage_contract.required_dimensions
        )
        new_tasks: list[ResearchTask] = []
        expected_task_index = next_task_index
        for task in decision.new_tasks[: settings.max_new_tasks_per_replan]:
            if (
                task.task_id in existing_ids
                or task.task_id != f"T{expected_task_index}"
            ):
                continue
            if not set(task.covered_dimensions) <= required_dimensions:
                continue
            existing_ids.add(task.task_id)
            new_tasks.append(task)
            expected_task_index += 1
            store.upsert_task(state["run_id"], task)

        log_step(
            "ResearchSupervisor",
            "replan.completed",
            run_id=state["run_id"],
            round=next_round,
            should_continue=decision.should_continue,
            rationale=decision.rationale,
            new_task_ids=[task.task_id for task in new_tasks],
            retired_task_ids=decision.retired_task_ids,
            unresolved_gaps=decision.unresolved_gaps,
        )
        tasks = [
            *state.get("tasks", []),
            *[task.model_dump(mode="json") for task in new_tasks],
        ]
        updated_plan = plan.model_copy(
            update={"tasks": [*plan.tasks, *new_tasks]}
        )
        statuses = dict(state.get("task_statuses", {}))
        for task in new_tasks:
            statuses[task.task_id] = TaskStatus.PENDING.value
        return {
            "tasks": tasks,
            "plan": updated_plan.model_dump(mode="json"),
            "task_statuses": statuses,
            "follow_up_round": next_round,
            "active_replan_task_ids": (
                [task.task_id for task in new_tasks]
                if decision.should_continue
                else []
            ),
        }

    def dispatch_replan_workers(state: GraphState) -> list[Send] | str:
        active_ids = set(state.get("active_replan_task_ids", []))
        tasks = [
            ResearchTask.model_validate(payload)
            for payload in state.get("tasks", [])
            if payload.get("task_id") in active_ids
        ]
        if not tasks:
            return "synthesize"
        return [
            Send(
                "research_worker",
                {
                    "run_id": state["run_id"],
                    "question": state["question"],
                    "active_task": task.model_dump(mode="json"),
                },
            )
            for task in tasks
        ]

    def synthesize_node(state: GraphState) -> dict[str, Any]:
        audit = AuditResult.model_validate(state["audit"])
        plan = ResearchPlan.model_validate(state["plan"])
        raw_evidence = store.list_evidence(state["run_id"])
        evidence = curate_evidence(raw_evidence)
        writer_evidence = select_report_evidence(plan, audit, evidence)
        try:
            draft = runtime.synthesize_report(
                state["run_id"],
                state["question"],
                plan,
                audit,
                writer_evidence,
            )
        except Exception as error:
            log_step(
                "ReportWriter",
                "report.synthesis_fallback",
                run_id=state["run_id"],
                error=f"{type(error).__name__}: {error}",
            )
            draft = fallback_report_draft(
                state["question"], plan, audit, writer_evidence
            )
        return {"report_draft": draft.model_dump(mode="json")}

    def report_node(state: GraphState) -> dict[str, Any]:
        audit = AuditResult.model_validate(state["audit"])
        plan = ResearchPlan.model_validate(state["plan"])
        raw_evidence = store.list_evidence(state["run_id"])
        evidence = curate_evidence(raw_evidence)
        draft = (
            ReportDraft.model_validate(state["report_draft"])
            if state.get("report_draft")
            else fallback_report_draft(
                state["question"], plan, audit, evidence
            )
        )
        report_path = render_report(
            settings=settings,
            store=store,
            run_id=state["run_id"],
            question=state["question"],
            draft=draft,
            audit=audit,
            evidence=evidence,
            plan=plan,
            task_statuses=state.get("task_statuses", {}),
        )
        visualization_path = report_path.parent / "run_view.html"
        terminal_reason = (
            "evidence_sufficient"
            if audit.sufficient
            else "research_complete_with_open_gaps"
        )
        store.set_run_status(
            state["run_id"], "completed", report_path=str(report_path)
        )
        log_step(
            "Reporter",
            "report.completed",
            run_id=state["run_id"],
            path=str(report_path),
            visualization_path=str(visualization_path),
            evidence=len(evidence),
            statements=sum(
                len(section.findings) for section in draft.sections
            ),
            terminal_reason=terminal_reason,
        )
        return {
            "report_path": str(report_path),
            "visualization_path": str(visualization_path),
            "terminal_reason": terminal_reason,
        }

    builder = StateGraph(GraphState)
    builder.add_node("reconnaissance", reconnaissance_node)
    builder.add_node("plan", plan_node)
    builder.add_node("research_worker", research_worker)
    builder.add_node("aggregate", aggregate_node)
    builder.add_node("audit", audit_node)
    builder.add_node("replan", replan_node)
    builder.add_node("synthesize", synthesize_node)
    builder.add_node("report", report_node)

    builder.add_edge(START, "reconnaissance")
    builder.add_edge("reconnaissance", "plan")
    builder.add_conditional_edges("plan", dispatch_initial_workers, ["research_worker"])
    builder.add_edge("research_worker", "aggregate")
    builder.add_edge("aggregate", "audit")
    builder.add_conditional_edges(
        "audit",
        route_after_audit,
        {"replan": "replan", "synthesize": "synthesize"},
    )
    builder.add_conditional_edges(
        "replan",
        dispatch_replan_workers,
        ["research_worker", "synthesize"],
    )
    builder.add_edge("synthesize", "report")
    builder.add_edge("report", END)
    return builder


def render_report(
    *,
    settings: Settings,
    store: EvidenceStore,
    run_id: str,
    question: str,
    draft: ReportDraft,
    audit: AuditResult,
    evidence: list[EvidenceRecord],
    plan: ResearchPlan,
    task_statuses: dict[str, str],
) -> Path:
    run_dir = settings.runs_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    evidence_by_id = {item.evidence_id: item for item in evidence}
    citation_number_by_origin: dict[str, int] = {}
    citation_records: list[dict[str, Any]] = []

    def citation_marker(evidence_ids: list[str]) -> str:
        """Map internal evidence IDs to stable, source-level paper citations."""
        numbers: list[int] = []
        for evidence_id in evidence_ids:
            item = evidence_by_id.get(evidence_id)
            if item is None:
                continue
            # Cite the page whose content was actually retrieved and used to
            # create this evidence record. attributed_source_url is a lead to
            # an underlying source mentioned by that page; it must not replace
            # the retrieved PDF/page unless a worker follows and extracts it as
            # its own EvidenceRecord.
            reference_url = item.source_url
            reference_title = item.source_title
            origin = canonicalize_url(reference_url) or evidence_id
            number = citation_number_by_origin.get(origin)
            if number is None:
                number = len(citation_records) + 1
                citation_number_by_origin[origin] = number
                citation_records.append(
                    {
                        "citation_number": number,
                        "title": reference_title,
                        "url": reference_url,
                        "evidence_ids": [],
                    }
                )
            record = citation_records[number - 1]
            if evidence_id not in record["evidence_ids"]:
                record["evidence_ids"].append(evidence_id)
            if number not in numbers:
                numbers.append(number)
        return f"[{', '.join(str(number) for number in numbers)}]" if numbers else ""

    def cited_text(text: str, marker: str) -> str:
        """Place a numeric citation before terminal punctuation."""
        normalized = text.strip()
        if not marker:
            return normalized
        if normalized and normalized[-1] in ".!?。！？":
            return f"{normalized[:-1].rstrip()} {marker}{normalized[-1]}"
        return f"{normalized} {marker}"

    def render_statement(statement) -> str:
        return cited_text(
            statement.text,
            citation_marker(statement.evidence_ids),
        )

    lines = [
        f"# {draft.title}",
        "",
        f"**Question:** {question}",
        "",
        "## Direct answer",
        "",
    ]
    if draft.direct_answer:
        for statement in draft.direct_answer:
            lines.extend([render_statement(statement), ""])
    else:
        lines.extend([
            "The available evidence is insufficient to provide a reliable direct answer.",
            "",
        ])

    if draft.executive_summary:
        lines.extend(["## Executive summary", ""])
        lines.extend([
            f"- {render_statement(statement)}"
            for statement in draft.executive_summary
        ])
        lines.append("")

    lines.extend(["## Findings", ""])
    for section in draft.sections:
        lines.extend([f"### {section.heading}", ""])
        if section.findings:
            for statement in section.findings:
                lines.extend([render_statement(statement), ""])
        else:
            lines.extend([
                "The available evidence does not support a reliable conclusion for this dimension.",
                "",
            ])

    if draft.comparisons:
        lines.extend(["## Comparisons", ""])
        lines.extend([
            f"- {render_statement(statement)}"
            for statement in draft.comparisons
        ])
        lines.append("")

    if draft.methodology:
        lines.extend(["## Methodology", ""])
        lines.extend([f"- {item}" for item in draft.methodology])
        lines.append("")

    lines.extend(["## Limitations", ""])
    if draft.limitations:
        lines.extend([f"- {item}" for item in draft.limitations])
    else:
        lines.append("- No material evidence limitation was identified.")
    lines.append("")

    lines.extend(["## References", ""])
    if citation_records:
        for record in citation_records:
            title = " ".join(str(record["title"]).split())
            lines.append(
                f'{record["citation_number"]}. [{title}]({record["url"]})'
            )
    else:
        lines.append("No source was cited in the retained findings.")
    lines.append("")

    report_path = run_dir / "report.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    (run_dir / "evidence.json").write_text(
        json.dumps(
            [item.model_dump(mode="json") for item in evidence],
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    (run_dir / "citations.json").write_text(
        json.dumps(citation_records, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (run_dir / "report_draft.json").write_text(
        draft.model_dump_json(indent=2), encoding="utf-8"
    )
    (run_dir / "audit.json").write_text(
        audit.model_dump_json(indent=2), encoding="utf-8"
    )
    audit_plan = store.get_latest_audit_plan(run_id)
    if audit_plan is not None:
        (run_dir / "audit_plan.json").write_text(
            json.dumps(audit_plan, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    (run_dir / "source_risks.json").write_text(
        json.dumps(
            source_risk_summary(store.list_evidence(run_id)),
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    (run_dir / "source_assessments.json").write_text(
        json.dumps(
            store.list_source_assessments(run_id),
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    (run_dir / "trace.json").write_text(
        json.dumps(store.list_operations(run_id), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    render_run_visualization(
        settings=settings,
        store=store,
        run_id=run_id,
        audit=audit,
        evidence=evidence,
        status_override="completed",
    )
    return report_path


def run_graph(
    *,
    settings: Settings,
    store: EvidenceStore,
    runtime: ResearchRuntime,
    run_id: str,
    question: str | None,
    resume: bool = False,
) -> GraphState:
    log_step(
        "Orchestrator",
        "run.started",
        run_id=run_id,
        resume=resume,
        search_budget="unbounded",
    )
    builder = build_graph(settings, store, runtime)
    config = {"configurable": {"thread_id": run_id}}

    with SqliteSaver.from_conn_string(str(settings.checkpoint_db)) as checkpointer:
        graph = builder.compile(checkpointer=checkpointer)
        if resume:
            result = graph.invoke(None, config=config)
           
        else:
            if not question:
                raise ValueError("question is required for a new run")
            initial_state: GraphState = {
                "run_id": run_id,
                "question": question,
                "worker_results": [],
                "task_statuses": {},
                "evidence_ids": [],
                "preliminary_evidence_ids": [],
                "active_replan_task_ids": [],
                "search_calls_used": 0,
                "follow_up_round": 0,
            }
            result = graph.invoke(initial_state, config=config)
    if not result.get("terminal_reason") or not result.get("report_path"):
        raise RuntimeError(
            "Graph stopped without the completion contract: terminal_reason "
            "and report_path are required"
        )
    log_step(
        "Orchestrator",
        "run.completed",
        run_id=run_id,
        terminal_reason=result.get("terminal_reason"),
    )
    return GraphState(result)
