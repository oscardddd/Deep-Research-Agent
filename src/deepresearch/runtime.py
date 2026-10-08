from __future__ import annotations

import hashlib
import json
import time
from typing import Any

from .config import Settings
from .eventlog import log_step
from .evidence_policy import supervisor_evidence_payload
from .report_policy import normalize_report_draft
from .model_gateway import model_call_scope
from .providers import LanguageModelProvider, SearchProvider
from .schemas import (
    AuditResult,
    EvidenceBatch,
    EvidenceRecord,
    PageExtractionBatch,
    QueryPlan,
    QuerySpec,
    ReportDraft,
    ReplanDecision,
    ResearchPlan,
    ResearchStateDigest,
    ResearchTask,
    SearchResult,
    SourceAssessment,
    SourceAssessmentBatch,
    SourceHardFlag,
    SourceScoreDimensions,
    SourceType,
    SourceDirectness,
    WorkerDecision,
)
from .store import EvidenceStore, canonicalize_url, stable_id
from .source_scoring import SOURCE_RUBRIC_VERSION, finalize_source_assessment


class ResearchRuntime:
    """Idempotent boundary around model and search side effects."""

    def __init__(
        self,
        settings: Settings,
        store: EvidenceStore,
        search_provider: SearchProvider,
        model_provider: LanguageModelProvider,
    ) -> None:
        self.settings = settings
        self.store = store
        self.search_provider = search_provider
        self.model_provider = model_provider

    def _cached_call(
        self,
        *,
        run_id: str,
        task_id: str | None,
        operation_type: str,
        request: dict[str, Any],
        invoke: Any,
    ) -> dict[str, Any]:
        request_json = json.dumps(request, ensure_ascii=False, sort_keys=True)
        operation_key = stable_id(
            "op", run_id, task_id or "-", operation_type, request_json
        )
        cached = self.store.get_successful_operation(operation_key)
        if cached is not None:
            log_step(
                self._agent_name(operation_type, task_id),
                f"{operation_type}.cache_hit",
                run_id=run_id,
                task_id=task_id,
            )
            return cached

        agent = self._agent_name(operation_type, task_id)
        started = time.perf_counter()
        log_step(
            agent,
            f"{operation_type}.started",
            run_id=run_id,
            task_id=task_id,
            **self._request_log_fields(operation_type, request),
        )
        self.store.start_operation(
            operation_key, run_id, task_id, operation_type, request
        )
        try:
            with model_call_scope(
                run_id=run_id,
                task_id=task_id,
                operation=operation_type,
            ):
                response = invoke()
            payload = (
                response.model_dump(mode="json")
                if hasattr(response, "model_dump")
                else response
            )
            self.store.complete_operation(operation_key, payload)
            log_step(
                agent,
                f"{operation_type}.completed",
                run_id=run_id,
                task_id=task_id,
                elapsed_ms=round((time.perf_counter() - started) * 1000),
                **self._response_log_fields(operation_type, payload),
            )
            return payload
        except Exception as error:
            self.store.fail_operation(operation_key, str(error))
            log_step(
                agent,
                f"{operation_type}.failed",
                run_id=run_id,
                task_id=task_id,
                elapsed_ms=round((time.perf_counter() - started) * 1000),
                error=str(error),
            )
            raise

    @staticmethod
    def _agent_name(operation_type: str, task_id: str | None) -> str:
        if operation_type in {"model.plan", "model.reconnaissance_queries"}:
            return "Planner"
        if operation_type.startswith("model.replan"):
            return "ResearchSupervisor"
        if operation_type == "model.assess_sources":
            return f"SourceEvaluator:{task_id or 'unknown'}"
        if operation_type.startswith("model.audit"):
            return f"Auditor:{task_id}" if task_id else "Auditor"
        if operation_type == "model.synthesize_report":
            return "ReportWriter"
        if operation_type in {
            "model.formulate_queries",
            "model.decide_worker_next_step",
            "search.tavily",
            "extract.tavily",
            "model.extract",
        }:
            return f"ResearchWorker:{task_id or 'unknown'}"
        return "Orchestrator"

    @staticmethod
    def _request_log_fields(
        operation_type: str, request: dict[str, Any]
    ) -> dict[str, Any]:
        if operation_type == "search.tavily":
            return {
                "query": request.get("query"),
                "max_results": request.get("max_results"),
                "depth": request.get("depth"),
            }
        if operation_type == "extract.tavily":
            return {
                "urls": len(request.get("urls", [])),
                "extract_depth": request.get("extract_depth"),
                "format": request.get("format"),
            }
        if operation_type == "model.plan":
            return {
                "model": request.get("model"),
                "preliminary_evidence": len(
                    request.get("preliminary_evidence", [])
                ),
            }
        if operation_type == "model.reconnaissance_queries":
            return {
                "model": request.get("model"),
                "max_queries": request.get("max_queries"),
            }
        if operation_type.startswith("model.replan"):
            digest = request.get("research_digest", {})
            return {
                "model": request.get("model"),
                "round": request.get("round"),
                "evidence_items": len(request.get("drilldown_evidence", [])),
                "corpus_evidence_items": digest.get("evidence_count", 0),
                "worker_digests": len(digest.get("workers", [])),
                "current_tasks": len(request.get("plan", {}).get("tasks", [])),
            }
        if operation_type == "model.formulate_queries":
            task = request.get("task", {})
            return {
                "model": request.get("model"),
                "role": task.get("research_role"),
                "max_candidates": request.get("max_candidates"),
            }
        if operation_type == "model.extract":
            return {
                "model": request.get("model"),
                "search_results": len(request.get("results", [])),
            }
        if operation_type == "model.assess_sources":
            return {
                "model": request.get("model"),
                "sources": len(request.get("results", [])),
                "rubric": request.get("rubric_version"),
            }
        if operation_type == "model.decide_worker_next_step":
            return {
                "model": request.get("model"),
                "step": request.get("step_number"),
                "evidence_items": len(request.get("evidence", [])),
                "queries_executed": len(request.get("executed_queries", [])),
            }
        if operation_type.startswith("model.audit"):
            return {
                "model": request.get("model"),
                "evidence_items": len(request.get("evidence", [])),
                "round": request.get("round"),
            }
        if operation_type == "model.synthesize_report":
            return {
                "model": request.get("model"),
                "evidence_items": len(request.get("evidence", [])),
                "dimensions": len(
                    request.get("plan", {})
                    .get("coverage_contract", {})
                    .get("required_dimensions", [])
                ),
            }
        return {}

    @staticmethod
    def _response_log_fields(
        operation_type: str, response: dict[str, Any]
    ) -> dict[str, Any]:
        if operation_type == "model.plan":
            return {"tasks": len(response.get("tasks", []))}
        if operation_type == "model.reconnaissance_queries":
            return {"query_candidates": len(response.get("queries", []))}
        if operation_type.startswith("model.replan"):
            return {
                "continue": response.get("should_continue"),
                "new_tasks": len(response.get("new_tasks", [])),
                "unresolved_gaps": len(response.get("unresolved_gaps", [])),
            }
        if operation_type == "model.formulate_queries":
            return {"query_candidates": len(response.get("queries", []))}
        if operation_type == "search.tavily":
            return {"results": len(response.get("results", []))}
        if operation_type == "extract.tavily":
            return {
                "pages": len(response.get("pages", [])),
                "failed_urls": len(response.get("failed_urls", [])),
                "credits_used": response.get("credits_used", 0),
            }
        if operation_type == "model.extract":
            return {"evidence_candidates": len(response.get("evidence", []))}
        if operation_type == "model.assess_sources":
            assessments = response.get("assessments", [])
            return {
                "assessments": len(assessments),
                "final_evidence": sum(
                    item.get("eligibility") == "final_evidence"
                    for item in assessments
                ),
                "discovery_only": sum(
                    item.get("eligibility") == "discovery_only"
                    for item in assessments
                ),
                "rejected": sum(
                    item.get("eligibility") == "reject"
                    for item in assessments
                ),
            }
        if operation_type == "model.decide_worker_next_step":
            return {
                "action": response.get("action"),
                "stop_reason": response.get("stop_reason"),
                "unresolved_gaps": len(response.get("unresolved_gaps", [])),
            }
        if operation_type.startswith("model.audit"):
            return {
                "sufficient": response.get("sufficient"),
                "claims": len(response.get("claims", [])),
                "gaps": len(response.get("gaps", [])),
            }
        if operation_type == "model.synthesize_report":
            return {
                "sections": len(response.get("sections", [])),
                "direct_answers": len(response.get("direct_answer", [])),
            }
        return {}

    def reconnaissance_queries(
        self, run_id: str, question: str
    ) -> QueryPlan:
        request = {
            "question": question,
            "max_queries": self.settings.max_reconnaissance_queries,
            "model": self.settings.planner_model,
        }
        payload = self._cached_call(
            run_id=run_id,
            task_id="T0",
            operation_type="model.reconnaissance_queries",
            request=request,
            invoke=lambda: self.model_provider.reconnaissance_queries(
                question, self.settings.max_reconnaissance_queries
            ),
        )
        plan = QueryPlan.model_validate(payload)
        return plan.model_copy(
            update={"queries": plan.queries[: self.settings.max_reconnaissance_queries]}
        )

    def plan(
        self,
        run_id: str,
        question: str,
        preliminary_evidence: list[EvidenceRecord] | None = None,
    ) -> ResearchPlan:
        preliminary_evidence = preliminary_evidence or []
        request = {
            "question": question,
            "max_tasks": self.settings.max_initial_tasks,
            "model": self.settings.planner_model,
            "preliminary_evidence": [
                item.model_dump(mode="json") for item in preliminary_evidence
            ],
        }
        payload = self._cached_call(
            run_id=run_id,
            task_id=None,
            operation_type="model.plan",
            request=request,
            invoke=lambda: (
                self.model_provider.plan_with_context(
                    question,
                    self.settings.max_initial_tasks,
                    preliminary_evidence,
                )
                if preliminary_evidence
                else self.model_provider.plan(
                    question, self.settings.max_initial_tasks
                )
            ),
        )
        plan = ResearchPlan.model_validate(payload)
        if len(plan.tasks) > self.settings.max_initial_tasks:
            raise RuntimeError(
                "Planner exceeded MAX_INITIAL_TASKS: "
                f"{len(plan.tasks)} > {self.settings.max_initial_tasks}"
            )
        return plan

    def replan(
        self,
        run_id: str,
        question: str,
        plan: ResearchPlan,
        audit: AuditResult,
        research_digest: ResearchStateDigest,
        evidence: list[EvidenceRecord],
        task_statuses: dict[str, str],
        source_risks: dict[str, object],
        round_number: int,
        next_task_index: int,
    ) -> ReplanDecision:
        request = {
            "question": question,
            "plan": plan.model_dump(mode="json"),
            "audit_handoff": {
                "sufficient": audit.sufficient,
                "actionable_gaps": [
                    gap.model_dump(mode="json") for gap in audit.actionable_gaps
                ],
            },
            "research_digest": research_digest.model_dump(mode="json"),
            "drilldown_evidence": [supervisor_evidence_payload(item) for item in evidence],
            "task_statuses": task_statuses,
            "source_risks": source_risks,
            "round": round_number,
            "next_task_index": next_task_index,
            "max_new_tasks": self.settings.max_new_tasks_per_replan,
            "model": self.settings.planner_model,
        }
        payload = self._cached_call(
            run_id=run_id,
            task_id=None,
            operation_type=f"model.replan.{round_number}",
            request=request,
            invoke=lambda: self.model_provider.replan(
                question,
                plan,
                audit,
                research_digest,
                evidence,
                task_statuses,
                source_risks,
                next_task_index,
                self.settings.max_new_tasks_per_replan,
            ),
        )
        return ReplanDecision.model_validate(payload)

    def formulate_queries(self, run_id: str, task: ResearchTask) -> QueryPlan:
        request = {
            "task": task.model_dump(mode="json"),
            "max_candidates": self.settings.max_query_candidates_per_worker,
            "model": self.settings.fast_model,
        }
        payload = self._cached_call(
            run_id=run_id,
            task_id=task.task_id,
            operation_type="model.formulate_queries",
            request=request,
            invoke=lambda: self.model_provider.formulate_queries(
                task, self.settings.max_query_candidates_per_worker
            ),
        )
        return QueryPlan.model_validate(payload)

    def search(
        self, run_id: str, task: ResearchTask, query: QuerySpec
    ) -> list[SearchResult]:
        request = {
            "query": query.query,
            "intent": query.intent.value,
            "provider": query.provider,
            "max_results": self.settings.max_results_per_search,
            "depth": "basic",
        }
        payload = self._cached_call(
            run_id=run_id,
            task_id=task.task_id,
            operation_type="search.tavily",
            request=request,
            invoke=lambda: {
                "results": [
                    result.model_dump(mode="json")
                    for result in self.search_provider.search(
                        query.query, self.settings.max_results_per_search
                    )
                ]
            },
        )
        return [
            SearchResult.model_validate(item).model_copy(
                update={"query": query.query, "provider": query.provider}
            )
            for item in payload["results"]
        ]

    def extract_pages(
        self,
        run_id: str,
        task: ResearchTask,
        results: list[SearchResult],
    ) -> list[SearchResult]:
        urls = list(dict.fromkeys(item.url for item in results))
        request = {
            "urls": urls,
            "extract_depth": "basic",
            "format": "markdown",
        }
        payload = self._cached_call(
            run_id=run_id,
            task_id=task.task_id,
            operation_type="extract.tavily",
            request=request,
            invoke=lambda: self.search_provider.extract_pages(urls),
        )
        batch = PageExtractionBatch.model_validate(payload)
        content_by_url = {
            canonicalize_url(item.url): item.raw_content for item in batch.pages
        }
        enriched: list[SearchResult] = []
        for result in results:
            raw_content = content_by_url.get(canonicalize_url(result.url))
            if raw_content:
                enriched.append(
                    result.model_copy(
                        update={
                            "snippet": result.snippet or result.content,
                            "content": raw_content,
                            "content_source": "full_page",
                        }
                    )
                )
            else:
                enriched.append(
                    result.model_copy(
                        update={
                            "snippet": result.snippet or result.content,
                            "content_source": "search_snippet",
                        }
                    )
                )
        return enriched

    def extract(
        self, run_id: str, task: ResearchTask, results: list[SearchResult]
    ) -> EvidenceBatch:
        request = {
            "task": task.model_dump(mode="json"),
            "results": [
                {
                    "title": item.title,
                    "url": item.url,
                    "rank": item.rank,
                    "query": item.query,
                    "content_source": item.content_source,
                    "content_sha256": hashlib.sha256(
                        item.content.encode("utf-8")
                    ).hexdigest(),
                    "content_chars": len(item.content),
                }
                for item in results
            ],
            "model": self.settings.fast_model,
            "chunk_selection": {
                "algorithm": (
                    "child_hybrid_bm25_embedding_weighted_rrf_parent_window_v4"
                    if self.settings.hybrid_retrieval_enabled
                    else "child_bm25_parent_window_v2"
                ),
                "retrieval_chunk_tokens": self.settings.retrieval_chunk_tokens,
                "retrieval_chunk_overlap_tokens": (
                    self.settings.retrieval_chunk_overlap_tokens
                ),
                "extraction_window_tokens": self.settings.extraction_window_tokens,
                "max_retrieval_chunks_per_source": (
                    self.settings.max_relevant_chunks_per_source
                ),
                "max_extraction_windows_per_source": (
                    self.settings.max_extraction_windows_per_source
                ),
                "embedding_model": (
                    self.settings.embedding_model
                    if self.settings.hybrid_retrieval_enabled
                    else None
                ),
                "embedding_dimensions": (
                    self.settings.embedding_dimensions
                    if self.settings.hybrid_retrieval_enabled
                    else None
                ),
                "embedding_weight": (
                    self.settings.hybrid_embedding_weight
                    if self.settings.hybrid_retrieval_enabled
                    else 0.0
                ),
                "rrf_k": (
                    self.settings.hybrid_rrf_k
                    if self.settings.hybrid_retrieval_enabled
                    else None
                ),
            },
        }
        payload = self._cached_call(
            run_id=run_id,
            task_id=task.task_id,
            operation_type="model.extract",
            request=request,
            invoke=lambda: self.model_provider.extract(task, results),
        )
        batch = EvidenceBatch.model_validate(payload)
        allowed = set(task.covered_dimensions)
        normalized = []
        for item in batch.evidence:
            dimensions = [
                dimension
                for dimension in item.dimension_ids
                if dimension in allowed
            ]
            if not dimensions and len(allowed) == 1:
                dimensions = list(allowed)
            normalized.append(item.model_copy(update={
                "dimension_ids": list(dict.fromkeys(dimensions)),
            }))
        return EvidenceBatch(evidence=normalized)

    def assess_sources(
        self,
        run_id: str,
        task: ResearchTask,
        results: list[SearchResult],
    ) -> SourceAssessmentBatch:
        def finalize_batch(batch: SourceAssessmentBatch) -> SourceAssessmentBatch:
            by_rank = {item.source_rank: item for item in batch.assessments}
            finalized: list[SourceAssessment] = []
            for rank, result in enumerate(results, start=1):
                assessment = by_rank.get(rank)
                if assessment is None:
                    assessment = SourceAssessment(
                        source_rank=rank,
                        source_type=SourceType.SECONDARY,
                        source_directness=SourceDirectness.UNCLEAR,
                        scores=SourceScoreDimensions(
                            provenance_directness=0,
                            authority_for_claim=0,
                            authorship_transparency=0,
                            methodology_transparency=0,
                            publication_controls=0,
                            citation_traceability=0,
                            recency_for_claim=0,
                            independence=0,
                            claim_relevance=0,
                        ),
                        hard_flags=[SourceHardFlag.ANONYMOUS_UNSOURCED],
                        rationale=[
                            "The evaluator omitted this source, so it is conservatively retained only as a lead."
                        ],
                    )
                finalized.append(
                    finalize_source_assessment(result, assessment)
                )
            return SourceAssessmentBatch(assessments=finalized)

        request = {
            "task": task.model_dump(mode="json"),
            "results": [
                {
                    "title": item.title,
                    "url": item.url,
                    "rank": index,
                    "content_source": item.content_source,
                    "content_sha256": hashlib.sha256(
                        item.content.encode("utf-8")
                    ).hexdigest(),
                    "content_chars": len(item.content),
                }
                for index, item in enumerate(results, start=1)
            ],
            "rubric_version": SOURCE_RUBRIC_VERSION,
            "model": self.settings.fast_model,
        }
        payload = self._cached_call(
            run_id=run_id,
            task_id=task.task_id,
            operation_type="model.assess_sources",
            request=request,
            invoke=lambda: finalize_batch(
                self.model_provider.assess_sources(task, results)
            ),
        )
        return finalize_batch(SourceAssessmentBatch.model_validate(payload))

    def decide_worker_next_step(
        self,
        run_id: str,
        task: ResearchTask,
        executed_queries: list[QuerySpec],
        evidence: list[EvidenceRecord],
        step_number: int,
        search_errors: list[str],
    ) -> WorkerDecision:
        request = {
            "task": task.model_dump(mode="json"),
            "executed_queries": [
                item.model_dump(mode="json") for item in executed_queries
            ],
            "evidence": [item.model_dump(mode="json") for item in evidence],
            "step_number": step_number,
            "max_steps": self.settings.max_adaptive_steps,
            "search_errors": search_errors,
            "model": self.settings.fast_model,
        }
        payload = self._cached_call(
            run_id=run_id,
            task_id=task.task_id,
            operation_type="model.decide_worker_next_step",
            request=request,
            invoke=lambda: self.model_provider.decide_worker_next_step(
                task,
                executed_queries,
                evidence,
                step_number,
                self.settings.max_adaptive_steps,
                search_errors,
            ),
        )
        return WorkerDecision.model_validate(payload)

    def audit(
        self,
        run_id: str,
        question: str,
        evidence: list[EvidenceRecord],
        round_number: int,
        research_context: dict[str, Any] | None = None,
        audit_scope: str | None = None,
    ) -> AuditResult:
        request = {
            "question": question,
            "evidence": [item.model_dump(mode="json") for item in evidence],
            "round": round_number,
            "model": (
                self.settings.fast_model if audit_scope
                else self.settings.audit_model
            ),
            "research_context": research_context or {},
        }
        operation_type = f"model.audit.{round_number}"
        if audit_scope:
            operation_type = f"{operation_type}.{audit_scope}"
        payload = self._cached_call(
            run_id=run_id,
            task_id=audit_scope,
            operation_type=operation_type,
            request=request,
            invoke=lambda: self.model_provider.audit(
                question, evidence, round_number, research_context
            ),
        )
        return AuditResult.model_validate(payload)

    def synthesize_report(
        self,
        run_id: str,
        question: str,
        plan: ResearchPlan,
        audit: AuditResult,
        evidence: list[EvidenceRecord],
    ) -> ReportDraft:
        request = {
            "question": question,
            "plan": plan.model_dump(mode="json"),
            "audit": audit.model_dump(mode="json"),
            "evidence": [
                {
                    "evidence_id": item.evidence_id,
                    "task_id": item.task_id,
                    "dimension_ids": item.dimension_ids,
                    "claim_candidate": item.claim_candidate,
                    "verbatim_excerpt": item.verbatim_excerpt[:1200],
                    "source_title": item.source_title,
                    "source_url": item.source_url,
                    "credibility_tier": item.credibility_tier.value,
                    "source_directness": item.source_directness.value,
                }
                for item in evidence
            ],
            "model": self.settings.audit_model,
        }
        payload = self._cached_call(
            run_id=run_id,
            task_id=None,
            operation_type="model.synthesize_report",
            request=request,
            invoke=lambda: self.model_provider.synthesize_report(
                question, plan, audit, evidence
            ),
        )
        return normalize_report_draft(
            ReportDraft.model_validate(payload),
            plan=plan,
            audit=audit,
            evidence=evidence,
        )
