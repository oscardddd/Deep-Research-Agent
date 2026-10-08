from __future__ import annotations

from typing import TypeVar

from pydantic import BaseModel

from ..config import Settings
from ..model_gateway import ModelGateway
from ..schemas import (
    AuditResult,
    EvidenceBatch,
    EvidenceRecord,
    QueryPlan,
    QuerySpec,
    ReportDraft,
    ResearchPlan,
    ResearchStateDigest,
    ResearchTask,
    ReplanDecision,
    SearchResult,
    SourceAssessmentBatch,
    WorkerDecision,
)
from .auditor import AuditorAgent
from .planner import PlannerAgent
from .source_evaluator import SourceEvaluatorAgent
from .worker import ResearchWorkerAgent
from .writer import ReportWriterAgent


SchemaT = TypeVar("SchemaT", bound=BaseModel)


class DeepSeekModelProvider:
    """Compatibility facade that composes the role-specific agents.

    Runtime and CLI callers keep one stable provider interface, while role
    prompts, model selection, and policies live in independently testable
    classes. Optional injection also makes role-level test doubles possible.
    """

    def __init__(
        self,
        settings: Settings,
        *,
        planner: PlannerAgent | None = None,
        worker: ResearchWorkerAgent | None = None,
        source_evaluator: SourceEvaluatorAgent | None = None,
        auditor: AuditorAgent | None = None,
        writer: ReportWriterAgent | None = None,
    ):
        gateway = ModelGateway(settings)
        self.planner = planner or PlannerAgent(settings, gateway)
        self.worker = worker or ResearchWorkerAgent(settings, gateway)
        self.source_evaluator = source_evaluator or SourceEvaluatorAgent(
            settings, gateway
        )
        self.auditor = auditor or AuditorAgent(settings, gateway)
        self.writer = writer or ReportWriterAgent(settings, gateway)
        self.gateway = gateway

        # Preserve the old public attributes for existing callers.
        self.api_key = self.worker.api_key
        self.planner_model = self.planner.model
        self.fast_model = self.worker.model
        self.audit_model = self.auditor.model
        self.timeout = self.worker.timeout
        self.page_chunk_chars = self.worker.page_chunk_chars

    def plan(self, question: str, max_tasks: int) -> ResearchPlan:
        return self.planner.plan(question, max_tasks)

    def reconnaissance_queries(
        self, question: str, max_queries: int
    ) -> QueryPlan:
        return self.planner.reconnaissance_queries(question, max_queries)

    def plan_with_context(
        self,
        question: str,
        max_tasks: int,
        preliminary_evidence: list[EvidenceRecord],
    ) -> ResearchPlan:
        return self.planner.plan(question, max_tasks, preliminary_evidence)

    def replan(
        self,
        question: str,
        plan: ResearchPlan,
        audit: AuditResult,
        research_digest: ResearchStateDigest,
        evidence: list[EvidenceRecord],
        task_statuses: dict[str, str],
        source_risks: dict[str, object],
        next_task_index: int,
        max_new_tasks: int,
    ) -> ReplanDecision:
        return self.planner.replan(
            question,
            plan,
            audit,
            research_digest,
            evidence,
            task_statuses,
            source_risks,
            next_task_index,
            max_new_tasks,
        )

    def formulate_queries(
        self, task: ResearchTask, max_candidates: int
    ) -> QueryPlan:
        return self.worker.formulate_queries(task, max_candidates)

    def assess_sources(
        self, task: ResearchTask, results: list[SearchResult]
    ) -> SourceAssessmentBatch:
        return self.source_evaluator.assess_sources(task, results)

    def extract(
        self, task: ResearchTask, results: list[SearchResult]
    ) -> EvidenceBatch:
        return self.worker.extract(task, results)

    def decide_worker_next_step(
        self,
        task: ResearchTask,
        executed_queries: list[QuerySpec],
        evidence: list[EvidenceRecord],
        step_number: int,
        max_steps: int,
        search_errors: list[str],
    ) -> WorkerDecision:
        return self.worker.decide_worker_next_step(
            task,
            executed_queries,
            evidence,
            step_number,
            max_steps,
            search_errors,
        )

    def audit(
        self,
        question: str,
        evidence: list[EvidenceRecord],
        round_number: int,
        research_context: dict | None = None,
    ) -> AuditResult:
        return self.auditor.audit(
            question,
            evidence,
            round_number,
            research_context,
        )

    def synthesize_report(
        self,
        question: str,
        plan: ResearchPlan,
        audit: AuditResult,
        evidence: list[EvidenceRecord],
    ) -> ReportDraft:
        return self.writer.synthesize(question, plan, audit, evidence)

    def _json_completion(
        self,
        *,
        model: str,
        system_prompt: str,
        user_prompt: str,
        schema: type[SchemaT],
        max_tokens: int,
        profile: str = "standard_research",
    ) -> SchemaT:
        """Legacy escape hatch; new code should call a concrete role agent."""
        return self.worker._json_completion(
            model=model,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            schema=schema,
            max_tokens=max_tokens,
            profile=profile,
        )
