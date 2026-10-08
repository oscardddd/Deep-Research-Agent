from __future__ import annotations

import json

from ..config import Settings
from ..evidence_policy import supervisor_evidence_payload
from ..model_gateway import ModelGateway
from ..schemas import (
    AuditResult, EvidenceRecord, QueryPlan, ReplanDecision, ResearchPlan,
    ResearchStateDigest,
)
from .base import BaseDeepSeekAgent


class PlannerAgent(BaseDeepSeekAgent):
    """Decomposes the user's question into complementary research tasks."""

    def __init__(self, settings: Settings, gateway: ModelGateway | None = None):
        super().__init__(settings, gateway)
        self.model = settings.planner_model

    def reconnaissance_queries(
        self, question: str, max_queries: int
    ) -> QueryPlan:
        system = """
You are a research reconnaissance planner. Return JSON only. Create a small
set of broad, complementary searches that map the information landscape before
detailed task planning. Include, when applicable: an overview query, a query
targeting official or primary evidence, and a challenge/controversy query.
Avoid prematurely assuming the answer or an exhaustive taxonomy.

Use the QueryPlan JSON schema. The only provider is tavily. Return no more
queries than requested. Do not include markdown or commentary.
""".strip()
        user = json.dumps(
            {"question": question, "maximum_queries": max_queries},
            ensure_ascii=False,
        )
        return self._json_completion(
            model=self.model,
            system_prompt=system,
            user_prompt=user,
            schema=QueryPlan,
            max_tokens=1000,
            profile="research_planning",
        )

    def plan(
        self,
        question: str,
        max_tasks: int,
        preliminary_evidence: list[EvidenceRecord] | None = None,
    ) -> ResearchPlan:
        system = """
You are a research planner. Return JSON only. First define a global coverage
contract for the original question, then choose the smallest useful number of
complementary research tasks, never exceeding the supplied Maximum tasks.
The number of tasks must follow the independent evidence dimensions required
by the question; do not force a fixed number. Create a separate task only when
it has a distinct evidence objective, evidence type, or comparison role. Query
variants belong inside one worker. Tasks may overlap when independent
verification or comparison is valuable. Assign each task an epistemic
research role, a concrete objective, evidence requirements, exclusions, and a
small query budget. Do not generate search queries: each worker will formulate
its own queries from the assigned role and objective.

Treat preliminary reconnaissance as fallible orientation, not established
fact. Use it to discover terminology, candidate dimensions, disagreements,
and likely primary sources. Do not let one prominent result define the plan,
and do not repeat its conclusions as assumptions.

Roles must describe research responsibilities, not fictional personalities.
Prefer complementary coverage roles. Use a challenge role only when the user
asks about risks, limitations, controversy, or conflicting evidence.

Required JSON shape:
{
  "coverage_contract": {
    "decision_type": "benefit-risk assessment",
    "required_dimensions": ["dimension that must be answered"],
    "required_comparisons": ["comparison needed for a useful conclusion"]
  },
  "tasks": [
    {
      "task_id": "T1",
      "question": "specific sub-question",
      "research_role": "empirical evidence reviewer",
      "objective": "specific research objective",
      "must_find": ["required evidence type"],
      "avoid": ["out-of-scope or weak evidence"],
      "query_budget": 2,
      "search_mode": "coverage",
      "importance": "high",
      "execution_profile": "standard_research",
      "success_criteria": {
        "min_relevant_evidence": 1,
        "min_independent_sources": 2,
        "primary_source_preferred": true
      },
      "covered_dimensions": ["exact entry from required_dimensions"]
    }
  ]
}
Use sequential IDs T1 through TN. Do not include markdown or commentary.
Choose execution_profile from lightweight_extraction, standard_research, or
deep_reasoning. Use deep_reasoning only when the task requires substantial
cross-source reasoning; use lightweight_extraction for bounded extraction.
""".strip()
        user = json.dumps(
            {
                "research_question": question,
                "maximum_tasks": max_tasks,
                "preliminary_reconnaissance": [
                    item.model_dump(mode="json")
                    for item in (preliminary_evidence or [])
                ],
            },
            ensure_ascii=False,
        )
        return self._json_completion(
            model=self.model,
            system_prompt=system,
            user_prompt=user,
            schema=ResearchPlan,
            max_tokens=5000,
            profile="research_planning",
        )

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
        system = """
You are the lead research supervisor revising a plan after a complete parallel
research wave. Return JSON only. Use intermediate findings to decide whether
additional research has positive expected value. Do not repeat completed work.

Rules:
- Treat research_state_digest as the complete map of the research wave. Raw
  excerpts in drilldown_evidence are a deliberately small supporting view;
  their absence does not mean the underlying evidence does not exist.
- Treat audit_handoff.actionable_gaps as the authoritative list of candidate
  follow-up work. Each gap already states its dimension, related tasks and
  evidence, missing evidence, priority, and suggested query. Do not reconstruct
  a different gap merely because the raw drill-down view is small.
- Ground references to individual sources in evidence IDs from the digest or
  drill-down view. Do not infer that a frequently represented task is stronger.
- Continue only for report-changing coverage gaps, weak provenance, unresolved
  contradictions, missing counterevidence, or a newly discovered dimension.
- New tasks must be narrow, mutually complementary, and grounded in the audit
  gaps or source risks. Prefer verification, challenge, or original-source
  recovery over another broad overview.
- Never treat discovery-only sources, repeated excerpts, or many claims from
  one URL as independent evidence.
- Use sequential task IDs beginning with the supplied next_task_index.
- Return no more new tasks than maximum_new_tasks.
- Preserve entries from the current coverage contract in covered_dimensions.
  A newly discovered dimension may be investigated by a task, but must not be
  placed in covered_dimensions until the contract itself is explicitly revised
  in a future design.
- Stop when remaining gaps are unlikely to change the answer or the evidence
  cannot realistically be obtained within another bounded wave.

Required JSON shape:
{
  "should_continue": true,
  "rationale": "why another wave is warranted",
  "new_tasks": [{
    "task_id": "T5",
    "question": "specific gap-closing question",
    "research_role": "targeted verification reviewer",
    "objective": "specific objective",
    "must_find": ["required evidence"],
    "avoid": ["already-covered work"],
    "query_budget": 1,
    "query_hint": null,
    "search_mode": "verify",
    "importance": "high",
    "execution_profile": "deep_reasoning",
    "success_criteria": {
      "min_relevant_evidence": 1,
      "min_independent_sources": 2,
      "primary_source_preferred": true
    },
    "covered_dimensions": ["existing required dimension"]
  }],
  "retired_task_ids": [],
  "unresolved_gaps": ["gap targeted by this revision"]
}
When stopping, set should_continue=false and new_tasks=[].
""".strip()
        payload = {
            "question": question,
            "current_plan": plan.model_dump(mode="json"),
            "task_statuses": task_statuses,
            "research_state_digest": research_digest.model_dump(mode="json"),
            "audit_handoff": {
                "sufficient": audit.sufficient,
                "coverage_sufficient": audit.coverage_sufficient,
                "source_sufficient": audit.source_sufficient,
                "task_execution_sufficient": audit.task_execution_sufficient,
                "challenge_sufficient": audit.challenge_sufficient,
                "actionable_gaps": [
                    gap.model_dump(mode="json") for gap in audit.actionable_gaps
                ],
            },
            "drilldown_evidence": [supervisor_evidence_payload(item) for item in evidence],
            "next_task_index": next_task_index,
            "maximum_new_tasks": max_new_tasks,
        }
        return self._json_completion(
            model=self.model,
            system_prompt=system,
            user_prompt=json.dumps(payload, ensure_ascii=False),
            schema=ReplanDecision,
            max_tokens=2200,
            profile="deep_reasoning",
        )
