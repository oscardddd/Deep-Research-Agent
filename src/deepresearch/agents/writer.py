from __future__ import annotations

import json

from ..config import Settings
from ..model_gateway import ModelGateway
from ..schemas import AuditResult, EvidenceRecord, ReportDraft, ResearchPlan
from .base import BaseDeepSeekAgent


class ReportWriterAgent(BaseDeepSeekAgent):
    """Turns audited evidence into a reader-facing, citation-grounded answer."""

    def __init__(self, settings: Settings, gateway: ModelGateway | None = None):
        super().__init__(settings, gateway)
        self.model = settings.audit_model

    def synthesize(
        self,
        question: str,
        plan: ResearchPlan,
        audit: AuditResult,
        evidence: list[EvidenceRecord],
    ) -> ReportDraft:
        system = """
You are the final research report writer. Return JSON only. Produce a direct,
professional answer to the user's question from the supplied evidence. The
auditor is an internal quality-control input, not the report format.

Rules:
- Lead with the answer, not process metadata.
- Cover every required dimension using the exact dimension string supplied in
  the coverage contract. Do not invent new required dimensions.
- Every factual statement, source projection, derived estimate, or
  interpretation must cite one or more supplied evidence_ids.
- Keep every ReportStatement atomic: one independently verifiable factual
  proposition and no more than one sentence. If two clauses require different
  evidence, emit two ReportStatements instead of combining them.
- Every cited evidence_id must independently support the complete statement.
  Do not attach several partial sources to one compound statement and assume
  their separate facts will be evaluated jointly.
- Evidence IDs may be reused across statements and sections. Source
  independence affects confidence; it does not make a direct official fact
  unusable.
- A fact supported by one authoritative direct source may be reported with
  medium confidence and explicit attribution. Do not erase useful information
  merely because independent corroboration is absent.
- Distinguish observed facts, source projections, derived estimates, and
  interpretations. Never perform or invent an unstated quantitative
  calculation. A derived estimate is allowed only when the evidence itself
  supplies the estimate or the provided evidence text states the calculation.
- Preserve material disagreement and qualifications.
- Put unresolved audit gaps in limitations. Do not mention sufficiency gates,
  task IDs, evidence quality scores, model calls, audit rounds, or internal
  workflow details.
- Keep statements concise and decision-relevant. Do not include markdown.

Required JSON shape:
{
  "title": "reader-facing report title",
  "direct_answer": [{
    "text": "direct answer statement",
    "evidence_ids": ["ev_..."],
    "statement_type": "source_projection",
    "confidence": "medium"
  }],
  "executive_summary": [{
    "text": "high-value summary statement",
    "evidence_ids": ["ev_..."],
    "statement_type": "observed_fact",
    "confidence": "high"
  }],
  "sections": [{
    "dimension": "exact required dimension",
    "heading": "clear reader-facing heading",
    "findings": [{
      "text": "grounded finding",
      "evidence_ids": ["ev_..."],
      "statement_type": "observed_fact",
      "confidence": "medium"
    }]
  }],
  "comparisons": [],
  "methodology": ["short explanation of how the answer was synthesized"],
  "limitations": ["material evidence limitation"]
}

Allowed statement_type values: observed_fact, source_projection,
derived_estimate, interpretation.
Allowed confidence values: high, medium, low.
Do not cite an evidence ID that is not in the input.
""".strip()
        payload = {
            "question": question,
            "coverage_contract": plan.coverage_contract.model_dump(mode="json"),
            "audit": {
                "claims": [
                    claim.model_dump(mode="json") for claim in audit.claims
                ],
                "gaps": audit.gaps,
            },
            "evidence": [
                {
                    "evidence_id": item.evidence_id,
                    "task_id": item.task_id,
                    "dimension_ids": item.dimension_ids,
                    "claim": item.claim_candidate,
                    "excerpt": item.verbatim_excerpt[:1200],
                    "stance": item.stance.value,
                    "source_title": item.source_title,
                    "source_url": item.source_url,
                    "source_type": item.source_type.value,
                    "source_directness": item.source_directness.value,
                    "credibility_tier": item.credibility_tier.value,
                }
                for item in evidence
            ],
        }
        return self._json_completion(
            model=self.model,
            system_prompt=system,
            user_prompt=json.dumps(payload, ensure_ascii=False),
            schema=ReportDraft,
            max_tokens=6000,
            profile="deep_reasoning",
        )
