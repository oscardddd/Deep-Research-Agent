from __future__ import annotations

import json

from ..config import Settings
from ..model_gateway import ModelGateway
from ..schemas import AuditResult, EvidenceRecord
from ..source_scoring import SOURCE_SCORING_GUIDE
from .base import BaseDeepSeekAgent


class AuditorAgent(BaseDeepSeekAgent):
    """Judges cross-worker evidence sufficiency and identifies open gaps."""

    def __init__(self, settings: Settings, gateway: ModelGateway | None = None):
        super().__init__(settings, gateway)
        self.model = settings.audit_model
        self.specialist_model = settings.fast_model

    def audit(
        self,
        question: str,
        evidence: list[EvidenceRecord],
        round_number: int,
        research_context: dict | None = None,
    ) -> AuditResult:
        system = """
You are an evidence auditor. Return JSON only. Synthesize the evidence into a
small set of atomic claims. Every factual claim must reference supplied
evidence IDs. Preserve important disagreement rather than forcing consensus.
Return at most 12 claims, merge overlapping findings, and order the most
decision-relevant claims first. Return at most 6 gaps. Keep `gaps` as plain
descriptions for report compatibility, and also return the same unresolved
issues as structured `actionable_gaps` for the Replanner.

Assign each claim a dimension copied exactly from the coverage contract's
required_dimensions when it fits one dimension. Use null only for a genuinely
cross-cutting claim.

Use status:
- supported: meaningful support and no material counterevidence
- mixed: both support and contradiction, or important qualifications
- unsupported: supplied evidence contradicts the claim
- unresolved: evidence is insufficient

Confidence must be high, medium, or low. Judge whether the original question,
coverage contract, task outcomes, evidence quality, and counterevidence are
collectively sufficient. A failed or partial required dimension is an open
gap even if available evidence supports some claims. Do not reuse one evidence
ID across multiple claims, and do not count multiple excerpts sharing a
provenance_key as independent support. Prefer direct A/B-tier origins over
secondary attribution; source quantity cannot compensate for poor provenance.
The formal SourceEvaluator supplies source_quality_score (0..15),
evidence_fitness_score (0..12), eligibility, hard flags, and observable
rationale. Treat these as prior assessments that you may question or downgrade,
but never upgrade merely because a source agrees with the emerging answer.
If insufficient, each actionable gap must state what evidence is missing, cite
only relevant supplied evidence IDs, and provide a targeted suggested_query
when another search could help. Use the current audit_check check_id, task_ids,
dimensions, and priority when present in research_context. Also provide the
single highest-value concrete follow_up_query; otherwise it may be null.

Required JSON shape:
{
  "sufficient": true,
  "claims": [
    {
      "claim_id": "C1",
      "claim": "atomic synthesized claim",
      "dimension": "exact coverage dimension or null",
      "status": "supported",
      "confidence": "medium",
      "supporting_evidence_ids": ["ev_..."],
      "contradicting_evidence_ids": [],
      "reasoning": "why the evidence warrants this status"
    }
  ],
  "gaps": ["plain description of the unresolved issue"],
  "actionable_gaps": [{
    "gap_id": "A1-G1",
    "check_id": "A1",
    "dimension": "exact coverage dimension or null",
    "task_ids": ["T1"],
    "gap_type": "source_quality",
    "priority": "high",
    "description": "same plain description as gaps",
    "related_evidence_ids": ["ev_..."],
    "missing_evidence": "specific evidence needed to close the gap",
    "suggested_query": "targeted search query or null"
  }],
  "follow_up_query": null
}
Do not cite an evidence ID that is not in the input.
""".strip()
        system += (
            "\n\nUse this same rubric when checking whether an assessment "
            "fits the cited claim. Do not silently recompute totals:\n"
            + SOURCE_SCORING_GUIDE
        )
        user = json.dumps(
            {
                "question": question,
                "audit_round": round_number,
                "evidence": [item.model_dump(mode="json") for item in evidence],
                "research_context": research_context or {},
            },
            ensure_ascii=False,
        )
        specialist = bool((research_context or {}).get("audit_check"))
        return self._json_completion(
            model=self.specialist_model if specialist else self.model,
            system_prompt=system,
            user_prompt=user,
            schema=AuditResult,
            max_tokens=3200,
            profile="audit_specialist" if specialist else "research_audit",
        )
