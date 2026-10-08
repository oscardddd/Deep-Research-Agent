from __future__ import annotations

from ..schemas import (
    AuditResult,
    ClaimAssessment,
    ClaimStatus,
    Confidence,
    CredibilityTier,
    CoverageContract,
    EvidenceBatch,
    EvidenceRecord,
    EvidenceStance,
    ExtractedEvidence,
    QueryIntent,
    QueryPlan,
    QuerySpec,
    ReportDraft,
    ReportSection,
    ReportStatement,
    ReportStatementType,
    ResearchPlan,
    ResearchTask,
    ReplanDecision,
    ResearchStateDigest,
    SearchMode,
    SearchResult,
    SourceDirectness,
    SourceAssessment,
    SourceAssessmentBatch,
    SourceScoreDimensions,
    SourceType,
    WorkerDecision,
    WorkerDecisionAction,
    WorkerStopReason,
)


class FakeLanguageModelProvider:
    """Deterministic all-role facade for tests and the local demo."""

    def reconnaissance_queries(
        self, question: str, max_queries: int
    ) -> QueryPlan:
        candidates = [
            QuerySpec(
                intent=QueryIntent.OVERVIEW,
                query=f"{question} research landscape",
                expected_evidence="Terminology, major dimensions, and candidate sources",
                priority=3,
            ),
            QuerySpec(
                intent=QueryIntent.PRIMARY_EVIDENCE,
                query=f"{question} official primary evidence",
                expected_evidence="Direct or official evidence",
                priority=2,
            ),
        ]
        return QueryPlan(queries=candidates[:max_queries])

    def plan(
        self,
        question: str,
        max_tasks: int,
        preliminary_evidence: list[EvidenceRecord] | None = None,
    ) -> ResearchPlan:
        templates = [
            (
                "What empirical evidence addresses the central claim?",
                "empirical evidence reviewer",
                "Assess direct empirical evidence for the central claim.",
                ["primary empirical evidence", "measured outcomes"],
                SearchMode.COVERAGE,
            ),
            (
                "What limitations or counterevidence have been reported?",
                "risk and failure reviewer",
                "Identify material limitations and counterevidence.",
                ["documented limitations", "counterevidence"],
                SearchMode.CHALLENGE,
            ),
            (
                "How well do independent sources replicate the finding?",
                "independent replication reviewer",
                "Assess whether independent sources replicate the finding.",
                ["independent replication evidence"],
                SearchMode.COVERAGE,
            ),
        ]
        dimensions = ["direct effects", "limitations", "independent replication"]
        tasks = [
            ResearchTask(
                task_id=f"T{index}",
                question=task_question,
                research_role=role,
                objective=objective,
                must_find=must_find,
                avoid=["unsupported claims"],
                query_budget=2,
                search_mode=search_mode,
                covered_dimensions=[dimensions[index - 1]],
            )
            for index, (task_question, role, objective, must_find, search_mode) in enumerate(
                templates[:max_tasks], start=1
            )
        ]
        return ResearchPlan(
            coverage_contract=CoverageContract(
                decision_type="balanced evidence assessment",
                required_dimensions=dimensions[:max_tasks],
                required_comparisons=["support versus counterevidence"],
            ),
            tasks=tasks,
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
        if audit.sufficient or not audit.gaps or max_new_tasks < 1:
            return ReplanDecision(
                should_continue=False,
                rationale="The bounded fake research state does not require another wave.",
                unresolved_gaps=list(audit.gaps),
            )
        dimension = plan.coverage_contract.required_dimensions[0]
        gap = audit.gaps[0]
        return ReplanDecision(
            should_continue=True,
            rationale="One targeted verification wave can address the highest-value gap.",
            new_tasks=[
                ResearchTask(
                    task_id=f"T{next_task_index}",
                    question=f"What independent evidence resolves this gap: {gap}",
                    research_role="targeted verification reviewer",
                    objective=gap,
                    must_find=[gap],
                    avoid=["repeating broad background research"],
                    query_budget=1,
                    search_mode=SearchMode.VERIFY,
                    covered_dimensions=[dimension],
                )
            ],
            unresolved_gaps=[gap],
        )

    def plan_with_context(
        self,
        question: str,
        max_tasks: int,
        preliminary_evidence: list[EvidenceRecord],
    ) -> ResearchPlan:
        return self.plan(question, max_tasks, preliminary_evidence)

    def formulate_queries(
        self, task: ResearchTask, max_candidates: int
    ) -> QueryPlan:
        objective = task.objective or task.question
        intents = (
            [QueryIntent.VERIFICATION]
            if task.query_hint
            else [
                QueryIntent.PRIMARY_EVIDENCE,
                QueryIntent.CHALLENGE,
                QueryIntent.OVERVIEW,
            ]
        )
        suffixes = {
            QueryIntent.PRIMARY_EVIDENCE: "empirical primary evidence",
            QueryIntent.CHALLENGE: "limitations counterevidence",
            QueryIntent.OVERVIEW: "independent overview",
            QueryIntent.VERIFICATION: "verification evidence",
        }
        queries = [
            QuerySpec(
                intent=intent,
                query=(task.query_hint or f"{objective} {suffixes[intent]}"),
                expected_evidence=f"Evidence for {intent.value}",
                priority=max(1, 3 - index),
            )
            for index, intent in enumerate(intents[:max_candidates])
        ]
        return QueryPlan(queries=queries)

    def extract(
        self, task: ResearchTask, results: list[SearchResult]
    ) -> EvidenceBatch:
        items: list[ExtractedEvidence] = []
        for index, result in enumerate(results, start=1):
            stance = (
                EvidenceStance.CONTRADICTS
                if "limited" in result.content or "limitations" in result.title.lower()
                else EvidenceStance.SUPPORTS
            )
            items.append(
                ExtractedEvidence(
                    source_rank=index,
                    claim_candidate=result.content,
                    verbatim_excerpt=result.content,
                    stance=stance,
                    relevance="high",
                    source_type=SourceType.PRIMARY_RESEARCH,
                    source_directness=SourceDirectness.DIRECT,
                    credibility_tier=CredibilityTier.B,
                    dimension_ids=task.covered_dimensions,
                )
            )
        return EvidenceBatch(evidence=items)

    def assess_sources(
        self, task: ResearchTask, results: list[SearchResult]
    ) -> SourceAssessmentBatch:
        return SourceAssessmentBatch(
            assessments=[
                SourceAssessment(
                    source_rank=index,
                    source_type=SourceType.PRIMARY_RESEARCH,
                    source_directness=SourceDirectness.DIRECT,
                    scores=SourceScoreDimensions(
                        provenance_directness=3,
                        authority_for_claim=3,
                        authorship_transparency=2,
                        methodology_transparency=3,
                        publication_controls=2,
                        citation_traceability=3,
                        recency_for_claim=2,
                        independence=2,
                        claim_relevance=3,
                    ),
                    rationale=[
                        "The deterministic fixture presents direct, task-relevant evidence.",
                        "Methods and provenance are visible in the fixture.",
                    ],
                )
                for index, _result in enumerate(results, start=1)
            ]
        )

    def decide_worker_next_step(
        self,
        task: ResearchTask,
        executed_queries: list[QuerySpec],
        evidence: list[EvidenceRecord],
        step_number: int,
        max_steps: int,
        search_errors: list[str],
    ) -> WorkerDecision:
        if task.search_mode == SearchMode.VERIFY or len(executed_queries) >= 2:
            return WorkerDecision(
                action=WorkerDecisionAction.STOP,
                decision_summary="The deterministic fixture has sufficient independent evidence.",
                unresolved_gaps=[],
                stop_reason=WorkerStopReason.SUFFICIENT,
            )
        objective = task.objective or task.question
        return WorkerDecision(
            action=WorkerDecisionAction.SEARCH,
            decision_summary="A second independent search can verify the first evidence set.",
            unresolved_gaps=["independent corroboration"],
            next_query=QuerySpec(
                intent=QueryIntent.VERIFICATION,
                query=f"{objective} independent verification follow-up",
                expected_evidence="Independent evidence that verifies or challenges the first result set",
                priority=3,
            ),
        )

    def audit(
        self,
        question: str,
        evidence: list[EvidenceRecord],
        round_number: int,
        research_context: dict | None = None,
    ) -> AuditResult:
        support = [e.evidence_id for e in evidence if e.stance == EvidenceStance.SUPPORTS]
        contradict = [
            e.evidence_id for e in evidence if e.stance == EvidenceStance.CONTRADICTS
        ]
        plan_payload = (research_context or {}).get("plan", {})
        dimensions = (
            plan_payload.get("coverage_contract", {}).get(
                "required_dimensions", []
            )
            or ["direct effects"]
        )
        claims: list[ClaimAssessment] = []
        for index, dimension in enumerate(dimensions):
            claim_support = support[index * 2 : index * 2 + 2]
            claim_contradict = contradict[index : index + 1]
            has_both = bool(claim_support and claim_contradict)
            claims.append(
                ClaimAssessment(
                    claim_id=f"C{index + 1}",
                    claim=(
                        f"For {dimension}, available evidence yields a finding "
                        "with material qualifications."
                    ),
                    dimension=dimension,
                    status=(
                        ClaimStatus.MIXED
                        if has_both
                        else ClaimStatus.UNRESOLVED
                    ),
                    confidence=(
                        Confidence.MEDIUM if has_both else Confidence.LOW
                    ),
                    supporting_evidence_ids=claim_support,
                    contradicting_evidence_ids=claim_contradict,
                    reasoning=(
                        "Independent support exists alongside a material limitation."
                        if has_both
                        else "The fixture lacks enough distinct evidence for this dimension."
                    ),
                )
            )
        all_dimensions_covered = all(
            claim.status in {ClaimStatus.SUPPORTED, ClaimStatus.MIXED}
            for claim in claims
        )
        return AuditResult(
            sufficient=all_dimensions_covered,
            claims=claims,
            gaps=(
                []
                if all_dimensions_covered
                else ["One or more dimensions lack distinct evidence."]
            ),
            follow_up_query=(
                None
                if all_dimensions_covered
                else f"{question} missing dimension evidence"
            ),
        )

    def synthesize_report(
        self,
        question: str,
        plan: ResearchPlan,
        audit: AuditResult,
        evidence: list[EvidenceRecord],
    ) -> ReportDraft:
        task_dimensions = {
            task.task_id: task.covered_dimensions for task in plan.tasks
        }
        sections: list[ReportSection] = []
        for dimension in plan.coverage_contract.required_dimensions:
            candidates = [
                item
                for item in evidence
                if dimension in (
                    item.dimension_ids
                    or task_dimensions.get(item.task_id, [])
                )
            ]
            findings = [
                ReportStatement(
                    text=item.claim_candidate,
                    evidence_ids=[item.evidence_id],
                    statement_type=ReportStatementType.OBSERVED_FACT,
                    confidence=(
                        Confidence.MEDIUM
                        if item.credibility_tier in {
                            CredibilityTier.A,
                            CredibilityTier.B,
                        }
                        else Confidence.LOW
                    ),
                )
                for item in candidates[:2]
            ]
            sections.append(ReportSection(
                dimension=dimension,
                heading=dimension,
                findings=findings,
            ))
        all_findings = [
            finding for section in sections for finding in section.findings
        ]
        direct_answer = all_findings[:2]
        return ReportDraft(
            title=f"Research report: {question}"[:300],
            direct_answer=direct_answer,
            executive_summary=direct_answer,
            sections=sections,
            methodology=[
                "The report synthesizes direct, traceable evidence by required dimension."
            ],
            limitations=list(audit.gaps),
        )
