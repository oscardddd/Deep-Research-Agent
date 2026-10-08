from __future__ import annotations

from enum import StrEnum
from typing import Annotated, Literal

from pydantic import BaseModel, Field, field_validator, model_validator


class SearchMode(StrEnum):
    COVERAGE = "coverage"
    CHALLENGE = "challenge"
    VERIFY = "verify"


class TaskStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    PARTIAL = "partial"
    FAILED_RETRYABLE = "failed_retryable"
    FAILED_TERMINAL = "failed_terminal"


class EvidenceStance(StrEnum):
    SUPPORTS = "supports"
    CONTRADICTS = "contradicts"
    NEUTRAL = "neutral"


class SourceType(StrEnum):
    OFFICIAL = "official"
    PRIMARY_RESEARCH = "primary_research"
    DIRECT_INTERVIEW = "direct_interview"
    REPUTABLE_SECONDARY = "reputable_secondary"
    SECONDARY = "secondary"
    REFERENCE = "reference"
    AGGREGATOR = "aggregator"
    SOCIAL = "social"
    AI_SYNTHESIS = "ai_synthesis"


class SourceDirectness(StrEnum):
    DIRECT = "direct"
    ATTRIBUTED = "attributed"
    UNCLEAR = "unclear"


class CredibilityTier(StrEnum):
    A = "A"
    B = "B"
    C = "C"
    D = "D"


class SourceEligibility(StrEnum):
    FINAL_EVIDENCE = "final_evidence"
    DISCOVERY_ONLY = "discovery_only"
    REJECT = "reject"


class SourceHardFlag(StrEnum):
    AI_SYNTHESIS = "ai_synthesis"
    SOCIAL_CONTENT = "social_content"
    CONTENT_FARM = "content_farm"
    ANONYMOUS_UNSOURCED = "anonymous_unsourced"
    ATTRIBUTED_ONLY = "attributed_only"
    INACCESSIBLE = "inaccessible"
    IRRELEVANT = "irrelevant"
    POSSIBLE_RETRACTION = "possible_retraction"


class ClaimStatus(StrEnum):
    SUPPORTED = "supported"
    MIXED = "mixed"
    UNSUPPORTED = "unsupported"
    UNRESOLVED = "unresolved"


class Confidence(StrEnum):
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class SuccessCriteria(BaseModel):
    min_relevant_evidence: int = Field(default=1, ge=1, le=5)
    min_independent_sources: int = Field(default=2, ge=1, le=3)
    primary_source_preferred: bool = True


class CoverageContract(BaseModel):
    """Global dimensions that must survive task decomposition."""

    decision_type: str = Field(default="evidence assessment", min_length=3, max_length=200)
    required_dimensions: list[str] = Field(min_length=1, max_length=8)
    required_comparisons: list[str] = Field(default_factory=list, max_length=8)


class QueryIntent(StrEnum):
    OVERVIEW = "overview"
    PRIMARY_EVIDENCE = "primary_evidence"
    CHALLENGE = "challenge"
    IMPLEMENTATION = "implementation"
    VERIFICATION = "verification"


class QuerySpec(BaseModel):
    intent: QueryIntent
    query: str = Field(min_length=3, max_length=400)
    expected_evidence: str = Field(min_length=3, max_length=500)
    priority: int = Field(default=1, ge=1, le=3)
    provider: Literal["tavily"] = "tavily"


class QueryPlan(BaseModel):
    queries: list[QuerySpec] = Field(min_length=1, max_length=4)


class WorkerDecisionAction(StrEnum):
    SEARCH = "search"
    FOLLOW_SOURCE = "follow_source"
    STOP = "stop"


class WorkerStopReason(StrEnum):
    SUFFICIENT = "sufficient"
    SATURATED = "saturated"
    BLOCKED = "blocked"


class WorkerDecision(BaseModel):
    action: WorkerDecisionAction
    decision_summary: str = Field(min_length=3, max_length=800)
    unresolved_gaps: list[str] = Field(default_factory=list)
    next_query: QuerySpec | None = None
    target_source_url: str | None = Field(default=None, max_length=1000)
    stop_reason: WorkerStopReason | None = None

    @model_validator(mode="after")
    def action_payload_is_consistent(self) -> "WorkerDecision":
        if self.action == WorkerDecisionAction.SEARCH:
            if self.next_query is None:
                raise ValueError("search action requires next_query")
            if self.target_source_url is not None:
                raise ValueError("search action cannot include target_source_url")
            if self.stop_reason is not None:
                raise ValueError("search action cannot include stop_reason")
        elif self.action == WorkerDecisionAction.FOLLOW_SOURCE:
            if self.target_source_url is None:
                raise ValueError("follow_source action requires target_source_url")
            if self.next_query is not None:
                raise ValueError("follow_source action cannot include next_query")
            if self.stop_reason is not None:
                raise ValueError("follow_source action cannot include stop_reason")
        else:
            if self.next_query is not None:
                raise ValueError("stop action cannot include next_query")
            if self.target_source_url is not None:
                raise ValueError("stop action cannot include target_source_url")
            if self.stop_reason is None:
                raise ValueError("stop action requires stop_reason")
        return self


class ResearchTask(BaseModel):
    task_id: str = Field(pattern=r"^T[0-9]+$")
    question: str = Field(min_length=8, max_length=500)
    research_role: str = Field(
        default="evidence_researcher", min_length=3, max_length=120
    )
    objective: str | None = Field(default=None, max_length=800)
    must_find: list[str] = Field(default_factory=list, max_length=6)
    avoid: list[str] = Field(default_factory=list, max_length=6)
    query_budget: int = Field(default=2, ge=1, le=3)
    query_hint: str | None = Field(default=None, min_length=3, max_length=400)
    search_mode: SearchMode = SearchMode.COVERAGE
    importance: Literal["high", "medium", "low"] = "high"
    execution_profile: Literal[
        "lightweight_extraction",
        "standard_research",
        "deep_reasoning",
    ] = "standard_research"
    success_criteria: SuccessCriteria = Field(default_factory=SuccessCriteria)
    covered_dimensions: list[str] = Field(default_factory=list, max_length=8)


class ResearchPlan(BaseModel):
    coverage_contract: CoverageContract
    tasks: list[ResearchTask] = Field(min_length=1)

    @field_validator("tasks")
    @classmethod
    def unique_task_ids(cls, tasks: list[ResearchTask]) -> list[ResearchTask]:
        ids = [task.task_id for task in tasks]
        if len(ids) != len(set(ids)):
            raise ValueError("task_id values must be unique")
        return tasks

    @model_validator(mode="after")
    def required_dimensions_are_assigned(self) -> "ResearchPlan":
        required = set(self.coverage_contract.required_dimensions)
        assigned = {
            dimension
            for task in self.tasks
            for dimension in task.covered_dimensions
        }
        unknown = assigned - required
        missing = required - assigned
        if unknown:
            raise ValueError(
                "covered_dimensions must come from required_dimensions: "
                + ", ".join(sorted(unknown))
            )
        if missing:
            raise ValueError(
                "every required dimension must be assigned to a task: "
                + ", ".join(sorted(missing))
            )
        return self


class ReplanDecision(BaseModel):
    """A bounded global plan revision after one complete research wave."""

    should_continue: bool
    rationale: str = Field(min_length=3, max_length=1200)
    new_tasks: list[ResearchTask] = Field(default_factory=list, max_length=4)
    retired_task_ids: list[str] = Field(default_factory=list, max_length=12)
    unresolved_gaps: list[str] = Field(default_factory=list, max_length=8)

    @model_validator(mode="after")
    def continuation_has_work(self) -> "ReplanDecision":
        if self.should_continue and not self.new_tasks:
            raise ValueError("continued research requires at least one new task")
        if not self.should_continue and self.new_tasks:
            raise ValueError("completed research cannot include new tasks")
        return self


class DigestClaim(BaseModel):
    """Compact claim reference; full evidence remains in the evidence store."""

    claim: str = Field(min_length=3, max_length=500)
    stance: EvidenceStance
    evidence_ids: list[str] = Field(default_factory=list, max_length=8)
    provenance_count: int = Field(default=0, ge=0)
    best_credibility_tier: CredibilityTier = CredibilityTier.C


class WorkerDigest(BaseModel):
    task_id: str
    question: str = Field(min_length=3, max_length=500)
    status: str
    covered_dimensions: list[str] = Field(default_factory=list, max_length=8)
    evidence_count: int = Field(default=0, ge=0)
    source_count: int = Field(default=0, ge=0)
    discovery_only_count: int = Field(default=0, ge=0)
    claims: list[DigestClaim] = Field(default_factory=list, max_length=8)
    unresolved_gaps: list[str] = Field(default_factory=list, max_length=6)


class ResearchStateDigest(BaseModel):
    """Bounded map passed to the supervisor instead of the full evidence corpus."""

    version: Literal[1] = 1
    round: int = Field(ge=0)
    coverage_contract: CoverageContract
    workers: list[WorkerDigest] = Field(default_factory=list)
    audit_summary: dict[str, object] = Field(default_factory=dict)
    source_risks: dict[str, object] = Field(default_factory=dict)
    evidence_count: int = Field(default=0, ge=0)
    drilldown_evidence_ids: list[str] = Field(default_factory=list)


class SearchResult(BaseModel):
    title: str
    url: str
    content: str
    snippet: str | None = None
    content_source: Literal["search_snippet", "full_page"] = "search_snippet"
    score: float = 0.0
    rank: int = 0
    query: str | None = None
    provider: str = "tavily"


class ExtractedPage(BaseModel):
    url: str
    raw_content: str


class PageExtractionBatch(BaseModel):
    pages: list[ExtractedPage] = Field(default_factory=list)
    failed_urls: list[str] = Field(default_factory=list)
    credits_used: float = Field(default=0.0, ge=0)


class SourceScoreDimensions(BaseModel):
    """Auditable source-quality and claim-fitness sub-scores (0..3)."""

    provenance_directness: int = Field(ge=0, le=3)
    authority_for_claim: int = Field(ge=0, le=3)
    authorship_transparency: int = Field(ge=0, le=3)
    methodology_transparency: int = Field(ge=0, le=3)
    publication_controls: int = Field(ge=0, le=3)
    citation_traceability: int = Field(ge=0, le=3)
    recency_for_claim: int = Field(ge=0, le=3)
    independence: int = Field(ge=0, le=3)
    claim_relevance: int = Field(ge=0, le=3)


class SourceAssessment(BaseModel):
    """One claim-contextual evaluation of one retrieved source."""

    source_rank: int = Field(ge=1)
    source_type: SourceType
    source_directness: SourceDirectness
    scores: SourceScoreDimensions
    hard_flags: list[SourceHardFlag] = Field(default_factory=list, max_length=8)
    attributed_source_name: str | None = Field(default=None, max_length=300)
    attributed_source_url: str | None = Field(default=None, max_length=1000)
    follow_up_query: str | None = Field(default=None, max_length=400)
    rationale: list[str] = Field(min_length=1, max_length=8)
    rubric_version: str = "source-rubric-v1"
    source_quality_score: int = Field(default=0, ge=0, le=15)
    evidence_fitness_score: int = Field(default=0, ge=0, le=12)
    credibility_tier: CredibilityTier = CredibilityTier.C
    eligibility: SourceEligibility = SourceEligibility.DISCOVERY_ONLY

    @model_validator(mode="after")
    def calculate_outcome(self) -> "SourceAssessment":
        self.source_quality_score = sum(
            [
                self.scores.authorship_transparency,
                self.scores.methodology_transparency,
                self.scores.publication_controls,
                self.scores.citation_traceability,
                self.scores.independence,
            ]
        )
        self.evidence_fitness_score = sum(
            [
                self.scores.provenance_directness,
                self.scores.authority_for_claim,
                self.scores.recency_for_claim,
                self.scores.claim_relevance,
            ]
        )
        reject_flags = {
            SourceHardFlag.INACCESSIBLE,
            SourceHardFlag.IRRELEVANT,
            SourceHardFlag.POSSIBLE_RETRACTION,
        }
        disqualifying_flags = {
            SourceHardFlag.AI_SYNTHESIS,
            SourceHardFlag.SOCIAL_CONTENT,
            SourceHardFlag.CONTENT_FARM,
            SourceHardFlag.ANONYMOUS_UNSOURCED,
        }
        flags = set(self.hard_flags)
        if flags & reject_flags:
            self.credibility_tier = CredibilityTier.D
            self.eligibility = SourceEligibility.REJECT
        elif flags & disqualifying_flags:
            self.credibility_tier = CredibilityTier.D
            self.eligibility = SourceEligibility.DISCOVERY_ONLY
        elif (
            SourceHardFlag.ATTRIBUTED_ONLY in flags
            or self.source_directness == SourceDirectness.ATTRIBUTED
        ):
            self.credibility_tier = (
                CredibilityTier.C
                if self.source_quality_score >= 6
                else CredibilityTier.D
            )
            self.eligibility = SourceEligibility.DISCOVERY_ONLY
        elif (
            self.source_quality_score >= 13
            and self.evidence_fitness_score >= 10
        ):
            self.credibility_tier = CredibilityTier.A
            self.eligibility = SourceEligibility.FINAL_EVIDENCE
        elif (
            self.source_quality_score >= 9
            and self.evidence_fitness_score >= 8
        ):
            self.credibility_tier = CredibilityTier.B
            self.eligibility = SourceEligibility.FINAL_EVIDENCE
        elif (
            self.source_quality_score >= 6
            and self.evidence_fitness_score >= 5
        ):
            self.credibility_tier = CredibilityTier.C
            self.eligibility = SourceEligibility.FINAL_EVIDENCE
        else:
            self.credibility_tier = CredibilityTier.D
            self.eligibility = SourceEligibility.DISCOVERY_ONLY
        return self


class SourceAssessmentBatch(BaseModel):
    assessments: list[SourceAssessment] = Field(default_factory=list)


class ExtractedEvidence(BaseModel):
    source_rank: int = Field(ge=1)
    claim_candidate: str = Field(min_length=5, max_length=800)
    verbatim_excerpt: str = Field(min_length=5, max_length=2000)
    stance: EvidenceStance
    relevance: Literal["high", "medium", "low"]
    source_type: SourceType = SourceType.SECONDARY
    source_directness: SourceDirectness = SourceDirectness.UNCLEAR
    credibility_tier: CredibilityTier = CredibilityTier.C
    discovery_only: bool = False
    attributed_source_name: str | None = Field(default=None, max_length=300)
    attributed_source_url: str | None = Field(default=None, max_length=1000)
    follow_up_query: str | None = Field(default=None, max_length=400)
    dimension_ids: list[str] = Field(default_factory=list, max_length=8)


class EvidenceBatch(BaseModel):
    evidence: list[ExtractedEvidence] = Field(default_factory=list)


class EvidenceRecord(BaseModel):
    evidence_id: str
    source_id: str
    task_id: str
    source_title: str
    source_url: str
    source_domain: str
    claim_candidate: str
    verbatim_excerpt: str
    stance: EvidenceStance
    relevance: Literal["high", "medium", "low"]
    source_type: SourceType = SourceType.SECONDARY
    source_directness: SourceDirectness = SourceDirectness.UNCLEAR
    credibility_tier: CredibilityTier = CredibilityTier.C
    discovery_only: bool = False
    attributed_source_name: str | None = None
    attributed_source_url: str | None = None
    follow_up_query: str | None = None
    provenance_key: str | None = None
    source_assessment_id: str | None = None
    source_quality_score: int | None = Field(default=None, ge=0, le=15)
    evidence_fitness_score: int | None = Field(default=None, ge=0, le=12)
    source_assessment_rationale: list[str] = Field(default_factory=list)
    source_eligibility: SourceEligibility | None = None
    source_hard_flags: list[SourceHardFlag] = Field(default_factory=list)
    source_score_dimensions: SourceScoreDimensions | None = None
    dimension_ids: list[str] = Field(default_factory=list, max_length=8)


class WorkerResult(BaseModel):
    task_id: str
    execution_outcome: Literal["succeeded", "partial", "failed_retryable"]
    evidence_ids: list[str] = Field(default_factory=list)
    source_ids: list[str] = Field(default_factory=list)
    search_calls_used: int = Field(default=0, ge=0)
    new_evidence_count: int = Field(default=0, ge=0)
    executed_queries: list[QuerySpec] = Field(default_factory=list)
    adaptive_steps: int = Field(default=0, ge=0)
    stop_reason: str | None = None
    unresolved_gaps: list[str] = Field(default_factory=list)
    error: str | None = None


class ClaimAssessment(BaseModel):
    claim_id: str = Field(pattern=r"^C[0-9]+$")
    claim: str = Field(min_length=5, max_length=1000)
    dimension: str | None = Field(default=None, max_length=200)
    status: ClaimStatus
    confidence: Confidence
    supporting_evidence_ids: list[str] = Field(default_factory=list)
    contradicting_evidence_ids: list[str] = Field(default_factory=list)
    reasoning: str = Field(min_length=3, max_length=1200)


class AuditGap(BaseModel):
    """A replan-ready evidence gap emitted by an Auditor."""

    gap_id: str = Field(min_length=2, max_length=80)
    check_id: str | None = Field(default=None, max_length=40)
    dimension: str | None = Field(default=None, max_length=200)
    task_ids: list[str] = Field(default_factory=list, max_length=12)
    gap_type: Literal[
        "coverage",
        "source_quality",
        "task_execution",
        "counterevidence",
        "contradiction",
        "verification",
        "other",
    ] = "other"
    priority: Literal["high", "medium", "low"] = "medium"
    description: str = Field(min_length=3, max_length=1000)
    related_evidence_ids: list[str] = Field(default_factory=list, max_length=16)
    missing_evidence: str | None = Field(default=None, max_length=1000)
    suggested_query: str | None = Field(default=None, max_length=400)


class AuditResult(BaseModel):
    sufficient: bool
    coverage_sufficient: bool = True
    source_sufficient: bool = True
    task_execution_sufficient: bool = True
    challenge_sufficient: bool = True
    claims: list[ClaimAssessment] = Field(default_factory=list)
    gaps: list[str] = Field(default_factory=list)
    actionable_gaps: list[AuditGap] = Field(default_factory=list, max_length=6)
    follow_up_query: str | None = Field(default=None, max_length=400)

    @field_validator("claims", mode="before")
    @classmethod
    def keep_bounded_claim_set(cls, value):
        """Keep reports bounded even when a model emits one claim per source."""
        if isinstance(value, list):
            return value[:12]
        return value

    @field_validator("actionable_gaps", mode="before")
    @classmethod
    def keep_bounded_actionable_gaps(cls, value):
        """Bound specialist handoff even when a model overproduces gaps."""
        if isinstance(value, list):
            return value[:6]
        return value

    @field_validator("gaps", mode="before")
    @classmethod
    def normalize_gap_shapes(cls, value):
        """Accept common structured gap output and store canonical text."""
        if not isinstance(value, list):
            return value

        normalized: list[str] = []
        for item in value:
            if isinstance(item, str):
                text = item.strip()
            elif isinstance(item, dict):
                text = next(
                    (
                        str(item[key]).strip()
                        for key in (
                            "description",
                            "gap",
                            "reason",
                            "issue",
                            "summary",
                            "question",
                        )
                        if item.get(key)
                    ),
                    "",
                )
            else:
                text = str(item).strip()
            if text:
                normalized.append(text)
        return normalized[:6]


class AuditCheck(BaseModel):
    """One bounded semantic inspection over one or more coverage dimensions."""

    check_id: str = Field(pattern=r"^A[0-9]+$")
    dimensions: list[str] = Field(min_length=1, max_length=8)
    task_ids: list[str] = Field(default_factory=list, max_length=12)
    priority: Literal["high", "medium", "low"] = "medium"
    reasons: list[str] = Field(default_factory=list, max_length=8)
    evidence_ids: list[str] = Field(default_factory=list)
    estimated_evidence_tokens: int = Field(default=0, ge=0)


class AuditPlan(BaseModel):
    """Durable coordinator output for either a fast or hierarchical audit."""

    mode: Literal["single", "hierarchical"]
    round: int = Field(ge=0)
    rationale: str = Field(min_length=3, max_length=1000)
    estimated_single_audit_tokens: int = Field(ge=0)
    checks: list[AuditCheck] = Field(default_factory=list, max_length=8)


class ReportStatementType(StrEnum):
    OBSERVED_FACT = "observed_fact"
    SOURCE_PROJECTION = "source_projection"
    DERIVED_ESTIMATE = "derived_estimate"
    INTERPRETATION = "interpretation"


class ReportStatement(BaseModel):
    """One reader-facing statement with explicit evidence provenance."""

    text: str = Field(min_length=5, max_length=1500)
    evidence_ids: list[str] = Field(default_factory=list, max_length=8)
    statement_type: ReportStatementType = ReportStatementType.OBSERVED_FACT
    confidence: Confidence = Confidence.MEDIUM


class ReportSection(BaseModel):
    dimension: str = Field(min_length=1, max_length=200)
    heading: str = Field(min_length=1, max_length=240)
    findings: list[ReportStatement] = Field(default_factory=list, max_length=10)


class ReportDraft(BaseModel):
    """Typed, grounding-addressable input to the final Markdown renderer."""

    title: str = Field(min_length=3, max_length=300)
    direct_answer: list[ReportStatement] = Field(default_factory=list, max_length=6)
    executive_summary: list[ReportStatement] = Field(default_factory=list, max_length=8)
    sections: list[ReportSection] = Field(default_factory=list, max_length=10)
    comparisons: list[ReportStatement] = Field(default_factory=list, max_length=8)
    methodology: list[str] = Field(default_factory=list, max_length=8)
    limitations: list[str] = Field(default_factory=list, max_length=10)


class ResearchState(dict):
    """Documentation-only state shape; the graph uses a TypedDict in graph.py."""


EvidenceIdList = Annotated[list[str], "References into the evidence store"]
