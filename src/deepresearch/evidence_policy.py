from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from .schemas import (
    AuditResult,
    CredibilityTier,
    EvidenceRecord,
    DigestClaim,
    ResearchPlan,
    ResearchStateDigest,
    SourceDirectness,
    SourceType,
    WorkerDigest,
)
from .store import canonicalize_url


SOCIAL_DOMAINS = {
    "facebook.com",
    "reddit.com",
    "x.com",
    "twitter.com",
    "tiktok.com",
}

REPUTABLE_SECONDARY_DOMAINS = {
    "apnews.com",
    "bbc.com",
    "britannica.com",
    "espn.com",
    "ft.com",
    "nytimes.com",
    "reuters.com",
    "theguardian.com",
    "washingtonpost.com",
}

OFFICIAL_DOMAINS = {
    "nba.com",
    "who.int",
}

AI_SYNTHESIS_TITLE_MARKERS = (
    "using openai's deep research",
    "using openai’s deep research",
    "using chatgpt",
    "according to ai",
    "ai-generated report",
)


def _root_domain(domain: str) -> str:
    return domain.casefold().removeprefix("www.")


def _matches_domain(domain: str, candidates: set[str]) -> bool:
    return any(domain == item or domain.endswith(f".{item}") for item in candidates)


def apply_deterministic_source_policy(item: EvidenceRecord) -> EvidenceRecord:
    """Apply conservative source rules that cannot be waived by an extractor."""
    domain = _root_domain(item.source_domain)
    title = item.source_title.casefold()
    source_type = item.source_type
    tier = item.credibility_tier
    discovery_only = item.discovery_only
    has_formal_assessment = bool(item.source_assessment_id)

    if source_type in {SourceType.SOCIAL, SourceType.AI_SYNTHESIS}:
        tier = CredibilityTier.D
        discovery_only = True
    elif source_type == SourceType.AGGREGATOR:
        tier = CredibilityTier.D
        discovery_only = True
    elif _matches_domain(domain, SOCIAL_DOMAINS):
        source_type = SourceType.SOCIAL
        tier = CredibilityTier.D
        discovery_only = True
    elif any(marker in title for marker in AI_SYNTHESIS_TITLE_MARKERS):
        source_type = SourceType.AI_SYNTHESIS
        tier = CredibilityTier.D
        discovery_only = True
    elif domain == "wikipedia.org" or domain.endswith(".wikipedia.org"):
        source_type = SourceType.REFERENCE
        if not has_formal_assessment:
            tier = CredibilityTier.C
    elif _matches_domain(domain, OFFICIAL_DOMAINS) or domain.endswith(".gov"):
        source_type = SourceType.OFFICIAL
        if not has_formal_assessment:
            tier = CredibilityTier.A
    elif domain.endswith(".edu"):
        if not has_formal_assessment and tier in {
            CredibilityTier.C,
            CredibilityTier.D,
        }:
            tier = CredibilityTier.B
    elif _matches_domain(domain, REPUTABLE_SECONDARY_DOMAINS):
        source_type = SourceType.REPUTABLE_SECONDARY
        if not has_formal_assessment and tier in {
            CredibilityTier.C,
            CredibilityTier.D,
        }:
            tier = CredibilityTier.B

    if item.source_directness == SourceDirectness.ATTRIBUTED:
        # The page is useful for locating the origin, but does not itself
        # establish the attributed underlying fact.
        if tier in {CredibilityTier.A, CredibilityTier.B}:
            tier = CredibilityTier.C
        discovery_only = True

    if (
        not has_formal_assessment
        and source_type in {SourceType.SECONDARY, SourceType.REFERENCE}
    ):
        if tier == CredibilityTier.A:
            tier = CredibilityTier.C
    if tier == CredibilityTier.D:
        discovery_only = True

    provenance_key = item.provenance_key
    if (
        item.source_directness == SourceDirectness.ATTRIBUTED
        and item.attributed_source_url
    ):
        provenance_key = canonicalize_url(item.attributed_source_url)
    else:
        provenance_key = canonicalize_url(item.source_url)

    return item.model_copy(
        update={
            "source_type": source_type,
            "credibility_tier": tier,
            "discovery_only": discovery_only,
            "provenance_key": provenance_key,
        }
    )


def curate_evidence(
    evidence: list[EvidenceRecord],
    *,
    max_per_source: int = 4,
    include_discovery: bool = False,
) -> list[EvidenceRecord]:
    """Select a diverse context while retaining raw evidence in storage."""
    assessed = [apply_deterministic_source_policy(item) for item in evidence]
    eligible = [
        item for item in assessed
        if include_discovery or not item.discovery_only
    ]
    tier_score = {
        CredibilityTier.A: 4,
        CredibilityTier.B: 3,
        CredibilityTier.C: 2,
        CredibilityTier.D: 1,
    }
    relevance_score = {"high": 3, "medium": 2, "low": 1}
    directness_score = {
        SourceDirectness.DIRECT: 3,
        SourceDirectness.UNCLEAR: 2,
        SourceDirectness.ATTRIBUTED: 1,
    }
    ordered = sorted(
        enumerate(eligible),
        key=lambda pair: (
            -tier_score[pair[1].credibility_tier],
            -directness_score[pair[1].source_directness],
            -relevance_score[pair[1].relevance],
            pair[0],
        ),
    )

    selected: list[tuple[int, EvidenceRecord]] = []
    source_counts: Counter[str] = Counter()
    for original_index, item in ordered:
        source_key = canonicalize_url(item.source_url)
        if source_counts[source_key] >= max_per_source:
            continue
        source_counts[source_key] += 1
        selected.append((original_index, item))

    return [item for _, item in sorted(selected, key=lambda pair: pair[0])]


def select_supervisor_evidence(
    evidence: list[EvidenceRecord],
    audit: AuditResult,
    *,
    max_items: int = 48,
) -> list[EvidenceRecord]:
    """Bound replan context while preserving evidence cited by the Auditor."""
    by_id = {item.evidence_id: item for item in evidence}
    cited_ids = [
        evidence_id
        for claim in audit.claims
        for evidence_id in (
            *claim.supporting_evidence_ids,
            *claim.contradicting_evidence_ids,
        )
    ]
    selected: list[EvidenceRecord] = []
    seen: set[str] = set()
    for evidence_id in cited_ids:
        item = by_id.get(evidence_id)
        if item is not None and evidence_id not in seen:
            selected.append(item)
            seen.add(evidence_id)

    tier_score = {
        CredibilityTier.A: 4,
        CredibilityTier.B: 3,
        CredibilityTier.C: 2,
        CredibilityTier.D: 1,
    }
    relevance_score = {"high": 3, "medium": 2, "low": 1}
    remaining = sorted(
        (item for item in evidence if item.evidence_id not in seen),
        key=lambda item: (
            -tier_score[item.credibility_tier],
            -relevance_score[item.relevance],
            item.evidence_id,
        ),
    )
    selected.extend(remaining[: max(0, max_items - len(selected))])
    return selected[:max_items]


def build_research_digest(
    plan: ResearchPlan,
    audit: AuditResult,
    evidence: list[EvidenceRecord],
    task_statuses: dict[str, str],
    worker_results: list[dict[str, object]],
    source_risks: dict[str, object],
    round_number: int,
) -> ResearchStateDigest:
    """Build a bounded, evidence-addressable map of the current research state."""
    by_task: dict[str, list[EvidenceRecord]] = defaultdict(list)
    for item in evidence:
        by_task[item.task_id].append(item)
    result_gaps: dict[str, list[str]] = defaultdict(list)
    for result in worker_results:
        task_id = str(result.get("task_id") or "")
        gaps = result.get("unresolved_gaps") or []
        if task_id and isinstance(gaps, list):
            result_gaps[task_id].extend(str(gap) for gap in gaps if gap)

    tier_score = {
        CredibilityTier.A: 4,
        CredibilityTier.B: 3,
        CredibilityTier.C: 2,
        CredibilityTier.D: 1,
    }
    workers: list[WorkerDigest] = []
    for task in plan.tasks:
        items = by_task.get(task.task_id, [])
        ordered = sorted(
            items,
            key=lambda item: (
                item.discovery_only,
                -tier_score[item.credibility_tier],
                {"high": 0, "medium": 1, "low": 2}[item.relevance],
                item.evidence_id,
            ),
        )
        claims = [
            DigestClaim(
                claim=item.claim_candidate,
                stance=item.stance,
                evidence_ids=[item.evidence_id],
                provenance_count=1,
                best_credibility_tier=item.credibility_tier,
            )
            for item in ordered[:6]
        ]
        workers.append(
            WorkerDigest(
                task_id=task.task_id,
                question=task.question,
                status=task_statuses.get(task.task_id, "unknown"),
                covered_dimensions=task.covered_dimensions,
                evidence_count=len(items),
                source_count=len({item.provenance_key or item.source_url for item in items}),
                discovery_only_count=sum(item.discovery_only for item in items),
                claims=claims,
                unresolved_gaps=list(dict.fromkeys(result_gaps[task.task_id]))[:6],
            )
        )

    audit_summary = {
        "sufficient": audit.sufficient,
        "coverage_sufficient": audit.coverage_sufficient,
        "source_sufficient": audit.source_sufficient,
        "task_execution_sufficient": audit.task_execution_sufficient,
        "challenge_sufficient": audit.challenge_sufficient,
        "gaps": audit.gaps,
        "actionable_gaps": [
            gap.model_dump(mode="json") for gap in audit.actionable_gaps
        ],
        "claims": [
            {
                "claim_id": claim.claim_id,
                "claim": claim.claim,
                "dimension": claim.dimension,
                "status": claim.status.value,
                "confidence": claim.confidence.value,
                "supporting_evidence_ids": claim.supporting_evidence_ids,
                "contradicting_evidence_ids": claim.contradicting_evidence_ids,
            }
            for claim in audit.claims
        ],
    }
    return ResearchStateDigest(
        round=round_number,
        coverage_contract=plan.coverage_contract,
        workers=workers,
        audit_summary=audit_summary,
        source_risks=source_risks,
        evidence_count=len(evidence),
    )


def select_drilldown_evidence(
    evidence: list[EvidenceRecord],
    audit: AuditResult,
    question: str,
    *,
    token_budget: int,
) -> list[EvidenceRecord]:
    """Retrieve a bounded evidence slice for each actionable audit gap."""
    cited = {
        evidence_id
        for claim in audit.claims
        for evidence_id in (
            *claim.supporting_evidence_ids,
            *claim.contradicting_evidence_ids,
        )
    }
    tier_score = {CredibilityTier.A: 4, CredibilityTier.B: 3,
                  CredibilityTier.C: 2, CredibilityTier.D: 1}

    def terms(value: str) -> set[str]:
        return set(re.findall(r"[\w\u4e00-\u9fff]+", value.casefold()))

    priority_score = {"high": 3, "medium": 2, "low": 1}
    actionable = sorted(
        audit.actionable_gaps,
        key=lambda gap: (-priority_score[gap.priority], gap.gap_id),
    )[:3]

    # Old checkpoints and single-auditor fixtures may not have structured gaps.
    if not actionable:
        query_terms = terms(" ".join([
            question,
            *audit.gaps,
            audit.follow_up_query or "",
        ]))

        def legacy_score(item: EvidenceRecord) -> tuple[int, int, int, str]:
            text_terms = terms(f"{item.claim_candidate} {item.verbatim_excerpt}")
            return (
                1 if item.evidence_id in cited else 0,
                len(query_terms & text_terms),
                tier_score[item.credibility_tier],
                item.evidence_id,
            )

        selected: list[EvidenceRecord] = []
        used_tokens = 0
        provenance_counts: Counter[str] = Counter()
        for item in sorted(evidence, key=legacy_score, reverse=True):
            provenance = item.provenance_key or item.source_url
            if provenance_counts[provenance] >= 2 and item.evidence_id not in cited:
                continue
            payload = supervisor_evidence_payload(item)
            estimated_tokens = max(
                1, (len(json.dumps(payload, ensure_ascii=False)) + 2) // 3
            )
            if used_tokens + estimated_tokens > token_budget:
                continue
            selected.append(item)
            used_tokens += estimated_tokens
            provenance_counts[provenance] += 1
        return selected

    selected: list[EvidenceRecord] = []
    selected_ids: set[str] = set()
    used_tokens = 0
    provenance_counts: Counter[str] = Counter()
    per_gap_budget = max(600, token_budget // len(actionable))

    def add(item: EvidenceRecord, *, local_tokens: int) -> tuple[int, bool]:
        nonlocal used_tokens
        if item.evidence_id in selected_ids:
            return local_tokens, False
        provenance = item.provenance_key or item.source_url
        related = any(
            item.evidence_id in gap.related_evidence_ids for gap in actionable
        )
        if provenance_counts[provenance] >= 2 and not related:
            return local_tokens, False
        item_tokens = max(1, (
            len(json.dumps(supervisor_evidence_payload(item), ensure_ascii=False)) + 2
        ) // 3)
        if used_tokens + item_tokens > token_budget:
            return local_tokens, False
        if local_tokens + item_tokens > per_gap_budget and local_tokens > 0:
            return local_tokens, False
        selected.append(item)
        selected_ids.add(item.evidence_id)
        provenance_counts[provenance] += 1
        used_tokens += item_tokens
        return local_tokens + item_tokens, True

    evidence_by_id = {item.evidence_id: item for item in evidence}
    for gap in actionable:
        local_tokens = 0
        for evidence_id in gap.related_evidence_ids:
            item = evidence_by_id.get(evidence_id)
            if item is not None:
                local_tokens, _ = add(item, local_tokens=local_tokens)

        gap_terms = terms(" ".join(filter(None, [
            gap.dimension,
            gap.description,
            gap.missing_evidence,
            gap.suggested_query,
        ])))
        task_ids = set(gap.task_ids)
        candidates = [
            item for item in evidence
            if not task_ids or item.task_id in task_ids
        ]

        def gap_score(item: EvidenceRecord) -> tuple[int, int, int, str]:
            text_terms = terms(
                f"{item.source_title} {item.claim_candidate} {item.verbatim_excerpt}"
            )
            return (
                1 if item.evidence_id in gap.related_evidence_ids else 0,
                len(gap_terms & text_terms),
                tier_score[item.credibility_tier],
                item.evidence_id,
            )

        for item in sorted(candidates, key=gap_score, reverse=True):
            local_tokens, added = add(item, local_tokens=local_tokens)
            if not added and local_tokens >= per_gap_budget:
                break
    return selected


def supervisor_evidence_payload(item: EvidenceRecord) -> dict[str, object]:
    """Serialize only fields needed to decide whether another wave is useful."""
    return {
        "evidence_id": item.evidence_id,
        "task_id": item.task_id,
        "claim": item.claim_candidate,
        "excerpt": item.verbatim_excerpt[:600],
        "source_title": item.source_title,
        "source_url": item.source_url,
        "source_domain": item.source_domain,
        "source_type": item.source_type.value,
        "source_directness": item.source_directness.value,
        "credibility_tier": item.credibility_tier.value,
        "relevance": item.relevance,
        "provenance_key": item.provenance_key,
        "source_quality_score": item.source_quality_score,
        "evidence_fitness_score": item.evidence_fitness_score,
        "source_eligibility": (
            item.source_eligibility.value if item.source_eligibility else None
        ),
        "source_hard_flags": [flag.value for flag in item.source_hard_flags],
    }


def source_risk_summary(evidence: list[EvidenceRecord]) -> dict[str, object]:
    assessed = [apply_deterministic_source_policy(item) for item in evidence]
    url_counts = Counter(canonicalize_url(item.source_url) for item in assessed)
    discovery_urls = sorted(
        {item.source_url for item in assessed if item.discovery_only}
    )
    attributed_without_url = sum(
        item.source_directness == SourceDirectness.ATTRIBUTED
        and not item.attributed_source_url
        for item in assessed
    )
    tier_counts = Counter(item.credibility_tier.value for item in assessed)
    hard_flag_counts = Counter(
        flag.value for item in assessed for flag in item.source_hard_flags
    )
    return {
        "unique_urls": len(url_counts),
        "overrepresented_urls": [
            {"url": url, "evidence_count": count}
            for url, count in url_counts.most_common()
            if count > 4
        ][:8],
        "discovery_only_urls": discovery_urls[:8],
        "attributed_claims_missing_original_url": attributed_without_url,
        "tier_counts": dict(sorted(tier_counts.items())),
        "unassessed_evidence_items": sum(
            not item.source_assessment_id for item in assessed
        ),
        "low_fitness_evidence_items": sum(
            item.evidence_fitness_score is not None
            and item.evidence_fitness_score < 5
            for item in assessed
        ),
        "hard_flag_counts": dict(sorted(hard_flag_counts.items())),
    }
