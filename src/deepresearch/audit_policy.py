from __future__ import annotations

import json
import re
from collections import Counter, defaultdict

from .schemas import (
    AuditCheck,
    AuditGap,
    AuditPlan,
    AuditResult,
    ClaimAssessment,
    ClaimStatus,
    Confidence,
    CredibilityTier,
    EvidenceRecord,
    EvidenceStance,
    ResearchPlan,
    SourceDirectness,
    TaskStatus,
)


_PRIORITY_SCORE = {"high": 3, "medium": 2, "low": 1}
_CLAIM_STATUS_SCORE = {
    ClaimStatus.SUPPORTED: 4,
    ClaimStatus.MIXED: 3,
    ClaimStatus.UNSUPPORTED: 2,
    ClaimStatus.UNRESOLVED: 1,
}
_CONFIDENCE_SCORE = {
    Confidence.HIGH: 3,
    Confidence.MEDIUM: 2,
    Confidence.LOW: 1,
}


def _normalized_text(value: str) -> str:
    return " ".join(value.casefold().split())


def _merge_claims(
    audit_plan: AuditPlan,
    partial_audits: list[AuditResult],
) -> list[ClaimAssessment]:
    """Deduplicate claims and preserve every audited coverage dimension."""
    merged: dict[tuple[str, str], ClaimAssessment] = {}
    order: list[tuple[str, str]] = []
    for partial_index, partial in enumerate(partial_audits):
        check = (
            audit_plan.checks[partial_index]
            if partial_index < len(audit_plan.checks)
            else None
        )
        for claim in partial.claims:
            if check and len(check.dimensions) == 1 and (
                claim.dimension not in check.dimensions
            ):
                claim = claim.model_copy(
                    update={"dimension": check.dimensions[0]}
                )
            key = (claim.dimension or "", _normalized_text(claim.claim))
            existing = merged.get(key)
            if existing is None:
                merged[key] = claim
                order.append(key)
                continue
            statuses = {existing.status, claim.status}
            status = existing.status if len(statuses) == 1 else ClaimStatus.MIXED
            confidence = (
                existing.confidence
                if existing.confidence == claim.confidence and len(statuses) == 1
                else Confidence.LOW
            )
            merged[key] = existing.model_copy(update={
                "status": status,
                "confidence": confidence,
                "supporting_evidence_ids": list(dict.fromkeys([
                    *existing.supporting_evidence_ids,
                    *claim.supporting_evidence_ids,
                ])),
                "contradicting_evidence_ids": list(dict.fromkeys([
                    *existing.contradicting_evidence_ids,
                    *claim.contradicting_evidence_ids,
                ])),
                "reasoning": " ".join(dict.fromkeys([
                    existing.reasoning,
                    claim.reasoning,
                ]))[:1200],
            })

    order_index = {key: index for index, key in enumerate(order)}

    def quality(key: tuple[str, str]) -> tuple[int, int, int, int]:
        claim = merged[key]
        evidence_count = len(set([
            *claim.supporting_evidence_ids,
            *claim.contradicting_evidence_ids,
        ]))
        return (
            _CLAIM_STATUS_SCORE[claim.status],
            _CONFIDENCE_SCORE[claim.confidence],
            evidence_count,
            -order_index[key],
        )

    dimensions = list(dict.fromkeys(
        dimension
        for check in audit_plan.checks
        for dimension in check.dimensions
    ))
    selected: list[tuple[str, str]] = []
    selected_set: set[tuple[str, str]] = set()
    for dimension in dimensions:
        candidates = [key for key in order if key[0] == dimension]
        if not candidates:
            continue
        best = max(candidates, key=quality)
        selected.append(best)
        selected_set.add(best)

    remaining = sorted(
        (key for key in order if key not in selected_set),
        key=quality,
        reverse=True,
    )
    selected.extend(remaining[:max(0, 12 - len(selected))])
    return [
        merged[key].model_copy(update={"claim_id": f"C{index}"})
        for index, key in enumerate(selected[:12], start=1)
    ]


def aggregate_audits(
    audit_plan: AuditPlan,
    partial_audits: list[AuditResult],
) -> AuditResult:
    """Combine specialist results locally and preserve replan-ready gaps."""
    checks = audit_plan.checks
    collected: list[AuditGap] = []
    plain_gaps: list[str] = []
    follow_up_query: str | None = None

    for index, partial in enumerate(partial_audits):
        check = checks[index] if index < len(checks) else None
        explicit_descriptions: set[str] = set()
        for gap in partial.actionable_gaps:
            description_key = _normalized_text(gap.description)
            explicit_descriptions.add(description_key)
            collected.append(gap.model_copy(update={
                "check_id": gap.check_id or (check.check_id if check else None),
                "dimension": gap.dimension or (
                    check.dimensions[0]
                    if check and len(check.dimensions) == 1
                    else None
                ),
                "task_ids": gap.task_ids or (check.task_ids if check else []),
                "priority": (
                    gap.priority
                    if gap.priority != "medium" or not check
                    else check.priority
                ),
            }))
        for gap_index, description in enumerate(partial.gaps, start=1):
            plain_gaps.append(description)
            if _normalized_text(description) in explicit_descriptions:
                continue
            collected.append(AuditGap(
                gap_id=(
                    f"{check.check_id}-G{gap_index}"
                    if check else f"G{index + 1}-{gap_index}"
                ),
                check_id=check.check_id if check else None,
                dimension=(
                    check.dimensions[0]
                    if check and len(check.dimensions) == 1
                    else None
                ),
                task_ids=check.task_ids if check else [],
                priority=check.priority if check else "medium",
                description=description,
                suggested_query=partial.follow_up_query,
            ))
        if follow_up_query is None and partial.follow_up_query:
            follow_up_query = partial.follow_up_query

    deduplicated: dict[str, AuditGap] = {}
    for gap in collected:
        key = _normalized_text(gap.description)
        existing = deduplicated.get(key)
        if existing is None:
            deduplicated[key] = gap
            continue
        preferred = (
            gap if _PRIORITY_SCORE[gap.priority] > _PRIORITY_SCORE[existing.priority]
            else existing
        )
        deduplicated[key] = preferred.model_copy(update={
            "task_ids": list(dict.fromkeys([*existing.task_ids, *gap.task_ids])),
            "related_evidence_ids": list(dict.fromkeys([
                *existing.related_evidence_ids,
                *gap.related_evidence_ids,
            ])),
            "suggested_query": existing.suggested_query or gap.suggested_query,
            "missing_evidence": existing.missing_evidence or gap.missing_evidence,
        })

    actionable = sorted(
        deduplicated.values(),
        key=lambda gap: (-_PRIORITY_SCORE[gap.priority], gap.gap_id),
    )[:6]
    gaps = list(dict.fromkeys([
        *plain_gaps,
        *(gap.description for gap in actionable),
    ]))[:6]
    all_sufficient = bool(partial_audits) and all(
        item.sufficient for item in partial_audits
    )
    return AuditResult(
        sufficient=all_sufficient and not gaps,
        coverage_sufficient=bool(partial_audits) and all(
            item.coverage_sufficient for item in partial_audits
        ),
        source_sufficient=bool(partial_audits) and all(
            item.source_sufficient for item in partial_audits
        ),
        task_execution_sufficient=bool(partial_audits) and all(
            item.task_execution_sufficient for item in partial_audits
        ),
        challenge_sufficient=bool(partial_audits) and all(
            item.challenge_sufficient for item in partial_audits
        ),
        claims=_merge_claims(audit_plan, partial_audits),
        gaps=gaps,
        actionable_gaps=actionable,
        follow_up_query=next(
            (gap.suggested_query for gap in actionable if gap.suggested_query),
            follow_up_query,
        ),
    )


def estimate_single_audit_tokens(
    evidence: list[EvidenceRecord], research_context: dict[str, object]
) -> int:
    payload = {
        "evidence": [item.model_dump(mode="json") for item in evidence],
        "research_context": research_context,
    }
    return max(1, (len(json.dumps(payload, ensure_ascii=False)) + 2) // 3)


def build_audit_plan(
    plan: ResearchPlan,
    evidence: list[EvidenceRecord],
    task_statuses: dict[str, str],
    *,
    round_number: int,
    estimated_single_tokens: int,
    single_context_budget: int,
    max_specialists: int,
) -> AuditPlan:
    """Create a deterministic, coverage-safe coordinator plan."""
    if estimated_single_tokens <= single_context_budget:
        return AuditPlan(
            mode="single",
            round=round_number,
            rationale=(
                f"Estimated audit input {estimated_single_tokens} tokens fits the "
                f"single-auditor budget of {single_context_budget}."
            ),
            estimated_single_audit_tokens=estimated_single_tokens,
        )

    dimensions = list(plan.coverage_contract.required_dimensions)
    if not dimensions:
        dimensions = ["overall evidence"]
    tasks_by_dimension: dict[str, list[str]] = defaultdict(list)
    for task in plan.tasks:
        for dimension in task.covered_dimensions:
            tasks_by_dimension[dimension].append(task.task_id)
    evidence_by_task: dict[str, list[EvidenceRecord]] = defaultdict(list)
    for item in evidence:
        evidence_by_task[item.task_id].append(item)

    ranked_dimensions: list[tuple[int, str, list[str], list[str]]] = []
    for dimension in dimensions:
        task_ids = tasks_by_dimension.get(dimension, [])
        items = [item for task_id in task_ids for item in evidence_by_task[task_id]]
        provenance = {item.provenance_key or item.source_url for item in items}
        stances = {item.stance for item in items}
        credible = sum(
            item.credibility_tier in {CredibilityTier.A, CredibilityTier.B}
            and not item.discovery_only
            for item in items
        )
        weak_tasks = [
            task_id
            for task_id in task_ids
            if task_statuses.get(task_id) not in {
                TaskStatus.SUCCEEDED.value,
            }
        ]
        score = 0
        reasons: list[str] = []
        if not items:
            score += 10
            reasons.append("No eligible evidence is assigned to this dimension.")
        if len(provenance) < 2:
            score += 6
            reasons.append("Fewer than two independent provenances are available.")
        if weak_tasks:
            score += 5
            reasons.append("Required tasks are partial, failed, or not completed.")
        if EvidenceStance.SUPPORTS in stances and EvidenceStance.CONTRADICTS in stances:
            score += 4
            reasons.append("Supporting and contradicting evidence both exist.")
        if items and credible == 0:
            score += 5
            reasons.append("No A/B-tier eligible evidence is available.")
        if not reasons:
            reasons.append("Required coverage dimension needs a bounded semantic check.")
        ranked_dimensions.append((score, dimension, task_ids, reasons))

    ranked_dimensions.sort(key=lambda item: (-item[0], item[1]))
    group_count = min(max_specialists, len(ranked_dimensions))
    groups: list[dict[str, object]] = [
        {"score": 0, "dimensions": [], "task_ids": [], "reasons": []}
        for _ in range(group_count)
    ]
    for index, (score, dimension, task_ids, reasons) in enumerate(ranked_dimensions):
        target = index if index < group_count else min(
            range(group_count),
            key=lambda group_index: (
                int(groups[group_index]["score"]),
                len(groups[group_index]["dimensions"]),
            ),
        )
        group = groups[target]
        group["score"] = int(group["score"]) + score
        group["dimensions"] = [*group["dimensions"], dimension]
        group["task_ids"] = list(dict.fromkeys([*group["task_ids"], *task_ids]))
        group["reasons"] = list(dict.fromkeys([*group["reasons"], *reasons]))[:8]

    checks: list[AuditCheck] = []
    for index, group in enumerate(groups, start=1):
        score = int(group["score"])
        checks.append(
            AuditCheck(
                check_id=f"A{index}",
                dimensions=list(group["dimensions"]),
                task_ids=list(group["task_ids"]),
                priority="high" if score >= 8 else "medium" if score >= 3 else "low",
                reasons=list(group["reasons"]),
            )
        )
    return AuditPlan(
        mode="hierarchical",
        round=round_number,
        rationale=(
            f"Estimated single-auditor input {estimated_single_tokens} exceeds "
            f"the {single_context_budget}-token budget; split all required "
            f"dimensions across {len(checks)} bounded checks."
        ),
        estimated_single_audit_tokens=estimated_single_tokens,
        checks=checks,
    )


def compact_audit_evidence(item: EvidenceRecord) -> EvidenceRecord:
    """Keep source semantics and citation identity while bounding verbose fields."""
    return item.model_copy(
        update={
            "claim_candidate": item.claim_candidate[:600],
            "verbatim_excerpt": item.verbatim_excerpt[:1200],
            "source_assessment_rationale": item.source_assessment_rationale[:3],
        }
    )


def select_audit_evidence(
    evidence: list[EvidenceRecord],
    check: AuditCheck,
    *,
    token_budget: int,
) -> tuple[list[EvidenceRecord], int]:
    """Read a diverse, dimension-local slice from evidence memory."""
    task_ids = set(check.task_ids)
    task_candidates = (
        [item for item in evidence if item.task_id in task_ids]
        if task_ids
        else list(evidence)
    )
    selected_dimensions = set(check.dimensions)
    candidates = [
        item
        for item in task_candidates
        if not item.dimension_ids
        or bool(selected_dimensions & set(item.dimension_ids))
    ]
    query_terms = set(re.findall(
        r"[\w\u4e00-\u9fff]+",
        " ".join(check.dimensions).casefold(),
    ))
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

    def rank(item: EvidenceRecord) -> tuple[int, int, int, int, int, int, str]:
        terms = set(re.findall(
            r"[\w\u4e00-\u9fff]+",
            f"{item.source_title} {item.claim_candidate} {item.verbatim_excerpt}".casefold(),
        ))
        numeric = min(4, len(re.findall(
            r"\b\d+(?:[.,]\d+)?%?\b",
            f"{item.claim_candidate} {item.verbatim_excerpt}",
        )))
        return (
            len(query_terms & terms),
            directness_score[item.source_directness],
            tier_score[item.credibility_tier],
            numeric,
            item.evidence_fitness_score or 0,
            relevance_score[item.relevance],
            item.evidence_id,
        )

    ordered = sorted(candidates, key=rank, reverse=True)

    selected: list[EvidenceRecord] = []
    selected_ids: set[str] = set()
    used_tokens = 0
    provenance_counts: Counter[str] = Counter()
    deferred: list[EvidenceRecord] = []

    def add(item: EvidenceRecord) -> bool:
        nonlocal used_tokens
        if item.evidence_id in selected_ids:
            return False
        compact = compact_audit_evidence(item)
        item_tokens = max(1, (len(compact.model_dump_json()) + 2) // 3)
        if used_tokens + item_tokens > token_budget:
            return False
        selected.append(compact)
        selected_ids.add(item.evidence_id)
        used_tokens += item_tokens
        provenance = item.provenance_key or item.source_url
        provenance_counts[provenance] += 1
        return True

    # Reserve the strongest supporting and challenging evidence before applying
    # provenance diversity. This prevents a central numeric result from being
    # displaced by tangential evidence from many domains.
    for dimension in check.dimensions:
        dimension_candidates = [
            item
            for item in candidates
            if not item.dimension_ids or dimension in item.dimension_ids
        ]
        for stance in (EvidenceStance.SUPPORTS, EvidenceStance.CONTRADICTS):
            stance_candidates = [
                item for item in dimension_candidates if item.stance == stance
            ]
            if stance_candidates:
                add(max(stance_candidates, key=rank))

    for item in ordered:
        if item.evidence_id in selected_ids:
            continue
        provenance = item.provenance_key or item.source_url
        if provenance_counts[provenance] >= 1:
            deferred.append(item)
            continue
        add(item)
    for item in deferred:
        provenance = item.provenance_key or item.source_url
        if provenance_counts[provenance] >= 3:
            continue
        add(item)
    return selected, used_tokens


def scoped_research_context(
    plan: ResearchPlan,
    task_statuses: dict[str, str],
    source_risks: dict[str, object],
    check: AuditCheck,
) -> dict[str, object]:
    selected_dimensions = set(check.dimensions)
    tasks = []
    for task in plan.tasks:
        covered = [
            dimension
            for dimension in task.covered_dimensions
            if dimension in selected_dimensions
        ]
        if not covered:
            continue
        payload = task.model_dump(mode="json")
        payload["covered_dimensions"] = covered
        tasks.append(payload)
    return {
        "plan": {
            "coverage_contract": {
                **plan.coverage_contract.model_dump(mode="json"),
                "required_dimensions": check.dimensions,
            },
            "tasks": tasks,
        },
        "task_statuses": {
            task_id: task_statuses.get(task_id, "unknown")
            for task_id in check.task_ids
        },
        "source_risks": source_risks,
        "audit_check": check.model_dump(mode="json"),
    }
