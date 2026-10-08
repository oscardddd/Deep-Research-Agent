from __future__ import annotations

import re
from collections import Counter

from .schemas import (
    AuditResult,
    Confidence,
    CredibilityTier,
    EvidenceRecord,
    ReportDraft,
    ReportSection,
    ReportStatement,
    ReportStatementType,
    ResearchPlan,
    SourceDirectness,
)


_STOPWORDS = {
    "about", "across", "analysis", "and", "changes", "current", "for",
    "from", "future", "japan", "patterns", "required", "the", "with",
}

_NUMBER_RE = re.compile(r"(?<!\w)\d+(?:[.,]\d+)*%?")
_SENTENCE_BOUNDARY_RE = re.compile(
    r"(?<=[.!?。！？])\s+(?=[A-Z0-9\u4e00-\u9fff])"
)


def _terms(value: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[\w\u4e00-\u9fff]+", value.casefold())
        if len(token) > 2 and token not in _STOPWORDS
    }


def evidence_dimensions(
    item: EvidenceRecord,
    plan: ResearchPlan,
) -> list[str]:
    required = plan.coverage_contract.required_dimensions
    explicit = [dimension for dimension in item.dimension_ids if dimension in required]
    if explicit:
        return explicit
    task = next((task for task in plan.tasks if task.task_id == item.task_id), None)
    if task is None:
        return []
    if len(task.covered_dimensions) == 1:
        return task.covered_dimensions
    text_terms = _terms(f"{item.claim_candidate} {item.verbatim_excerpt}")
    scored = [
        (len(text_terms & _terms(dimension)), dimension)
        for dimension in task.covered_dimensions
    ]
    best_score = max((score for score, _dimension in scored), default=0)
    if best_score == 0:
        return []
    return [dimension for score, dimension in scored if score == best_score]


def _rank(item: EvidenceRecord, dimension: str | None = None) -> tuple[int, ...]:
    tier = {
        CredibilityTier.A: 4,
        CredibilityTier.B: 3,
        CredibilityTier.C: 2,
        CredibilityTier.D: 1,
    }[item.credibility_tier]
    directness = {
        SourceDirectness.DIRECT: 3,
        SourceDirectness.UNCLEAR: 2,
        SourceDirectness.ATTRIBUTED: 1,
    }[item.source_directness]
    relevance = {"high": 3, "medium": 2, "low": 1}[item.relevance]
    text = f"{item.claim_candidate} {item.verbatim_excerpt}"
    numeric = min(4, len(re.findall(r"\b\d+(?:[.,]\d+)?%?\b", text)))
    overlap = len(_terms(text) & _terms(dimension or ""))
    return (
        overlap,
        directness,
        tier,
        numeric,
        item.evidence_fitness_score or 0,
        relevance,
    )


def select_report_evidence(
    plan: ResearchPlan,
    audit: AuditResult,
    evidence: list[EvidenceRecord],
    *,
    max_items: int = 48,
) -> list[EvidenceRecord]:
    """Build a coverage-first writer context without enforcing claim independence."""
    eligible = [item for item in evidence if not item.discovery_only]
    by_id = {item.evidence_id: item for item in eligible}
    selected: list[EvidenceRecord] = []
    seen: set[str] = set()

    def add(item: EvidenceRecord | None) -> None:
        if item is not None and item.evidence_id not in seen and len(selected) < max_items:
            selected.append(item)
            seen.add(item.evidence_id)

    # Guarantee that audit citations cannot consume the entire context before
    # each required dimension contributes at least one strong evidence item.
    for dimension in plan.coverage_contract.required_dimensions:
        candidates = [
            item
            for item in eligible
            if dimension in evidence_dimensions(item, plan)
        ]
        if candidates:
            add(max(candidates, key=lambda value: _rank(value, dimension)))

    for claim in audit.claims:
        for evidence_id in [
            *claim.supporting_evidence_ids,
            *claim.contradicting_evidence_ids,
        ]:
            add(by_id.get(evidence_id))

    for dimension in plan.coverage_contract.required_dimensions:
        candidates = [
            item
            for item in eligible
            if dimension in evidence_dimensions(item, plan)
        ]
        already_selected = [
            item
            for item in selected
            if dimension in evidence_dimensions(item, plan)
        ]
        provenance_counts = Counter(
            item.provenance_key or item.source_url
            for item in already_selected
        )
        added = len(already_selected)
        for item in sorted(
            candidates,
            key=lambda value: _rank(value, dimension),
            reverse=True,
        ):
            provenance = item.provenance_key or item.source_url
            if provenance_counts[provenance] >= 3:
                continue
            before = len(selected)
            add(item)
            if len(selected) > before:
                provenance_counts[provenance] += 1
                added += 1
            if added >= 6:
                break

    for item in sorted(eligible, key=_rank, reverse=True):
        add(item)
    return selected


def _reader_limitation(value: str) -> str:
    """Translate deterministic QC language into reader-facing limitations."""
    normalized = " ".join(value.split())
    replacements = (
        (
            "One or more high-priority research tasks are incomplete.",
            "Evidence collection remains incomplete for one or more high-priority aspects.",
        ),
        (
            "Required dimensions lack completed research:",
            "Evidence collection remains incomplete for:",
        ),
        (
            "Required dimensions lack an evidence-backed audited conclusion:",
            "The available evidence does not support a firm conclusion for:",
        ),
        (
            "The planned counterevidence search is incomplete.",
            "Counterevidence coverage remains incomplete.",
        ),
        (
            "One or more core claims lack two independent origins including an A/B-tier source.",
            "Some core findings have limited independent corroboration.",
        ),
    )
    for internal, reader_facing in replacements:
        if normalized.startswith(internal):
            return normalized.replace(internal, reader_facing, 1)
    return normalized


def _numbers(value: str) -> set[str]:
    return {
        match.group(0).replace(",", "").rstrip(".%")
        for match in _NUMBER_RE.finditer(value)
    }


def _support_rank(text: str, item: EvidenceRecord) -> tuple[int, float, int]:
    """Rank how completely one evidence item supports one reader sentence."""
    evidence_text = f"{item.claim_candidate} {item.verbatim_excerpt}"
    statement_numbers = _numbers(text)
    evidence_numbers = _numbers(evidence_text)
    statement_terms = _terms(text)
    evidence_terms = _terms(evidence_text)
    overlap = len(statement_terms & evidence_terms)
    coverage = overlap / max(1, len(statement_terms))
    return (
        len(statement_numbers & evidence_numbers),
        coverage,
        overlap,
    )


def _independently_supports(text: str, item: EvidenceRecord) -> bool:
    """Conservatively reject citations that cover only part of a compound fact."""
    statement_numbers = _numbers(text)
    evidence_text = f"{item.claim_candidate} {item.verbatim_excerpt}"
    if statement_numbers and not statement_numbers <= _numbers(evidence_text):
        return False
    _numeric_overlap, coverage, overlap = _support_rank(text, item)
    threshold = 0.45 if statement_numbers else 0.60
    return overlap >= 2 and coverage >= threshold


def _statement_fragments(value: str) -> list[str]:
    normalized = " ".join(value.split())
    return [
        fragment.strip()
        for fragment in _SENTENCE_BOUNDARY_RE.split(normalized)
        if fragment.strip()
    ]


def _normalize_statement(
    statement: ReportStatement,
    evidence_by_id: dict[str, EvidenceRecord],
) -> list[ReportStatement]:
    ids = list(dict.fromkeys(
        evidence_id
        for evidence_id in statement.evidence_ids
        if evidence_id in evidence_by_id
        and not evidence_by_id[evidence_id].discovery_only
    ))[:8]
    if not ids:
        return []
    items = [evidence_by_id[evidence_id] for evidence_id in ids]

    def normalized_fragment(text: str, fragment_ids: list[str]) -> ReportStatement:
        fragment_items = [evidence_by_id[evidence_id] for evidence_id in fragment_ids]
        provenances = {item.provenance_key or item.source_url for item in fragment_items}
        has_credible = any(
            item.credibility_tier in {CredibilityTier.A, CredibilityTier.B}
            for item in fragment_items
        )
        confidence = statement.confidence
        if len(provenances) < 2 and confidence == Confidence.HIGH:
            confidence = Confidence.MEDIUM if has_credible else Confidence.LOW
        elif not has_credible:
            confidence = Confidence.LOW
        return statement.model_copy(update={
            "text": text,
            "evidence_ids": fragment_ids,
            "confidence": confidence,
        })

    fragments = _statement_fragments(statement.text)
    # A single atomic statement with one citation needs no heuristic rewrite.
    if len(fragments) == 1 and len(ids) == 1:
        return [normalized_fragment(fragments[0], ids)]

    normalized: list[ReportStatement] = []
    seen: set[tuple[str, tuple[str, ...]]] = set()
    for fragment in fragments:
        supporting_ids = [
            item.evidence_id
            for item in items
            if _independently_supports(fragment, item)
        ]
        text = fragment
        if not supporting_ids:
            # Do not retain a synthesized compound claim when no cited source
            # supports it independently. Fall back to the strongest atomic
            # claim already attached to the evidence record.
            best = max(items, key=lambda item: _support_rank(fragment, item))
            supporting_ids = [best.evidence_id]
            text = " ".join(best.claim_candidate.split())
        key = (text.casefold(), tuple(supporting_ids))
        if key in seen:
            continue
        seen.add(key)
        normalized.append(normalized_fragment(text, supporting_ids))
    return normalized


def normalize_report_draft(
    draft: ReportDraft,
    *,
    plan: ResearchPlan,
    audit: AuditResult,
    evidence: list[EvidenceRecord],
) -> ReportDraft:
    """Remove ungrounded prose and guarantee one section per required dimension."""
    evidence_by_id = {item.evidence_id: item for item in evidence}

    def normalize_many(
        statements: list[ReportStatement], *, max_items: int
    ) -> list[ReportStatement]:
        normalized = [
            fragment
            for statement in statements
            for fragment in _normalize_statement(statement, evidence_by_id)
        ]
        return normalized[:max_items]

    supplied = {
        section.dimension: section
        for section in draft.sections
        if section.dimension in plan.coverage_contract.required_dimensions
    }
    sections: list[ReportSection] = []
    for dimension in plan.coverage_contract.required_dimensions:
        section = supplied.get(dimension)
        findings = normalize_many(
            section.findings if section else [], max_items=10
        )
        if not findings:
            candidates = [
                item
                for item in evidence
                if not item.discovery_only
                and dimension in evidence_dimensions(item, plan)
            ]
            if candidates:
                item = max(candidates, key=lambda value: _rank(value, dimension))
                statement_type = (
                    ReportStatementType.SOURCE_PROJECTION
                    if re.search(r"\b(project|forecast|scenario)", item.claim_candidate, re.I)
                    else ReportStatementType.OBSERVED_FACT
                )
                findings = [ReportStatement(
                    text=item.claim_candidate,
                    evidence_ids=[item.evidence_id],
                    statement_type=statement_type,
                    confidence=(
                        Confidence.MEDIUM
                        if item.credibility_tier in {CredibilityTier.A, CredibilityTier.B}
                        else Confidence.LOW
                    ),
                )]
        sections.append(ReportSection(
            dimension=dimension,
            heading=" ".join((section.heading if section else dimension).split()),
            findings=findings,
        ))

    direct_answer = normalize_many(draft.direct_answer, max_items=6)
    if not direct_answer:
        direct_answer = [
            finding
            for section in sections
            for finding in section.findings
        ][:3]
    executive_summary = normalize_many(draft.executive_summary, max_items=8)
    if not executive_summary:
        executive_summary = direct_answer[:4]
    comparisons = normalize_many(draft.comparisons, max_items=8)
    limitations = list(dict.fromkeys(
        _reader_limitation(item)
        for item in [*audit.gaps, *draft.limitations]
        if item.strip()
    ))[:10]
    methodology = [
        " ".join(item.split())
        for item in draft.methodology
        if item.strip()
    ] or [
        "The report separates direct observations, source projections, and interpretations, and cites the evidence used for each factual statement."
    ]
    return draft.model_copy(update={
        "title": " ".join(draft.title.lstrip("# ").split()),
        "direct_answer": direct_answer,
        "executive_summary": executive_summary,
        "sections": sections,
        "comparisons": comparisons,
        "methodology": methodology,
        "limitations": limitations,
    })


def fallback_report_draft(
    question: str,
    plan: ResearchPlan,
    audit: AuditResult,
    evidence: list[EvidenceRecord],
) -> ReportDraft:
    return normalize_report_draft(
        ReportDraft(
            title=f"Research report: {question}"[:300],
            limitations=list(audit.gaps),
        ),
        plan=plan,
        audit=audit,
        evidence=evidence,
    )
