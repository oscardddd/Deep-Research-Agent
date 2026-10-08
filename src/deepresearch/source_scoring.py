from __future__ import annotations

from urllib.parse import urlsplit

from .evidence_policy import (
    AI_SYNTHESIS_TITLE_MARKERS,
    OFFICIAL_DOMAINS,
    REPUTABLE_SECONDARY_DOMAINS,
    SOCIAL_DOMAINS,
)
from .schemas import (
    SearchResult,
    SourceAssessment,
    SourceDirectness,
    SourceHardFlag,
    SourceType,
)


SOURCE_RUBRIC_VERSION = "source-rubric-v1"

SOURCE_SCORING_GUIDE = """
Score each dimension from 0 to 3 using only visible page evidence.

Intrinsic source quality:
- authorship_transparency: 0 anonymous; 1 name only; 2 identifiable author or
  institution; 3 identity, role, and accountability are clear.
- methodology_transparency: 0 no method; 1 assertions with minimal method;
  2 material method/data limitations described; 3 reproducible method or
  primary record with clear definitions and limitations.
- publication_controls: 0 self-published/unknown; 1 basic editorial context;
  2 accountable institutional or professional editorial controls; 3 formal
  official, peer-review, standards, or audited publication controls.
- citation_traceability: 0 no traceable basis; 1 named but unlinked sources;
  2 traceable citations/data; 3 citations resolve to original records.
- independence: 0 direct conflict or promotional source; 1 material conflict
  is possible; 2 no material conflict is apparent; 3 clearly independent of
  the subject and evidence producer.

Claim-specific evidence fitness:
- provenance_directness: 0 hearsay/unknown; 1 secondary attribution; 2 direct
  reporting with some ambiguity; 3 original record/data/research/interview.
- authority_for_claim: 0 no relevant authority; 1 adjacent expertise; 2
  credible authority for part of the claim; 3 authoritative for this exact
  kind of claim. Authority must be evaluated relative to the assigned task.
- recency_for_claim: 0 materially outdated; 1 date unclear or aging; 2 current
  enough; 3 current/versioned as required. Historical facts may score 3 when
  age is immaterial.
- claim_relevance: 0 irrelevant; 1 tangential; 2 materially relevant; 3
  directly answers the assigned evidence requirement.

Hard flags override totals. AI synthesis, social content, content farms,
anonymous unsourced pages, and attributed-only pages are discovery leads.
Inaccessible, irrelevant, or plausibly retracted sources are rejected. Do not
infer peer review, authorship, methods, or independence from polished prose or
domain reputation alone. Provide short observable reasons for the scores.
""".strip()


def _domain_matches(domain: str, candidates: set[str]) -> bool:
    normalized = domain.casefold().removeprefix("www.")
    return any(
        normalized == item or normalized.endswith(f".{item}")
        for item in candidates
    )


def finalize_source_assessment(
    result: SearchResult,
    assessment: SourceAssessment,
) -> SourceAssessment:
    """Apply non-negotiable URL/title rules and recompute derived outcomes."""
    payload = assessment.model_dump(mode="json")
    flags = {SourceHardFlag(item) for item in payload.get("hard_flags", [])}
    domain = urlsplit(result.url).netloc
    title = result.title.casefold()
    source_type = assessment.source_type
    directness = assessment.source_directness

    if source_type == SourceType.AI_SYNTHESIS:
        flags.add(SourceHardFlag.AI_SYNTHESIS)
    if source_type == SourceType.SOCIAL:
        flags.add(SourceHardFlag.SOCIAL_CONTENT)
    if source_type == SourceType.AGGREGATOR:
        flags.add(SourceHardFlag.CONTENT_FARM)
    if _domain_matches(domain, SOCIAL_DOMAINS):
        flags.add(SourceHardFlag.SOCIAL_CONTENT)
        source_type = SourceType.SOCIAL
    if any(marker in title for marker in AI_SYNTHESIS_TITLE_MARKERS):
        flags.add(SourceHardFlag.AI_SYNTHESIS)
        source_type = SourceType.AI_SYNTHESIS
    if directness == SourceDirectness.ATTRIBUTED:
        flags.add(SourceHardFlag.ATTRIBUTED_ONLY)
    if (
        _domain_matches(domain, OFFICIAL_DOMAINS) or domain.endswith(".gov")
    ) and SourceHardFlag.AI_SYNTHESIS not in flags:
        source_type = SourceType.OFFICIAL
    elif (
        _domain_matches(domain, REPUTABLE_SECONDARY_DOMAINS)
        and SourceHardFlag.AI_SYNTHESIS not in flags
    ):
        source_type = SourceType.REPUTABLE_SECONDARY

    if result.content_source != "full_page":
        scores = payload["scores"]
        for key in (
            "provenance_directness",
            "authorship_transparency",
            "methodology_transparency",
            "publication_controls",
            "citation_traceability",
            "recency_for_claim",
        ):
            scores[key] = min(int(scores[key]), 1)
        rationale = list(payload.get("rationale", []))
        rationale.append(
            "Only a search snippet was available, so unverifiable quality dimensions were capped."
        )
        payload["rationale"] = rationale[:8]

    payload.update(
        {
            "source_type": source_type,
            "hard_flags": sorted(item.value for item in flags),
            "rubric_version": SOURCE_RUBRIC_VERSION,
        }
    )
    return SourceAssessment.model_validate(payload)
