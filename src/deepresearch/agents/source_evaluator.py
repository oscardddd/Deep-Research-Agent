from __future__ import annotations

import json

from ..config import Settings
from ..model_gateway import ModelGateway
from ..schemas import ResearchTask, SearchResult, SourceAssessmentBatch
from ..source_scoring import SOURCE_RUBRIC_VERSION, SOURCE_SCORING_GUIDE
from .base import BaseDeepSeekAgent


class SourceEvaluatorAgent(BaseDeepSeekAgent):
    """Scores retrieved sources with a shared, auditable rubric."""

    def __init__(self, settings: Settings, gateway: ModelGateway | None = None):
        super().__init__(settings, gateway)
        self.model = settings.fast_model
        self.max_content_chars = min(settings.page_chunk_chars, 8000)

    def assess_sources(
        self,
        task: ResearchTask,
        results: list[SearchResult],
    ) -> SourceAssessmentBatch:
        system = f"""
You are a source evaluator, separate from the evidence extractor. Return JSON
only. Evaluate every supplied source relative to the assigned research task.
Do not decide whether the task's conclusion is true. Do not reward agreement
with the expected answer. Use rubric {SOURCE_RUBRIC_VERSION}.

{SOURCE_SCORING_GUIDE}

Return one assessment per supplied source_rank. Do not calculate the final
tier, total scores, or eligibility; deterministic code derives those fields.
Copy an attributed original URL only when it visibly appears in the content.
Never invent bibliographic metadata or a URL.
Allowed hard_flags values are: ai_synthesis, social_content, content_farm,
anonymous_unsourced, attributed_only, inaccessible, irrelevant, and
possible_retraction. Allowed source_directness values are direct, attributed,
and unclear. Allowed source_type values are official, primary_research,
direct_interview, reputable_secondary, secondary, reference, aggregator,
social, and ai_synthesis.

Required JSON shape:
{{
  "assessments": [{{
    "source_rank": 1,
    "source_type": "primary_research",
    "source_directness": "direct",
    "scores": {{
      "provenance_directness": 3,
      "authority_for_claim": 3,
      "authorship_transparency": 2,
      "methodology_transparency": 3,
      "publication_controls": 2,
      "citation_traceability": 3,
      "recency_for_claim": 2,
      "independence": 2,
      "claim_relevance": 3
    }},
    "hard_flags": [],
    "attributed_source_name": null,
    "attributed_source_url": null,
    "follow_up_query": null,
    "rationale": ["short observable reason", "another reason"]
  }}]
}}
""".strip()
        user = json.dumps(
            {
                "task": task.model_dump(mode="json"),
                "sources": [
                    {
                        "source_rank": index,
                        "title": result.title,
                        "url": result.url,
                        "content_source": result.content_source,
                        "content": result.content[: self.max_content_chars],
                        "content_truncated": len(result.content)
                        > self.max_content_chars,
                    }
                    for index, result in enumerate(results, start=1)
                ],
            },
            ensure_ascii=False,
        )
        return self._json_completion(
            model=self.model,
            system_prompt=system,
            user_prompt=user,
            schema=SourceAssessmentBatch,
            max_tokens=4200,
            profile="lightweight_extraction",
        )
