from __future__ import annotations

from typing import Protocol

import httpx

from .agents import (
    DeepSeekModelProvider,
    estimate_text_tokens,
    FakeLanguageModelProvider,
    split_page_content,
    rank_page_chunks,
    split_retrieval_chunks,
)
from .config import Settings
from .schemas import (
    AuditResult,
    EvidenceBatch,
    EvidenceRecord,
    ExtractedPage,
    PageExtractionBatch,
    QueryPlan,
    QuerySpec,
    ReportDraft,
    ReplanDecision,
    ResearchStateDigest,
    ResearchPlan,
    ResearchTask,
    SearchResult,
    SourceAssessmentBatch,
    WorkerDecision,
)


class SearchProvider(Protocol):
    def search(self, query: str, max_results: int) -> list[SearchResult]: ...

    def extract_pages(self, urls: list[str]) -> PageExtractionBatch: ...


class LanguageModelProvider(Protocol):
    """Stable role facade consumed by ResearchRuntime."""

    def plan(self, question: str, max_tasks: int) -> ResearchPlan: ...

    def reconnaissance_queries(
        self, question: str, max_queries: int
    ) -> QueryPlan: ...

    def plan_with_context(
        self,
        question: str,
        max_tasks: int,
        preliminary_evidence: list[EvidenceRecord],
    ) -> ResearchPlan: ...

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
    ) -> ReplanDecision: ...

    def formulate_queries(
        self, task: ResearchTask, max_candidates: int
    ) -> QueryPlan: ...

    def assess_sources(
        self, task: ResearchTask, results: list[SearchResult]
    ) -> SourceAssessmentBatch: ...

    def extract(
        self, task: ResearchTask, results: list[SearchResult]
    ) -> EvidenceBatch: ...

    def decide_worker_next_step(
        self,
        task: ResearchTask,
        executed_queries: list[QuerySpec],
        evidence: list[EvidenceRecord],
        step_number: int,
        max_steps: int,
        search_errors: list[str],
    ) -> WorkerDecision: ...

    def audit(
        self,
        question: str,
        evidence: list[EvidenceRecord],
        round_number: int,
        research_context: dict | None = None,
    ) -> AuditResult: ...

    def synthesize_report(
        self,
        question: str,
        plan: ResearchPlan,
        audit: AuditResult,
        evidence: list[EvidenceRecord],
    ) -> ReportDraft: ...


class TavilySearchProvider:
    def __init__(self, settings: Settings):
        if not settings.tavily_api_key:
            raise RuntimeError("TAVILY_API_KEY is required")
        self.api_key = settings.tavily_api_key
        self.timeout = settings.request_timeout_seconds

    def search(self, query: str, max_results: int) -> list[SearchResult]:
        response = httpx.post(
            "https://api.tavily.com/search",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={
                "query": query,
                "search_depth": "basic",
                "max_results": max_results,
                "include_answer": False,
                "include_raw_content": False,
                "auto_parameters": False,
            },
            timeout=self.timeout,
        )
        response.raise_for_status()
        payload = response.json()
        return [
            SearchResult(
                title=item.get("title", "Untitled source"),
                url=item["url"],
                content=item.get("content", ""),
                snippet=item.get("content", ""),
                content_source="search_snippet",
                score=float(item.get("score", 0.0)),
                rank=index,
            )
            for index, item in enumerate(payload.get("results", []), start=1)
            if item.get("url") and item.get("content")
        ]

    def extract_pages(self, urls: list[str]) -> PageExtractionBatch:
        if not urls:
            return PageExtractionBatch()
        response = httpx.post(
            "https://api.tavily.com/extract",
            headers={"Authorization": f"Bearer {self.api_key}"},
            json={
                "urls": urls,
                "extract_depth": "basic",
                "format": "markdown",
                "include_images": False,
                "include_usage": True,
            },
            timeout=min(max(self.timeout, 1.0), 60.0),
        )
        response.raise_for_status()
        payload = response.json()
        failed_urls = []
        for item in payload.get("failed_results", []):
            if isinstance(item, str):
                failed_urls.append(item)
            elif isinstance(item, dict) and item.get("url"):
                failed_urls.append(str(item["url"]))
        return PageExtractionBatch(
            pages=[
                ExtractedPage(url=item["url"], raw_content=item["raw_content"])
                for item in payload.get("results", [])
                if item.get("url") and item.get("raw_content")
            ],
            failed_urls=failed_urls,
            credits_used=float(payload.get("usage", {}).get("credits", 0.0)),
        )


class FakeSearchProvider:
    """Deterministic provider for tests and a no-key local demo."""

    def __init__(self) -> None:
        self.call_count = 0
        self.extract_call_count = 0

    def search(self, query: str, max_results: int) -> list[SearchResult]:
        self.call_count += 1
        fixtures = [
            (
                "Primary study reports a measured benefit",
                "https://example.org/primary-study",
                "The controlled study reported a measurable benefit under its evaluated conditions.",
            ),
            (
                "Independent review identifies limitations",
                "https://review.example.net/limitations",
                "The review found that the evidence remains limited outside the evaluated settings.",
            ),
            (
                "Replication provides qualified support",
                "https://science.example.com/replication",
                "An independent replication observed a smaller but directionally similar effect.",
            ),
        ]
        return [
            SearchResult(
                title=title,
                url=url,
                content=content,
                score=0.9 - index * 0.1,
                rank=index + 1,
            )
            for index, (title, url, content) in enumerate(fixtures[:max_results])
        ]

    def extract_pages(self, urls: list[str]) -> PageExtractionBatch:
        self.extract_call_count += 1
        bodies = {
            "https://example.org/primary-study": (
                "The controlled study reported a measurable benefit under its evaluated conditions.\n\n"
                "The full page describes the study design and evaluated population."
            ),
            "https://review.example.net/limitations": (
                "The review found that the evidence remains limited outside the evaluated settings.\n\n"
                "The full page discusses external-validity limitations and open questions."
            ),
            "https://science.example.com/replication": (
                "An independent replication observed a smaller but directionally similar effect.\n\n"
                "The full page reports replication methods and qualified conclusions."
            ),
        }
        pages = [
            ExtractedPage(url=url, raw_content=bodies[url])
            for url in urls
            if url in bodies
        ]
        return PageExtractionBatch(
            pages=pages,
            failed_urls=[url for url in urls if url not in bodies],
        )


__all__ = [
    "DeepSeekModelProvider",
    "FakeLanguageModelProvider",
    "FakeSearchProvider",
    "LanguageModelProvider",
    "SearchProvider",
    "TavilySearchProvider",
    "split_page_content",
    "split_retrieval_chunks",
    "estimate_text_tokens",
    "rank_page_chunks",
]
