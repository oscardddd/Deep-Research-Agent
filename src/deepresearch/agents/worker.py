from __future__ import annotations

import json
import math
import re
from bisect import bisect_left, bisect_right
from collections.abc import Callable
from collections import Counter
from dataclasses import dataclass

from ..config import Settings
from ..embeddings import EmbeddingProvider, OpenAICompatibleEmbeddingProvider
from ..eventlog import log_step
from ..model_gateway import ModelGateway
from ..schemas import (
    EvidenceBatch,
    EvidenceRecord,
    ExtractedEvidence,
    QueryPlan,
    QuerySpec,
    ResearchTask,
    SearchResult,
    WorkerDecision,
)
from .base import BaseDeepSeekAgent


_TOKEN_RE = re.compile(r"[a-z0-9][a-z0-9'-]*|[\u4e00-\u9fff]+", re.IGNORECASE)
_STOP_WORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
    "how", "in", "is", "it", "of", "on", "or", "that", "the", "this",
    "to", "was", "were", "what", "when", "where", "which", "who", "why",
    "with", "vs", "versus",
}


@dataclass(frozen=True)
class RankedPageChunk:
    """A page chunk selected locally before paid model inference."""

    original_index: int
    total_chunks: int
    content: str
    relevance_score: float
    retrieval_indexes: tuple[int, ...] = ()
    start_offset: int = 0
    end_offset: int = 0
    bm25_score: float = 0.0
    embedding_score: float | None = None


@dataclass(frozen=True)
class PageRetrievalChunk:
    original_index: int
    total_chunks: int
    content: str
    start_offset: int
    end_offset: int


@dataclass(frozen=True)
class _ScoredPageChunk:
    chunk: PageRetrievalChunk
    bm25_score: float
    embedding_score: float | None
    relevance_score: float


def estimate_text_tokens(text: str) -> int:
    """Dependency-free conservative estimate for multilingual chunk sizing."""
    cjk = sum("\u4e00" <= char <= "\u9fff" for char in text)
    non_cjk_chars = len(text) - cjk
    return max(1, cjk + (non_cjk_chars + 3) // 4)


def _token_unit_prefix(text: str) -> list[int]:
    prefix = [0]
    for char in text:
        prefix.append(prefix[-1] + (4 if "\u4e00" <= char <= "\u9fff" else 1))
    return prefix


def _span_tokens(prefix: list[int], start: int, end: int) -> int:
    return max(1, (prefix[end] - prefix[start] + 3) // 4)


def _semantic_boundaries(text: str) -> list[int]:
    boundaries = {0, len(text)}
    for match in re.finditer(r"\n\s*\n|\n|(?<=[.!?。！？])\s+", text):
        boundaries.add(match.end())
    return sorted(boundaries)


def _furthest_end(
    text: str, start: int, token_limit: int, boundaries: list[int],
    token_prefix: list[int],
) -> int:
    first = bisect_right(boundaries, start)
    low, high = first, len(boundaries) - 1
    best: int | None = None
    while low <= high:
        middle = (low + high) // 2
        boundary = boundaries[middle]
        if _span_tokens(token_prefix, start, boundary) <= token_limit:
            best = boundary
            low = middle + 1
        else:
            high = middle - 1
    if best is not None:
        return best
    low, high = start + 1, len(text)
    while low < high:
        middle = (low + high + 1) // 2
        if _span_tokens(token_prefix, start, middle) <= token_limit:
            low = middle
        else:
            high = middle - 1
    return low


def _overlap_start(
    text: str, previous_start: int, end: int, overlap_tokens: int,
    boundaries: list[int], token_prefix: list[int],
) -> int:
    if overlap_tokens <= 0:
        return end
    first = bisect_right(boundaries, previous_start)
    last = bisect_left(boundaries, end)
    candidates = boundaries[first:last]
    if not candidates:
        low, high = previous_start + 1, end
        while low < high:
            middle = (low + high) // 2
            if _span_tokens(token_prefix, middle, end) > overlap_tokens:
                low = middle + 1
            else:
                high = middle
        return low
    target_units = token_prefix[end] - overlap_tokens * 4
    candidate_units = [token_prefix[boundary] for boundary in candidates]
    position = bisect_left(candidate_units, target_units)
    nearby = candidates[max(0, position - 1):min(len(candidates), position + 2)]
    return min(nearby, key=lambda boundary: abs(
        _span_tokens(token_prefix, boundary, end) - overlap_tokens
    ))


def split_retrieval_chunks(
    content: str, *, max_tokens: int = 500, overlap_tokens: int = 75
) -> list[PageRetrievalChunk]:
    """Create paragraph/sentence-aware child chunks with stable source offsets."""
    if not content:
        return []
    if overlap_tokens >= max_tokens:
        overlap_tokens = max_tokens // 5
    boundaries = _semantic_boundaries(content)
    token_prefix = _token_unit_prefix(content)
    spans: list[tuple[int, int]] = []
    start = 0
    while start < len(content):
        end = _furthest_end(
            content, start, max_tokens, boundaries, token_prefix
        )
        if end <= start:
            end = min(len(content), start + 1)
        spans.append((start, end))
        if end >= len(content):
            break
        next_start = _overlap_start(
            content, start, end, overlap_tokens, boundaries, token_prefix
        )
        if next_start <= start or next_start >= end:
            next_start = end
        start = next_start
    total = len(spans)
    return [
        PageRetrievalChunk(index, total, content[start:end], start, end)
        for index, (start, end) in enumerate(spans, start=1)
    ]


def _tokenize_for_ranking(text: str) -> list[str]:
    tokens: list[str] = []
    for match in _TOKEN_RE.findall(text.casefold()):
        if re.fullmatch(r"[\u4e00-\u9fff]+", match):
            if len(match) == 1:
                tokens.append(match)
            else:
                tokens.extend(match[index : index + 2] for index in range(len(match) - 1))
            continue
        if match not in _STOP_WORDS and (len(match) > 1 or match.isdigit()):
            tokens.append(match)
    return tokens


def _task_ranking_terms(
    task: ResearchTask,
    *,
    title: str,
    search_query: str | None,
) -> tuple[Counter[str], list[str], set[str]]:
    weighted_fields = [
        (task.question, 2),
        (task.objective or "", 3),
        (task.query_hint or "", 3),
        (search_query or "", 4),
        (title, 1),
    ]
    weighted_fields.extend((value, 4) for value in task.must_find)
    weighted_fields.extend((value, 3) for value in task.covered_dimensions)

    terms: Counter[str] = Counter()
    for text, weight in weighted_fields:
        for token in _tokenize_for_ranking(text):
            terms[token] += weight

    phrases = [
        value.casefold().strip()
        for value in [search_query or "", *task.must_find, *task.covered_dimensions]
        if len(value.strip()) >= 4
    ]
    avoided = {
        token
        for value in task.avoid
        for token in _tokenize_for_ranking(value)
    }
    return terms, phrases, avoided


def _embedding_query_text(
    task: ResearchTask,
    *,
    title: str,
    search_query: str | None,
) -> str:
    parts = [
        task.question,
        task.objective or "",
        task.query_hint or "",
        search_query or "",
        title,
        *task.must_find,
        *task.covered_dimensions,
    ]
    query = "\n".join(part.strip() for part in parts if part.strip())
    return (
        "Instruct: Retrieve passages that contain evidence for this research "
        f"task.\nQuery: {query}"
    )


def _cosine_similarity(left: list[float], right: list[float]) -> float:
    if not left or len(left) != len(right):
        raise ValueError("Embedding vectors must have the same non-zero dimension")
    left_norm = math.sqrt(sum(value * value for value in left))
    right_norm = math.sqrt(sum(value * value for value in right))
    if left_norm == 0 or right_norm == 0:
        raise ValueError("Embedding vectors must have non-zero magnitude")
    return sum(a * b for a, b in zip(left, right)) / (left_norm * right_norm)


def _descending_ranks(scores: list[float]) -> list[int]:
    """Return one-based competition ranks, giving equal scores equal ranks."""
    ordered = sorted(range(len(scores)), key=lambda index: (-scores[index], index))
    ranks = [0] * len(scores)
    previous_score: float | None = None
    current_rank = 0
    for position, index in enumerate(ordered, start=1):
        score = scores[index]
        if previous_score is None or not math.isclose(score, previous_score):
            current_rank = position
            previous_score = score
        ranks[index] = current_rank
    return ranks


def _weighted_rrf_scores(
    bm25_scores: list[float],
    embedding_scores: list[float],
    *,
    embedding_weight: float,
    rrf_k: int,
) -> list[float]:
    """Fuse lexical and semantic result ranks with weighted RRF."""
    if len(bm25_scores) != len(embedding_scores):
        raise ValueError("BM25 and embedding score counts must match")
    weight = min(1.0, max(0.0, embedding_weight))
    lexical_weight = 1.0 - weight
    offset = max(1, rrf_k)
    bm25_ranks = _descending_ranks(bm25_scores)
    embedding_ranks = _descending_ranks(embedding_scores)
    return [
        round(
            lexical_weight / (offset + bm25_rank)
            + weight / (offset + embedding_rank),
            8,
        )
        for bm25_rank, embedding_rank in zip(bm25_ranks, embedding_ranks)
    ]


def rank_page_chunks(
    content: str,
    *,
    task: ResearchTask,
    title: str = "",
    search_query: str | None = None,
    max_chars: int | None = None,
    max_chunks: int = 4,
    retrieval_chunk_tokens: int = 500,
    retrieval_chunk_overlap_tokens: int = 75,
    extraction_window_tokens: int = 1800,
    max_windows: int = 2,
    embed_texts: Callable[[list[str]], list[list[float]]] | None = None,
    embedding_weight: float = 0.35,
    rrf_k: int = 60,
) -> list[RankedPageChunk]:
    """Rank child chunks with BM25 or weighted RRF, then expand hits."""
    chunks = split_retrieval_chunks(
        content,
        max_tokens=retrieval_chunk_tokens,
        overlap_tokens=retrieval_chunk_overlap_tokens,
    )
    if not chunks:
        return []

    query_terms, phrases, avoided_terms = _task_ranking_terms(
        task,
        title=title,
        search_query=search_query,
    )
    tokenized_chunks = [_tokenize_for_ranking(chunk.content) for chunk in chunks]
    frequencies = [Counter(tokens) for tokens in tokenized_chunks]
    average_length = sum(map(len, tokenized_chunks)) / max(1, len(tokenized_chunks))
    document_frequency = Counter(
        token for frequencies_for_chunk in frequencies for token in frequencies_for_chunk
    )
    chunk_count = len(chunks)

    bm25_scores: list[float] = []
    for chunk, tokens, token_counts in zip(chunks, tokenized_chunks, frequencies):
        score = 0.0
        length_normalizer = 1.0 - 0.75 + 0.75 * (
            len(tokens) / max(1.0, average_length)
        )
        for token, query_weight in query_terms.items():
            frequency = token_counts.get(token, 0)
            if not frequency:
                continue
            inverse_frequency = math.log(
                1.0
                + (chunk_count - document_frequency[token] + 0.5)
                / (document_frequency[token] + 0.5)
            )
            score += query_weight * inverse_frequency * (
                frequency * 2.2 / (frequency + 1.2 * length_normalizer)
            )
        folded_chunk = chunk.content.casefold()
        score += 2.0 * sum(phrase in folded_chunk for phrase in phrases)
        score -= 0.35 * sum(token_counts.get(token, 0) for token in avoided_terms)
        bm25_scores.append(round(score, 6))

    dense_scores: list[float | None] = [None] * len(chunks)
    final_scores = list(bm25_scores)
    weight = min(1.0, max(0.0, embedding_weight))
    if embed_texts is not None and weight > 0:
        vectors = embed_texts([
            _embedding_query_text(task, title=title, search_query=search_query),
            *(chunk.content for chunk in chunks),
        ])
        if len(vectors) != len(chunks) + 1:
            raise ValueError("Embedding response count does not match query and chunks")
        query_vector = vectors[0]
        embedding_scores = [
            _cosine_similarity(query_vector, vector) for vector in vectors[1:]
        ]
        dense_scores = embedding_scores
        final_scores = _weighted_rrf_scores(
            bm25_scores,
            embedding_scores,
            embedding_weight=weight,
            rrf_k=rrf_k,
        )

    scored = [
        _ScoredPageChunk(
            chunk=chunk,
            bm25_score=bm25_score,
            embedding_score=dense_score,
            relevance_score=relevance_score,
        )
        for chunk, bm25_score, dense_score, relevance_score in zip(
            chunks, bm25_scores, dense_scores, final_scores
        )
    ]

    ordered = sorted(
        scored,
        key=lambda item: (-item.relevance_score, item.chunk.original_index),
    )
    selected: list[_ScoredPageChunk] = []
    for candidate in ordered:
        child = candidate.chunk
        # Overlapping child chunks usually describe the same passage. Do not
        # let duplicate windows consume every retrieval slot.
        if any(
            child.start_offset < existing.end_offset
            and existing.start_offset < child.end_offset
            for existing in (item.chunk for item in selected)
        ):
            continue
        selected.append(candidate)
        if len(selected) >= max_chunks:
            break
    return _expanded_extraction_windows(
        content,
        selected,
        token_limit=extraction_window_tokens,
        max_windows=max_windows,
    )


def _expand_child_span(
    content: str,
    child: PageRetrievalChunk,
    *,
    token_limit: int,
) -> tuple[int, int]:
    """Center a larger, contiguous source window around one retrieval child."""
    if estimate_text_tokens(content) <= token_limit:
        return 0, len(content)
    boundaries = _semantic_boundaries(content)
    remaining = max(0, token_limit - estimate_text_tokens(child.content))
    left_budget = remaining // 2
    right_budget = remaining - left_budget

    left_candidates = [boundary for boundary in boundaries if boundary < child.start_offset]
    start = child.start_offset
    for boundary in reversed(left_candidates):
        if estimate_text_tokens(content[boundary:child.start_offset]) <= left_budget:
            start = boundary
        else:
            break
    right_candidates = [boundary for boundary in boundaries if boundary > child.end_offset]
    end = child.end_offset
    for boundary in right_candidates:
        if estimate_text_tokens(content[child.end_offset:boundary]) <= right_budget:
            end = boundary
        else:
            break
    return start, end


def _expanded_extraction_windows(
    content: str,
    selected: list[_ScoredPageChunk],
    *,
    token_limit: int,
    max_windows: int,
) -> list[RankedPageChunk]:
    # Cluster neighboring child hits before expansion. Expanding each hit first
    # would create large, redundant windows that are difficult to merge.
    clusters: list[dict[str, object]] = []
    for scored in sorted(selected, key=lambda item: item.chunk.start_offset):
        child = scored.chunk
        if clusters:
            previous = clusters[-1]
            gap = content[int(previous["end"]):child.start_offset]
            union_start = int(previous["start"])
            union_end = max(int(previous["end"]), child.end_offset)
            if (
                estimate_text_tokens(gap) <= 100
                and estimate_text_tokens(content[union_start:union_end]) <= token_limit
            ):
                previous["end"] = max(int(previous["end"]), child.end_offset)
                previous["score"] = max(
                    float(previous["score"]), scored.relevance_score
                )
                previous["bm25_score"] = max(
                    float(previous["bm25_score"]), scored.bm25_score
                )
                if scored.embedding_score is not None:
                    prior_dense = previous["embedding_score"]
                    previous["embedding_score"] = max(
                        float(prior_dense) if prior_dense is not None else -1.0,
                        scored.embedding_score,
                    )
                previous["indexes"] = set(previous["indexes"]) | {child.original_index}
                continue
        clusters.append({
            "start": child.start_offset,
            "end": child.end_offset,
            "score": scored.relevance_score,
            "bm25_score": scored.bm25_score,
            "embedding_score": scored.embedding_score,
            "indexes": {child.original_index},
            "total": child.total_chunks,
        })

    merged: list[dict[str, object]] = []
    for cluster in clusters:
        cluster_start = int(cluster["start"])
        cluster_end = int(cluster["end"])
        representative = PageRetrievalChunk(
            original_index=min(set(cluster["indexes"])),
            total_chunks=int(cluster["total"]),
            content=content[cluster_start:cluster_end],
            start_offset=cluster_start,
            end_offset=cluster_end,
        )
        start, end = _expand_child_span(
            content, representative, token_limit=token_limit
        )
        merged.append({**cluster, "start": start, "end": end})

    strongest = sorted(
        merged,
        key=lambda item: (-float(item["score"]), int(item["start"])),
    )[:max_windows]
    return [
        RankedPageChunk(
            original_index=min(set(item["indexes"])),
            total_chunks=int(item["total"]),
            content=content[int(item["start"]):int(item["end"])],
            relevance_score=float(item["score"]),
            bm25_score=float(item["bm25_score"]),
            embedding_score=(
                float(item["embedding_score"])
                if item["embedding_score"] is not None
                else None
            ),
            retrieval_indexes=tuple(sorted(set(item["indexes"]))),
            start_offset=int(item["start"]),
            end_offset=int(item["end"]),
        )
        for item in strongest
    ]


def split_page_content(content: str, max_chars: int) -> list[str]:
    """Partition a page without dropping text; excerpts remain source substrings."""
    if not content:
        return []
    if len(content) <= max_chars:
        return [content]

    chunks: list[str] = []
    start = 0
    while start < len(content):
        end = min(start + max_chars, len(content))
        if end < len(content):
            search_start = start + max_chars // 2
            paragraph_end = content.rfind("\n\n", search_start, end)
            line_end = content.rfind("\n", search_start, end)
            boundary = max(paragraph_end + 2, line_end + 1)
            if boundary > start:
                end = boundary
        chunks.append(content[start:end])
        start = end
    return chunks


class ResearchWorkerAgent(BaseDeepSeekAgent):
    """Forms queries, extracts evidence, and controls adaptive follow-ups."""

    def __init__(
        self,
        settings: Settings,
        gateway: ModelGateway | None = None,
        embedding_provider: EmbeddingProvider | None = None,
    ):
        super().__init__(settings, gateway)
        self.model = settings.fast_model
        self.page_chunk_chars = settings.page_chunk_chars
        self.retrieval_chunk_tokens = settings.retrieval_chunk_tokens
        self.retrieval_chunk_overlap_tokens = settings.retrieval_chunk_overlap_tokens
        self.extraction_window_tokens = settings.extraction_window_tokens
        self.max_relevant_chunks_per_source = settings.max_relevant_chunks_per_source
        self.max_extraction_windows_per_source = (
            settings.max_extraction_windows_per_source
        )
        self.hybrid_embedding_weight = settings.hybrid_embedding_weight
        self.hybrid_rrf_k = settings.hybrid_rrf_k
        self.embedding_provider = embedding_provider
        if self.embedding_provider is None and settings.hybrid_retrieval_enabled:
            self.embedding_provider = OpenAICompatibleEmbeddingProvider(
                base_url=settings.embedding_base_url or "",
                api_key=settings.embedding_api_key,
                model=settings.embedding_model,
                dimensions=settings.embedding_dimensions,
                batch_size=settings.embedding_batch_size,
                timeout=settings.request_timeout_seconds,
            )

    def formulate_queries(
        self, task: ResearchTask, max_candidates: int
    ) -> QueryPlan:
        system = """
You are a bounded research worker. Return JSON only. Based on your assigned
research role, objective, evidence requirements, and exclusions, formulate
search query candidates with meaningfully different intents.

Keep the original question and the task's covered_dimensions in view. Search
for conditional findings and applicability boundaries, not merely documents
that mention the topic.

Rules:
- Return no more candidates than requested.
- Order candidates from highest to lowest expected research value.
- Use priority 3 for the most valuable query and 1 for the least valuable.
- Prefer primary evidence when the task asks for it.
- A challenge query should actively seek limitations or counterevidence.
- The only currently available provider is tavily.
- If query_hint is present, use it as the highest-priority candidate, refining
  wording only when necessary.

Required JSON shape:
{
  "queries": [
    {
      "intent": "primary_evidence",
      "query": "concise standalone web search query",
      "expected_evidence": "what useful evidence this query should retrieve",
      "priority": 3,
      "provider": "tavily"
    }
  ]
}
Allowed intent values: overview, primary_evidence, challenge, implementation,
verification. Do not include markdown or commentary.
""".strip()
        user = json.dumps(
            {
                "task": task.model_dump(mode="json"),
                "maximum_candidates": max_candidates,
            },
            ensure_ascii=False,
        )
        return self._json_completion(
            model=self.model,
            system_prompt=system,
            user_prompt=user,
            schema=QueryPlan,
            max_tokens=1400,
            profile=task.execution_profile,
        )

    def extract(
        self, task: ResearchTask, results: list[SearchResult]
    ) -> EvidenceBatch:
        system = """
You are an evidence extractor. Return JSON only. Extract atomic evidence for
the assigned research task from the supplied full-page content chunk.

Critical rules:
- verbatim_excerpt must be copied exactly and contiguously from content.
- Never add facts that are absent from content.
- source_rank is the one-based rank supplied with the source.
- stance is supports, contradicts, or neutral relative to the task question.
- Omit irrelevant or content-free results.
- Return only the strongest task-relevant atomic evidence in this chunk, with
  no more than 4 items. Avoid navigation text, repeated facts, and tangential
  claims even when they are technically related.
- A content chunk is part of the page at its URL; do not infer facts from chunk
  boundaries or assume omitted neighboring text.
- Judge whether the page is the direct origin of the information or merely
  attributes it to another publication, study, interview, dataset, or official
  record. An exact quote from a secondary page proves only that the page made
  the statement; it does not make the underlying fact primary evidence.
- AI-generated reports, social posts, SEO aggregators, and unsourced opinion
  pages are discovery leads, not final supporting evidence.
- Assign credibility conservatively: A for an authoritative official record or
  clearly identified original high-quality research; B for direct evidence
  from a credible institution/publication with transparent authorship and
  methods; C for ordinary secondary explanation or unclear provenance; D for
  AI synthesis, anonymous/unsourced opinion, social content, or content-farm
  aggregation. A polished writing style is not evidence of credibility.
- When an attributed original URL is present in the supplied content, copy it
  into attributed_source_url. Otherwise provide a concise follow_up_query that
  could locate the named original source. Never invent a URL.
- Assign dimension_ids using only exact strings from the task's
  covered_dimensions. Include every dimension directly addressed by the
  evidence and do not tag a merely tangential dimension.

Required JSON shape:
{
  "evidence": [
    {
      "source_rank": 1,
      "claim_candidate": "atomic claim supported by the excerpt",
      "verbatim_excerpt": "exact quote from content",
      "stance": "supports",
      "relevance": "high",
      "source_type": "primary_research",
      "source_directness": "direct",
      "credibility_tier": "A",
      "discovery_only": false,
      "attributed_source_name": null,
      "attributed_source_url": null,
      "follow_up_query": null,
      "dimension_ids": ["exact task covered dimension"]
    }
  ]
}
""".strip()
        extracted: list[ExtractedEvidence] = []
        embed_texts = (
            self.embedding_provider.embed
            if self.embedding_provider is not None
            else None
        )
        for source_rank, result in enumerate(results, start=1):
            retrieval_mode = "hybrid_bm25_embedding_rrf" if embed_texts else "bm25"
            try:
                selected_chunks = rank_page_chunks(
                    result.content,
                    task=task,
                    title=result.title,
                    search_query=result.query,
                    max_chunks=self.max_relevant_chunks_per_source,
                    retrieval_chunk_tokens=self.retrieval_chunk_tokens,
                    retrieval_chunk_overlap_tokens=(
                        self.retrieval_chunk_overlap_tokens
                    ),
                    extraction_window_tokens=self.extraction_window_tokens,
                    max_windows=self.max_extraction_windows_per_source,
                    embed_texts=embed_texts,
                    embedding_weight=self.hybrid_embedding_weight,
                    rrf_k=self.hybrid_rrf_k,
                )
            except Exception as error:
                if embed_texts is None:
                    raise
                log_step(
                    f"ResearchWorker:{task.task_id}",
                    "chunks.embedding_fallback",
                    source_rank=source_rank,
                    url=result.url,
                    error=f"{type(error).__name__}: {error}",
                )
                embed_texts = None
                retrieval_mode = "bm25_fallback"
                selected_chunks = rank_page_chunks(
                    result.content,
                    task=task,
                    title=result.title,
                    search_query=result.query,
                    max_chunks=self.max_relevant_chunks_per_source,
                    retrieval_chunk_tokens=self.retrieval_chunk_tokens,
                    retrieval_chunk_overlap_tokens=(
                        self.retrieval_chunk_overlap_tokens
                    ),
                    extraction_window_tokens=self.extraction_window_tokens,
                    max_windows=self.max_extraction_windows_per_source,
                )
            log_step(
                f"ResearchWorker:{task.task_id}",
                "chunks.ranked",
                source_rank=source_rank,
                url=result.url,
                retrieval_mode=retrieval_mode,
                total_chunks=(selected_chunks[0].total_chunks if selected_chunks else 0),
                selected_chunks=len(selected_chunks),
                selected_indexes=[chunk.original_index for chunk in selected_chunks],
                retrieval_indexes=[
                    list(chunk.retrieval_indexes) for chunk in selected_chunks
                ],
                extraction_window_tokens=[
                    estimate_text_tokens(chunk.content) for chunk in selected_chunks
                ],
                relevance_scores=[chunk.relevance_score for chunk in selected_chunks],
                bm25_scores=[chunk.bm25_score for chunk in selected_chunks],
                embedding_scores=[
                    chunk.embedding_score for chunk in selected_chunks
                ],
            )
            for selected_rank, selected_chunk in enumerate(selected_chunks, start=1):
                user = json.dumps(
                    {
                        "task": task.model_dump(mode="json"),
                        "sources": [
                            {
                                "source_rank": source_rank,
                                "title": result.title,
                                "url": result.url,
                                "content_source": result.content_source,
                                "chunk_index": selected_chunk.original_index,
                                "retrieval_chunk_indexes": list(
                                    selected_chunk.retrieval_indexes
                                ),
                                "total_chunks": selected_chunk.total_chunks,
                                "source_start_offset": selected_chunk.start_offset,
                                "source_end_offset": selected_chunk.end_offset,
                                "selected_chunk_rank": selected_rank,
                                "chunks_selected": len(selected_chunks),
                                "local_relevance_score": selected_chunk.relevance_score,
                                "content": selected_chunk.content,
                            }
                        ],
                    },
                    ensure_ascii=False,
                )
                batch = self._json_completion(
                    model=self.model,
                    system_prompt=system,
                    user_prompt=user,
                    schema=EvidenceBatch,
                    max_tokens=5000,
                    profile=task.execution_profile,
                )
                extracted.extend(batch.evidence)
        return EvidenceBatch(evidence=extracted)

    def decide_worker_next_step(
        self,
        task: ResearchTask,
        executed_queries: list[QuerySpec],
        evidence: list[EvidenceRecord],
        step_number: int,
        max_steps: int,
        search_errors: list[str],
    ) -> WorkerDecision:
        system = """
You control one evidence-first research worker. Return JSON only. Review the
worker's objective, evidence requirements, queries already executed, accepted
atomic evidence, source diversity, and errors. Decide whether to run one more
targeted search or stop this worker.

Rules:
- Search only when a concrete unresolved evidence gap remains.
- The next query must be atomic, standalone, materially different from all
  executed queries, and targeted at the highest-value unresolved gap.
- Use challenge or verification intent for conflicts and weak corroboration.
- Stop as sufficient only when the task's must_find and success_criteria are
  covered by accepted direct external evidence from sufficiently credible,
  independent provenance. A large number of excerpts from one page is still
  one source.
- Prefer following an attributed original source over issuing another broad
  search. Use follow_source only for an explicit URL present in accepted
  evidence; never invent a URL.
- Treat discovery_only evidence as a lead, not as support for sufficiency.
- If one URL dominates the evidence, search for independent verification and
  exclude that source from the next query where practical.
- Stop as saturated when another query has low expected marginal value.
- Stop as blocked when errors or unavailable evidence prevent useful progress.
- Parametric model knowledge may guide decomposition, but it is never evidence.
- The only available search provider is tavily.

Required JSON shape when continuing:
{
  "action": "search",
  "decision_summary": "why this is the next highest-value step",
  "unresolved_gaps": ["specific open evidence gap"],
  "next_query": {
    "intent": "verification",
    "query": "standalone targeted web search query",
    "expected_evidence": "evidence that would close the gap",
    "priority": 3,
    "provider": "tavily"
  },
  "stop_reason": null
}

Required JSON shape when following an explicit original source:
{
  "action": "follow_source",
  "decision_summary": "why the attributed original is higher value",
  "unresolved_gaps": ["claim still relies on a secondary attribution"],
  "next_query": null,
  "target_source_url": "https://original.example/source",
  "stop_reason": null
}

Required JSON shape when stopping:
{
  "action": "stop",
  "decision_summary": "why further search is unnecessary or low value",
  "unresolved_gaps": [],
  "next_query": null,
  "target_source_url": null,
  "stop_reason": "sufficient"
}
Allowed action values: search, follow_source, stop.
Allowed stop_reason values: sufficient, saturated, blocked.
Do not include markdown or hidden chain-of-thought.
""".strip()
        user = json.dumps(
            {
                "task": task.model_dump(mode="json"),
                "step_number": step_number,
                "maximum_steps": max_steps,
                "executed_queries": [
                    item.model_dump(mode="json") for item in executed_queries
                ],
                "accepted_evidence": [
                    item.model_dump(mode="json") for item in evidence
                ],
                "source_domains": sorted(
                    {item.source_domain for item in evidence}
                ),
                "source_url_counts": {
                    url: sum(item.source_url == url for item in evidence)
                    for url in sorted({item.source_url for item in evidence})
                },
                "attributed_source_urls": sorted(
                    {
                        item.attributed_source_url
                        for item in evidence
                        if item.attributed_source_url
                    }
                ),
                "search_errors": search_errors,
            },
            ensure_ascii=False,
        )
        return self._json_completion(
            model=self.model,
            system_prompt=system,
            user_prompt=user,
            schema=WorkerDecision,
            max_tokens=1400,
            profile=task.execution_profile,
        )
