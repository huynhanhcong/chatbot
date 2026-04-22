from typing import Any

from .bm25_store import BM25Store
from .entity_linker import EntityLink, normalize_for_linking
from .embeddings import GeminiEmbedder
from .models import SearchResult
from .text import tokenize_vi
from .vector_store import QdrantVectorStore


def hybrid_retrieve(
    *,
    query: str,
    embedder: GeminiEmbedder,
    vector_store: QdrantVectorStore,
    bm25_store: BM25Store,
    bm25_top_k: int = 30,
    vector_top_k: int = 30,
    candidate_limit: int = 10,
    entity_links: list[EntityLink] | None = None,
) -> list[SearchResult]:
    query_vector = embedder.embed_text(query)
    vector_results = vector_store.search(query_vector, limit=vector_top_k)
    bm25_results = bm25_store.search(query, limit=bm25_top_k)
    return rrf_fuse(
        vector_results=vector_results,
        bm25_results=bm25_results,
        entity_links=entity_links or [],
        candidate_limit=candidate_limit,
    )


def rrf_fuse(
    *,
    vector_results: list[SearchResult],
    bm25_results: list[SearchResult],
    entity_links: list[EntityLink] | None = None,
    candidate_limit: int = 10,
    rrf_k: int = 60,
) -> list[SearchResult]:
    merged: dict[str, SearchResult] = {}
    rrf_scores: dict[str, float] = {}

    for ranked_results, score_attr in (
        (vector_results, "vector_score"),
        (bm25_results, "bm25_score"),
    ):
        for rank, result in enumerate(ranked_results, start=1):
            current = merged.get(result.id)
            if current is None:
                merged[result.id] = result.model_copy(deep=True)
                current = merged[result.id]
            setattr(current, score_attr, max(float(getattr(current, score_attr)), float(getattr(result, score_attr))))
            rrf_scores[result.id] = rrf_scores.get(result.id, 0.0) + (1.0 / (rrf_k + rank))

    links = entity_links or []
    for result_id, result in merged.items():
        source_boost = 0.02 if result.payload.get("source") == "hanhphuc" else 0.0
        result.score = rrf_scores.get(result_id, 0.0) + source_boost + _entity_boost(result, links)
    return sorted(merged.values(), key=lambda item: item.score, reverse=True)[:candidate_limit]


def diversify_results(
    query: str,
    candidates: list[SearchResult],
    *,
    top_k: int = 5,
    max_per_parent: int = 2,
    diversity_weight: float = 0.28,
) -> list[SearchResult]:
    if not candidates:
        return []

    query_tokens = set(tokenize_vi(normalize_for_linking(query)) or tokenize_vi(query))
    selected: list[SearchResult] = []
    selected_tokens: list[set[str]] = []
    parent_counts: dict[str, int] = {}
    pool = [candidate.model_copy(deep=True) for candidate in candidates]

    while pool and len(selected) < top_k:
        best_index = -1
        best_score = float("-inf")
        for index, candidate in enumerate(pool):
            parent_id = candidate.parent_id or candidate.id
            if parent_counts.get(parent_id, 0) >= max_per_parent:
                continue
            candidate_tokens = set(tokenize_vi(normalize_for_linking(candidate.text or candidate.title)))
            relevance = candidate.score + _query_overlap(query_tokens, candidate_tokens) * 0.04
            redundancy = max((_jaccard(candidate_tokens, tokens) for tokens in selected_tokens), default=0.0)
            mmr_score = relevance - (diversity_weight * redundancy)
            if mmr_score > best_score:
                best_index = index
                best_score = mmr_score

        if best_index == -1:
            break
        chosen = pool.pop(best_index)
        selected.append(chosen)
        selected_tokens.append(set(tokenize_vi(normalize_for_linking(chosen.text or chosen.title))))
        parent_id = chosen.parent_id or chosen.id
        parent_counts[parent_id] = parent_counts.get(parent_id, 0) + 1

    return selected or candidates[:top_k]


def _entity_boost(result: SearchResult, links: list[EntityLink]) -> float:
    boost = 0.0
    result_parent = result.parent_id or result.id
    result_category = normalize_for_linking(result.category or "")
    for link in links:
        if link.entity_id in {result.id, result_parent}:
            boost += 0.22 * link.confidence
        elif link.entity_type == result.entity_type:
            boost += 0.025 * link.confidence
        if link.category and result_category:
            link_category = normalize_for_linking(link.category)
            if link_category and (link_category in result_category or result_category in link_category):
                boost += 0.03 * link.confidence
        if link.price_intent and result.price_vnd:
            boost += 0.02
    return min(boost, 0.35)


def _query_overlap(query_tokens: set[str], candidate_tokens: set[str]) -> float:
    if not query_tokens or not candidate_tokens:
        return 0.0
    return len(query_tokens & candidate_tokens) / len(query_tokens)


def _jaccard(left: set[Any], right: set[Any]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)
