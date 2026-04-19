from .bm25_store import BM25Store
from .embeddings import GeminiEmbedder
from .models import SearchResult
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
) -> list[SearchResult]:
    query_vector = embedder.embed_text(query)
    vector_results = vector_store.search(query_vector, limit=vector_top_k)
    bm25_results = bm25_store.search(query, limit=bm25_top_k)

    merged: dict[str, SearchResult] = {}
    for result in vector_results:
        current = merged.get(result.id)
        if current:
            current.vector_score = max(current.vector_score, result.vector_score)
        else:
            merged[result.id] = result

    for result in bm25_results:
        current = merged.get(result.id)
        if current:
            current.bm25_score = max(current.bm25_score, result.bm25_score)
        else:
            merged[result.id] = result

    for result in merged.values():
        source_boost = 0.05 if result.payload.get("source") == "hanhphuc" else 0.0
        result.score = (0.62 * result.vector_score) + (0.38 * result.bm25_score) + source_boost

    return sorted(merged.values(), key=lambda item: item.score, reverse=True)[:candidate_limit]
