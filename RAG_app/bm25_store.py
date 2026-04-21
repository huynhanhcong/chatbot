import json
from pathlib import Path

from .models import RagDocument, SearchResult
from .text import tokenize_vi


class BM25Store:
    def __init__(self, docs: list[dict]):
        self.docs = docs
        self.tokens = [doc["tokens"] for doc in docs]
        self._bm25 = None
        try:
            from rank_bm25 import BM25Okapi

            self._bm25 = BM25Okapi(self.tokens)
        except ImportError:
            self._bm25 = None

    @classmethod
    def from_documents(cls, documents: list[RagDocument]) -> "BM25Store":
        return cls(
            [
                {
                    "id": doc.id,
                    "tokens": tokenize_vi(doc.search_text or doc.text),
                    "payload": doc.payload,
                }
                for doc in documents
            ]
        )

    @classmethod
    def load(cls, path: Path) -> "BM25Store":
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        return cls(data["documents"])

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            json.dump({"documents": self.docs}, handle, ensure_ascii=False)

    def search(self, query: str, limit: int = 30) -> list[SearchResult]:
        query_tokens = tokenize_vi(query)
        if not query_tokens:
            return []

        if self._bm25:
            scores = self._bm25.get_scores(query_tokens)
        else:
            scores = [_fallback_score(query_tokens, doc["tokens"]) for doc in self.docs]

        ranked = sorted(enumerate(scores), key=lambda item: float(item[1]), reverse=True)[:limit]
        max_score = max((float(score) for _, score in ranked), default=0.0) or 1.0
        results: list[SearchResult] = []
        for index, score in ranked:
            if score <= 0:
                continue
            payload = self.docs[index]["payload"]
            results.append(
                SearchResult(
                    id=payload["entity_id"],
                    parent_id=payload.get("parent_id"),
                    score=float(score) / max_score,
                    bm25_score=float(score) / max_score,
                    title=payload["title"],
                    entity_type=payload["entity_type"],
                    category=payload.get("category"),
                    chunk_type=payload.get("chunk_type"),
                    price_vnd=payload.get("price_vnd"),
                    source_url=payload.get("source_url"),
                    text=payload["text"],
                    payload=payload,
                )
            )
        return results


def _fallback_score(query_tokens: list[str], doc_tokens: list[str]) -> float:
    doc_counts = {}
    for token in doc_tokens:
        doc_counts[token] = doc_counts.get(token, 0) + 1
    return float(sum(doc_counts.get(token, 0) for token in query_tokens))
