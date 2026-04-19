from .bm25_store import BM25Store
from .config import Settings
from .embeddings import GeminiEmbedder
from .formatter import build_sources
from .gemini_client import GeminiTextClient
from .guardrails import apply_guardrails
from .intent import detect_intent
from .models import RagAnswer, SearchResult
from .retriever import hybrid_retrieve
from .vector_store import QdrantVectorStore


class RagPipeline:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.embedder = GeminiEmbedder(settings.gemini_api_key, settings.gemini_embedding_model)
        self.gemini = GeminiTextClient(settings.gemini_api_key, settings.gemini_generation_model)
        self.vector_store = QdrantVectorStore(
            settings.qdrant_url,
            settings.qdrant_collection,
            path=str(settings.qdrant_path) if settings.qdrant_path else None,
        )
        self.bm25_store = BM25Store.load(settings.bm25_path)

    def answer(self, question: str) -> RagAnswer:
        intent = detect_intent(question)
        rewritten_query = self.gemini.rewrite_query(question, intent)
        candidates = hybrid_retrieve(
            query=rewritten_query,
            embedder=self.embedder,
            vector_store=self.vector_store,
            bm25_store=self.bm25_store,
        )
        reranked = self._rerank(question, candidates)
        generated = self._generate(question, intent, reranked)
        guarded = apply_guardrails(generated, question, has_context=bool(reranked))
        return RagAnswer(
            answer=guarded,
            sources=build_sources(reranked),
            confidence=_confidence(reranked),
            intent=intent,
            used_context_ids=[result.id for result in reranked],
        )

    def _rerank(self, question: str, candidates: list[SearchResult], top_k: int = 5) -> list[SearchResult]:
        if not candidates:
            return []
        candidate_dicts = [result.model_dump() for result in candidates[:10]]
        ordered_ids = self.gemini.rerank(question, candidate_dicts, top_k=top_k)
        by_id = {result.id: result for result in candidates}
        reranked = [by_id[result_id] for result_id in ordered_ids if result_id in by_id]
        return reranked or candidates[:top_k]

    def _generate(self, question: str, intent: str, contexts: list[SearchResult]) -> str:
        if not contexts:
            return ""
        context_text = "\n\n".join(
            f"[{index}] {item.title} ({item.entity_type})\nURL: {item.source_url}\n{item.text[:2500]}"
            for index, item in enumerate(contexts, start=1)
        )
        prompt = (
            "Bạn là trợ lý RAG cho Bệnh viện Đa khoa Quốc tế Hạnh Phúc.\n"
            "Quy tắc bắt buộc:\n"
            "- Chỉ trả lời dựa trên CONTEXT.\n"
            "- Nếu CONTEXT không có thông tin, nói rõ chưa có thông tin trong dữ liệu hiện tại.\n"
            "- Không chẩn đoán chắc chắn, không kê đơn, không tự thêm giá/dịch vụ ngoài context.\n"
            "- Trả lời ngắn gọn, rõ ràng, tiếng Việt.\n\n"
            f"Intent: {intent}\n"
            f"Câu hỏi: {question}\n\n"
            f"CONTEXT:\n{context_text}\n\n"
            "Câu trả lời:"
        )
        return self.gemini.generate(prompt)


def _confidence(results: list[SearchResult]) -> str:
    if not results:
        return "low"
    top_score = results[0].score
    if top_score >= 0.75:
        return "high"
    if top_score >= 0.4:
        return "medium"
    return "low"
