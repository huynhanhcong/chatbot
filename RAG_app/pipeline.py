import logging
import time

from .bm25_store import BM25Store
from .cache import build_cache, stable_cache_key
from .config import Settings
from .data_loader import load_hanhphuc_chunk_documents
from .entity_linker import HanhPhucEntityLinker, normalize_for_linking
from .embeddings import GeminiEmbedder
from .formatter import build_sources, compose_answer_text
from .gemini_client import GeminiTextClient
from .guardrails import apply_guardrails
from .intent import detect_intent
from .models import RagAnswer, SearchResult
from .retriever import diversify_results, hybrid_retrieve
from .vector_store import QdrantVectorStore


logger = logging.getLogger("uvicorn.error")


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
        self.active_vector_store = self.vector_store
        self.active_bm25_store = self.bm25_store
        self.active_collection = settings.qdrant_collection
        self._configure_chunk_index()
        self.cache = build_cache(settings, "hanhphuc")
        self.entity_linker = HanhPhucEntityLinker.from_jsonl(settings.entities_path)

    def answer(
        self,
        question: str,
        conversation_context: str | None = None,
        original_question: str | None = None,
        memory_context: str | None = None,
        resolved_entities: list[dict] | None = None,
        answer_mode: str | None = None,
    ) -> RagAnswer:
        answer_cache_key = _answer_cache_key(
            question=question,
            original_question=original_question,
            conversation_context=conversation_context,
        )
        if answer_cache_key:
            cached_answer = self.cache.get_json(answer_cache_key)
            if isinstance(cached_answer, dict):
                return RagAnswer.model_validate(cached_answer)

        started_at = time.perf_counter()
        intent = answer_mode or detect_intent(question)
        rewritten_query = self._retrieval_query(
            question,
            intent,
            conversation_context,
            memory_context=memory_context,
            resolved_entities=resolved_entities,
        )
        _log_step("hospital_query_prepare", started_at, intent=intent)

        entity_links = self.entity_linker.link(rewritten_query)
        _log_step("hospital_entity_linking", started_at, link_count=len(entity_links))

        candidates = self._retrieve_with_cache(rewritten_query, entity_links)
        _log_step("hospital_retrieval", started_at, candidate_count=len(candidates))

        reranked = self._rerank(question, candidates)
        _log_step("hospital_rerank", started_at, reranked_count=len(reranked))

        generated = self._generate(
            question=original_question or question,
            retrieval_question=question,
            intent=intent,
            contexts=reranked,
            conversation_context=conversation_context,
            memory_context=memory_context,
        )
        _log_step("hospital_generation", started_at)

        styled = compose_answer_text(generated)
        guarded = apply_guardrails(
            styled,
            original_question or question,
            has_context=bool(reranked),
        )
        answer = RagAnswer(
            answer=guarded,
            sources=build_sources(reranked),
            confidence=_confidence(reranked),
            intent=intent,
            used_context_ids=[result.id for result in reranked],
        )
        if answer_cache_key:
            self.cache.set_json(
                answer_cache_key,
                answer.model_dump(mode="json"),
                self.settings.cache_ttl_seconds,
            )
        return answer

    def _rerank(self, question: str, candidates: list[SearchResult], top_k: int = 5) -> list[SearchResult]:
        if not candidates:
            return []
        if self.settings.reranker_provider != "llm":
            return diversify_results(question, candidates, top_k=top_k, max_per_parent=2)
        candidate_dicts = [result.model_dump() for result in candidates[:10]]
        ordered_ids = self.gemini.rerank(question, candidate_dicts, top_k=top_k)
        by_id = {result.id: result for result in candidates}
        reranked = [by_id[result_id] for result_id in ordered_ids if result_id in by_id]
        return reranked or candidates[:top_k]

    def _retrieval_query(
        self,
        question: str,
        intent: str,
        conversation_context: str | None,
        memory_context: str | None = None,
        resolved_entities: list[dict] | None = None,
    ) -> str:
        resolved_titles = [
            str(item.get("title") or "").strip()
            for item in resolved_entities or []
            if isinstance(item, dict) and str(item.get("title") or "").strip()
        ]
        if resolved_titles:
            return f"{' ; '.join(resolved_titles[:2])}. {question}"
        if not _needs_llm_rewrite(question, conversation_context):
            return question
        try:
            rewritten = self.gemini.rewrite_query(question, intent)
        except Exception:
            return question
        if rewritten and memory_context:
            return f"{rewritten}\n{memory_context}"
        return rewritten or question

    def _retrieve_with_cache(
        self,
        rewritten_query: str,
        entity_links: list,
    ) -> list[SearchResult]:
        link_signature = ",".join(link.entity_id for link in entity_links)
        cache_key = stable_cache_key(
            "retrieval",
            self.active_collection,
            rewritten_query,
            link_signature,
        )
        cached = self.cache.get_json(cache_key)
        if isinstance(cached, list):
            return [SearchResult.model_validate(item) for item in cached]

        candidates = hybrid_retrieve(
            query=rewritten_query,
            embedder=self.embedder,
            vector_store=self.active_vector_store,
            bm25_store=self.active_bm25_store,
            entity_links=entity_links,
        )
        self.cache.set_json(
            cache_key,
            [candidate.model_dump(mode="json") for candidate in candidates],
            self.settings.cache_ttl_seconds,
        )
        return candidates

    def _configure_chunk_index(self) -> None:
        if not self.settings.rag_use_chunk_index:
            return
        try:
            chunk_vector_store = QdrantVectorStore(
                self.settings.qdrant_url,
                self.settings.chunk_qdrant_collection,
                path=str(self.settings.qdrant_path) if self.settings.qdrant_path else None,
                client=self.vector_store.client,
            )
            if not chunk_vector_store.collection_exists():
                return
            if self.settings.chunk_bm25_path.exists():
                chunk_bm25_store = BM25Store.load(self.settings.chunk_bm25_path)
            elif self.settings.chunks_path.exists():
                chunk_bm25_store = BM25Store.from_documents(
                    load_hanhphuc_chunk_documents(self.settings.chunks_path)
                )
            else:
                return
        except Exception as exc:
            logger.warning("chunk_index_unavailable error=%s", exc)
            return

        self.active_vector_store = chunk_vector_store
        self.active_bm25_store = chunk_bm25_store
        self.active_collection = self.settings.chunk_qdrant_collection

    def _generate(
        self,
        question: str,
        retrieval_question: str,
        intent: str,
        contexts: list[SearchResult],
        conversation_context: str | None = None,
        memory_context: str | None = None,
    ) -> str:
        if not contexts:
            return ""
        context_text = "\n\n".join(
            f"[{index}] {item.title} ({item.entity_type})\nURL: {item.source_url}\n{item.text[:1800]}"
            for index, item in enumerate(contexts, start=1)
        )
        history_block = (
            f"LICH_SU_HOI_THOAI:\n{conversation_context}\n\n"
            if conversation_context and conversation_context.strip()
            else ""
        )
        memory_block = (
            f"MEMORY_RESOLUTION:\n{memory_context}\n\n"
            if memory_context and memory_context.strip()
            else ""
        )
        prompt = (
            "Bạn là trợ lý RAG cho Bệnh viện Đa khoa Quốc tế Hạnh Phúc.\n"
            "Quy tắc bắt buộc:\n"
            "- Chỉ trả lời dựa trên CONTEXT.\n"
            "- Nếu MEMORY_RESOLUTION có resolved_reference, phải trả lời đúng item đó.\n"
            "- Nếu Intent là compare_question, chỉ so sánh các item đã resolve và có trong CONTEXT.\n"
            "- Dùng LICH_SU_HOI_THOAI chỉ để hiểu đại từ, chủ đề và câu hỏi tiếp nối.\n"
            "- Nếu CONTEXT không có thông tin, nói rõ chưa có thông tin trong dữ liệu hiện tại.\n"
            "- Không chẩn đoán chắc chắn, không kê đơn, không tự thêm giá hoặc dịch vụ ngoài context.\n"
            "- Trả lời tự nhiên, ngắn gọn, tiếng Việt.\n"
            "- Câu đầu tiên phải trả lời trực tiếp đúng ý người dùng hỏi.\n"
            "- Tối đa 2 đoạn ngắn. Không nhắc đến CONTEXT, nguồn, độ tin cậy, hay các câu như 'theo thông tin được cung cấp'.\n\n"
            f"Intent: {intent}\n"
            f"Câu hỏi người dùng: {question}\n"
            f"Câu hỏi độc lập dùng để truy xuất: {retrieval_question}\n\n"
            f"{history_block}"
            f"{memory_block}"
            f"CONTEXT:\n{context_text}\n\n"
            "Câu trả lời:"
        )
        return self.gemini.generate(prompt)


def _confidence(results: list[SearchResult]) -> str:
    if not results:
        return "low"
    top_score = results[0].score
    if top_score >= 0.18:
        return "high"
    if top_score >= 0.04:
        return "medium"
    return "low"


def _needs_llm_rewrite(question: str, conversation_context: str | None) -> bool:
    if not conversation_context or not conversation_context.strip():
        return False
    normalized = normalize_for_linking(question)
    return any(
        phrase in normalized
        for phrase in [
            "goi nay",
            "goi do",
            "bac si do",
            "bac si nay",
            "dich vu nay",
            "dich vu do",
            "nguoi dau tien",
            "nguoi thu hai",
            "bac si kia",
            "goi dau tien",
            "goi thu hai",
            "loai dau tien",
            "loai thu hai",
            "khac gi",
            "so sanh",
            "gia bao nhieu",
            "bao nhieu tien",
            "chi phi",
            "o tren",
            "vua roi",
            "cai nay",
            "muc nay",
            "co gi",
            "gom gi",
        ]
    )


def _answer_cache_key(
    *,
    question: str,
    original_question: str | None,
    conversation_context: str | None,
) -> str | None:
    if conversation_context and conversation_context.strip():
        return None
    return stable_cache_key("answer", question, original_question or "")


def _log_step(event: str, started_at: float, **fields: object) -> None:
    elapsed_ms = (time.perf_counter() - started_at) * 1000
    suffix = " ".join(f"{key}={value}" for key, value in fields.items())
    logger.info("%s latency_ms=%.2f %s", event, elapsed_ms, suffix)
