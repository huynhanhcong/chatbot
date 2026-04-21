from __future__ import annotations

from typing import Any, Callable

from Flow_code.conversation_memory import (
    ConversationSession,
    InMemoryConversationStore,
    format_conversation_context,
)
from Flow_code.dialogue_state import (
    DialogueStateStore,
    entity_from_pharmacity_response,
    entity_from_sources,
    update_state_after_turn,
)
from Flow_code.drug_service import DrugInfoService
from Flow_code.hospital_session import HospitalSession, HospitalTurn, InMemoryHospitalSessionStore
from Flow_code.observability import ChatObserver
from Flow_code.router_service import IntentRouter, RouteContext, looks_like_contextual_follow_up, normalize_vi
from Flow_code.service_contracts import ChatEnvelope, IntentDecision


class ChatOrchestrator:
    def __init__(
        self,
        *,
        conversation_store: InMemoryConversationStore,
        hospital_session_store: InMemoryHospitalSessionStore,
        dialogue_state_store: DialogueStateStore,
        drug_service: DrugInfoService,
        rag_pipeline_provider: Callable[[], Any],
        router: IntentRouter | None = None,
        observer: ChatObserver | None = None,
    ) -> None:
        self.conversation_store = conversation_store
        self.hospital_session_store = hospital_session_store
        self.dialogue_state_store = dialogue_state_store
        self.drug_service = drug_service
        self.rag_pipeline_provider = rag_pipeline_provider
        self.router = router or IntentRouter()
        self.observer = observer or ChatObserver()

    def handle(self, payload: Any) -> dict[str, Any]:
        trace = self.observer.start(path="/chat")
        conversation = self.conversation_store.get_or_create(payload.conversation_id)
        state = self.dialogue_state_store.get_or_create(conversation.conversation_id)
        try:
            pharmacity_session = self.drug_service.get_session(conversation.conversation_id)
            decision = self.router.classify(
                message=payload.message,
                selected_index=payload.selected_index,
                selected_sku=payload.selected_sku,
                context=RouteContext(
                    conversation=conversation,
                    state=state,
                    pharmacity_session=pharmacity_session,
                    hospital_active=self.hospital_session_store.has_active_session(
                        conversation.conversation_id
                    ),
                ),
            )
            self.observer.route_selected(
                trace=trace,
                conversation_id=conversation.conversation_id,
                decision=decision,
            )

            if decision.route == "pharmacity":
                envelope = self._handle_pharmacity(payload, conversation, decision)
            elif decision.route == "hospital_rag":
                envelope = self._handle_hospital_rag(payload, conversation, decision)
            else:
                envelope = self._handle_out_of_scope(conversation, decision)

            self.observer.finish(
                trace=trace,
                conversation_id=envelope["conversation_id"],
                status=str(envelope.get("status")),
                route=str(envelope.get("route")),
                intent=envelope.get("intent"),
                retrieval_count=len(envelope.get("sources") or []),
            )
            return envelope
        except Exception as exc:
            self.observer.error(
                trace=trace,
                conversation_id=conversation.conversation_id,
                error=exc,
            )
            raise

    def _handle_pharmacity(
        self,
        payload: Any,
        conversation: ConversationSession,
        decision: IntentDecision,
    ) -> dict[str, Any]:
        envelope = self.drug_service.handle_envelope(
            message=payload.message,
            conversation_id=conversation.conversation_id,
            selected_index=payload.selected_index,
            selected_sku=payload.selected_sku,
        )
        conversation_id = envelope.get("conversation_id") or conversation.conversation_id
        self._remember_chat_turn(
            conversation_id=conversation_id,
            route="pharmacity",
            user_message=payload.message,
            assistant_message=envelope.get("answer") or envelope.get("message"),
            sources=envelope.get("sources") or [],
            metadata={
                "status": envelope.get("status"),
                "selected_product": envelope.get("selected_product"),
                "intent_decision": decision.reason,
            },
        )
        state = self.dialogue_state_store.get_or_create(conversation_id)
        active_entity = entity_from_pharmacity_response(envelope)
        unresolved_slots = (
            {"product_selection": "required"}
            if envelope.get("status") == "need_selection"
            else {}
        )
        self.dialogue_state_store.save(
            update_state_after_turn(
                state,
                domain="pharmacity",
                intent=decision.intent,
                active_entity=active_entity,
                unresolved_slots=unresolved_slots,
            )
        )
        envelope["intent"] = decision.intent
        envelope["confidence"] = _decision_confidence_label(decision)
        return envelope

    def _handle_hospital_rag(
        self,
        payload: Any,
        conversation: ConversationSession,
        decision: IntentDecision,
    ) -> dict[str, Any]:
        pipeline = self.rag_pipeline_provider()
        session = self.hospital_session_store.get_or_create(conversation.conversation_id)
        standalone_question = standalone_hospital_question(
            pipeline,
            session,
            payload.message,
            conversation,
        )
        conversation_context = format_conversation_context(conversation)
        answer = answer_hospital_pipeline(
            pipeline=pipeline,
            standalone_question=standalone_question,
            original_question=payload.message,
            conversation_context=conversation_context,
        )
        self.hospital_session_store.save_turn(
            conversation_id=session.conversation_id,
            question=payload.message,
            standalone_question=standalone_question,
            answer=answer.answer,
            sources=answer.sources,
        )
        self._remember_chat_turn(
            conversation_id=session.conversation_id,
            route="hospital_rag",
            user_message=payload.message,
            assistant_message=answer.answer,
            standalone_question=standalone_question,
            sources=answer.sources,
            metadata={
                "intent": getattr(answer, "intent", decision.intent),
                "confidence": getattr(answer, "confidence", None),
                "intent_decision": decision.reason,
            },
        )
        state = self.dialogue_state_store.get_or_create(session.conversation_id)
        self.dialogue_state_store.save(
            update_state_after_turn(
                state,
                domain="hospital",
                intent=decision.intent,
                active_entity=entity_from_sources(answer.sources),
            )
        )
        return {
            "status": "answered",
            "route": "hospital_rag",
            "conversation_id": session.conversation_id,
            "message": None,
            "options": [],
            "answer": answer.answer,
            "sources": answer.sources,
            "confidence": answer.confidence,
            "intent": answer.intent,
        }

    def _handle_out_of_scope(
        self,
        conversation: ConversationSession,
        decision: IntentDecision,
    ) -> dict[str, Any]:
        answer = (
            "Tôi chưa xác định được bạn muốn hỏi về thuốc Pharmacity hay dữ liệu "
            "Bệnh viện Hạnh Phúc. Bạn vui lòng nói rõ tên thuốc, tên gói khám, "
            "bác sĩ hoặc dịch vụ cần tra cứu."
        )
        state = self.dialogue_state_store.get_or_create(conversation.conversation_id)
        self.dialogue_state_store.save(
            update_state_after_turn(
                state,
                domain="unknown",
                intent=decision.intent,
                unresolved_slots={"clarification": "domain_or_entity"},
            )
        )
        return ChatEnvelope(
            status="needs_clarification",
            route="out_of_scope",
            conversation_id=conversation.conversation_id,
            answer=answer,
            confidence="low",
            intent=decision.intent,
        ).to_response()

    def _remember_chat_turn(
        self,
        *,
        conversation_id: str,
        route: str,
        user_message: str,
        assistant_message: str | None,
        standalone_question: str | None = None,
        sources: list[dict[str, Any]] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        if route not in {"hospital_rag", "pharmacity"}:
            return
        self.conversation_store.save_turn(
            conversation_id=conversation_id,
            route=route,  # type: ignore[arg-type]
            user_message=user_message,
            assistant_message=assistant_message,
            standalone_question=standalone_question,
            sources=sources or [],
            metadata=metadata or {},
        )


def answer_hospital_pipeline(
    *,
    pipeline: Any,
    standalone_question: str,
    original_question: str,
    conversation_context: str,
) -> Any:
    try:
        return pipeline.answer(
            standalone_question,
            conversation_context=conversation_context,
            original_question=original_question,
        )
    except TypeError as exc:
        if "conversation_context" not in str(exc) and "original_question" not in str(exc):
            raise
        return pipeline.answer(standalone_question)


def standalone_hospital_question(
    pipeline: Any,
    session: HospitalSession,
    message: str,
    conversation: ConversationSession | None = None,
) -> str:
    if not session.turns and not session.summary:
        return message

    fallback = fallback_hospital_question(session, message)
    gemini = getattr(pipeline, "gemini", None)
    generate = getattr(gemini, "generate", None)
    if not callable(generate):
        return fallback

    prompt = build_hospital_rewrite_prompt(session, message, conversation)
    try:
        rewritten = generate(prompt)
    except Exception:
        return fallback

    rewritten = clean_rewritten_question(rewritten)
    return rewritten or fallback


def fallback_hospital_question(session: HospitalSession, message: str) -> str:
    if not looks_like_contextual_follow_up(normalize_vi(message)):
        return message

    subject = last_hospital_subject(session)
    if not subject:
        return message
    normalized_message = message.strip()
    if subject.lower() in normalized_message.lower():
        return normalized_message
    return f"{subject}. {normalized_message}"


def build_hospital_rewrite_prompt(
    session: HospitalSession,
    message: str,
    conversation: ConversationSession | None = None,
) -> str:
    compact_history = "\n\n".join(format_hospital_turn(turn) for turn in session.turns[-4:])
    summary = session.summary or ""
    conversation_context = format_conversation_context(conversation, max_turns=4, max_chars=2800)
    return (
        "You rewrite Vietnamese hospital chatbot follow-up questions for RAG retrieval.\n"
        "Use CHAT_HISTORY only to resolve references like this package, that doctor, price, cost, "
        "it, above, or the service mentioned previously.\n"
        "If the new question is already standalone or changes topic, return it unchanged.\n"
        "Return only one standalone Vietnamese question. Do not answer the question.\n\n"
        f"CHAT_SUMMARY:\n{summary or 'None'}\n\n"
        f"FULL_CONVERSATION_CONTEXT:\n{conversation_context or 'None'}\n\n"
        f"CHAT_HISTORY:\n{compact_history}\n\n"
        f"NEW_QUESTION: {message}\n"
        "STANDALONE_QUESTION:"
    )


def format_hospital_turn(turn: HospitalTurn) -> str:
    source_titles = [
        str(source.get("title"))
        for source in turn.sources
        if isinstance(source, dict) and source.get("title")
    ]
    sources_text = ", ".join(source_titles[:3]) if source_titles else "None"
    return (
        f"User: {turn.question}\n"
        f"Standalone: {turn.standalone_question}\n"
        f"Assistant: {turn.answer[:1000]}\n"
        f"Sources: {sources_text}"
    )


def clean_rewritten_question(value: str) -> str:
    cleaned = (value or "").strip().strip('"').strip("'").strip()
    if not cleaned:
        return ""

    prefixes = [
        "STANDALONE_QUESTION:",
        "Standalone question:",
        "Cau hoi doc lap:",
        "Cau hoi:",
    ]
    for prefix in prefixes:
        if cleaned.lower().startswith(prefix.lower()):
            cleaned = cleaned[len(prefix) :].strip()
            break

    first_line = next((line.strip() for line in cleaned.splitlines() if line.strip()), "")
    return first_line.strip().strip('"').strip("'")


def last_hospital_subject(session: HospitalSession) -> str | None:
    for turn in reversed(session.turns):
        for source in turn.sources:
            if isinstance(source, dict) and source.get("title"):
                return str(source["title"])
    return None


def _decision_confidence_label(decision: IntentDecision) -> str:
    if decision.confidence >= 0.85:
        return "high"
    if decision.confidence >= 0.55:
        return "medium"
    return "low"
