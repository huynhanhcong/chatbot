from __future__ import annotations

from typing import Any, Callable

from Flow_code.conversation_memory import (
    ConversationSession,
    InMemoryConversationStore,
    format_conversation_context,
)
from Flow_code.dialogue_state import (
    DialogueStateStore,
    displayed_item_from_active_entity,
    displayed_items_from_pharmacity_options,
    displayed_items_from_sources,
    entity_from_pharmacity_response,
    entity_from_sources,
    update_state_after_turn,
)
from Flow_code.drug_service import DrugInfoService, pharmacity_envelope
from Flow_code.hospital_session import HospitalSession, HospitalTurn, InMemoryHospitalSessionStore
from Flow_code.observability import ChatObserver
from Flow_code.mention_resolver import MentionResolver, ResolvedReference, render_memory_context
from Flow_code.router_service import IntentRouter, RouteContext, looks_like_contextual_follow_up, normalize_vi
from Flow_code.service_contracts import ChatEnvelope, DisplayedItem, IntentDecision


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
        self.mention_resolver = MentionResolver()

    def handle(self, payload: Any) -> dict[str, Any]:
        trace = self.observer.start(path="/chat")
        conversation = self.conversation_store.get_or_create(payload.conversation_id)
        state = self.dialogue_state_store.get_or_create(conversation.conversation_id)
        try:
            pharmacity_session = self.drug_service.get_session(conversation.conversation_id)
            resolved = self.mention_resolver.resolve(payload.message, state)
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
                if resolved.needs_clarification:
                    envelope = self._handle_reference_clarification(conversation, decision, resolved)
                else:
                    envelope = self._handle_hospital_rag(payload, conversation, decision, resolved)
            else:
                envelope = self._handle_out_of_scope(conversation, decision)

            self.observer.finish(
                trace=trace,
                conversation_id=envelope["conversation_id"],
                status=str(envelope.get("status")),
                route=str(envelope.get("route")),
                intent=envelope.get("intent"),
                retrieval_count=len(envelope.get("sources") or []),
                selection_step=envelope.get("status") == "need_selection",
                clarification_step=envelope.get("status") == "needs_clarification",
                empty_context=envelope.get("route") == "hospital_rag" and not (envelope.get("sources") or []),
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
        raw_response = self.drug_service.handle_raw(
            message=payload.message,
            conversation_id=conversation.conversation_id,
            selected_index=payload.selected_index,
            selected_sku=payload.selected_sku,
        )
        envelope = pharmacity_envelope(raw_response)
        conversation_id = envelope.get("conversation_id") or conversation.conversation_id
        self._remember_chat_turn(
            conversation_id=conversation_id,
            route="pharmacity",
            user_message=payload.message,
            assistant_message=envelope.get("answer") or envelope.get("message"),
            metadata={
                "status": raw_response.get("status"),
                "selected_product": raw_response.get("selected_product"),
                "internal_grounding": raw_response.get("internal_grounding"),
                "intent_decision": decision.reason,
            },
        )
        state = self.dialogue_state_store.get_or_create(conversation_id)
        active_entity = entity_from_pharmacity_response(raw_response)
        shown_items = displayed_items_from_pharmacity_options(raw_response.get("options"))
        selected_item = displayed_item_from_active_entity(active_entity) if active_entity is not None else None
        unresolved_slots = (
            {"product_selection": "required"}
            if raw_response.get("status") == "need_selection"
            else {}
        )
        self.dialogue_state_store.save(
            update_state_after_turn(
                state,
                domain="pharmacity",
                intent=decision.intent,
                active_entity=active_entity,
                last_selected_item=selected_item,
                last_shown_items=shown_items if shown_items else None,
                pending_question=raw_response.get("internal_grounding", {}).get("drug_name")
                if raw_response.get("status") == "need_selection"
                else None,
                unresolved_slots=unresolved_slots,
            )
        )
        if shown_items:
            envelope["displayed_items"] = [_displayed_item_response(item) for item in shown_items]
        envelope["intent"] = decision.intent
        envelope["confidence"] = _decision_confidence_label(decision)
        return envelope

    def _handle_hospital_rag(
        self,
        payload: Any,
        conversation: ConversationSession,
        decision: IntentDecision,
        resolved: ResolvedReference,
    ) -> dict[str, Any]:
        pipeline = self.rag_pipeline_provider()
        session = self.hospital_session_store.get_or_create(conversation.conversation_id)
        state = self.dialogue_state_store.get_or_create(conversation.conversation_id)
        standalone_question = standalone_hospital_question(
            pipeline,
            session,
            payload.message,
            conversation,
            resolved,
        )
        conversation_context = format_conversation_context(conversation)
        memory_context = render_memory_context(state, resolved)
        answer = answer_hospital_pipeline(
            pipeline=pipeline,
            standalone_question=standalone_question,
            original_question=payload.message,
            conversation_context=conversation_context,
            memory_context=memory_context,
            resolved_items=resolved.resolved_items,
            answer_mode=resolved.intent or decision.intent,
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
        shown_items = displayed_items_from_sources(answer.sources)
        active_entity = entity_from_sources(answer.sources)
        selected_item = resolved.primary_item
        compared_items = resolved.resolved_items[:2] if (resolved.intent == "compare_question" and len(resolved.resolved_items) >= 2) else None
        self.dialogue_state_store.save(
            update_state_after_turn(
                state,
                domain="hospital",
                intent=decision.intent,
                active_entity=active_entity,
                last_selected_item=selected_item,
                last_shown_items=shown_items if shown_items else None,
                last_compared_items=compared_items,
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
            "displayed_items": [_displayed_item_response(item) for item in shown_items],
        }

    def _handle_reference_clarification(
        self,
        conversation: ConversationSession,
        decision: IntentDecision,
        resolved: ResolvedReference,
    ) -> dict[str, Any]:
        options = [_displayed_item_response(item) for item in resolved.resolved_items[:4]]
        answer = _reference_clarification_message(resolved.resolved_items)
        state = self.dialogue_state_store.get_or_create(conversation.conversation_id)
        self.dialogue_state_store.save(
            update_state_after_turn(
                state,
                domain=state.active_domain,
                intent="clarification",
                ambiguity_candidates=resolved.resolved_items[:4],
                unresolved_slots={"clarification": resolved.reason or "ambiguous_reference"},
            )
        )
        return ChatEnvelope(
            status="needs_clarification",
            route=decision.route,
            conversation_id=conversation.conversation_id,
            answer=answer,
            options=options,
            confidence="medium",
            intent=decision.intent,
            displayed_items=options,
        ).to_response()

    def _handle_out_of_scope(
        self,
        conversation: ConversationSession,
        decision: IntentDecision,
    ) -> dict[str, Any]:
        answer = (
            "Tôi chưa xác định được bạn muốn hỏi về thuốc hay dữ liệu "
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
    memory_context: str | None = None,
    resolved_items: list[DisplayedItem] | None = None,
    answer_mode: str | None = None,
) -> Any:
    try:
        return pipeline.answer(
            standalone_question,
            conversation_context=conversation_context,
            original_question=original_question,
            memory_context=memory_context,
            resolved_entities=[_displayed_item_response(item) for item in resolved_items or []],
            answer_mode=answer_mode,
        )
    except TypeError as exc:
        message = str(exc)
        if (
            "conversation_context" not in message
            and "original_question" not in message
            and "memory_context" not in message
            and "resolved_entities" not in message
            and "answer_mode" not in message
        ):
            raise
    try:
        return pipeline.answer(
            standalone_question,
            conversation_context=conversation_context,
            original_question=original_question,
        )
    except TypeError as exc:
        message = str(exc)
        if "conversation_context" not in message and "original_question" not in message:
            raise
        return pipeline.answer(standalone_question)


def standalone_hospital_question(
    pipeline: Any,
    session: HospitalSession,
    message: str,
    conversation: ConversationSession | None = None,
    resolved: ResolvedReference | None = None,
) -> str:
    if resolved and resolved.primary_item is not None:
        return f"{resolved.primary_item.title}. {message.strip()}"
    if not session.turns and not session.summary:
        return message
    if not _should_rewrite_hospital_follow_up(session, message):
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


def _should_rewrite_hospital_follow_up(session: HospitalSession, message: str) -> bool:
    if not last_hospital_subject(session):
        return False
    return looks_like_contextual_follow_up(normalize_vi(message))


def _decision_confidence_label(decision: IntentDecision) -> str:
    if decision.confidence >= 0.85:
        return "high"
    if decision.confidence >= 0.55:
        return "medium"
    return "low"


def _displayed_item_response(item: DisplayedItem) -> dict[str, Any]:
    return {
        "index": item.index,
        "entity_id": item.entity_id,
        "entity_type": item.entity_type,
        "title": item.title,
        "source": item.source,
        "source_url": item.source_url,
        "price_vnd": item.price_vnd,
        "payload": item.payload,
    }


def _reference_clarification_message(items: list[DisplayedItem]) -> str:
    if not items:
        return "Bạn muốn hỏi về gói, dịch vụ hoặc bác sĩ nào? Bạn vui lòng nói rõ tên giúp mình."
    names = [item.title for item in items[:4]]
    if len(names) == 1:
        return f"Bạn muốn hỏi về {names[0]} đúng không?"
    return "Bạn đang muốn hỏi mục nào: " + "; ".join(
        f"{index}. {name}" for index, name in enumerate(names, start=1)
    )
