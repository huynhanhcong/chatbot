from __future__ import annotations

from Flow_code.conversation_memory import InMemoryConversationStore
from Flow_code.dialogue_state import InMemoryDialogueStateStore, update_state_after_turn
from Flow_code.router_service import IntentRouter, RouteContext


def _context(conversation_id: str = "conv-1", pharmacity_session=None):
    conversations = InMemoryConversationStore()
    states = InMemoryDialogueStateStore()
    conversation = conversations.get_or_create(conversation_id)
    state = states.get_or_create(conversation.conversation_id)
    return RouteContext(
        conversation=conversation,
        state=state,
        pharmacity_session=pharmacity_session,
        hospital_active=False,
    )


def test_router_detects_drug_search() -> None:
    decision = IntentRouter().classify(
        message="Hay cho toi biet thong tin ve thuoc Oresol",
        context=_context(),
    )

    assert decision.intent == "drug_search"
    assert decision.route == "pharmacity"
    assert decision.confidence >= 0.9


def test_router_detects_hospital_intents() -> None:
    router = IntentRouter()

    doctor = router.classify(message="Bac si nao ho tro sinh san?", context=_context())
    package = router.classify(message="Goi IVF Standard gom gi?", context=_context())
    price = router.classify(message="Goi IVF Standard bao nhieu tien?", context=_context())

    assert doctor.intent == "doctor_search"
    assert package.intent == "package_search"
    assert price.intent == "price_question"
    assert {doctor.route, package.route, price.route} == {"hospital_rag"}


def test_router_preserves_explicit_product_selection() -> None:
    decision = IntentRouter().classify(
        message="1",
        selected_index=1,
        context=_context(),
    )

    assert decision.intent == "drug_followup"
    assert decision.route == "pharmacity"
    assert decision.confidence == 1.0


def test_router_uses_dialogue_state_for_context_followup() -> None:
    conversations = InMemoryConversationStore()
    states = InMemoryDialogueStateStore()
    conversation = conversations.get_or_create("conv-1")
    state = states.get_or_create("conv-1")
    states.save(update_state_after_turn(state, domain="hospital", intent="package_search"))

    decision = IntentRouter().classify(
        message="Goi nay bao nhieu tien",
        context=RouteContext(
            conversation=conversation,
            state=states.get_or_create("conv-1"),
            hospital_active=True,
        ),
    )

    assert decision.route == "hospital_rag"
    assert decision.intent in {"context_followup", "price_question"}


def test_router_returns_out_of_scope_for_low_signal_message() -> None:
    decision = IntentRouter().classify(message="xin chao", context=_context())

    assert decision.intent == "out_of_scope"
    assert decision.route == "out_of_scope"
    assert decision.confidence < 0.5
