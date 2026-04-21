from __future__ import annotations

import logging
from types import SimpleNamespace

from Flow_code.chat_orchestrator import ChatOrchestrator
from Flow_code.conversation_memory import InMemoryConversationStore
from Flow_code.dialogue_state import InMemoryDialogueStateStore
from Flow_code.drug_service import DrugInfoService
from Flow_code.hospital_session import InMemoryHospitalSessionStore
from Flow_code.observability import ChatObserver
from Flow_code.router_service import IntentRouter


class FakeFlow:
    def __init__(self) -> None:
        self.sessions = set()

    def handle_message(
        self,
        message: str,
        conversation_id: str | None = None,
        selected_index: int | None = None,
        selected_sku: str | None = None,
    ) -> dict:
        conversation_id = conversation_id or "conv-1"
        self.sessions.add(conversation_id)
        if selected_index or selected_sku:
            return {
                "status": "answered",
                "conversation_id": conversation_id,
                "answer": "Thong tin thuoc.",
                "selected_product": {
                    "sku": selected_sku or "P00219",
                    "name": "Thuoc bot Oresol 245 DHG",
                    "detail_url": "https://www.pharmacity.vn/oresol-245.html",
                },
                "source_url": "https://www.pharmacity.vn/oresol-245.html",
            }
        return {
            "status": "need_selection",
            "conversation_id": conversation_id,
            "message": "Toi tim thay cac thuoc sau, ban muon hoi thuoc nao?",
            "options": [{"index": 1, "sku": "P00219", "name": "Thuoc bot Oresol 245 DHG"}],
        }

    def has_active_session(self, conversation_id: str | None) -> bool:
        return conversation_id in self.sessions


class FakeRagPipeline:
    def answer(
        self,
        question: str,
        conversation_context: str | None = None,
        original_question: str | None = None,
    ):
        return SimpleNamespace(
            answer=f"RAG answer: {question}",
            sources=[{"id": "package_goi-ivf-standard", "title": "Goi IVF Standard", "url": "https://example.test"}],
            confidence="high",
            intent="package",
        )


def _payload(message: str, conversation_id: str | None = None, **kwargs):
    return SimpleNamespace(
        message=message,
        conversation_id=conversation_id,
        selected_index=kwargs.get("selected_index"),
        selected_sku=kwargs.get("selected_sku"),
    )


def _orchestrator(flow: FakeFlow | None = None, logger: logging.Logger | None = None):
    flow = flow or FakeFlow()
    states = InMemoryDialogueStateStore()
    return (
        ChatOrchestrator(
            conversation_store=InMemoryConversationStore(),
            hospital_session_store=InMemoryHospitalSessionStore(),
            dialogue_state_store=states,
            drug_service=DrugInfoService(
                flow_provider=lambda: flow,
                existing_flow_provider=lambda: flow,
            ),
            rag_pipeline_provider=lambda: FakeRagPipeline(),
            router=IntentRouter(),
            observer=ChatObserver(logger),
        ),
        states,
    )


def test_orchestrator_returns_frontend_compatible_hospital_envelope() -> None:
    orchestrator, states = _orchestrator()

    response = orchestrator.handle(_payload("Goi IVF Standard gom gi?"))

    assert response["status"] == "answered"
    assert response["route"] == "hospital_rag"
    assert response["conversation_id"]
    assert response["answer"].startswith("RAG answer:")
    assert response["sources"][0]["title"] == "Goi IVF Standard"
    state = states.get_or_create(response["conversation_id"])
    assert state.active_domain == "hospital"
    assert state.active_entity
    assert state.active_entity.name == "Goi IVF Standard"


def test_orchestrator_stores_selected_pharmacity_product_in_state() -> None:
    flow = FakeFlow()
    orchestrator, states = _orchestrator(flow)
    start = orchestrator.handle(_payload("Hay cho toi biet thong tin ve thuoc Oresol"))

    response = orchestrator.handle(
        _payload("1", conversation_id=start["conversation_id"], selected_index=1)
    )

    assert response["route"] == "pharmacity"
    assert response["selected_product"]["sku"] == "P00219"
    state = states.get_or_create(response["conversation_id"])
    assert state.active_domain == "pharmacity"
    assert state.active_entity
    assert state.active_entity.entity_id == "P00219"


def test_orchestrator_clarifies_low_confidence_unknown_message() -> None:
    orchestrator, _ = _orchestrator()

    response = orchestrator.handle(_payload("xin chao"))

    assert response["status"] == "needs_clarification"
    assert response["route"] == "out_of_scope"
    assert response["intent"] == "out_of_scope"


def test_observability_logs_route_and_latency(caplog) -> None:
    logger = logging.getLogger("test.chat_observer")
    orchestrator, _ = _orchestrator(logger=logger)

    with caplog.at_level(logging.INFO, logger="test.chat_observer"):
        orchestrator.handle(_payload("Goi IVF Standard gom gi?"))

    text = caplog.text
    assert "chat_route" in text
    assert "route=hospital_rag" in text
    assert "latency_ms=" in text
