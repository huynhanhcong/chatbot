from __future__ import annotations

from types import SimpleNamespace

from fastapi.testclient import TestClient

from Flow_code import api
from Flow_code.conversation_memory import InMemoryConversationStore
from Flow_code.dialogue_state import InMemoryDialogueStateStore
from Flow_code.hospital_session import InMemoryHospitalSessionStore
from Flow_code.session_store import InMemorySessionStore


class FakeApiFlow:
    def handle_message(
        self,
        message: str,
        conversation_id: str | None = None,
        selected_index: int | None = None,
        selected_sku: str | None = None,
    ) -> dict:
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
            "conversation_id": conversation_id or "conv-1",
            "message": "Toi tim thay cac thuoc sau, ban muon hoi thuoc nao?",
            "options": [
                {
                    "index": 1,
                    "sku": "P00219",
                    "name": "Thuoc bot Oresol 245 DHG",
                    "brand": "DHG Pharma",
                    "price": "1.350 VND/Goi",
                    "detail_url": "https://www.pharmacity.vn/oresol-245.html",
                }
            ],
        }

    def has_active_session(self, conversation_id: str | None) -> bool:
        return conversation_id == "conv-1"


class FakeRagPipeline:
    def __init__(self, source_title: str = "Nguon Hanh Phuc") -> None:
        self.source_title = source_title
        self.questions: list[str] = []

    def answer(self, question: str):
        self.questions.append(question)
        return SimpleNamespace(
            answer=f"RAG answer: {question}",
            sources=[{"title": self.source_title, "url": "https://example.test"}],
            confidence="high",
            intent="package",
        )


class ContextAwareFakeRagPipeline(FakeRagPipeline):
    def __init__(self, source_title: str = "Nguon Hanh Phuc") -> None:
        super().__init__(source_title)
        self.contexts: list[str | None] = []
        self.original_questions: list[str | None] = []

    def answer(
        self,
        question: str,
        conversation_context: str | None = None,
        original_question: str | None = None,
    ):
        self.contexts.append(conversation_context)
        self.original_questions.append(original_question)
        return super().answer(question)


def _reset_runtime(monkeypatch, *, flow=None, rag=None) -> None:
    monkeypatch.setattr(api, "_flow", flow)
    monkeypatch.setattr(api, "_rag_pipeline", rag)
    monkeypatch.setattr(api, "_conversations", InMemoryConversationStore())
    monkeypatch.setattr(api, "_hospital_sessions", InMemoryHospitalSessionStore())
    monkeypatch.setattr(api, "_dialogue_states", InMemoryDialogueStateStore())
    monkeypatch.setattr(api, "_drug_sessions", InMemorySessionStore())


def test_health() -> None:
    client = TestClient(api.app)

    response = client.get("/health")

    assert response.status_code == 200
    assert float(response.headers["x-response-time-ms"]) >= 0
    assert response.json() == {"status": "ok"}


def test_root_serves_webapp() -> None:
    client = TestClient(api.app)

    response = client.get("/")

    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "Trợ lý Bệnh viện Hạnh Phúc" in response.text


def test_app_serves_webapp() -> None:
    client = TestClient(api.app)

    response = client.get("/app")

    assert response.status_code == 200
    assert "Trợ lý Bệnh viện Hạnh Phúc" in response.text


def test_favicon_returns_no_content() -> None:
    client = TestClient(api.app)

    response = client.get("/favicon.ico")

    assert response.status_code == 204


def test_drug_info_single_endpoint_start_and_select(monkeypatch) -> None:
    _reset_runtime(monkeypatch, flow=FakeApiFlow())
    client = TestClient(api.app)

    start = client.post(
        "/chat/drug-info",
        json={"message": "Hay cho toi biet thong tin ve thuoc Oresol"},
    )
    select = client.post(
        "/chat/drug-info",
        json={"message": "1", "conversation_id": "conv-1", "selected_index": 1},
    )

    assert start.status_code == 200
    assert start.json()["status"] == "need_selection"
    assert "detail_url" not in start.json()["options"][0]
    assert select.status_code == 200
    assert select.json()["status"] == "answered"
    assert select.json()["selected_product"] == {
        "sku": "P00219",
        "name": "Thuoc bot Oresol 245 DHG",
    }
    assert "source_url" not in select.json()
    assert "sources" not in select.json()


def test_unified_chat_routes_pharmacity_search(monkeypatch) -> None:
    _reset_runtime(monkeypatch, flow=FakeApiFlow())
    client = TestClient(api.app)

    response = client.post(
        "/chat",
        json={"message": "Hay cho toi biet thong tin ve thuoc Oresol"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["route"] == "pharmacity"
    assert body["status"] == "need_selection"
    assert body["options"][0]["sku"] == "P00219"
    assert "detail_url" not in body["options"][0]
    assert body["sources"] == []


def test_new_unified_chat_clears_pharmacity_export_files(monkeypatch, tmp_path) -> None:
    raw_path = tmp_path / "pharmacity.txt"
    extracted_path = tmp_path / "pharmacity_1.txt"
    raw_path.write_text("old raw", encoding="utf-8")
    extracted_path.write_text("old extracted", encoding="utf-8")
    monkeypatch.setattr(api, "PHARMACITY_EXPORT_PATHS", (raw_path, extracted_path))
    _reset_runtime(monkeypatch, flow=None, rag=FakeRagPipeline())
    client = TestClient(api.app)

    response = client.post("/chat", json={"message": "Goi IVF Standard gom gi?"})

    assert response.status_code == 200
    assert raw_path.read_text(encoding="utf-8") == ""
    assert extracted_path.read_text(encoding="utf-8") == ""


def test_existing_unified_chat_keeps_pharmacity_export_files(monkeypatch, tmp_path) -> None:
    raw_path = tmp_path / "pharmacity.txt"
    extracted_path = tmp_path / "pharmacity_1.txt"
    raw_path.write_text("keep raw", encoding="utf-8")
    extracted_path.write_text("keep extracted", encoding="utf-8")
    monkeypatch.setattr(api, "PHARMACITY_EXPORT_PATHS", (raw_path, extracted_path))
    _reset_runtime(monkeypatch, flow=None, rag=FakeRagPipeline())
    client = TestClient(api.app)

    response = client.post(
        "/chat",
        json={"message": "Goi IVF Standard gom gi?", "conversation_id": "conv-existing"},
    )

    assert response.status_code == 200
    assert raw_path.read_text(encoding="utf-8") == "keep raw"
    assert extracted_path.read_text(encoding="utf-8") == "keep extracted"


def test_unified_chat_routes_pharmacity_selection(monkeypatch) -> None:
    _reset_runtime(monkeypatch, flow=FakeApiFlow())
    client = TestClient(api.app)

    response = client.post(
        "/chat",
        json={"message": "1", "conversation_id": "conv-1", "selected_index": 1},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["route"] == "pharmacity"
    assert body["status"] == "answered"
    assert body["sources"] == []
    assert body["selected_product"] == {
        "sku": "P00219",
        "name": "Thuoc bot Oresol 245 DHG",
    }
    assert body["source_url"] is None


def test_unified_chat_routes_hospital_rag(monkeypatch) -> None:
    _reset_runtime(monkeypatch, flow=None, rag=FakeRagPipeline())
    client = TestClient(api.app)

    response = client.post("/chat", json={"message": "Goi IVF Standard gom gi?"})

    assert response.status_code == 200
    body = response.json()
    assert body["route"] == "hospital_rag"
    assert body["status"] == "answered"
    assert body["conversation_id"]
    assert body["sources"][0]["title"] == "Nguon Hanh Phuc"


def test_unified_chat_hospital_rag_uses_previous_context(monkeypatch) -> None:
    rag = FakeRagPipeline(source_title="Goi kham thai Tam An (Goi kham thai 10-40 tuan)")
    _reset_runtime(monkeypatch, flow=None, rag=rag)
    client = TestClient(api.app)

    start = client.post(
        "/chat",
        json={"message": "Toi mang thai 12 tuan thi nen chon goi kham nao"},
    )
    conversation_id = start.json()["conversation_id"]
    follow_up = client.post(
        "/chat",
        json={"message": "Goi nay bao nhieu tien", "conversation_id": conversation_id},
    )

    assert follow_up.status_code == 200
    assert rag.questions[1].startswith("Goi kham thai Tam An")
    assert "bao nhieu tien" in rag.questions[1]


def test_unified_chat_switches_from_pharmacity_session_to_hospital_rag(monkeypatch) -> None:
    _reset_runtime(monkeypatch, flow=FakeApiFlow(), rag=FakeRagPipeline())
    client = TestClient(api.app)

    response = client.post(
        "/chat",
        json={"message": "Goi IVF Standard gom gi?", "conversation_id": "conv-1"},
    )

    assert response.status_code == 200
    assert response.json()["route"] == "hospital_rag"


def test_unified_chat_switches_from_hospital_rag_to_pharmacity(monkeypatch) -> None:
    _reset_runtime(monkeypatch, flow=FakeApiFlow(), rag=FakeRagPipeline())
    client = TestClient(api.app)

    start = client.post("/chat", json={"message": "Goi IVF Standard gom gi?"})
    conversation_id = start.json()["conversation_id"]
    response = client.post(
        "/chat",
        json={
            "message": "Hay cho toi biet thong tin ve thuoc Oresol",
            "conversation_id": conversation_id,
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["route"] == "pharmacity"
    assert body["conversation_id"] == conversation_id


def test_unified_chat_passes_conversation_context_to_rag_pipeline(monkeypatch) -> None:
    rag = ContextAwareFakeRagPipeline(source_title="Goi IVF Standard")
    _reset_runtime(monkeypatch, flow=None, rag=rag)
    client = TestClient(api.app)

    start = client.post("/chat", json={"message": "Goi IVF Standard gom gi?"})
    conversation_id = start.json()["conversation_id"]
    follow_up = client.post(
        "/chat",
        json={"message": "Goi nay bao nhieu tien", "conversation_id": conversation_id},
    )

    assert follow_up.status_code == 200
    assert rag.original_questions[1] == "Goi nay bao nhieu tien"
    assert rag.contexts[1] and "Goi IVF Standard gom gi?" in rag.contexts[1]
