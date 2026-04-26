from __future__ import annotations

import httpx
import json

from Chat_Voice.agent.chat_bridge import (
    FALLBACK_ASSISTANT_TEXT,
    PRODUCT_SELECTION_TEXT,
    VoiceChatBridge,
    assistant_text_for_voice,
    parse_spoken_selection,
)
from Chat_Voice.agent.models import VoiceTurnRequest


def test_voice_turn_posts_transcript_and_preserves_conversation_id() -> None:
    seen_payloads: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen_payloads.append(json.loads(request.content.decode("utf-8")))
        return httpx.Response(
            200,
            json={
                "status": "answered",
                "route": "hospital_rag",
                "conversation_id": "conv-1",
                "answer": "Gói IVF Standard gồm các bước chính.",
                "intent": "package_search",
                "confidence": "high",
            },
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    bridge = VoiceChatBridge("https://chat.test/chat", client=client)

    response = bridge.handle_turn(
        VoiceTurnRequest(
            transcript="Gói IVF Standard gồm gì?",
            conversation_id="conv-1",
            voice_session_id="voice-1",
        )
    )

    assert seen_payloads == [
        {
            "message": "Gói IVF Standard gồm gì?",
            "conversation_id": "conv-1",
        }
    ]
    assert response.conversation_id == "conv-1"
    assert response.voice_session_id == "voice-1"
    assert response.assistant_text == "Gói IVF Standard gồm các bước chính."
    assert response.route == "hospital_rag"
    assert response.intent == "package_search"
    assert response.latency_ms >= 0


def test_need_selection_response_is_rendered_for_speech() -> None:
    text = assistant_text_for_voice(
        {
            "status": "need_selection",
            "message": "Tôi tìm thấy các thuốc sau.",
            "options": [
                {"index": 1, "name": "Oresol 245", "price": "1.350 VND/Gói"},
                {"index": 2, "name": "Oresol cam"},
            ],
        }
    )

    assert text == PRODUCT_SELECTION_TEXT
    assert "Oresol" not in text


def test_spoken_selection_maps_to_selected_index_payload() -> None:
    seen_payloads: list[dict] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen_payloads.append(json.loads(request.content.decode("utf-8")))
        return httpx.Response(
            200,
            json={
                "status": "answered",
                "route": "pharmacity",
                "conversation_id": "conv-1",
                "answer": "Thông tin thuốc.",
                "intent": "drug_followup",
            },
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    bridge = VoiceChatBridge("https://chat.test/chat", client=client)

    bridge.handle_turn(
        VoiceTurnRequest(
            transcript="chọn số một",
            conversation_id="conv-1",
            voice_session_id="voice-1",
        )
    )

    assert seen_payloads == [
        {
            "message": "1",
            "conversation_id": "conv-1",
            "selected_index": 1,
        }
    ]


def test_provider_failure_returns_spoken_fallback_without_losing_session() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, json={"detail": "provider down"})

    client = httpx.Client(transport=httpx.MockTransport(handler))
    bridge = VoiceChatBridge("https://chat.test/chat", client=client)

    response = bridge.handle_turn(
        VoiceTurnRequest(
            transcript="Gói khám thai giá bao nhiêu?",
            conversation_id="conv-existing",
            voice_session_id="voice-existing",
        )
    )

    assert response.conversation_id == "conv-existing"
    assert response.voice_session_id == "voice-existing"
    assert response.assistant_text == FALLBACK_ASSISTANT_TEXT
    assert response.status == "voice_error"


def test_parse_spoken_selection_accepts_vietnamese_number_words() -> None:
    assert parse_spoken_selection("chọn số hai") == 2
    assert parse_spoken_selection("lua chon 5") == 5
    assert parse_spoken_selection("2") == 2
    assert parse_spoken_selection("Cho tôi thông tin về gói khám thai 12 tuần") is None
    assert parse_spoken_selection("tôi muốn hỏi giá") is None
