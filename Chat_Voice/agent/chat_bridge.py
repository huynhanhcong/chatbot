from __future__ import annotations

import re
import time
from typing import Any

import httpx

from .models import VoiceTurnRequest, VoiceTurnResponse


FALLBACK_ASSISTANT_TEXT = (
    "Xin lỗi, hiện tôi chưa xử lý được yêu cầu bằng giọng nói. "
    "Bạn vui lòng thử lại hoặc nhập câu hỏi bằng văn bản."
)
PRODUCT_SELECTION_TEXT = "Tôi tìm được các sản phẩm sau hãy chọn sản phẩm của bạn."


class VoiceChatBridge:
    def __init__(
        self,
        chat_api_url: str,
        *,
        timeout_seconds: float = 60.0,
        client: httpx.Client | None = None,
    ) -> None:
        self.chat_api_url = chat_api_url
        self.timeout_seconds = timeout_seconds
        self._client = client

    def handle_turn(self, request: VoiceTurnRequest) -> VoiceTurnResponse:
        started_at = time.perf_counter()
        payload = _chat_payload(request)
        try:
            data = self._post_chat(payload)
            assistant_text = assistant_text_for_voice(data)
        except Exception as exc:
            data = {
                "status": "voice_error",
                "route": None,
                "intent": None,
                "detail": str(exc),
            }
            assistant_text = FALLBACK_ASSISTANT_TEXT

        latency_ms = (time.perf_counter() - started_at) * 1000
        return VoiceTurnResponse(
            conversation_id=str(data.get("conversation_id") or request.conversation_id or ""),
            voice_session_id=request.voice_session_id,
            transcript=request.transcript,
            assistant_text=assistant_text,
            route=data.get("route"),
            intent=data.get("intent"),
            status=data.get("status"),
            confidence=data.get("confidence"),
            latency_ms=latency_ms,
            raw_response=data,
        )

    def _post_chat(self, payload: dict[str, Any]) -> dict[str, Any]:
        if self._client is not None:
            response = self._client.post(self.chat_api_url, json=payload, timeout=self.timeout_seconds)
            response.raise_for_status()
            return response.json()

        with httpx.Client(timeout=self.timeout_seconds) as client:
            response = client.post(self.chat_api_url, json=payload)
            response.raise_for_status()
            return response.json()


def assistant_text_for_voice(data: dict[str, Any]) -> str:
    if data.get("status") == "need_selection":
        return PRODUCT_SELECTION_TEXT

    text = data.get("answer") or data.get("message")
    if isinstance(text, str) and text.strip():
        return _compact_for_speech(text)
    return FALLBACK_ASSISTANT_TEXT


def _chat_payload(request: VoiceTurnRequest) -> dict[str, Any]:
    selected_index = request.selected_index or parse_spoken_selection(request.transcript)
    payload: dict[str, Any] = {"message": request.transcript}
    if request.conversation_id:
        payload["conversation_id"] = request.conversation_id
    if selected_index is not None:
        payload["selected_index"] = selected_index
        payload["message"] = str(selected_index)
    if request.selected_sku:
        payload["selected_sku"] = request.selected_sku
    return payload


def parse_spoken_selection(transcript: str) -> int | None:
    normalized = _normalize_digits(transcript).lower().strip()
    if re.fullmatch(r"[1-9][0-9]?", normalized):
        return int(normalized)

    match = re.search(
        r"\b(?:chon|chọn|lua chon|lựa chọn|lay|lấy|xem)\s*"
        r"(?:so|số|option|lua chon|lựa chọn)?\s*([1-9][0-9]?)\b",
        normalized,
    )
    if match:
        return int(match.group(1))
    return None


def _normalize_digits(text: str) -> str:
    replacements = {
        "một": "1",
        "mot": "1",
        "hai": "2",
        "ba": "3",
        "bốn": "4",
        "bon": "4",
        "tư": "4",
        "tu": "4",
        "năm": "5",
        "nam": "5",
        "sáu": "6",
        "sau": "6",
        "bảy": "7",
        "bay": "7",
        "tám": "8",
        "tam": "8",
        "chín": "9",
        "chin": "9",
    }
    normalized = text
    for word, digit in replacements.items():
        normalized = re.sub(rf"\b{re.escape(word)}\b", digit, normalized, flags=re.IGNORECASE)
    return normalized


def _compact_for_speech(text: str) -> str:
    compact = re.sub(r"\s+", " ", text).strip()
    compact = compact.replace("**", "")
    return compact
