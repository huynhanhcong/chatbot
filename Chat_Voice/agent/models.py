from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4


@dataclass(frozen=True)
class VoiceTurnRequest:
    transcript: str
    conversation_id: str | None = None
    voice_session_id: str = field(default_factory=lambda: f"voice-{uuid4().hex}")
    selected_index: int | None = None
    selected_sku: str | None = None


@dataclass(frozen=True)
class VoiceTurnResponse:
    conversation_id: str
    voice_session_id: str
    transcript: str
    assistant_text: str
    route: str | None
    intent: str | None
    status: str | None
    confidence: str | None
    latency_ms: float
    raw_response: dict[str, Any]

    def to_metadata(self) -> dict[str, Any]:
        return {
            "conversation_id": self.conversation_id,
            "voice_session_id": self.voice_session_id,
            "transcript": self.transcript,
            "assistant_text": self.assistant_text,
            "route": self.route,
            "intent": self.intent,
            "status": self.status,
            "confidence": self.confidence,
            "latency_ms": self.latency_ms,
        }

