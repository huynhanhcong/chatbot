from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field


@dataclass
class VoiceTrace:
    voice_session_id: str
    started_at: float = field(default_factory=time.perf_counter)
    stt_latency_ms: float | None = None
    chatbot_latency_ms: float | None = None
    tts_latency_ms: float | None = None
    interruption_count: int = 0


class VoiceObserver:
    def __init__(self, logger: logging.Logger | None = None) -> None:
        self.logger = logger or logging.getLogger("voice")

    def start(self, voice_session_id: str) -> VoiceTrace:
        return VoiceTrace(voice_session_id=voice_session_id)

    def finish(
        self,
        *,
        trace: VoiceTrace,
        conversation_id: str | None,
        route: str | None,
        intent: str | None,
        status: str | None,
    ) -> None:
        elapsed_ms = (time.perf_counter() - trace.started_at) * 1000
        self.logger.info(
            "voice_turn voice_session_id=%s conversation_id=%s route=%s intent=%s status=%s "
            "stt_latency_ms=%s chatbot_latency_ms=%s tts_latency_ms=%s interruptions=%s total_latency_ms=%.2f",
            trace.voice_session_id,
            conversation_id,
            route,
            intent,
            status,
            _format_ms(trace.stt_latency_ms),
            _format_ms(trace.chatbot_latency_ms),
            _format_ms(trace.tts_latency_ms),
            trace.interruption_count,
            elapsed_ms,
        )


def _format_ms(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.2f}"

