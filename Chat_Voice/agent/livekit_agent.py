from __future__ import annotations

"""
LiveKit production scaffold.

The runnable browser demo in Chat_Voice/web uses browser STT/TTS so the project
works without paid services. This module documents the production integration
point: a LiveKit worker should call VoiceChatBridge for every final transcript
and stream VoiceTurnResponse.assistant_text through the configured TTS provider.
"""

import logging

from .chat_bridge import VoiceChatBridge
from .config import load_voice_settings
from .models import VoiceTurnRequest
from .observability import VoiceObserver


logger = logging.getLogger("voice.livekit")


def build_bridge() -> VoiceChatBridge:
    settings = load_voice_settings()
    return VoiceChatBridge(settings.chat_api_url)


def handle_final_transcript(
    transcript: str,
    *,
    conversation_id: str | None,
    voice_session_id: str,
) -> dict:
    observer = VoiceObserver(logger)
    trace = observer.start(voice_session_id)
    bridge = build_bridge()
    response = bridge.handle_turn(
        VoiceTurnRequest(
            transcript=transcript,
            conversation_id=conversation_id,
            voice_session_id=voice_session_id,
        )
    )
    trace.chatbot_latency_ms = response.latency_ms
    observer.finish(
        trace=trace,
        conversation_id=response.conversation_id,
        route=response.route,
        intent=response.intent,
        status=response.status,
    )
    return response.to_metadata()


if __name__ == "__main__":
    raise SystemExit(
        "This is a LiveKit integration scaffold. Run the current web voice demo "
        "through the main FastAPI app at /voice, or wire this bridge into a "
        "LiveKit Agents worker with your chosen STT/TTS plugins."
    )

