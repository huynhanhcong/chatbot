from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv


@dataclass(frozen=True)
class VoiceSettings:
    chat_api_url: str
    livekit_url: str | None
    livekit_api_key: str | None
    livekit_api_secret: str | None
    stt_provider: str
    tts_provider: str
    groq_api_key: str | None


def load_voice_settings() -> VoiceSettings:
    load_dotenv()
    return VoiceSettings(
        chat_api_url=os.getenv("VOICE_CHAT_API_URL", "http://127.0.0.1:8000/chat"),
        livekit_url=os.getenv("LIVEKIT_URL"),
        livekit_api_key=os.getenv("LIVEKIT_API_KEY"),
        livekit_api_secret=os.getenv("LIVEKIT_API_SECRET"),
        stt_provider=os.getenv("VOICE_STT_PROVIDER", "browser"),
        tts_provider=os.getenv("VOICE_TTS_PROVIDER", "browser"),
        groq_api_key=os.getenv("GROQ_API_KEY"),
    )

