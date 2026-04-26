# Chat Voice

Voice expansion for the medical/pharmacy/customer-support chatbot.

## Architecture

The voice layer is an adapter over the existing chatbot. It does not reimplement
routing, memory, RAG, Pharmacity lookup, or medical guardrails.

```text
Browser microphone
  -> STT transcript
  -> existing POST /chat
  -> assistant text + metadata
  -> TTS playback
```

The runnable v1 UI uses browser speech recognition and speech synthesis so it
can be tested without paid services. The production path is prepared in
`agent/livekit_agent.py`: wire LiveKit Agents STT/TTS plugins to
`VoiceChatBridge`, then stream `assistant_text` back to the user.

## Run

Install the current chatbot dependencies first, then run the main FastAPI app:

```powershell
python -m pip install -r requirements-rag.txt
python -m uvicorn Flow_code.api:app --reload
```

Open:

```text
http://127.0.0.1:8000/voice
```

Chrome/Edge work best for browser Vietnamese speech recognition. If the browser
does not support speech recognition, use the text fallback input on the same
page.

## Voice Environment

```env
VOICE_CHAT_API_URL=http://127.0.0.1:8000/chat
VOICE_STT_PROVIDER=browser
VOICE_TTS_PROVIDER=browser

LIVEKIT_URL=
LIVEKIT_API_KEY=
LIVEKIT_API_SECRET=
GROQ_API_KEY=
```

## Provider Matrix

| Layer | v1 default | Free/open fallback | Production notes |
| --- | --- | --- | --- |
| Transport | Browser + `/chat` | LiveKit self-host | LiveKit is the target for WebRTC, interruption, and telephony later. |
| STT | Browser STT | `faster-whisper`, PhoWhisper | Groq Whisper free tier can be used when API quota is acceptable. |
| Chat brain | Existing `/chat` | Existing `/chat` | Keep all routing/RAG/session behavior centralized. |
| TTS | Browser TTS | VieNeu-TTS, Kokoro, Piper | Use Vietnamese-first TTS for production voice quality. |
| Observability | Browser latency + API logs | `VoiceObserver` | Track STT, chatbot, TTS, route, intent, interruptions. |

## GitHub Findings

- LiveKit Agents: best fit for production web microphone/WebRTC voice agents.
  https://github.com/livekit/agents
- Pipecat: strong alternative for composable multimodal voice pipelines.
  https://github.com/pipecat-ai/pipecat
- Bolna: useful telephony-oriented reference.
  https://github.com/bolna-ai/bolna
- faster-whisper: efficient local Whisper inference.
  https://github.com/SYSTRAN/faster-whisper
- PhoWhisper: Vietnamese ASR model family.
  https://github.com/VinAIResearch/PhoWhisper
- VieNeu-TTS: Vietnamese local TTS candidate.
  https://github.com/pnnbao97/VieNeu-TTS
- RealtimeSTT: useful VAD/wake-word reference, but community-driven now.
  https://github.com/KoljaB/RealtimeSTT

## Roadmap

1. Replace browser STT/TTS with LiveKit Agents in `agent/livekit_agent.py`.
2. Add Groq Whisper STT and Vietnamese TTS provider adapters behind
   `VOICE_STT_PROVIDER` and `VOICE_TTS_PROVIDER`.
3. Add barge-in support: cancel TTS when the user starts speaking.
4. Add structured metrics export for STT latency, chatbot latency, TTS latency,
   route, intent, and interruption count.
5. Fix mojibake in existing Vietnamese UI/backend strings before production
   demos; spoken output magnifies encoding mistakes.
6. Require Redis for multi-worker deployment so text and voice sessions share
   durable memory.

## Safety

Voice answers are for information lookup only. For urgent symptoms, medication
allergy, pregnancy, severe pain, breathing trouble, or suspected overdose, the
assistant should direct the user to a clinician, pharmacist, emergency service,
or official hospital contact channel.

