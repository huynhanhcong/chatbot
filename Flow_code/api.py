from __future__ import annotations

import logging
import time
from typing import Any

from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict, Field

from Flow_code.arduino_service import (
    ArduinoBusyError,
    ArduinoDispenseError,
    ArduinoDispenseService,
    ArduinoTimeoutError,
)
from Flow_code.chat_orchestrator import ChatOrchestrator
from Flow_code.conversation_memory import InMemoryConversationStore, RedisConversationStore
from Flow_code.dialogue_state import InMemoryDialogueStateStore, RedisDialogueStateStore
from Flow_code.drug_service import DrugInfoService
from Flow_code.hospital_session import InMemoryHospitalSessionStore, RedisHospitalSessionStore
from Flow_code.observability import ChatObserver
from Flow_code.pharmacity_client import (
    PharmacityClientError,
    PharmacityNetworkError,
    PharmacityParsingError,
)
from Flow_code.pharmacity_flow import (
    FlowDependencyError,
    FlowNotFoundError,
    FlowValidationError,
    PharmacityFlow,
)
from Flow_code.pharmacity_export import (
    DEFAULT_PHARMACITY_EXPORT_PATH,
    DEFAULT_PHARMACITY_EXTRACTED_PATH,
    clear_pharmacity_export_files,
)
from Flow_code.session_store import InMemorySessionStore, RedisSessionStore
from Flow_code.router_service import (
    IntentRouter,
    RouteContext,
    has_strong_drug_lookup_signal,
    looks_like_contextual_follow_up,
    looks_like_drug_question,
    looks_like_hospital_question,
    looks_like_product_selection,
    normalize_vi,
)
from RAG_app.config import ROOT, load_settings


WEB_APP_DIR = ROOT / "Web_app"
CHAT_VOICE_WEB_DIR = ROOT / "Chat_Voice" / "web"
ROBOT_WEB_DIR = ROOT / "Web_new"
ROBOT_LEGACY_WEB_DIR = ROOT / "Web_robot"
SPEED_LOG_PATHS = {"/chat", "/chat/drug-info"}
PHARMACITY_EXPORT_PATHS = (
    DEFAULT_PHARMACITY_EXPORT_PATH,
    DEFAULT_PHARMACITY_EXTRACTED_PATH,
)

logger = logging.getLogger("uvicorn.error")

app = FastAPI(title="Medical Chatbot API", version="0.1.0")
if WEB_APP_DIR.exists():
    app.mount("/static", StaticFiles(directory=WEB_APP_DIR), name="static")
if CHAT_VOICE_WEB_DIR.exists():
    app.mount("/voice/static", StaticFiles(directory=CHAT_VOICE_WEB_DIR), name="voice-static")
if ROBOT_WEB_DIR.exists():
    app.mount("/robot/static", StaticFiles(directory=ROBOT_WEB_DIR), name="robot-static")
if ROBOT_LEGACY_WEB_DIR.exists():
    app.mount("/robot-legacy/static", StaticFiles(directory=ROBOT_LEGACY_WEB_DIR), name="robot-legacy-static")

_flow: PharmacityFlow | None = None
_rag_pipeline: Any | None = None
_settings = load_settings()
_arduino_service = ArduinoDispenseService.from_env()


def _build_conversation_store() -> Any:
    if _settings.use_redis and _settings.redis_url:
        return RedisConversationStore(_settings.redis_url)
    return InMemoryConversationStore()


def _build_hospital_session_store() -> Any:
    if _settings.use_redis and _settings.redis_url:
        return RedisHospitalSessionStore(_settings.redis_url)
    return InMemoryHospitalSessionStore()


def _build_dialogue_state_store() -> Any:
    if _settings.use_redis and _settings.redis_url:
        return RedisDialogueStateStore(_settings.redis_url)
    return InMemoryDialogueStateStore()


def _build_drug_session_store() -> Any:
    if _settings.use_redis and _settings.redis_url:
        return RedisSessionStore(_settings.redis_url)
    return InMemorySessionStore()


_conversations = _build_conversation_store()
_hospital_sessions = _build_hospital_session_store()
_dialogue_states = _build_dialogue_state_store()
_drug_sessions = _build_drug_session_store()


class ChatRequest(BaseModel):
    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {"message": "Hãy cho tôi biết thông tin về thuốc Oresol"},
                {"message": "Gói IVF Standard gồm gì?"},
            ]
        }
    )

    message: str = Field(..., min_length=1)
    conversation_id: str | None = None
    selected_index: int | None = Field(default=None, ge=1)
    selected_sku: str | None = None


DrugInfoRequest = ChatRequest


class RobotDispenseRequest(BaseModel):
    code: str = Field(..., min_length=1)
    source: str = Field(default="manual", min_length=1)


class VoiceMetricRequest(BaseModel):
    metric: str = Field(..., min_length=1)
    elapsed_ms: float = Field(..., ge=0)
    conversation_id: str | None = None
    label: str | None = None


@app.middleware("http")
async def log_response_speed(request: Request, call_next: Any) -> Response:
    start_time = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        elapsed_ms = _elapsed_ms(start_time)
        if _should_log_speed(request.url.path):
            logger.exception(
                "SPEED %s %s failed after %.2f ms (%.2f s)",
                request.method,
                request.url.path,
                elapsed_ms,
                elapsed_ms / 1000,
            )
        raise

    elapsed_ms = _elapsed_ms(start_time)
    response.headers["X-Response-Time-Ms"] = f"{elapsed_ms:.2f}"
    if _should_log_speed(request.url.path):
        logger.info(
            "SPEED %s %s -> %s in %.2f ms (%.2f s)",
            request.method,
            request.url.path,
            response.status_code,
            elapsed_ms,
            elapsed_ms / 1000,
        )
    return response


def get_flow() -> PharmacityFlow:
    global _flow
    if _flow is None:
        _flow = PharmacityFlow(session_store=_drug_sessions)
    return _flow


def get_existing_flow() -> PharmacityFlow | None:
    return _flow


def get_rag_pipeline() -> Any:
    global _rag_pipeline
    if _rag_pipeline is None:
        from RAG_app.pipeline import RagPipeline

        _rag_pipeline = RagPipeline(_settings)
    return _rag_pipeline


def get_drug_service() -> DrugInfoService:
    return DrugInfoService(
        flow_provider=get_flow,
        existing_flow_provider=get_existing_flow,
    )


def get_chat_orchestrator() -> ChatOrchestrator:
    return ChatOrchestrator(
        conversation_store=_conversations,
        hospital_session_store=_hospital_sessions,
        dialogue_state_store=_dialogue_states,
        drug_service=get_drug_service(),
        rag_pipeline_provider=get_rag_pipeline,
        router=IntentRouter(),
        observer=ChatObserver(logger),
    )


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/")
def root() -> FileResponse:
    return _index_response()


@app.get("/app")
def web_app() -> FileResponse:
    return _index_response()


@app.get("/voice")
def voice_app() -> FileResponse:
    index_path = CHAT_VOICE_WEB_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="Voice app not found.")
    return FileResponse(index_path, headers={"Cache-Control": "no-store"})


@app.get("/robot")
def robot_app() -> FileResponse:
    index_path = ROBOT_WEB_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="Robot app not found.")
    return FileResponse(index_path, headers={"Cache-Control": "no-store"})


@app.get("/robot-legacy")
def robot_legacy_app() -> HTMLResponse:
    index_path = ROBOT_LEGACY_WEB_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="Robot legacy app not found.")
    html = index_path.read_text(encoding="utf-8").replace("/robot/static/", "/robot-legacy/static/")
    return HTMLResponse(html, headers={"Cache-Control": "no-store"})


@app.get("/favicon.ico", include_in_schema=False)
def favicon() -> Response:
    return Response(status_code=204)


@app.post("/robot/api/medicine/dispense")
def robot_dispense_medicine(payload: RobotDispenseRequest) -> dict[str, str]:
    try:
        return _arduino_service.dispense(code=payload.code, source=payload.source)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except ArduinoBusyError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    except ArduinoTimeoutError as exc:
        raise HTTPException(status_code=504, detail=str(exc)) from exc
    except ArduinoDispenseError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/robot/api/voice-metrics")
def robot_voice_metrics(payload: VoiceMetricRequest) -> dict[str, str]:
    logger.info(
        "ROBOT VOICE metric=%s elapsed_ms=%.2f conversation_id=%s label=%s",
        payload.metric,
        payload.elapsed_ms,
        payload.conversation_id or "",
        payload.label or "",
    )
    return {"status": "ok"}


@app.post("/chat")
def chat(payload: ChatRequest) -> dict[str, Any]:
    try:
        _clear_pharmacity_exports_for_new_conversation(payload)
        return get_chat_orchestrator().handle(payload)
    except FlowNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except FlowValidationError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except (PharmacityNetworkError, PharmacityParsingError) as exc:
        raise HTTPException(status_code=502, detail=f"Drug data provider error: {exc}") from exc
    except FlowDependencyError as exc:
        raise HTTPException(status_code=500, detail=f"Dependency error: {exc}") from exc
    except PharmacityClientError as exc:
        raise HTTPException(status_code=500, detail=f"Drug information service error: {exc}") from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@app.post("/chat/drug-info")
def chat_drug_info(payload: DrugInfoRequest) -> dict[str, Any]:
    try:
        _clear_pharmacity_exports_for_new_conversation(payload)
        return _call_pharmacity(payload)
    except FlowNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except FlowValidationError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    except (PharmacityNetworkError, PharmacityParsingError) as exc:
        raise HTTPException(status_code=502, detail=f"Drug data provider error: {exc}") from exc
    except FlowDependencyError as exc:
        raise HTTPException(status_code=500, detail=f"Dependency error: {exc}") from exc
    except PharmacityClientError as exc:
        raise HTTPException(status_code=500, detail=f"Drug information service error: {exc}") from exc


def _call_pharmacity(
    payload: ChatRequest,
    conversation_id: str | None = None,
) -> dict[str, Any]:
    return get_drug_service().handle_public(
        message=payload.message,
        conversation_id=conversation_id or payload.conversation_id,
        selected_index=payload.selected_index,
        selected_sku=payload.selected_sku,
    )


def _clear_pharmacity_exports_for_new_conversation(payload: ChatRequest) -> None:
    if payload.conversation_id:
        return
    try:
        clear_pharmacity_export_files(PHARMACITY_EXPORT_PATHS)
    except OSError as exc:
        logger.warning("pharmacity_export_clear_failed error=%s", exc)


def _should_route_to_pharmacity(payload: ChatRequest) -> bool:
    return _route_chat(payload, _conversations.get_or_create(payload.conversation_id)) == "pharmacity"


def _route_chat(payload: ChatRequest, conversation: Any) -> str:
    if conversation is None:
        conversation = _conversations.get_or_create(payload.conversation_id)
    state = _dialogue_states.get_or_create(conversation.conversation_id)
    drug_service = get_drug_service()
    decision = IntentRouter().classify(
        message=payload.message,
        selected_index=payload.selected_index,
        selected_sku=payload.selected_sku,
        context=RouteContext(
            conversation=conversation,
            state=state,
            pharmacity_session=drug_service.get_session(conversation.conversation_id),
            hospital_active=_hospital_sessions.has_active_session(conversation.conversation_id),
        ),
    )
    return decision.route


def _looks_like_drug_question(message: str) -> bool:
    return looks_like_drug_question(normalize_vi(message))


def _has_strong_drug_lookup_signal(message: str) -> bool:
    return has_strong_drug_lookup_signal(normalize_vi(message))


def _looks_like_hospital_question(message: str) -> bool:
    return looks_like_hospital_question(normalize_vi(message))


def _looks_like_product_selection(message: str) -> bool:
    return looks_like_product_selection(normalize_vi(message))


def _looks_like_contextual_follow_up(message: str) -> bool:
    return looks_like_contextual_follow_up(normalize_vi(message))


def _normalize_vi(value: str) -> str:
    return normalize_vi(value)


def _index_response() -> FileResponse:
    index_path = WEB_APP_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=404, detail="Web app not found.")
    return FileResponse(index_path, headers={"Cache-Control": "no-store"})


def _elapsed_ms(start_time: float) -> float:
    return (time.perf_counter() - start_time) * 1000


def _should_log_speed(path: str) -> bool:
    return path in SPEED_LOG_PATHS
