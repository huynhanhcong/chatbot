from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from threading import RLock
from typing import Any, Callable, Literal

from .redis_utils import create_redis_client, redis_get_json, redis_set_json


RouteName = Literal["hospital_rag", "pharmacity"]


@dataclass(frozen=True)
class ConversationTurn:
    route: RouteName
    user_message: str
    assistant_message: str | None
    timestamp: float
    standalone_question: str | None = None
    sources: list[dict[str, Any]] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ConversationSession:
    conversation_id: str
    expires_at: float
    active_route: RouteName | None = None
    summary: str = ""
    turns: list[ConversationTurn] = field(default_factory=list)


class InMemoryConversationStore:
    def __init__(
        self,
        ttl_seconds: int = 30 * 60,
        max_sessions: int = 1000,
        max_recent_turns: int = 8,
        max_summary_chars: int = 2500,
        time_func: Callable[[], float] | None = None,
    ) -> None:
        self.ttl_seconds = ttl_seconds
        self.max_sessions = max_sessions
        self.max_recent_turns = max_recent_turns
        self.max_summary_chars = max_summary_chars
        self._time_func = time_func or time.monotonic
        self._sessions: dict[str, ConversationSession] = {}
        self._lock = RLock()

    def get_or_create(self, conversation_id: str | None = None) -> ConversationSession:
        with self._lock:
            self._cleanup_expired_locked()
            if conversation_id:
                session = self._get_locked(conversation_id)
                if session is not None:
                    return session

            if len(self._sessions) >= self.max_sessions:
                self._drop_oldest_locked()

            session_id = conversation_id or str(uuid.uuid4())
            session = ConversationSession(
                conversation_id=session_id,
                expires_at=self._time_func() + self.ttl_seconds,
            )
            self._sessions[session_id] = session
            return session

    def get(self, conversation_id: str | None) -> ConversationSession | None:
        if not conversation_id:
            return None
        with self._lock:
            return self._get_locked(conversation_id)

    def save_turn(
        self,
        *,
        conversation_id: str,
        route: RouteName,
        user_message: str,
        assistant_message: str | None,
        standalone_question: str | None = None,
        sources: list[dict[str, Any]] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ConversationSession:
        with self._lock:
            session = self.get_or_create(conversation_id)
            turn = ConversationTurn(
                route=route,
                user_message=user_message,
                assistant_message=assistant_message,
                standalone_question=standalone_question,
                sources=sources or [],
                metadata=metadata or {},
                timestamp=self._time_func(),
            )
            session.turns.append(turn)
            session.active_route = route
            session.expires_at = self._time_func() + self.ttl_seconds

            if len(session.turns) > self.max_recent_turns:
                overflow_count = len(session.turns) - self.max_recent_turns
                overflow = session.turns[:overflow_count]
                session.summary = _merge_summary(
                    existing=session.summary,
                    turns=overflow,
                    max_chars=self.max_summary_chars,
                )
                del session.turns[:overflow_count]
            return session

    def set_active_route(self, conversation_id: str, route: RouteName) -> ConversationSession:
        with self._lock:
            session = self.get_or_create(conversation_id)
            session.active_route = route
            session.expires_at = self._time_func() + self.ttl_seconds
            return session

    def delete(self, conversation_id: str) -> None:
        with self._lock:
            self._sessions.pop(conversation_id, None)

    def _get_locked(self, conversation_id: str) -> ConversationSession | None:
        session = self._sessions.get(conversation_id)
        if session is None:
            return None
        if session.expires_at <= self._time_func():
            self._sessions.pop(conversation_id, None)
            return None
        return session

    def _cleanup_expired_locked(self) -> None:
        now = self._time_func()
        expired_ids = [
            conversation_id
            for conversation_id, session in self._sessions.items()
            if session.expires_at <= now
        ]
        for conversation_id in expired_ids:
            self._sessions.pop(conversation_id, None)

    def _drop_oldest_locked(self) -> None:
        if not self._sessions:
            return
        oldest_id = min(self._sessions.items(), key=lambda item: item[1].expires_at)[0]
        self._sessions.pop(oldest_id, None)


class RedisConversationStore:
    def __init__(
        self,
        redis_url: str,
        ttl_seconds: int = 30 * 60,
        max_recent_turns: int = 8,
        max_summary_chars: int = 2500,
        client: Any | None = None,
        time_func: Callable[[], float] | None = None,
    ) -> None:
        self.ttl_seconds = ttl_seconds
        self.max_recent_turns = max_recent_turns
        self.max_summary_chars = max_summary_chars
        self._time_func = time_func or time.monotonic
        self._client = client or create_redis_client(redis_url)
        self._prefix = "chatbot:conversation:"

    def get_or_create(self, conversation_id: str | None = None) -> ConversationSession:
        if conversation_id:
            session = self.get(conversation_id)
            if session is not None:
                return session

        session_id = conversation_id or str(uuid.uuid4())
        session = ConversationSession(
            conversation_id=session_id,
            expires_at=self._time_func() + self.ttl_seconds,
        )
        self._save(session)
        return session

    def get(self, conversation_id: str | None) -> ConversationSession | None:
        if not conversation_id:
            return None
        payload = redis_get_json(self._client, self._key(conversation_id))
        if not isinstance(payload, dict):
            return None
        return _conversation_session_from_json(payload)

    def save_turn(
        self,
        *,
        conversation_id: str,
        route: RouteName,
        user_message: str,
        assistant_message: str | None,
        standalone_question: str | None = None,
        sources: list[dict[str, Any]] | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> ConversationSession:
        session = self.get_or_create(conversation_id)
        turn = ConversationTurn(
            route=route,
            user_message=user_message,
            assistant_message=assistant_message,
            standalone_question=standalone_question,
            sources=sources or [],
            metadata=metadata or {},
            timestamp=self._time_func(),
        )
        session.turns.append(turn)
        session.active_route = route
        session.expires_at = self._time_func() + self.ttl_seconds

        if len(session.turns) > self.max_recent_turns:
            overflow_count = len(session.turns) - self.max_recent_turns
            overflow = session.turns[:overflow_count]
            session.summary = _merge_summary(
                existing=session.summary,
                turns=overflow,
                max_chars=self.max_summary_chars,
            )
            del session.turns[:overflow_count]

        self._save(session)
        return session

    def set_active_route(self, conversation_id: str, route: RouteName) -> ConversationSession:
        session = self.get_or_create(conversation_id)
        session.active_route = route
        session.expires_at = self._time_func() + self.ttl_seconds
        self._save(session)
        return session

    def delete(self, conversation_id: str) -> None:
        self._client.delete(self._key(conversation_id))

    def _save(self, session: ConversationSession) -> None:
        redis_set_json(self._client, self._key(session.conversation_id), session, self.ttl_seconds)

    def _key(self, conversation_id: str) -> str:
        return self._prefix + conversation_id


def format_conversation_context(
    session: ConversationSession | None,
    *,
    max_turns: int = 6,
    max_chars: int = 5000,
) -> str:
    if session is None:
        return ""

    parts: list[str] = []
    if session.summary:
        parts.append(f"TOM_TAT_TRUOC_DO:\n{session.summary}")

    recent_turns = session.turns[-max_turns:]
    if recent_turns:
        parts.append(
            "LUOT_GAN_DAY:\n"
            + "\n\n".join(_format_turn_for_context(turn) for turn in recent_turns)
        )

    context = "\n\n".join(part for part in parts if part.strip()).strip()
    return _truncate_middle(context, max_chars)


def _format_turn_for_context(turn: ConversationTurn) -> str:
    source_titles = [
        str(source.get("title"))
        for source in turn.sources
        if isinstance(source, dict) and source.get("title")
    ]
    source_text = ", ".join(source_titles[:3]) if source_titles else "None"
    lines = [
        f"Route: {turn.route}",
        f"User: {turn.user_message}",
    ]
    if turn.standalone_question and turn.standalone_question != turn.user_message:
        lines.append(f"Standalone question: {turn.standalone_question}")
    if turn.assistant_message:
        lines.append(f"Assistant: {_truncate_end(turn.assistant_message, 900)}")
    lines.append(f"Sources: {source_text}")
    return "\n".join(lines)


def _merge_summary(
    *,
    existing: str,
    turns: list[ConversationTurn],
    max_chars: int,
) -> str:
    additions = [_format_turn_for_summary(turn) for turn in turns]
    summary = "\n".join(part for part in [existing, *additions] if part.strip())
    return _truncate_middle(summary, max_chars)


def _format_turn_for_summary(turn: ConversationTurn) -> str:
    user = _truncate_end(turn.user_message, 180)
    answer = _truncate_end(turn.assistant_message or "", 260)
    subject = _subject_from_turn(turn)
    subject_part = f" | Chu de/nguon: {subject}" if subject else ""
    return f"- {turn.route}: User hoi `{user}`. Assistant tra loi `{answer}`.{subject_part}"


def _subject_from_turn(turn: ConversationTurn) -> str:
    for source in turn.sources:
        if isinstance(source, dict) and source.get("title"):
            return str(source["title"])
    selected = turn.metadata.get("selected_product")
    if isinstance(selected, dict) and selected.get("name"):
        return str(selected["name"])
    return ""


def _truncate_end(value: str, max_chars: int) -> str:
    value = " ".join(str(value).split())
    if len(value) <= max_chars:
        return value
    return value[: max_chars - 14].rstrip() + " ...[rut gon]"


def _truncate_middle(value: str, max_chars: int) -> str:
    value = value.strip()
    if len(value) <= max_chars:
        return value
    head_len = max_chars * 2 // 3
    tail_len = max_chars - head_len - 18
    return value[:head_len].rstrip() + "\n...[rut gon]...\n" + value[-tail_len:].lstrip()


def _conversation_turn_from_json(payload: dict[str, Any]) -> ConversationTurn:
    return ConversationTurn(
        route=str(payload.get("route") or "hospital_rag"),  # type: ignore[arg-type]
        user_message=str(payload.get("user_message") or ""),
        assistant_message=payload.get("assistant_message"),
        timestamp=float(payload.get("timestamp") or 0.0),
        standalone_question=payload.get("standalone_question"),
        sources=list(payload.get("sources") or []),
        metadata=dict(payload.get("metadata") or {}),
    )


def _conversation_session_from_json(payload: dict[str, Any]) -> ConversationSession:
    return ConversationSession(
        conversation_id=str(payload.get("conversation_id") or ""),
        expires_at=float(payload.get("expires_at") or 0.0),
        active_route=payload.get("active_route"),
        summary=str(payload.get("summary") or ""),
        turns=[_conversation_turn_from_json(item) for item in payload.get("turns") or [] if isinstance(item, dict)],
    )
