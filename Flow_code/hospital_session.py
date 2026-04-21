from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from threading import RLock
from typing import Any, Callable


@dataclass(frozen=True)
class HospitalTurn:
    question: str
    standalone_question: str
    answer: str
    sources: list[dict[str, Any]]
    timestamp: float


@dataclass
class HospitalSession:
    conversation_id: str
    expires_at: float
    summary: str = ""
    turns: list[HospitalTurn] = field(default_factory=list)


class InMemoryHospitalSessionStore:
    def __init__(
        self,
        ttl_seconds: int = 30 * 60,
        max_sessions: int = 1000,
        max_turns: int = 6,
        max_summary_chars: int = 2200,
        time_func: Callable[[], float] | None = None,
    ) -> None:
        self.ttl_seconds = ttl_seconds
        self.max_sessions = max_sessions
        self.max_turns = max_turns
        self.max_summary_chars = max_summary_chars
        self._time_func = time.monotonic
        if time_func is not None:
            self._time_func = time_func
        self._sessions: dict[str, HospitalSession] = {}
        self._lock = RLock()

    def get_or_create(self, conversation_id: str | None = None) -> HospitalSession:
        with self._lock:
            self._cleanup_expired_locked()
            if conversation_id:
                session = self._get_locked(conversation_id)
                if session is not None:
                    return session

            if len(self._sessions) >= self.max_sessions:
                self._drop_oldest_locked()

            session_id = conversation_id or str(uuid.uuid4())
            session = HospitalSession(
                conversation_id=session_id,
                expires_at=self._time_func() + self.ttl_seconds,
            )
            self._sessions[session_id] = session
            return session

    def get(self, conversation_id: str | None) -> HospitalSession | None:
        if not conversation_id:
            return None
        with self._lock:
            return self._get_locked(conversation_id)

    def _get_locked(self, conversation_id: str | None) -> HospitalSession | None:
        if not conversation_id:
            return None
        session = self._sessions.get(conversation_id)
        if session is None:
            return None
        if session.expires_at <= self._time_func():
            self._sessions.pop(conversation_id, None)
            return None
        return session

    def save_turn(
        self,
        conversation_id: str,
        question: str,
        standalone_question: str,
        answer: str,
        sources: list[dict[str, Any]],
    ) -> HospitalSession | None:
        with self._lock:
            session = self._get_locked(conversation_id)
            if session is None:
                return None

            turn = HospitalTurn(
                question=question,
                standalone_question=standalone_question,
                answer=answer,
                sources=sources,
                timestamp=self._time_func(),
            )
            session.turns.append(turn)
            if len(session.turns) > self.max_turns:
                overflow_count = len(session.turns) - self.max_turns
                overflow = session.turns[:overflow_count]
                session.summary = _merge_hospital_summary(
                    session.summary,
                    overflow,
                    max_chars=self.max_summary_chars,
                )
                del session.turns[:overflow_count]
            session.expires_at = self._time_func() + self.ttl_seconds
            return session

    def has_active_session(self, conversation_id: str | None) -> bool:
        return self.get(conversation_id) is not None

    def delete(self, conversation_id: str) -> None:
        with self._lock:
            self._sessions.pop(conversation_id, None)

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


def _merge_hospital_summary(
    existing: str,
    turns: list[HospitalTurn],
    *,
    max_chars: int,
) -> str:
    additions = [_format_hospital_summary_turn(turn) for turn in turns]
    summary = "\n".join(part for part in [existing, *additions] if part.strip())
    if len(summary) <= max_chars:
        return summary
    return summary[-max_chars:].lstrip()


def _format_hospital_summary_turn(turn: HospitalTurn) -> str:
    titles = [
        str(source.get("title"))
        for source in turn.sources
        if isinstance(source, dict) and source.get("title")
    ]
    title_text = f" Nguon/chu de: {', '.join(titles[:2])}." if titles else ""
    return (
        f"- User hoi: {_truncate(turn.question, 180)} | "
        f"Cau hoi doc lap: {_truncate(turn.standalone_question, 220)} | "
        f"Assistant: {_truncate(turn.answer, 260)}.{title_text}"
    )


def _truncate(value: str, max_chars: int) -> str:
    value = " ".join(str(value).split())
    if len(value) <= max_chars:
        return value
    return value[: max_chars - 13].rstrip() + " ...[rut gon]"
