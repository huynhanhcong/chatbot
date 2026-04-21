from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field, replace
from threading import RLock
from typing import Callable

from .models import ProductDetail, ProductOption


@dataclass(frozen=True)
class DrugTurn:
    question: str
    answer: str
    timestamp: float


@dataclass(frozen=True)
class SearchSession:
    conversation_id: str
    drug_name: str
    options: list[ProductOption]
    expires_at: float
    selected_detail: ProductDetail | None = None
    last_answer: str | None = None
    summary: str = ""
    turns: list[DrugTurn] = field(default_factory=list)


class InMemorySessionStore:
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
        self._time_func = time_func or time.monotonic
        self._sessions: dict[str, SearchSession] = {}
        self._lock = RLock()

    def save_search(
        self,
        drug_name: str,
        options: list[ProductOption],
        conversation_id: str | None = None,
    ) -> SearchSession:
        with self._lock:
            self._cleanup_expired_locked()
            if len(self._sessions) >= self.max_sessions:
                self._drop_oldest_locked()

            session_id = conversation_id or str(uuid.uuid4())
            session = SearchSession(
                conversation_id=session_id,
                drug_name=drug_name,
                options=options,
                expires_at=self._time_func() + self.ttl_seconds,
            )
            self._sessions[session_id] = session
            return session

    def save_selected_detail(
        self,
        conversation_id: str,
        selected_detail: ProductDetail,
        last_answer: str,
        question: str | None = None,
    ) -> SearchSession | None:
        with self._lock:
            session = self._get_locked(conversation_id)
            if session is None:
                return None

            turns = list(session.turns)
            summary = session.summary
            if question:
                turns.append(
                    DrugTurn(
                        question=question,
                        answer=last_answer,
                        timestamp=self._time_func(),
                    )
                )
                if len(turns) > self.max_turns:
                    overflow_count = len(turns) - self.max_turns
                    summary = _merge_drug_summary(
                        summary,
                        turns[:overflow_count],
                        max_chars=self.max_summary_chars,
                    )
                    turns = turns[overflow_count:]

            updated = replace(
                session,
                selected_detail=selected_detail,
                last_answer=last_answer,
                summary=summary,
                turns=turns,
                expires_at=self._time_func() + self.ttl_seconds,
            )
            self._sessions[conversation_id] = updated
            return updated

    def get(self, conversation_id: str | None) -> SearchSession | None:
        if not conversation_id:
            return None
        with self._lock:
            return self._get_locked(conversation_id)

    def _get_locked(self, conversation_id: str | None) -> SearchSession | None:
        if not conversation_id:
            return None
        session = self._sessions.get(conversation_id)
        if session is None:
            return None
        if session.expires_at <= self._time_func():
            self._sessions.pop(conversation_id, None)
            return None
        return session

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


def format_drug_history(session: SearchSession, max_turns: int = 4, max_chars: int = 3500) -> str:
    parts: list[str] = []
    if session.summary:
        parts.append(f"TOM_TAT_TRUOC_DO:\n{session.summary}")
    if session.turns:
        rendered_turns = []
        for turn in session.turns[-max_turns:]:
            rendered_turns.append(
                "User: "
                + _truncate(turn.question, 220)
                + "\nAssistant: "
                + _truncate(turn.answer, 700)
            )
        parts.append("LUOT_GAN_DAY:\n" + "\n\n".join(rendered_turns))
    text = "\n\n".join(parts).strip()
    if len(text) <= max_chars:
        return text
    return text[-max_chars:].lstrip()


def _merge_drug_summary(
    existing: str,
    turns: list[DrugTurn],
    *,
    max_chars: int,
) -> str:
    additions = [
        f"- User hoi: {_truncate(turn.question, 180)} | Assistant: {_truncate(turn.answer, 260)}"
        for turn in turns
    ]
    summary = "\n".join(part for part in [existing, *additions] if part.strip())
    if len(summary) <= max_chars:
        return summary
    return summary[-max_chars:].lstrip()


def _truncate(value: str, max_chars: int) -> str:
    value = " ".join(str(value).split())
    if len(value) <= max_chars:
        return value
    return value[: max_chars - 13].rstrip() + " ...[rut gon]"
