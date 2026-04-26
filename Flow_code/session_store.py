from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field, replace
from threading import RLock
from typing import Any, Callable

from .models import ProductDetail, ProductOption
from .redis_utils import create_redis_client, redis_get_json, redis_set_json


@dataclass(frozen=True)
class DrugTurn:
    question: str
    answer: str
    timestamp: float


@dataclass(frozen=True)
class SearchSession:
    conversation_id: str
    drug_name: str
    requested_question: str | None
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
        question: str | None = None,
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
                requested_question=(question or "").strip() or None,
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


class RedisSessionStore:
    def __init__(
        self,
        redis_url: str,
        ttl_seconds: int = 30 * 60,
        max_turns: int = 6,
        max_summary_chars: int = 2200,
        client: Any | None = None,
        time_func: Callable[[], float] | None = None,
    ) -> None:
        self.ttl_seconds = ttl_seconds
        self.max_turns = max_turns
        self.max_summary_chars = max_summary_chars
        self._time_func = time_func or time.monotonic
        self._client = client or create_redis_client(redis_url)
        self._prefix = "chatbot:drug:"

    def save_search(
        self,
        drug_name: str,
        options: list[ProductOption],
        question: str | None = None,
        conversation_id: str | None = None,
    ) -> SearchSession:
        session_id = conversation_id or str(uuid.uuid4())
        session = SearchSession(
            conversation_id=session_id,
            drug_name=drug_name,
            requested_question=(question or "").strip() or None,
            options=options,
            expires_at=self._time_func() + self.ttl_seconds,
        )
        self._save(session)
        return session

    def save_selected_detail(
        self,
        conversation_id: str,
        selected_detail: ProductDetail,
        last_answer: str,
        question: str | None = None,
    ) -> SearchSession | None:
        session = self.get(conversation_id)
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
        self._save(updated)
        return updated

    def get(self, conversation_id: str | None) -> SearchSession | None:
        if not conversation_id:
            return None
        payload = redis_get_json(self._client, self._key(conversation_id))
        if not isinstance(payload, dict):
            return None
        return _search_session_from_json(payload)

    def delete(self, conversation_id: str) -> None:
        self._client.delete(self._key(conversation_id))

    def _save(self, session: SearchSession) -> None:
        redis_set_json(self._client, self._key(session.conversation_id), session, self.ttl_seconds)

    def _key(self, conversation_id: str) -> str:
        return self._prefix + conversation_id


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


def _product_option_from_json(payload: dict[str, Any]) -> ProductOption:
    return ProductOption(
        index=int(payload.get("index") or 0),
        sku=str(payload.get("sku") or ""),
        slug=payload.get("slug"),
        name=str(payload.get("name") or ""),
        brand=payload.get("brand"),
        price=payload.get("price"),
        detail_url=payload.get("detail_url"),
        image_url=payload.get("image_url"),
        ingredients=list(payload.get("ingredients") or []),
        is_prescription_drug=bool(payload.get("is_prescription_drug")),
    )


def _product_detail_from_json(payload: dict[str, Any]) -> ProductDetail:
    return ProductDetail(
        sku=str(payload.get("sku") or ""),
        slug=payload.get("slug"),
        name=str(payload.get("name") or ""),
        brand=payload.get("brand"),
        product_type=payload.get("product_type"),
        category=payload.get("category"),
        short_description=payload.get("short_description"),
        long_description=payload.get("long_description"),
        ingredients=list(payload.get("ingredients") or []),
        variants=list(payload.get("variants") or []),
        is_prescription_drug=bool(payload.get("is_prescription_drug")),
        source_url=payload.get("source_url"),
        raw=dict(payload.get("raw") or {}),
    )


def _drug_turn_from_json(payload: dict[str, Any]) -> DrugTurn:
    return DrugTurn(
        question=str(payload.get("question") or ""),
        answer=str(payload.get("answer") or ""),
        timestamp=float(payload.get("timestamp") or 0.0),
    )


def _search_session_from_json(payload: dict[str, Any]) -> SearchSession:
    selected_detail = payload.get("selected_detail")
    return SearchSession(
        conversation_id=str(payload.get("conversation_id") or ""),
        drug_name=str(payload.get("drug_name") or ""),
        requested_question=payload.get("requested_question"),
        options=[_product_option_from_json(item) for item in payload.get("options") or [] if isinstance(item, dict)],
        expires_at=float(payload.get("expires_at") or 0.0),
        selected_detail=_product_detail_from_json(selected_detail) if isinstance(selected_detail, dict) else None,
        last_answer=payload.get("last_answer"),
        summary=str(payload.get("summary") or ""),
        turns=[_drug_turn_from_json(item) for item in payload.get("turns") or [] if isinstance(item, dict)],
    )
