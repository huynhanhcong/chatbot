from __future__ import annotations

import time
from dataclasses import replace
from threading import RLock
from typing import Any, Callable, Protocol

from .service_contracts import ActiveEntity, DialogueState, DomainName, IntentName


class DialogueStateStore(Protocol):
    def get_or_create(self, conversation_id: str) -> DialogueState:
        ...

    def save(self, state: DialogueState) -> DialogueState:
        ...

    def delete(self, conversation_id: str) -> None:
        ...


class InMemoryDialogueStateStore:
    def __init__(
        self,
        ttl_seconds: int = 30 * 60,
        max_sessions: int = 1000,
        max_entities: int = 10,
        time_func: Callable[[], float] | None = None,
    ) -> None:
        self.ttl_seconds = ttl_seconds
        self.max_sessions = max_sessions
        self.max_entities = max_entities
        self._time_func = time_func or time.monotonic
        self._states: dict[str, tuple[DialogueState, float]] = {}
        self._lock = RLock()

    def get_or_create(self, conversation_id: str) -> DialogueState:
        with self._lock:
            self._cleanup_expired_locked()
            entry = self._states.get(conversation_id)
            if entry is not None:
                state, expires_at = entry
                if expires_at > self._time_func():
                    self._states[conversation_id] = (state, self._expires_at())
                    return state

            if len(self._states) >= self.max_sessions:
                self._drop_oldest_locked()

            state = DialogueState(conversation_id=conversation_id)
            self._states[conversation_id] = (state, self._expires_at())
            return state

    def save(self, state: DialogueState) -> DialogueState:
        with self._lock:
            state.mentioned_entities = _dedupe_entities(
                state.mentioned_entities,
                max_entities=self.max_entities,
            )
            self._states[state.conversation_id] = (state, self._expires_at())
            return state

    def delete(self, conversation_id: str) -> None:
        with self._lock:
            self._states.pop(conversation_id, None)

    def _expires_at(self) -> float:
        return self._time_func() + self.ttl_seconds

    def _cleanup_expired_locked(self) -> None:
        now = self._time_func()
        expired_ids = [
            conversation_id
            for conversation_id, (_, expires_at) in self._states.items()
            if expires_at <= now
        ]
        for conversation_id in expired_ids:
            self._states.pop(conversation_id, None)

    def _drop_oldest_locked(self) -> None:
        if not self._states:
            return
        oldest_id = min(self._states.items(), key=lambda item: item[1][1])[0]
        self._states.pop(oldest_id, None)


class RedisDialogueStateStore:
    """Interface placeholder for production Redis state.

    Phase 1 keeps Redis optional. This class makes the adapter boundary explicit
    without adding a hard runtime dependency or changing local development.
    """

    def __init__(self, redis_url: str, ttl_seconds: int = 30 * 60) -> None:
        self.redis_url = redis_url
        self.ttl_seconds = ttl_seconds
        raise RuntimeError("RedisDialogueStateStore is planned for Phase 2; use local memory now.")


def update_state_after_turn(
    state: DialogueState,
    *,
    domain: DomainName,
    intent: IntentName,
    active_entity: ActiveEntity | None = None,
    unresolved_slots: dict[str, Any] | None = None,
) -> DialogueState:
    updated = replace(state)
    updated.active_domain = domain
    updated.last_intent = intent
    updated.unresolved_slots = unresolved_slots or {}
    if active_entity is not None:
        updated.active_entity = active_entity
        updated.mentioned_entities = [*updated.mentioned_entities, active_entity]
    return updated


def entity_from_pharmacity_response(response: dict[str, Any]) -> ActiveEntity | None:
    selected = response.get("selected_product")
    if not isinstance(selected, dict):
        return None
    name = _clean(selected.get("name"))
    sku = _clean(selected.get("sku"))
    if not name and not sku:
        return None
    return ActiveEntity(
        entity_type="product",
        entity_id=sku,
        name=name or sku or "Pharmacity product",
        source_url=_clean(selected.get("detail_url") or response.get("source_url")),
        metadata={"source": "pharmacity"},
    )


def entity_from_sources(sources: list[dict[str, Any]] | None) -> ActiveEntity | None:
    for source in sources or []:
        if not isinstance(source, dict):
            continue
        title = _clean(source.get("title"))
        if not title:
            continue
        return ActiveEntity(
            entity_type="rag_source",
            entity_id=_clean(source.get("id")),
            name=title,
            source_url=_clean(source.get("url") or source.get("source_url")),
            metadata={"source": "hanhphuc"},
        )
    return None


def _dedupe_entities(
    entities: list[ActiveEntity],
    *,
    max_entities: int,
) -> list[ActiveEntity]:
    seen: set[tuple[str, str | None, str]] = set()
    deduped: list[ActiveEntity] = []
    for entity in reversed(entities):
        key = (entity.entity_type, entity.entity_id, entity.name.lower())
        if key in seen:
            continue
        seen.add(key)
        deduped.append(entity)
        if len(deduped) >= max_entities:
            break
    return list(reversed(deduped))


def _clean(value: Any) -> str | None:
    if value is None:
        return None
    cleaned = str(value).strip()
    return cleaned or None
