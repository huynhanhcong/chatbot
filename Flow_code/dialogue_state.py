from __future__ import annotations

import time
from dataclasses import replace
from threading import RLock
from typing import Any, Callable, Protocol

from .redis_utils import create_redis_client, redis_get_json, redis_set_json
from .service_contracts import ActiveEntity, DialogueState, DisplayedItem, DomainName, IntentName


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
    def __init__(
        self,
        redis_url: str,
        ttl_seconds: int = 30 * 60,
        max_entities: int = 10,
        client: Any | None = None,
        time_func: Callable[[], float] | None = None,
    ) -> None:
        self.ttl_seconds = ttl_seconds
        self.max_entities = max_entities
        self._time_func = time_func or time.monotonic
        self._client = client or create_redis_client(redis_url)
        self._prefix = "chatbot:state:"

    def get_or_create(self, conversation_id: str) -> DialogueState:
        payload = redis_get_json(self._client, self._key(conversation_id))
        if isinstance(payload, dict):
            return _dialogue_state_from_json(payload)

        state = DialogueState(conversation_id=conversation_id)
        self.save(state)
        return state

    def save(self, state: DialogueState) -> DialogueState:
        state.mentioned_entities = _dedupe_entities(
            state.mentioned_entities,
            max_entities=self.max_entities,
        )
        redis_set_json(self._client, self._key(state.conversation_id), state, self.ttl_seconds)
        return state

    def delete(self, conversation_id: str) -> None:
        self._client.delete(self._key(conversation_id))

    def _key(self, conversation_id: str) -> str:
        return self._prefix + conversation_id


def update_state_after_turn(
    state: DialogueState,
    *,
    domain: DomainName,
    intent: IntentName,
    active_entity: ActiveEntity | None = None,
    last_selected_item: DisplayedItem | None = None,
    last_shown_items: list[DisplayedItem] | None = None,
    last_compared_items: list[DisplayedItem] | None = None,
    pending_question: str | None = None,
    ambiguity_candidates: list[DisplayedItem] | None = None,
    unresolved_slots: dict[str, Any] | None = None,
) -> DialogueState:
    updated = replace(state)
    updated.active_domain = domain
    updated.last_intent = intent
    updated.unresolved_slots = unresolved_slots or {}
    if active_entity is not None:
        updated.active_entity = active_entity
        updated.mentioned_entities = [*updated.mentioned_entities, active_entity]
    if last_selected_item is not None:
        updated.last_selected_item = last_selected_item
    if last_shown_items is not None:
        updated.last_shown_items = list(last_shown_items)
    if last_compared_items is not None:
        updated.last_compared_items = list(last_compared_items)
    if pending_question is not None or not updated.unresolved_slots:
        updated.pending_question = pending_question
    if ambiguity_candidates is not None:
        updated.ambiguity_candidates = list(ambiguity_candidates)
    return updated


def displayed_item_from_active_entity(entity: ActiveEntity, *, index: int = 1) -> DisplayedItem:
    source = entity.metadata.get("source") if isinstance(entity.metadata, dict) else None
    return DisplayedItem(
        index=index,
        entity_id=entity.entity_id,
        entity_type=entity.entity_type,
        title=entity.name,
        source="pharmacity" if source == "pharmacity" else "hospital",
        source_url=entity.source_url,
        payload=dict(entity.metadata or {}),
    )


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


def displayed_items_from_pharmacity_options(options: list[dict[str, Any]] | None) -> list[DisplayedItem]:
    items: list[DisplayedItem] = []
    for option in options or []:
        if not isinstance(option, dict):
            continue
        title = _clean(option.get("name"))
        if not title:
            continue
        items.append(
            DisplayedItem(
                index=_safe_int(option.get("index"), len(items) + 1),
                entity_id=_clean(option.get("sku")),
                entity_type="product",
                title=title,
                source="pharmacity",
                source_url=_clean(option.get("detail_url")),
                payload={key: value for key, value in option.items() if key != "detail_url"},
            )
        )
    return items


def displayed_items_from_sources(sources: list[dict[str, Any]] | None) -> list[DisplayedItem]:
    items: list[DisplayedItem] = []
    for source in sources or []:
        if not isinstance(source, dict):
            continue
        title = _clean(source.get("title"))
        if not title:
            continue
        items.append(
            DisplayedItem(
                index=len(items) + 1,
                entity_id=_clean(source.get("id")),
                entity_type=_entity_type_from_source(source, title),
                title=title,
                source="hospital",
                source_url=_clean(source.get("url") or source.get("source_url")),
                price_vnd=_safe_optional_int(source.get("price_vnd")),
                payload={key: value for key, value in source.items() if key not in {"url", "source_url"}},
            )
        )
    return items


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


def _safe_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _safe_optional_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _entity_type_from_source(source: dict[str, Any], title: str) -> str:
    explicit = _clean(source.get("entity_type") or source.get("type"))
    if explicit:
        return explicit
    source_id = (_clean(source.get("id")) or "").lower()
    normalized = f"{source_id} {title}".lower()
    if "doctor" in normalized or "bac si" in normalized or "bác sĩ" in normalized:
        return "doctor"
    if "package" in normalized or "goi" in normalized or "gói" in normalized:
        return "package"
    return "service"


def _active_entity_from_json(payload: dict[str, Any]) -> ActiveEntity:
    return ActiveEntity(
        entity_type=str(payload.get("entity_type") or ""),
        entity_id=_clean(payload.get("entity_id")),
        name=str(payload.get("name") or ""),
        source_url=_clean(payload.get("source_url")),
        metadata=dict(payload.get("metadata") or {}),
    )


def _displayed_item_from_json(payload: dict[str, Any]) -> DisplayedItem:
    return DisplayedItem(
        index=int(payload.get("index") or 0),
        entity_id=_clean(payload.get("entity_id")),
        entity_type=str(payload.get("entity_type") or ""),
        title=str(payload.get("title") or ""),
        source=payload.get("source") or "unknown",
        source_url=_clean(payload.get("source_url")),
        price_vnd=_safe_optional_int(payload.get("price_vnd")),
        payload=dict(payload.get("payload") or {}),
    )


def _dialogue_state_from_json(payload: dict[str, Any]) -> DialogueState:
    active_entity = payload.get("active_entity")
    mentioned_entities = [
        _active_entity_from_json(item)
        for item in payload.get("mentioned_entities") or []
        if isinstance(item, dict)
    ]
    return DialogueState(
        conversation_id=str(payload.get("conversation_id") or ""),
        active_domain=payload.get("active_domain") or "unknown",
        active_entity=_active_entity_from_json(active_entity) if isinstance(active_entity, dict) else None,
        last_intent=payload.get("last_intent"),
        mentioned_entities=mentioned_entities,
        last_selected_item=_displayed_item_from_json(payload["last_selected_item"])
        if isinstance(payload.get("last_selected_item"), dict)
        else None,
        last_shown_items=[
            _displayed_item_from_json(item)
            for item in payload.get("last_shown_items") or []
            if isinstance(item, dict)
        ],
        last_compared_items=[
            _displayed_item_from_json(item)
            for item in payload.get("last_compared_items") or []
            if isinstance(item, dict)
        ],
        pending_question=_clean(payload.get("pending_question")),
        ambiguity_candidates=[
            _displayed_item_from_json(item)
            for item in payload.get("ambiguity_candidates") or []
            if isinstance(item, dict)
        ],
        user_preferences=dict(payload.get("user_preferences") or {}),
        unresolved_slots=dict(payload.get("unresolved_slots") or {}),
    )
