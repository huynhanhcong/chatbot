from __future__ import annotations

import hashlib
import json
import time
from threading import RLock
from typing import Any

from .text import clean_text


class JsonCache:
    def get_json(self, key: str) -> Any | None:
        raise NotImplementedError

    def set_json(self, key: str, value: Any, ttl_seconds: int) -> None:
        raise NotImplementedError


class InMemoryJsonCache(JsonCache):
    def __init__(self, time_func: Any | None = None) -> None:
        self._time_func = time_func or time.monotonic
        self._items: dict[str, tuple[float, Any]] = {}
        self._lock = RLock()

    def get_json(self, key: str) -> Any | None:
        with self._lock:
            item = self._items.get(key)
            if item is None:
                return None
            expires_at, value = item
            if expires_at <= self._time_func():
                self._items.pop(key, None)
                return None
            return value

    def set_json(self, key: str, value: Any, ttl_seconds: int) -> None:
        if ttl_seconds <= 0:
            return
        with self._lock:
            self._items[key] = (self._time_func() + ttl_seconds, value)


class RedisJsonCache(JsonCache):
    def __init__(self, redis_url: str) -> None:
        import redis

        self._client = redis.Redis.from_url(redis_url, decode_responses=True)
        self._client.ping()

    def get_json(self, key: str) -> Any | None:
        raw = self._client.get(key)
        if raw is None:
            return None
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            return None

    def set_json(self, key: str, value: Any, ttl_seconds: int) -> None:
        if ttl_seconds <= 0:
            return
        self._client.setex(key, ttl_seconds, json.dumps(value, ensure_ascii=False))


def build_cache(settings: Any, namespace: str) -> JsonCache:
    if getattr(settings, "use_redis", False) and getattr(settings, "redis_url", None):
        try:
            return NamespacedJsonCache(RedisJsonCache(settings.redis_url), namespace)
        except Exception:
            pass
    return NamespacedJsonCache(InMemoryJsonCache(), namespace)


class NamespacedJsonCache(JsonCache):
    def __init__(self, inner: JsonCache, namespace: str) -> None:
        self.inner = inner
        self.namespace = namespace.strip(":")

    def get_json(self, key: str) -> Any | None:
        return self.inner.get_json(self._key(key))

    def set_json(self, key: str, value: Any, ttl_seconds: int) -> None:
        self.inner.set_json(self._key(key), value, ttl_seconds)

    def _key(self, key: str) -> str:
        return f"{self.namespace}:{key}"


def stable_cache_key(*parts: object) -> str:
    normalized = "\n".join(clean_text(part).lower() for part in parts)
    digest = hashlib.sha256(normalized.encode("utf-8")).hexdigest()
    return digest[:48]
