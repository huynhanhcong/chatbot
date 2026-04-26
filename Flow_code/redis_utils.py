from __future__ import annotations

import json
from dataclasses import asdict, is_dataclass
from typing import Any


def create_redis_client(redis_url: str) -> Any:
    try:
        import redis
    except ImportError as exc:
        raise RuntimeError("Missing dependency: install redis from requirements-rag.txt.") from exc
    return redis.from_url(redis_url, decode_responses=True)


def redis_get_json(client: Any, key: str) -> Any | None:
    raw = client.get(key)
    if raw is None:
        return None
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    return json.loads(raw)


def redis_set_json(client: Any, key: str, value: Any, ttl_seconds: int) -> None:
    client.set(key, json.dumps(_to_jsonable(value), ensure_ascii=False), ex=max(1, int(ttl_seconds)))


def _to_jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return asdict(value)
    return value
