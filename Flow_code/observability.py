from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any

from .service_contracts import IntentDecision


@dataclass(frozen=True)
class RequestTrace:
    started_at: float
    path: str


class ChatObserver:
    def __init__(self, logger: logging.Logger | None = None) -> None:
        self.logger = logger or logging.getLogger("uvicorn.error")

    def start(self, *, path: str = "/chat") -> RequestTrace:
        return RequestTrace(started_at=time.perf_counter(), path=path)

    def route_selected(
        self,
        *,
        trace: RequestTrace,
        conversation_id: str,
        decision: IntentDecision,
    ) -> None:
        self.logger.info(
            "chat_route path=%s conversation_id=%s route=%s intent=%s confidence=%.2f reason=%s",
            trace.path,
            conversation_id,
            decision.route,
            decision.intent,
            decision.confidence,
            decision.reason,
        )

    def finish(
        self,
        *,
        trace: RequestTrace,
        conversation_id: str,
        status: str,
        route: str,
        intent: str | None = None,
        retrieval_count: int | None = None,
    ) -> None:
        elapsed_ms = (time.perf_counter() - trace.started_at) * 1000
        self.logger.info(
            "chat_finish path=%s conversation_id=%s status=%s route=%s intent=%s retrieval_count=%s latency_ms=%.2f",
            trace.path,
            conversation_id,
            status,
            route,
            intent,
            retrieval_count,
            elapsed_ms,
        )

    def error(
        self,
        *,
        trace: RequestTrace,
        conversation_id: str | None,
        error: BaseException,
        extra: dict[str, Any] | None = None,
    ) -> None:
        elapsed_ms = (time.perf_counter() - trace.started_at) * 1000
        self.logger.exception(
            "chat_error path=%s conversation_id=%s error_type=%s latency_ms=%.2f extra=%s",
            trace.path,
            conversation_id,
            type(error).__name__,
            elapsed_ms,
            extra or {},
        )
