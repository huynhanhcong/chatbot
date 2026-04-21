from __future__ import annotations

from typing import Any, Callable


class DrugInfoService:
    def __init__(
        self,
        *,
        flow_provider: Callable[[], Any],
        existing_flow_provider: Callable[[], Any | None] | None = None,
    ) -> None:
        self._flow_provider = flow_provider
        self._existing_flow_provider = existing_flow_provider

    def handle_raw(
        self,
        *,
        message: str,
        conversation_id: str | None = None,
        selected_index: int | None = None,
        selected_sku: str | None = None,
    ) -> dict[str, Any]:
        return self._flow_provider().handle_message(
            message=message,
            conversation_id=conversation_id,
            selected_index=selected_index,
            selected_sku=selected_sku,
        )

    def handle_envelope(
        self,
        *,
        message: str,
        conversation_id: str | None = None,
        selected_index: int | None = None,
        selected_sku: str | None = None,
    ) -> dict[str, Any]:
        response = self.handle_raw(
            message=message,
            conversation_id=conversation_id,
            selected_index=selected_index,
            selected_sku=selected_sku,
        )
        return pharmacity_envelope(response)

    def get_session(self, conversation_id: str | None) -> Any | None:
        if not conversation_id:
            return None
        flow = self._existing_flow_provider() if self._existing_flow_provider else None
        if flow is None:
            return None
        get_session = getattr(flow, "get_session", None)
        if callable(get_session):
            return get_session(conversation_id)
        has_active_session = getattr(flow, "has_active_session", None)
        if callable(has_active_session) and has_active_session(conversation_id):
            return object()
        return None


def pharmacity_envelope(response: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": response.get("status"),
        "route": "pharmacity",
        "conversation_id": response.get("conversation_id"),
        "message": response.get("message"),
        "options": response.get("options", []),
        "answer": response.get("answer"),
        "sources": pharmacity_sources(response),
        "selected_product": response.get("selected_product"),
        "source_url": response.get("source_url"),
    }


def pharmacity_sources(response: dict[str, Any]) -> list[dict[str, str | None]]:
    source_url = response.get("source_url")
    selected_product = response.get("selected_product") or {}
    if not source_url:
        return []
    return [
        {
            "title": selected_product.get("name") or "Pharmacity",
            "url": source_url,
        }
    ]
