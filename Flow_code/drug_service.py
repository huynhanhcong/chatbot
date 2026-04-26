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

    def handle_public(
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
        return public_pharmacity_response(response)

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


def public_pharmacity_response(response: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": response.get("status"),
        "conversation_id": response.get("conversation_id"),
        "message": response.get("message"),
        "options": _public_options(response.get("options")),
        "answer": response.get("answer"),
        "selected_product": _public_selected_product(response.get("selected_product")),
    }


def pharmacity_envelope(response: dict[str, Any]) -> dict[str, Any]:
    return {
        "status": response.get("status"),
        "route": "pharmacity",
        "conversation_id": response.get("conversation_id"),
        "message": response.get("message"),
        "options": _public_options(response.get("options")),
        "answer": response.get("answer"),
        "sources": [],
        "selected_product": _public_selected_product(response.get("selected_product")),
        "source_url": None,
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


def _public_options(options: Any) -> list[dict[str, Any]]:
    public_options: list[dict[str, Any]] = []
    for option in options or []:
        if not isinstance(option, dict):
            continue
        public_options.append(
            {
                "index": option.get("index"),
                "sku": option.get("sku"),
                "name": option.get("name"),
                "brand": option.get("brand"),
                "price": option.get("price"),
                "image_url": option.get("image_url"),
            }
        )
    return public_options


def _public_selected_product(selected_product: Any) -> dict[str, Any] | None:
    if not isinstance(selected_product, dict):
        return None
    return {
        "sku": selected_product.get("sku"),
        "name": selected_product.get("name"),
    }
