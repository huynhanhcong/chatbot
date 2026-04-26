from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


IntentName = Literal[
    "drug_search",
    "drug_followup",
    "doctor_search",
    "package_search",
    "price_question",
    "compare_question",
    "medical_question",
    "context_followup",
    "clarification",
    "out_of_scope",
]

RouteName = Literal["hospital_rag", "pharmacity", "out_of_scope"]
DomainName = Literal["hospital", "pharmacity", "unknown"]


@dataclass(frozen=True)
class IntentDecision:
    intent: IntentName
    route: RouteName
    confidence: float
    reason: str


@dataclass(frozen=True)
class ActiveEntity:
    entity_type: str
    entity_id: str | None
    name: str
    source_url: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class DisplayedItem:
    index: int
    entity_id: str | None
    entity_type: str
    title: str
    source: DomainName
    source_url: str | None = None
    price_vnd: int | None = None
    payload: dict[str, Any] = field(default_factory=dict)


@dataclass
class DialogueState:
    conversation_id: str
    active_domain: DomainName = "unknown"
    active_entity: ActiveEntity | None = None
    last_intent: IntentName | None = None
    mentioned_entities: list[ActiveEntity] = field(default_factory=list)
    last_selected_item: DisplayedItem | None = None
    last_shown_items: list[DisplayedItem] = field(default_factory=list)
    last_compared_items: list[DisplayedItem] = field(default_factory=list)
    pending_question: str | None = None
    ambiguity_candidates: list[DisplayedItem] = field(default_factory=list)
    user_preferences: dict[str, Any] = field(default_factory=dict)
    unresolved_slots: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RetrievalResult:
    id: str
    text: str
    score: float
    parent_id: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class RerankResult:
    id: str
    score: float
    reason: str


@dataclass(frozen=True)
class GroundingResult:
    supported: bool
    confidence: float
    unsupported_claims: list[str] = field(default_factory=list)
    citations: list[dict[str, str | None]] = field(default_factory=list)


@dataclass(frozen=True)
class ChatEnvelope:
    status: str
    route: RouteName
    conversation_id: str
    message: str | None = None
    options: list[dict[str, Any]] = field(default_factory=list)
    answer: str | None = None
    sources: list[dict[str, Any]] = field(default_factory=list)
    confidence: str | None = None
    intent: str | None = None
    selected_product: dict[str, Any] | None = None
    source_url: str | None = None
    displayed_items: list[dict[str, Any]] = field(default_factory=list)

    def to_response(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "route": self.route,
            "conversation_id": self.conversation_id,
            "message": self.message,
            "options": self.options,
            "answer": self.answer,
            "sources": self.sources,
            "confidence": self.confidence,
            "intent": self.intent,
            "selected_product": self.selected_product,
            "source_url": self.source_url,
            "displayed_items": self.displayed_items,
        }
