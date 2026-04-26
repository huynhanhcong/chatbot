from __future__ import annotations

import re
from dataclasses import dataclass, field

from .router_service import normalize_vi
from .service_contracts import DialogueState, DisplayedItem


@dataclass(frozen=True)
class ResolvedReference:
    intent: str | None = None
    resolved_items: list[DisplayedItem] = field(default_factory=list)
    needs_clarification: bool = False
    reason: str = ""

    @property
    def primary_item(self) -> DisplayedItem | None:
        return self.resolved_items[0] if self.resolved_items else None


class MentionResolver:
    def resolve(self, message: str, state: DialogueState) -> ResolvedReference:
        normalized = normalize_vi(message)
        intent = _detect_followup_intent(normalized)

        compared = _resolve_compare(normalized, state)
        if compared:
            if len(compared) >= 2:
                return ResolvedReference(
                    intent="compare_question",
                    resolved_items=compared[:2],
                    reason="explicit_compare_ordinals",
                )
            return ResolvedReference(
                intent="compare_question",
                needs_clarification=True,
                reason="compare_missing_second_item",
            )

        ordinal = _extract_ordinal(normalized)
        if ordinal is not None:
            item = _item_by_ordinal(state.last_shown_items, ordinal, _wanted_type(normalized))
            if item is None:
                return ResolvedReference(
                    intent=intent or "context_followup",
                    needs_clarification=True,
                    reason="ordinal_not_available",
                )
            return ResolvedReference(
                intent=intent or "context_followup",
                resolved_items=[item],
                reason="ordinal_reference",
            )

        wanted_type = _wanted_type(normalized)
        if _has_pronoun_reference(normalized):
            item = _active_or_recent_item(state, wanted_type)
            if item is not None:
                return ResolvedReference(
                    intent=intent or "context_followup",
                    resolved_items=[item],
                    reason="active_reference",
                )
            candidates = _filter_items(state.last_shown_items, wanted_type)
            if len(candidates) == 1:
                return ResolvedReference(
                    intent=intent or "context_followup",
                    resolved_items=candidates,
                    reason="single_candidate_reference",
                )
            if len(candidates) > 1:
                return ResolvedReference(
                    intent=intent or "context_followup",
                    resolved_items=candidates[:4],
                    needs_clarification=True,
                    reason="ambiguous_reference",
                )

        if intent and _has_active_context(state):
            item = _active_or_recent_item(state, wanted_type)
            return ResolvedReference(
                intent=intent,
                resolved_items=[item] if item is not None else [],
                reason="followup_intent_active_context",
            )

        return ResolvedReference(intent=intent, reason="no_reference")


def render_memory_context(state: DialogueState, resolved: ResolvedReference | None = None) -> str:
    lines: list[str] = []
    if state.active_domain != "unknown":
        lines.append(f"active_domain: {state.active_domain}")
    if state.active_entity is not None:
        lines.append(
            "active_entity: "
            + _compact_item(
                DisplayedItem(
                    index=1,
                    entity_id=state.active_entity.entity_id,
                    entity_type=state.active_entity.entity_type,
                    title=state.active_entity.name,
                    source=state.active_domain,
                    source_url=state.active_entity.source_url,
                    payload=dict(state.active_entity.metadata or {}),
                )
            )
        )
    if state.last_selected_item is not None:
        lines.append(f"selected_item: {_compact_item(state.last_selected_item)}")
    if state.last_shown_items:
        shown = "; ".join(_compact_item(item) for item in state.last_shown_items[:6])
        lines.append(f"last_shown_items: {shown}")
    if resolved and resolved.resolved_items:
        resolved_text = "; ".join(_compact_item(item) for item in resolved.resolved_items[:3])
        lines.append(f"resolved_reference: {resolved_text}")
    if resolved and resolved.intent:
        lines.append(f"resolved_intent: {resolved.intent}")
    return "\n".join(lines)


def _compact_item(item: DisplayedItem) -> str:
    parts = [f"#{item.index}", item.entity_type, item.title]
    if item.entity_id:
        parts.append(f"id={item.entity_id}")
    return " | ".join(parts)


def _detect_followup_intent(normalized: str) -> str | None:
    if _looks_like_compare(normalized):
        return "compare_question"
    if any(keyword in normalized for keyword in ["gia", "chi phi", "bao nhieu tien", "bao nhieu", "phi"]):
        return "price_question"
    if any(keyword in normalized for keyword in ["kinh nghiem", "chuyen mon", "chuyen khoa", "bac si"]):
        return "doctor_search"
    if any(keyword in normalized for keyword in ["gom gi", "co gi", "chi tiet", "khac gi", "phu hop"]):
        return "context_followup"
    return None


def _resolve_compare(normalized: str, state: DialogueState) -> list[DisplayedItem]:
    if not _looks_like_compare(normalized):
        return []
    ordinals = _extract_all_ordinals(normalized)
    if len(ordinals) >= 2:
        items = []
        wanted_type = _wanted_type(normalized)
        for ordinal in ordinals[:2]:
            item = _item_by_ordinal(state.last_shown_items, ordinal, wanted_type)
            if item is not None:
                items.append(item)
        return items
    if len(state.last_compared_items) >= 2:
        return state.last_compared_items[:2]
    if len(state.last_shown_items) >= 2:
        return state.last_shown_items[:2]
    return []


def _looks_like_compare(normalized: str) -> bool:
    return any(keyword in normalized for keyword in ["khac gi", "so sanh", "khac nhau", "hon nhau"])


def _extract_all_ordinals(normalized: str) -> list[int]:
    values: list[int] = []
    for pattern, value in _ORDINAL_PATTERNS:
        for match in re.finditer(pattern, normalized):
            if _is_real_ordinal_reference(normalized, match.start(), match.group(0)):
                values.append(value)
    for match in re.finditer(
        r"\b(?:goi|loai|nguoi|bac si|dich vu|san pham|muc|so)\s+(?:thu\s+|so\s+)?([1-9][0-9]?)\b",
        normalized,
    ):
        values.append(int(match.group(1)))
    deduped: list[int] = []
    for value in values:
        if value not in deduped:
            deduped.append(value)
    return deduped


def _extract_ordinal(normalized: str) -> int | None:
    values = _extract_all_ordinals(normalized)
    return values[0] if values else None


_ORDINAL_PATTERNS = [
    (r"\b(?:dau tien|thu nhat|so mot|mot)\b", 1),
    (r"\b(?:thu hai|so hai|hai)\b", 2),
    (r"\b(?:thu ba|so ba|ba)\b", 3),
    (r"\b(?:thu tu|so bon|bon)\b", 4),
    (r"\b(?:thu nam|so nam|nam)\b", 5),
]


def _is_real_ordinal_reference(normalized: str, start: int, matched_text: str) -> bool:
    if re.search(r"\b(?:lan|lần)\s+" + re.escape(matched_text) + r"\b", normalized):
        return False
    prefix = normalized[max(0, start - 18) : start].strip()
    if re.search(r"\b(?:goi|loai|nguoi|bac si|dich vu|san pham|muc|cai)\s*$", prefix):
        return True
    short_reference = normalized.strip() in {
        "dau tien",
        "thu nhat",
        "so mot",
        "thu hai",
        "so hai",
        "thu ba",
        "so ba",
    }
    return short_reference


def _wanted_type(normalized: str) -> str | None:
    if any(keyword in normalized for keyword in ["bac si", "bs ", "nguoi"]):
        return "doctor"
    if any(keyword in normalized for keyword in ["goi", "dich vu"]):
        return "package"
    if any(keyword in normalized for keyword in ["thuoc", "san pham", "loai"]):
        return "product"
    return None


def _item_by_ordinal(
    items: list[DisplayedItem],
    ordinal: int,
    wanted_type: str | None,
) -> DisplayedItem | None:
    candidates = _filter_items(items, wanted_type)
    if ordinal < 1 or ordinal > len(candidates):
        return None
    return candidates[ordinal - 1]


def _filter_items(items: list[DisplayedItem], wanted_type: str | None) -> list[DisplayedItem]:
    if wanted_type is None:
        return list(items)
    if wanted_type == "package":
        return [item for item in items if item.entity_type in {"package", "service", "rag_source"}]
    return [item for item in items if item.entity_type == wanted_type]


def _has_pronoun_reference(normalized: str) -> bool:
    phrase_match = any(
        keyword in normalized
        for keyword in [
            "cai do",
            "cai nay",
            "luc nay",
            "vua roi",
            "o tren",
            "goi do",
            "goi nay",
            "dich vu do",
            "bac si do",
            "nguoi do",
            "loai do",
        ]
    )
    return phrase_match or bool(re.search(r"\b(?:do|nay|kia)\b", normalized))


def _active_or_recent_item(state: DialogueState, wanted_type: str | None) -> DisplayedItem | None:
    if state.last_selected_item and _type_matches(state.last_selected_item, wanted_type):
        return state.last_selected_item
    if state.last_shown_items:
        candidates = _filter_items(state.last_shown_items, wanted_type)
        if len(candidates) == 1:
            return candidates[0]
    if state.active_entity is not None:
        item = DisplayedItem(
            index=1,
            entity_id=state.active_entity.entity_id,
            entity_type=state.active_entity.entity_type,
            title=state.active_entity.name,
            source=state.active_domain,
            source_url=state.active_entity.source_url,
            payload=dict(state.active_entity.metadata or {}),
        )
        if _type_matches(item, wanted_type):
            return item
    return None


def _type_matches(item: DisplayedItem, wanted_type: str | None) -> bool:
    if wanted_type is None:
        return True
    if wanted_type == "package":
        return item.entity_type in {"package", "service", "rag_source"}
    return item.entity_type == wanted_type


def _has_active_context(state: DialogueState) -> bool:
    return (
        state.active_domain != "unknown"
        or state.active_entity is not None
        or bool(state.last_shown_items)
        or state.last_selected_item is not None
    )
