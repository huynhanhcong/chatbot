from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from typing import Any, Protocol

from .conversation_memory import ConversationSession
from .service_contracts import DialogueState, IntentDecision


class IntentFallbackClassifier(Protocol):
    def classify(self, message: str, state: DialogueState) -> IntentDecision | None:
        ...


@dataclass(frozen=True)
class RouteContext:
    conversation: ConversationSession
    state: DialogueState
    pharmacity_session: Any | None = None
    hospital_active: bool = False


class IntentRouter:
    def __init__(self, fallback_classifier: IntentFallbackClassifier | None = None) -> None:
        self.fallback_classifier = fallback_classifier

    def classify(
        self,
        *,
        message: str,
        context: RouteContext,
        selected_index: int | None = None,
        selected_sku: str | None = None,
    ) -> IntentDecision:
        if selected_index is not None or selected_sku:
            return IntentDecision(
                intent="drug_followup",
                route="pharmacity",
                confidence=1.0,
                reason="explicit_product_selection",
            )

        normalized = normalize_vi(message)
        if not normalized:
            return _out_of_scope("empty_message")

        drug_intent = looks_like_drug_question(normalized)
        hospital_intent = looks_like_hospital_question(normalized)
        price_intent = looks_like_price_question(normalized)
        compare_intent = looks_like_compare_question(normalized)
        doctor_intent = looks_like_doctor_question(normalized)
        package_intent = looks_like_package_question(normalized)
        ordinal_reference = looks_like_ordinal_reference(normalized)
        contextual = looks_like_contextual_follow_up(normalized)
        pharmacity_session = context.pharmacity_session

        if _pending_product_selection(context, pharmacity_session):
            if looks_like_product_selection(normalized):
                return IntentDecision("drug_followup", "pharmacity", 0.99, "pending_product_selection")
            if hospital_intent and not drug_intent:
                return _hospital_decision(normalized, price_intent, doctor_intent, package_intent, compare_intent)
            if drug_intent or _references_pending_product(normalized):
                return IntentDecision("drug_followup", "pharmacity", 0.88, "pending_product_context")

        if drug_intent and not hospital_intent:
            return IntentDecision("drug_search", "pharmacity", 0.95, "drug_lookup_signal")

        if hospital_intent and not drug_intent:
            return _hospital_decision(normalized, price_intent, doctor_intent, package_intent, compare_intent)

        if drug_intent and hospital_intent:
            if has_strong_drug_lookup_signal(normalized):
                return IntentDecision("drug_search", "pharmacity", 0.78, "mixed_intent_strong_drug")
            return _hospital_decision(normalized, price_intent, doctor_intent, package_intent, compare_intent)

        if _has_active_pharmacity_context(context, pharmacity_session) and _references_active_product(normalized):
            return IntentDecision("drug_followup", "pharmacity", 0.82, "active_product_context")

        if _has_active_hospital_context(context) and (contextual or ordinal_reference or compare_intent):
            if compare_intent:
                return IntentDecision("compare_question", "hospital_rag", 0.87, "hospital_context_compare")
            if price_intent:
                return IntentDecision("price_question", "hospital_rag", 0.84, "hospital_context_price")
            return IntentDecision("context_followup", "hospital_rag", 0.82, "active_hospital_context")

        if self.fallback_classifier is not None:
            fallback = self.fallback_classifier.classify(message, context.state)
            if fallback is not None:
                return fallback

        if looks_like_general_medical_question(normalized):
            return IntentDecision("medical_question", "hospital_rag", 0.62, "general_medical_signal")

        return _out_of_scope("no_supported_intent_signal")


def _hospital_decision(
    normalized: str,
    price_intent: bool,
    doctor_intent: bool,
    package_intent: bool,
    compare_intent: bool = False,
) -> IntentDecision:
    if compare_intent:
        return IntentDecision("compare_question", "hospital_rag", 0.91, "hospital_compare_signal")
    if price_intent:
        return IntentDecision("price_question", "hospital_rag", 0.91, "hospital_price_signal")
    if doctor_intent:
        return IntentDecision("doctor_search", "hospital_rag", 0.9, "doctor_signal")
    if package_intent:
        return IntentDecision("package_search", "hospital_rag", 0.9, "package_signal")
    if looks_like_general_medical_question(normalized):
        return IntentDecision("medical_question", "hospital_rag", 0.74, "hospital_medical_signal")
    return IntentDecision("package_search", "hospital_rag", 0.74, "hospital_signal")


def _out_of_scope(reason: str) -> IntentDecision:
    return IntentDecision("out_of_scope", "out_of_scope", 0.35, reason)


def _pending_product_selection(context: RouteContext, session: Any) -> bool:
    return (
        session is not None
        and _session_needs_product_selection(session)
        and context.state.unresolved_slots.get("product_selection") == "required"
    )


def _has_active_pharmacity_context(context: RouteContext, session: Any) -> bool:
    return (
        session is not None
        and (
            context.conversation.active_route == "pharmacity"
            or context.state.active_domain == "pharmacity"
        )
    )


def _has_active_hospital_context(context: RouteContext) -> bool:
    return (
        context.conversation.active_route == "hospital_rag"
        or context.hospital_active
        or context.state.active_domain == "hospital"
    )


def _session_needs_product_selection(session: Any) -> bool:
    return getattr(session, "selected_detail", None) is None


def looks_like_drug_question(normalized: str) -> bool:
    if has_strong_drug_lookup_signal(normalized):
        return True
    if re.search(r"\bthuoc\s+(?!gi\b|nao\b|nay\b|do\b|tren\b)([a-z0-9][\w.+-]{1,})", normalized):
        return True
    if re.search(r"\b(?:tim|tra cuu|thong tin|cho toi biet|xin|cho minh xin).{0,40}\bthuoc\b", normalized):
        return not re.search(r"\b(?:thuoc gi|thuoc nao|uong thuoc gi)\b", normalized)
    return False


def has_strong_drug_lookup_signal(normalized: str) -> bool:
    drug_keywords = [
        "duoc pham",
        "oresol",
        "paracetamol",
        "panadol",
        "berberin",
        "vitamin",
        "vien uong",
        "siro",
        "thuoc",
    ]
    if any(keyword in normalized for keyword in drug_keywords):
        return True
    return bool(re.search(r"\b(p\d{4,}|sku)\b", normalized))


def looks_like_hospital_question(normalized: str) -> bool:
    hospital_keywords = [
        "benh vien",
        "hanh phuc",
        "goi kham",
        "goi ivf",
        "ivf",
        "bac si",
        "bs ",
        "chuyen khoa",
        "khoa",
        "kham thai",
        "tam soat",
        "ung thu",
        "sinh san",
        "hiem muon",
        "phu khoa",
        "nhi khoa",
        "goi sinh",
        "sinh mo",
        "sinh thuong",
        "chuan bi sinh",
        "sap sinh",
        "dich vu trong goi",
        "mang thai",
    ]
    if any(keyword in normalized for keyword in hospital_keywords):
        return True
    return bool(re.search(r"\bgoi\s+(?:kham|sinh|ivf|tam soat|ho tro)\b", normalized))


def looks_like_doctor_question(normalized: str) -> bool:
    return any(keyword in normalized for keyword in ["bac si", "bs ", "chuyen gia", "chuyen khoa"])


def looks_like_package_question(normalized: str) -> bool:
    return any(
        keyword in normalized
        for keyword in ["goi", "ivf", "tam soat", "kham thai", "goi sinh", "dich vu"]
    )


def looks_like_price_question(normalized: str) -> bool:
    return any(
        keyword in normalized
        for keyword in ["gia", "chi phi", "bao nhieu tien", "bao nhieu", "phi"]
    )


def looks_like_compare_question(normalized: str) -> bool:
    return any(keyword in normalized for keyword in ["khac gi", "so sanh", "khac nhau", "hon nhau"])


def looks_like_ordinal_reference(normalized: str) -> bool:
    if re.search(
        r"\b(?:goi|loai|nguoi|bac si|dich vu|san pham|muc|cai)\s+"
        r"(?:dau tien|thu nhat|thu hai|thu ba|thu tu|thu nam)\b",
        normalized,
    ):
        return True
    if normalized.strip() in {"dau tien", "thu nhat", "thu hai", "thu ba", "thu tu", "thu nam"}:
        return True
    return bool(re.search(r"\b(?:goi|loai|nguoi|bac si|dich vu|san pham)\s+(?:thu\s+|so\s+)?[1-9]\b", normalized))


def looks_like_general_medical_question(normalized: str) -> bool:
    return any(
        keyword in normalized
        for keyword in ["trieu chung", "benh", "dau", "sot", "kho tho", "kham", "uong thuoc gi"]
    )


def looks_like_product_selection(normalized: str) -> bool:
    normalized = normalized.strip()
    if re.fullmatch(r"[1-9][0-9]?", normalized):
        return True
    return bool(
        re.fullmatch(
            r"(?:chon|toi chon|lay|xem)?\s*(?:thuoc|lua chon|option|so)?\s*[1-9][0-9]?",
            normalized,
        )
    )


def looks_like_contextual_follow_up(normalized: str) -> bool:
    contextual_keywords = [
        "tren",
        "vua roi",
        "goi nay",
        "goi kham nay",
        "goi do",
        "goi dau",
        "goi dau tien",
        "goi thu 2",
        "goi thu hai",
        "dich vu nay",
        "dich vu do",
        "bac si nay",
        "bac si kia",
        "nguoi dau tien",
        "nguoi thu 2",
        "nguoi thu hai",
        "loai dau tien",
        "loai thu 2",
        "loai thu hai",
        "khac gi",
        "so sanh",
        "san pham nay",
        "thuoc nay",
        "chi phi",
        "gia bao nhieu",
        "bao nhieu tien",
        "gom gi",
        "co gi",
        "cong dung",
        "thanh phan",
        "cach dung",
        "ke don",
        "xem them",
    ]
    if any(keyword in normalized for keyword in contextual_keywords):
        return True
    if re.search(r"\b(?:nay|do|kia)\b", normalized):
        return True
    return normalized in {
        "gia",
        "chi phi",
        "bao nhieu",
        "bao nhieu tien",
        "gom gi",
        "co gi",
        "cong dung gi",
        "thanh phan gi",
        "cach dung",
        "ke don khong",
    }


def _references_pending_product(normalized: str) -> bool:
    return any(
        keyword in normalized
        for keyword in [
            "san pham",
            "thuoc nao",
            "loai nao",
            "chon",
            "gia",
            "bao nhieu",
            "thanh phan",
            "cong dung",
            "cach dung",
            "ke don",
            "xem them",
        ]
    )


def _references_active_product(normalized: str) -> bool:
    return looks_like_contextual_follow_up(normalized) or any(
        keyword in normalized
        for keyword in [
            "thuoc",
            "san pham",
            "gia",
            "thanh phan",
            "cong dung",
            "cach dung",
            "ke don",
            "xem them",
        ]
    )


def normalize_vi(value: str) -> str:
    normalized = unicodedata.normalize("NFD", value or "")
    normalized = "".join(char for char in normalized if unicodedata.category(char) != "Mn")
    normalized = normalized.replace("đ", "d").replace("Đ", "D")
    return normalized.lower()
