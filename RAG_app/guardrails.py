from __future__ import annotations

from .text import clean_text


RISK_TERMS = [
    "đau ngực",
    "khó thở",
    "ngất",
    "co giật",
    "chảy máu",
    "sốt cao",
    "cấp cứu",
    "tự tử",
    "mang thai",
]


def has_medical_risk(query: str) -> bool:
    text = clean_text(query).lower()
    return any(term in text for term in RISK_TERMS)


def apply_guardrails(answer: str, query: str, has_context: bool) -> str:
    if not has_context:
        base = (
            "Tôi chưa tìm thấy thông tin phù hợp trong dữ liệu Bệnh viện Hạnh Phúc hiện tại. "
            "Bạn vui lòng liên hệ trực tiếp bệnh viện để được xác nhận."
        )
        if has_medical_risk(query):
            return (
                f"{base}\n\nNếu có triệu chứng nặng, diễn tiến nhanh hoặc tình huống cấp cứu, "
                "hãy đến cơ sở y tế gần nhất hoặc liên hệ cấp cứu."
            )
        return base

    guarded = clean_text(answer)
    if not guarded:
        return (
            "Tôi chưa tổng hợp được câu trả lời phù hợp từ dữ liệu hiện tại. "
            "Bạn vui lòng thử hỏi rõ hơn hoặc liên hệ trực tiếp bệnh viện."
        )

    if has_medical_risk(query):
        guarded = (
            f"{guarded}\n\nNếu có triệu chứng nặng, diễn tiến nhanh hoặc tình huống cấp cứu, "
            "hãy đến cơ sở y tế gần nhất hoặc liên hệ cấp cứu."
        )
    return guarded
