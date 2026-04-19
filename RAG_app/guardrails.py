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
        return (
            "Tôi chưa tìm thấy thông tin phù hợp trong dữ liệu Bệnh viện Hạnh Phúc hiện tại. "
            "Vui lòng liên hệ trực tiếp bệnh viện để được xác nhận."
        )

    guarded = clean_text(answer)
    disclaimer = (
        "Thông tin này chỉ dùng để tham khảo và không thay thế tư vấn trực tiếp của bác sĩ."
    )
    if disclaimer.lower() not in guarded.lower():
        guarded = f"{guarded}\n\n{disclaimer}"

    if has_medical_risk(query):
        guarded = (
            f"{guarded}\n\nNếu có triệu chứng nặng, diễn tiến nhanh hoặc tình huống cấp cứu, "
            "hãy đến cơ sở y tế gần nhất hoặc liên hệ cấp cứu."
        )
    return guarded
