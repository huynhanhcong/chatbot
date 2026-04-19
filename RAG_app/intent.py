from .text import clean_text


def detect_intent(query: str) -> str:
    text = clean_text(query).lower()
    if any(word in text for word in ["giá", "bao nhiêu tiền", "chi phí", "phí"]):
        return "price"
    if any(word in text for word in ["bao gồm", "gồm gì", "dịch vụ", "hạng mục"]):
        return "services"
    if any(word in text for word in ["chuẩn bị", "nhịn ăn", "trước khi"]):
        return "preparation"
    if any(word in text for word in ["điều khoản", "điều kiện", "chuyển nhượng", "phát sinh"]):
        return "terms"
    if any(word in text for word in ["bác sĩ", "bs.", "bs ", "tiến sĩ", "thạc sĩ"]):
        return "doctor_search"
    if any(word in text for word in ["gói", "ivf", "tầm soát", "khám", "sinh"]):
        return "package_search"
    if any(word in text for word in ["triệu chứng", "bệnh", "uống thuốc", "đau", "sốt"]):
        return "general_medical"
    return "unknown"
