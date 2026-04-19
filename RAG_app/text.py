import re
import unicodedata


def clean_text(value: object) -> str:
    if value is None:
        return ""
    text = unicodedata.normalize("NFC", str(value))
    return re.sub(r"\s+", " ", text).strip()


def flatten_list(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        return [clean_text(item) for item in value if clean_text(item)]
    text = clean_text(value)
    return [text] if text else []


def tokenize_vi(text: str) -> list[str]:
    return re.findall(r"[A-Za-zÀ-ỹ0-9][A-Za-zÀ-ỹ0-9+_.-]*", clean_text(text).lower())


def format_price(value: object) -> str:
    if value in (None, ""):
        return ""
    try:
        return f"{int(float(value)):,}đ".replace(",", ".")
    except (TypeError, ValueError):
        return clean_text(value)
