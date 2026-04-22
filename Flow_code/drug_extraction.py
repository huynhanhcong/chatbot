from __future__ import annotations

import re
import unicodedata

try:
    from rapidfuzz import fuzz, process
except ImportError:  # pragma: no cover
    fuzz = None
    process = None


KNOWN_DRUG_TERMS = [
    "oresol",
    "panadol",
    "paracetamol",
    "berberin",
    "vitamin c",
    "vitamin d",
    "amoxicillin",
    "augmentin",
    "smecta",
    "efferalgan",
    "hapacol",
    "ibuprofen",
    "loratadin",
    "cetirizin",
    "nacl",
]
CANONICAL_DRUG_TERMS = {
    "oresol": "Oresol",
    "panadol": "Panadol",
    "paracetamol": "Paracetamol",
    "berberin": "Berberin",
    "vitamin c": "Vitamin C",
    "vitamin d": "Vitamin D",
}

QUERY_PREFIX_RE = re.compile(
    r"(?:thuoc|tim|tra cuu|thong tin|cho toi biet|toi muon hoi|san pham)\s+(.+)",
    flags=re.IGNORECASE,
)


def extract_drug_name_local(message: str) -> str | None:
    text = (message or "").strip()
    if not text:
        return None

    sku = re.search(r"\bP\d{4,}\b", text, flags=re.IGNORECASE)
    if sku:
        return sku.group(0).upper()
    sku_digits = re.search(r"\bSKU\s*[:#]?\s*(P?\d{4,})\b", text, flags=re.IGNORECASE)
    if sku_digits:
        value = sku_digits.group(1).upper()
        return value if value.startswith("P") else f"P{value}"

    normalized = normalize_search_text(text)
    for term in sorted(KNOWN_DRUG_TERMS, key=len, reverse=True):
        if re.search(rf"\b{re.escape(term)}\b", normalized):
            return _canonical(term)

    match = QUERY_PREFIX_RE.search(normalized)
    if match:
        candidate = _strip_query_filler(match.group(1))
        if candidate:
            return candidate

    if 1 <= len(normalized.split()) <= 5:
        candidate = _strip_query_filler(normalized)
        if candidate and not _looks_like_generic_question(candidate):
            return candidate

    fuzzy = _fuzzy_known_term(normalized)
    return fuzzy


def normalize_search_text(value: str) -> str:
    text = _repair_mojibake(value.lower()) or value.lower()
    text = unicodedata.normalize("NFD", text)
    text = "".join(char for char in text if unicodedata.category(char) != "Mn")
    text = text.replace("đ", "d").replace("Đ", "d")
    text = re.sub(r"[^a-z0-9.+\-\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _strip_query_filler(value: str) -> str | None:
    value = re.split(r"[?.!,;:\n]", value, maxsplit=1)[0].strip()
    value = re.sub(r"^(?:ve|về)?\s*(?:thuoc|san pham)\s+", "", value, flags=re.IGNORECASE)
    value = re.sub(r"^thong tin\s+(?:ve\s+)?(?:thuoc|san pham)\s+", "", value, flags=re.IGNORECASE)
    value = re.sub(
        r"\b(la gi|co tac dung gi|cong dung|gia bao nhieu|bao nhieu tien|can tim|o pharmacity)\b",
        " ",
        value,
        flags=re.IGNORECASE,
    )
    value = re.sub(r"\s+", " ", value).strip()
    return value or None


def _looks_like_generic_question(value: str) -> bool:
    generic = {
        "xin chao",
        "cam on",
        "tu van",
        "thong tin",
        "thuoc gi",
        "thuoc nao",
        "uong thuoc gi",
    }
    return value in generic


def _fuzzy_known_term(normalized: str) -> str | None:
    if len(normalized) < 4:
        return None
    if process is not None and fuzz is not None:
        match = process.extractOne(normalized, KNOWN_DRUG_TERMS, scorer=fuzz.WRatio)
        if match and float(match[1]) >= 84:
            return _canonical(str(match[0]))
    return None


def _canonical(value: str) -> str:
    return CANONICAL_DRUG_TERMS.get(value, value)


def _repair_mojibake(value: str) -> str:
    try:
        return value.encode("cp1252").decode("utf-8")
    except UnicodeError:
        return ""
