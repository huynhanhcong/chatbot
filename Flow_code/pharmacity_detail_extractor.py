from __future__ import annotations

import html
import re
import unicodedata
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class DrugInfoSection:
    code: str
    name: str
    text: str


def extract_drug_info_sections(raw: dict[str, Any] | None) -> list[DrugInfoSection]:
    item = raw_product_item(raw)
    if not item:
        return []

    sections: list[DrugInfoSection] = []
    for group in item.get("attributes") or []:
        if not isinstance(group, dict):
            continue
        for attr in group.get("items") or []:
            if not isinstance(attr, dict):
                continue
            code = _clean(attr.get("code"))
            name = _html_to_text(_clean(attr.get("name"))) or code
            value = attr.get("value") if isinstance(attr.get("value"), dict) else {}
            text = _html_to_text(_clean(value.get("value")))
            if not code or not name or not text:
                continue
            sections.append(DrugInfoSection(code=code, name=name, text=text))

    return _dedupe_sections(sections)


def raw_product_item(raw: dict[str, Any] | None) -> dict[str, Any]:
    if not isinstance(raw, dict):
        return {}
    item = raw.get("item")
    if isinstance(item, dict):
        return item
    response_item = (
        raw.get("api_response", {})
        .get("data", {})
        .get("item")
        if isinstance(raw.get("api_response"), dict)
        else None
    )
    if isinstance(response_item, dict):
        return response_item
    if raw.get("sku") or raw.get("attributes"):
        return raw
    return {}


def render_all_crawled_drug_info(raw: dict[str, Any] | None, *, max_chars: int = 12000) -> str:
    item = raw_product_item(raw)
    sections = extract_drug_info_sections(raw)
    parts: list[str] = []

    scalar_labels = [
        ("erp_name", "Ten ERP"),
        ("brand_name", "Thuong hieu"),
        ("product_type_name", "Loai san pham"),
        ("category_name", "Danh muc"),
        ("sale_unit_name", "Don vi ban"),
        ("sale_unit_price", "Gia don vi ban"),
        ("sale_unit_original_price", "Gia goc"),
        ("stock_status", "Trang thai ton kho"),
        ("registration_number", "So dang ky"),
        ("date_published", "Ngay dang"),
        ("updated_at", "Ngay cap nhat"),
    ]
    registration = item.get("registration") if isinstance(item.get("registration"), dict) else {}
    if registration.get("registration_number") and not item.get("registration_number"):
        item = {**item, "registration_number": registration.get("registration_number")}

    for key, label in scalar_labels:
        value = _html_to_text(_clean(item.get(key)))
        if value:
            parts.append(f"{label}: {value}")

    for section in sections:
        parts.append(f"{section.name}:\n{section.text}")

    text = "\n\n".join(parts).strip()
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 18].rstrip() + "\n...[da rut gon]"


def find_section_text(
    raw: dict[str, Any] | None,
    *,
    codes: list[str],
    names: list[str] | None = None,
    max_chars: int = 900,
) -> str | None:
    normalized_codes = {normalize_key(code) for code in codes}
    normalized_names = [normalize_key(name) for name in names or []]
    matches: list[str] = []
    for section in extract_drug_info_sections(raw):
        section_code = normalize_key(section.code)
        section_name = normalize_key(section.name)
        if section_code in normalized_codes or any(name in section_name for name in normalized_names):
            matches.append(section.text)
    if not matches:
        return None
    text = "\n\n".join(matches)
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 18].rstrip() + "\n...[da rut gon]"


def normalize_key(value: str) -> str:
    value = _html_to_text(value) or ""
    value = unicodedata.normalize("NFD", value)
    value = "".join(char for char in value if unicodedata.category(char) != "Mn")
    value = value.replace("đ", "d").replace("Đ", "D")
    value = value.lower()
    value = re.sub(r"[^\w\s-]", " ", value, flags=re.UNICODE)
    value = re.sub(r"\s+", " ", value).strip()
    return value


def _dedupe_sections(sections: list[DrugInfoSection]) -> list[DrugInfoSection]:
    seen: set[tuple[str, str]] = set()
    deduped: list[DrugInfoSection] = []
    for section in sections:
        key = (normalize_key(section.code), normalize_key(section.text))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(section)
    return deduped


def _html_to_text(value: str | None) -> str | None:
    if not value:
        return None
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        text = re.sub(r"(?i)<br\s*/?>", "\n", value)
        text = re.sub(r"</(?:p|div|li|h[1-6])>", "\n", text)
        text = re.sub(r"<[^>]+>", " ", text)
        text = html.unescape(text)
    else:
        soup = BeautifulSoup(value, "html.parser")
        for tag in soup(["script", "style"]):
            tag.decompose()
        text = soup.get_text("\n", strip=True)

    text = html.unescape(text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip() or None


def _clean(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None
