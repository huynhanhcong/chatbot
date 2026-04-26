from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

from .models import ProductDetail, format_variants_price
from .pharmacity_detail_extractor import (
    extract_drug_info_sections,
    raw_product_item,
    render_all_crawled_drug_info,
)


DEFAULT_PHARMACITY_EXPORT_PATH = Path("pharmacity.txt")
DEFAULT_PHARMACITY_EXTRACTED_PATH = Path("pharmacity_1.txt")


def clear_pharmacity_export_files(
    paths: list[str | Path] | tuple[str | Path, ...] = (
        DEFAULT_PHARMACITY_EXPORT_PATH,
        DEFAULT_PHARMACITY_EXTRACTED_PATH,
    ),
) -> None:
    for path in paths:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text("", encoding="utf-8")


def export_product_detail(
    detail: ProductDetail,
    *,
    path: str | Path = DEFAULT_PHARMACITY_EXPORT_PATH,
) -> None:
    """Write every Pharmacity detail field currently available for inspection."""
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    section = render_product_detail_for_export(detail).strip()
    if not section:
        return

    existing = destination.read_text(encoding="utf-8") if destination.exists() else ""
    marker = _product_marker(detail)
    if marker and marker in existing:
        destination.write_text(_replace_existing_section(existing, marker, section), encoding="utf-8")
        return

    text = (existing.rstrip() + "\n\n" + section + "\n").lstrip()
    destination.write_text(text, encoding="utf-8")


def export_extracted_product_info(
    detail: ProductDetail,
    *,
    path: str | Path = DEFAULT_PHARMACITY_EXTRACTED_PATH,
) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    section = render_extracted_product_info(detail).strip()
    if not section:
        return

    existing = destination.read_text(encoding="utf-8") if destination.exists() else ""
    marker = _product_marker(detail)
    if marker and marker in existing:
        destination.write_text(_replace_existing_section(existing, marker, section), encoding="utf-8")
        return

    text = (existing.rstrip() + "\n\n" + section + "\n").lstrip()
    destination.write_text(text, encoding="utf-8")


def rebuild_extracted_file_from_raw_export(
    *,
    source_path: str | Path = DEFAULT_PHARMACITY_EXPORT_PATH,
    destination_path: str | Path = DEFAULT_PHARMACITY_EXTRACTED_PATH,
) -> int:
    source = Path(source_path)
    if not source.exists():
        Path(destination_path).write_text("", encoding="utf-8")
        return 0

    details = _details_from_export_text(source.read_text(encoding="utf-8"))
    rendered = [render_extracted_product_info(detail).strip() for detail in details]
    text = "\n\n".join(section for section in rendered if section).strip()
    Path(destination_path).write_text(text + ("\n" if text else ""), encoding="utf-8")
    return len(rendered)


def render_product_detail_for_export(detail: ProductDetail) -> str:
    marker = _product_marker(detail)
    parsed_json = json.dumps(asdict(detail), ensure_ascii=False, indent=2, default=str)
    raw_json = json.dumps(detail.raw or {}, ensure_ascii=False, indent=2, default=str)
    lines = [
        "============================================================",
        marker or "PHARMACITY_PRODUCT",
        "============================================================",
        "PARSED_FIELDS",
        "-------------",
        _field("Ten thuoc", detail.name),
        _field("SKU", detail.sku),
        _field("Slug", detail.slug),
        _field("Thuong hieu", detail.brand),
        _field("Loai san pham", detail.product_type),
        _field("Danh muc", detail.category),
        _field("Can ke don", "Co" if detail.is_prescription_drug else "Khong"),
        _field("Thanh phan", ", ".join(detail.ingredients) if detail.ingredients else None),
        _field("Dong goi/gia", format_variants_price(detail.variants)),
        _field("Mo ta ngan", detail.short_description),
        _field("Thong tin chi tiet", detail.long_description),
        _field("Nguon Pharmacity", detail.source_url),
        "",
        "PARSED_PRODUCT_DETAIL_JSON",
        "--------------------------",
        parsed_json,
        "",
        "RAW_CRAWLED_JSON",
        "----------------",
        raw_json,
    ]
    return "\n".join(line for line in lines if line is not None)


def render_extracted_product_info(detail: ProductDetail) -> str:
    item = raw_product_item(detail.raw)
    sections = extract_drug_info_sections(detail.raw)
    lines = [
        "============================================================",
        _product_marker(detail) or "PHARMACITY_PRODUCT",
        "============================================================",
        "EXTRACTED_DRUG_INFO",
        "-------------------",
        _field("Ten thuoc", detail.name or item.get("name")),
        _field("SKU", detail.sku or item.get("sku")),
        _field("Slug", detail.slug or item.get("slug")),
        _field("Ten ERP", item.get("erp_name")),
        _field("Thuong hieu", detail.brand or item.get("brand_name")),
        _field("Loai san pham", detail.product_type or item.get("product_type_name")),
        _field("Danh muc", detail.category or item.get("category_name")),
        _field("La thuoc", _bool_text(item.get("is_drug"))),
        _field("Can ke don", _bool_text(detail.is_prescription_drug or item.get("is_prescription_drug"))),
        _field("Thuoc dac biet", _bool_text(item.get("is_special_drug"))),
        _field("Dang ban", _bool_text(item.get("is_available_sale"))),
        _field("Dang hien thi", _bool_text(item.get("is_available_listing"))),
        _field("Don vi ban", item.get("sale_unit_name") or item.get("sale_unit")),
        _field("Gia ban", item.get("sale_unit_price")),
        _field("Gia goc", item.get("sale_unit_original_price")),
        _field("Dong goi/gia", format_variants_price(detail.variants)),
        _field("So dang ky", _registration_number(item)),
        _field("Ngay dang", item.get("date_published")),
        _field("Ngay cap nhat", item.get("updated_at")),
        _field("Mo ta ngan", detail.short_description or item.get("short_description")),
        _field("Mo ta chi tiet", detail.long_description or item.get("long_description")),
        _field("Nguon Pharmacity", detail.source_url),
        "",
        "IMAGES",
        "------",
        *_image_lines(item),
        "",
        "VARIANTS",
        "--------",
        *_variant_lines(item.get("variants") or detail.variants),
        "",
        "CRAWLED_ATTRIBUTE_SECTIONS",
        "--------------------------",
        *_section_lines(sections),
        "",
        "ALL_CRAWLED_TEXT_FOR_FLOW",
        "-------------------------",
        render_all_crawled_drug_info(detail.raw, max_chars=50000),
    ]
    return "\n".join(line for line in lines if line is not None)


def _field(label: str, value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    return f"{label}: {text}"


def _details_from_export_text(text: str) -> list[ProductDetail]:
    decoder = json.JSONDecoder()
    details: list[ProductDetail] = []
    search_from = 0
    while True:
        marker_index = text.find("PARSED_PRODUCT_DETAIL_JSON", search_from)
        if marker_index == -1:
            break
        json_start = text.find("{", marker_index)
        if json_start == -1:
            break
        try:
            parsed, end_offset = decoder.raw_decode(text[json_start:])
        except json.JSONDecodeError:
            search_from = json_start + 1
            continue
        if isinstance(parsed, dict):
            try:
                details.append(ProductDetail(**parsed))
            except TypeError:
                pass
        search_from = json_start + end_offset
    return details


def _bool_text(value: Any) -> str | None:
    if value is None:
        return None
    return "Co" if bool(value) else "Khong"


def _registration_number(item: dict[str, Any]) -> str | None:
    registration = item.get("registration") if isinstance(item.get("registration"), dict) else {}
    value = registration.get("registration_number") or item.get("registration_number")
    return str(value).strip() if value else None


def _image_lines(item: dict[str, Any]) -> list[str]:
    lines: list[str] = []
    thumbnail = item.get("thumbnail") if isinstance(item.get("thumbnail"), dict) else {}
    if thumbnail.get("image_url"):
        lines.append(f"- thumbnail: {thumbnail['image_url']}")
    for index, image in enumerate(item.get("images") or [], start=1):
        if isinstance(image, dict) and image.get("image_url"):
            alt = f" ({image.get('image_alt')})" if image.get("image_alt") else ""
            lines.append(f"- image_{index}: {image['image_url']}{alt}")
    return lines or ["- Khong co anh trong du lieu crawl"]


def _variant_lines(variants: Any) -> list[str]:
    if not isinstance(variants, list) or not variants:
        return ["- Khong co bien the/gia trong du lieu crawl"]
    lines: list[str] = []
    for index, variant in enumerate(variants, start=1):
        if not isinstance(variant, dict):
            continue
        parts = [f"{key}={value}" for key, value in variant.items() if value not in (None, "")]
        if parts:
            lines.append(f"- variant_{index}: " + "; ".join(parts))
    return lines or ["- Khong co bien the/gia trong du lieu crawl"]


def _section_lines(sections: Any) -> list[str]:
    lines: list[str] = []
    for section in sections:
        lines.append(f"[{section.code}] {section.name}")
        lines.append(section.text)
        lines.append("")
    return lines or ["Khong co attribute section trong du lieu crawl"]


def _product_marker(detail: ProductDetail) -> str:
    sku = (detail.sku or "").strip()
    if sku:
        return f"PHARMACITY_PRODUCT_SKU: {sku}"
    name = (detail.name or "").strip()
    return f"PHARMACITY_PRODUCT_NAME: {name}" if name else ""


def _replace_existing_section(existing: str, marker: str, section: str) -> str:
    blocks = existing.split("\n\n")
    replaced: list[str] = []
    did_replace = False
    for block in blocks:
        if marker in block:
            replaced.append(section)
            did_replace = True
        elif block.strip():
            replaced.append(block.strip())
    if not did_replace:
        replaced.append(section)
    return "\n\n".join(replaced).rstrip() + "\n"
