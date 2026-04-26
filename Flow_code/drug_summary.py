from __future__ import annotations

import re

from .drug_extraction import normalize_search_text
from .models import ProductDetail, format_variants_price
from .pharmacity_detail_extractor import find_section_text, render_all_crawled_drug_info


def render_product_summary(detail: ProductDetail) -> str:
    sentences: list[str] = []

    intro = detail.name
    if detail.product_type:
        intro += f" là {detail.product_type.lower()}"
    else:
        intro += " là một sản phẩm thuốc"
    if detail.brand:
        intro += f" của {detail.brand}"
    sentences.append(_ensure_sentence(intro))

    price = format_variants_price(detail.variants)
    if price:
        sentences.append(f"Giá hiện tại: {price}.")

    description = _description_excerpt(detail)
    if description:
        sentences.append(description)

    if detail.is_prescription_drug:
        sentences.append("Đây là thuốc kê đơn, bạn nên hỏi bác sĩ hoặc dược sĩ trước khi dùng.")

    return " ".join(sentence for sentence in sentences if sentence).strip()


def answer_product_follow_up_template(detail: ProductDetail, question: str) -> str | None:
    normalized = normalize_search_text(question)
    if not normalized:
        return None

    if _has_any(normalized, ["gia", "bao nhieu", "chi phi", "dong goi"]):
        price = format_variants_price(detail.variants)
        if price:
            return f'Sản phẩm "{detail.name}" có giá là: {price}'
        return f'Sản phẩm "{detail.name}" hiện chưa có thông tin giá.'

    if _has_any(normalized, ["thanh phan", "hoat chat"]):
        raw_text = find_section_text(
            detail.raw,
            codes=["thanh-phan"],
            names=["thanh phan", "hoat chat"],
            max_chars=1600,
        )
        ingredient_text = _format_ingredient_answer_text(raw_text)
        if ingredient_text:
            return f'Thành phần của "{detail.name}":\n{ingredient_text}'
        if detail.ingredients:
            return f'Thành phần của "{detail.name}":\n' + "\n\n".join(detail.ingredients)
        return f'Sản phẩm "{detail.name}" hiện chưa có thông tin thành phần.'

    if _has_any(normalized, ["ke don", "don thuoc", "can don"]):
        if detail.is_prescription_drug:
            return f'"{detail.name}" thuộc nhóm cần kê đơn.'
        return f'"{detail.name}" không thuộc nhóm cần kê đơn.'

    if _has_any(normalized, ["thuong hieu", "hang"]):
        if detail.brand:
            return f'Thương hiệu của "{detail.name}" là {detail.brand}.'
        return f'Sản phẩm "{detail.name}" hiện chưa có thông tin thương hiệu.'

    if _has_any(normalized, ["nguon", "link", "xem them"]):
        if detail.source_url:
            return f"Bạn có thể xem thêm thông tin sản phẩm tại đây: {detail.source_url}"
        return f'Sản phẩm "{detail.name}" hiện chưa có đường dẫn tham khảo.'

    if _has_any(normalized, ["cong dung", "chi dinh", "dung de lam gi", "mo ta"]):
        raw_text = find_section_text(
            detail.raw,
            codes=["chi-dinh", "cong-dung"],
            names=["chi dinh", "cong dung"],
            max_chars=1400,
        )
        formatted = _format_general_section_text(raw_text, drop_titles=["công dụng", "chỉ định"])
        if formatted:
            return f'Công dụng/chỉ định của "{detail.name}":\n{formatted}'
        description = _description_excerpt(detail, max_chars=500, max_sentences=3)
        if description:
            return f'Thông tin hiện có về "{detail.name}": {description}'
        return f'Sản phẩm "{detail.name}" hiện chưa có mô tả hoặc công dụng.'

    if _has_any(normalized, ["lieu dung", "cach dung", "huong dan su dung", "uong nhu the nao"]):
        raw_text = find_section_text(
            detail.raw,
            codes=["huong-dan-su-dung", "cach-dung", "lieu-dung"],
            names=["cach su dung", "huong dan su dung", "lieu luong", "cach dung"],
            max_chars=1400,
        )
        formatted = _format_general_section_text(raw_text, drop_titles=["hướng dẫn sử dụng", "cách sử dụng"])
        if formatted:
            return f'Cách dùng/liều dùng của "{detail.name}":\n{formatted}'
        return f'Sản phẩm "{detail.name}" hiện chưa có thông tin cách dùng/liều dùng.'

    if _has_any(normalized, ["than trong", "chong chi dinh", "luu y", "canh bao"]):
        raw_text = find_section_text(
            detail.raw,
            codes=["than-trong", "chong-chi-dinh", "canh-bao"],
            names=["than trong", "chong chi dinh", "canh bao", "luu y"],
            max_chars=1400,
        )
        formatted = _format_general_section_text(raw_text, drop_titles=["lưu ý", "thận trọng"])
        if formatted:
            return f'Lưu ý/thận trọng của "{detail.name}":\n{formatted}'
        return f'Sản phẩm "{detail.name}" hiện chưa có thông tin thận trọng/chống chỉ định.'

    if _has_any(normalized, ["tac dung phu", "tac dung khong mong muon", "phan ung phu"]):
        raw_text = find_section_text(
            detail.raw,
            codes=["tac-dung-khong-mong-muon"],
            names=["tac dung khong mong muon", "tac dung phu"],
            max_chars=1400,
        )
        formatted = _format_general_section_text(raw_text, drop_titles=["tác dụng không mong muốn"])
        if formatted:
            return f'Tác dụng không mong muốn của "{detail.name}":\n{formatted}'
        return f'Sản phẩm "{detail.name}" hiện chưa có thông tin tác dụng không mong muốn.'

    if _has_any(normalized, ["bao quan", "nha san xuat", "xuat xu", "san xuat"]):
        raw_text = find_section_text(
            detail.raw,
            codes=["thong-tin-san-xuat", "manufacturer", "origin"],
            names=["thong tin san xuat", "nha san xuat", "bao quan", "noi san xuat"],
            max_chars=1400,
        )
        formatted = _format_general_section_text(raw_text, drop_titles=["thông tin sản xuất"])
        if formatted:
            return f'Thông tin sản xuất/bảo quản của "{detail.name}":\n{formatted}'
        return f'Sản phẩm "{detail.name}" hiện chưa có thông tin sản xuất/bảo quản.'

    if _has_any(normalized, ["tat ca thong tin", "day du thong tin", "thong tin day du"]):
        raw_text = render_all_crawled_drug_info(detail.raw, max_chars=2200)
        formatted = _format_general_section_text(raw_text)
        if formatted:
            return f'Thông tin crawl được về "{detail.name}":\n{formatted}'
        return f'Thông tin hiện có về "{detail.name}": {detail.to_context_text(max_description_chars=1800)}'

    return None


def _format_ingredient_answer_text(value: str | None) -> str:
    lines = _clean_section_lines(value)
    lines = [
        line
        for line in lines
        if _normalized(line) not in {"thanh phan", "hoat chat"}
        and line not in {",", ";", "."}
    ]

    # Prefer precise ingredient rows with dosage/strength. This removes later
    # raw Pharmacity link lists like "Metronidazole, Neomycin sulfat, Nystatin".
    dosage_lines = [
        line
        for line in lines
        if ":" in line and re.search(r"\d", line)
    ]
    selected = dosage_lines or lines
    return "\n\n".join(_dedupe_preserve_order(selected)).strip()


def _format_general_section_text(value: str | None, *, drop_titles: list[str] | None = None) -> str:
    drop = {_normalized(title) for title in drop_titles or []}
    lines = [
        line
        for line in _clean_section_lines(value)
        if _normalized(line) not in drop and line not in {",", ";", "."}
    ]
    return "\n\n".join(_dedupe_preserve_order(lines)).strip()


def _clean_section_lines(value: str | None) -> list[str]:
    if not value:
        return []
    lines: list[str] = []
    for raw_line in str(value).replace("\r\n", "\n").splitlines():
        line = _clean_text(raw_line)
        if not line:
            continue
        # Normalize broken punctuation-only rows and accidental spaces before punctuation.
        line = re.sub(r"\s+([,.;:])", r"\1", line)
        lines.append(line)
    return lines


def _dedupe_preserve_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for value in values:
        key = _normalized(value)
        if key in seen:
            continue
        seen.add(key)
        deduped.append(value)
    return deduped


def _description_excerpt(
    detail: ProductDetail,
    *,
    max_chars: int = 260,
    max_sentences: int = 2,
) -> str | None:
    text = (
        detail.short_description
        or detail.long_description
        or find_section_text(detail.raw, codes=["chi-dinh", "thanh-phan"], names=["chi dinh", "thanh phan"])
    )
    cleaned = _clean_text(text)
    if not cleaned:
        return None

    sentences = re.split(r"(?<=[.!?])\s+", cleaned)
    excerpt = " ".join(sentence.strip() for sentence in sentences[:max_sentences] if sentence.strip())
    excerpt = excerpt or cleaned
    if len(excerpt) > max_chars:
        excerpt = excerpt[: max_chars - 14].rstrip() + " ...[rút gọn]"
    return _ensure_sentence(excerpt)


def _has_any(value: str, keywords: list[str]) -> bool:
    return any(keyword in value for keyword in keywords)


def _clean_text(value: str | None) -> str:
    if not value:
        return ""
    return " ".join(str(value).split()).strip()


def _ensure_sentence(value: str) -> str:
    cleaned = _clean_text(value)
    if not cleaned:
        return ""
    if cleaned[-1] in ".!?":
        return cleaned
    return cleaned + "."


def _normalized(value: str) -> str:
    return normalize_search_text(value)
