from __future__ import annotations

from .drug_extraction import normalize_search_text
from .models import ProductDetail, format_variants_price


DISCLAIMER = "Thông tin chỉ mang tính tham khảo, không thay thế tư vấn y tế."


def render_product_summary(detail: ProductDetail) -> str:
    parts = [
        ("Tên thuốc", detail.name),
        ("Thương hiệu", detail.brand),
        ("Loại sản phẩm", detail.product_type),
        ("Danh mục", detail.category),
        ("Thành phần", ", ".join(detail.ingredients) if detail.ingredients else None),
        ("Đóng gói/giá", format_variants_price(detail.variants)),
        ("Mô tả", detail.short_description or _truncate(detail.long_description, 900)),
        ("Nguồn", detail.source_url),
    ]
    lines = [f"{label}: {value}" for label, value in parts if value]
    if detail.is_prescription_drug:
        lines.append("Lưu ý: Sản phẩm này có thông tin là thuốc kê đơn, cần hỏi bác sĩ/dược sĩ trước khi dùng.")
    lines.append(DISCLAIMER)
    return "\n".join(lines)


def answer_product_follow_up_template(detail: ProductDetail, question: str) -> str | None:
    normalized = normalize_search_text(question)
    if not normalized:
        return None

    if _has_any(normalized, ["gia", "bao nhieu", "chi phi", "dong goi"]):
        price = format_variants_price(detail.variants)
        if price:
            return _with_disclaimer(f"Đóng gói/giá: {price}\nNguồn: {detail.source_url or 'Pharmacity'}")
        return _with_disclaimer("Dữ liệu Pharmacity hiện chưa có thông tin giá/đóng gói cho sản phẩm này.")

    if _has_any(normalized, ["thanh phan", "hoat chat"]):
        if detail.ingredients:
            return _with_disclaimer("Thành phần: " + ", ".join(detail.ingredients))
        return _with_disclaimer("Dữ liệu Pharmacity hiện chưa có thông tin thành phần cho sản phẩm này.")

    if _has_any(normalized, ["ke don", "don thuoc", "can don"]):
        value = "có" if detail.is_prescription_drug else "không"
        return _with_disclaimer(f"Thông tin Pharmacity ghi nhận sản phẩm này {value} thuộc nhóm cần kê đơn.")

    if _has_any(normalized, ["nguon", "link", "xem them"]):
        if detail.source_url:
            return _with_disclaimer(f"Nguồn Pharmacity: {detail.source_url}")
        return _with_disclaimer("Dữ liệu hiện chưa có đường dẫn nguồn cho sản phẩm này.")

    if _has_any(normalized, ["cong dung", "chi dinh", "dung de lam gi", "mo ta", "lieu dung", "cach dung"]):
        description = detail.long_description or detail.short_description
        if description:
            return _with_disclaimer(_truncate(description, 1200))
        return _with_disclaimer("Dữ liệu Pharmacity hiện chưa có mô tả/công dụng/ cách dùng cho sản phẩm này.")

    return None


def _with_disclaimer(value: str) -> str:
    return f"{value}\n{DISCLAIMER}"


def _has_any(value: str, keywords: list[str]) -> bool:
    return any(keyword in value for keyword in keywords)


def _truncate(value: str | None, max_chars: int) -> str | None:
    if not value:
        return None
    cleaned = " ".join(str(value).split())
    if len(cleaned) <= max_chars:
        return cleaned
    return cleaned[: max_chars - 14].rstrip() + " ...[rút gọn]"
