from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


PHARMACITY_WEB_BASE_URL = "https://www.pharmacity.vn"


def build_detail_url(slug: str | None) -> str | None:
    if not slug:
        return None
    return f"{PHARMACITY_WEB_BASE_URL}/{slug}.html"


@dataclass(frozen=True)
class ProductOption:
    index: int
    sku: str
    slug: str | None
    name: str
    brand: str | None = None
    price: str | None = None
    detail_url: str | None = None
    image_url: str | None = None
    ingredients: list[str] = field(default_factory=list)
    is_prescription_drug: bool = False

    @classmethod
    def from_api_item(cls, item: dict[str, Any], index: int) -> ProductOption:
        sku = str(item.get("sku") or "").strip()
        slug = _clean_optional_string(item.get("slug"))
        name = str(item.get("name") or "").strip()
        thumbnail = item.get("thumbnail") if isinstance(item.get("thumbnail"), dict) else {}
        ingredients = item.get("ingredients") if isinstance(item.get("ingredients"), list) else []
        return cls(
            index=index,
            sku=sku,
            slug=slug,
            name=name,
            brand=_clean_optional_string(item.get("brand_name")),
            price=format_variants_price(item.get("variants")),
            detail_url=build_detail_url(slug),
            image_url=_clean_optional_string(thumbnail.get("image_url")),
            ingredients=[str(value).strip() for value in ingredients if str(value).strip()],
            is_prescription_drug=bool(item.get("is_prescription_drug")),
        )

    def to_response(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "sku": self.sku,
            "name": self.name,
            "brand": self.brand,
            "price": self.price,
            "detail_url": self.detail_url,
            "image_url": self.image_url,
        }


@dataclass(frozen=True)
class ProductDetail:
    sku: str
    slug: str | None
    name: str
    brand: str | None = None
    product_type: str | None = None
    category: str | None = None
    short_description: str | None = None
    long_description: str | None = None
    ingredients: list[str] = field(default_factory=list)
    variants: list[dict[str, Any]] = field(default_factory=list)
    is_prescription_drug: bool = False
    source_url: str | None = None
    raw: dict[str, Any] = field(default_factory=dict)

    def selected_product_response(self) -> dict[str, Any]:
        return {
            "sku": self.sku,
            "name": self.name,
            "detail_url": self.source_url,
        }

    def to_context_text(self, max_description_chars: int = 12000) -> str:
        parts = [
            ("Ten thuoc", self.name),
            ("SKU", self.sku),
            ("Thuong hieu", self.brand),
            ("Loai san pham", self.product_type),
            ("Danh muc", self.category),
            ("Can ke don", "Co" if self.is_prescription_drug else "Khong"),
            ("Thanh phan", ", ".join(self.ingredients) if self.ingredients else None),
            ("Dong goi/gia", format_variants_price(self.variants)),
            ("Mo ta ngan", self.short_description),
            ("Thong tin chi tiet", _truncate(self.long_description, max_description_chars)),
            ("Nguon", self.source_url),
        ]
        return "\n".join(f"{label}: {value}" for label, value in parts if value)


def format_variants_price(variants: Any) -> str | None:
    if not isinstance(variants, list):
        return None

    values: list[str] = []
    for variant in variants:
        if not isinstance(variant, dict):
            continue
        price = variant.get("price")
        if price is None:
            continue
        unit_name = _clean_optional_string(variant.get("unit_name") or variant.get("unit"))
        formatted_price = _format_vnd(price)
        values.append(f"{formatted_price}/{unit_name}" if unit_name else formatted_price)

    return "; ".join(values[:3]) if values else None


def _format_vnd(value: Any) -> str:
    try:
        amount = int(float(value))
    except (TypeError, ValueError):
        return str(value)
    return f"{amount:,}".replace(",", ".") + " VND"


def _clean_optional_string(value: Any) -> str | None:
    if value is None:
        return None
    cleaned = str(value).strip()
    return cleaned or None


def _truncate(value: str | None, max_chars: int) -> str | None:
    if not value:
        return None
    if len(value) <= max_chars:
        return value
    return value[:max_chars].rstrip() + "\n...[da rut gon]"
