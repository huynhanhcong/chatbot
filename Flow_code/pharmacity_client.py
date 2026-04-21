from __future__ import annotations

import html
import re
from typing import Any

from .models import ProductDetail, ProductOption, build_detail_url


class PharmacityClientError(RuntimeError):
    pass


class PharmacityNetworkError(PharmacityClientError):
    pass


class PharmacityParsingError(PharmacityClientError):
    pass


class PharmacityApiClient:
    BASE_URL = "https://api-gateway.pharmacity.vn/"

    def __init__(
        self,
        base_url: str = BASE_URL,
        http_client: Any | None = None,
        timeout_seconds: float = 30.0,
    ) -> None:
        self.base_url = base_url.rstrip("/") + "/"
        self._owns_http_client = http_client is None
        self._http = http_client or self._create_http_client(timeout_seconds)

    def search_products(self, keyword: str, max_options: int = 5) -> list[ProductOption]:
        keyword = keyword.strip()
        if not keyword:
            return []

        payload = self._request_json(
            "pmc-ecm-product/api/public/search/index",
            params={
                "platform": 1,
                "keyword": keyword,
                "order": "desc",
                "order_by": "de-xuat",
                "index": 1,
                "limit": 20,
                "total": 0,
                "refresh": "true",
            },
        )
        data = payload.get("data")
        if not isinstance(data, dict):
            raise PharmacityParsingError("Missing data object in Pharmacity search response.")

        items = data.get("items")
        if items is None:
            return []
        if not isinstance(items, list):
            raise PharmacityParsingError("Invalid items list in Pharmacity search response.")

        options: list[ProductOption] = []
        for item in items:
            if not isinstance(item, dict):
                continue
            option = ProductOption.from_api_item(item, index=len(options) + 1)
            if option.sku and option.name:
                options.append(option)
            if len(options) >= max_options:
                break
        return options

    def fetch_product_detail(self, product: ProductOption) -> ProductDetail:
        if product.slug:
            path = f"pmc-ecm-product/api/public/product/with-slug/{product.slug}"
        elif product.sku:
            path = f"pmc-ecm-product/api/public/product/with-sku/{product.sku}"
        else:
            raise PharmacityParsingError("Product option has no slug or SKU.")

        payload = self._request_json(
            path,
            params={"limit": 15, "include_related": "false"},
        )
        data = payload.get("data")
        if not isinstance(data, dict) or not isinstance(data.get("item"), dict):
            raise PharmacityParsingError("Missing product item in Pharmacity detail response.")
        return _parse_detail_item(data["item"])

    def close(self) -> None:
        if self._owns_http_client and hasattr(self._http, "close"):
            self._http.close()

    def _create_http_client(self, timeout_seconds: float) -> Any:
        try:
            import httpx
        except ImportError as exc:
            raise PharmacityClientError(
                "Missing dependency: install httpx from requirements-rag.txt."
            ) from exc

        return httpx.Client(
            timeout=timeout_seconds,
            follow_redirects=True,
            headers={
                "Accept": "application/json",
                "Accept-Language": "vi",
                "Content-Type": "application/json",
                "Origin": "https://www.pharmacity.vn",
                "Referer": "https://www.pharmacity.vn/",
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123 Safari/537.36"
                ),
                "x-device-id": "pharmacity-drug-flow",
                "x-device-platform": "Chrome",
                "x-device-platform-version": "123",
                "X-Device-Timezone": "Asia/Ho_Chi_Minh",
            },
        )

    def _request_json(self, path: str, params: dict[str, Any]) -> dict[str, Any]:
        url = self.base_url + path.lstrip("/")
        try:
            response = self._http.get(url, params=params)
        except Exception as exc:
            raise PharmacityNetworkError(f"Failed to call Pharmacity API: {exc}") from exc

        status_code = getattr(response, "status_code", None)
        if status_code is not None and status_code >= 400:
            body = str(getattr(response, "text", ""))[:300]
            raise PharmacityNetworkError(
                f"Pharmacity API returned HTTP {status_code}. {body}".strip()
            )

        try:
            payload = response.json()
        except Exception as exc:
            raise PharmacityParsingError("Pharmacity API did not return valid JSON.") from exc
        if not isinstance(payload, dict):
            raise PharmacityParsingError("Pharmacity API JSON root is not an object.")
        return payload


def _parse_detail_item(item: dict[str, Any]) -> ProductDetail:
    slug = _clean_optional_string(item.get("slug"))
    variants = item.get("variants") if isinstance(item.get("variants"), list) else []
    ingredients = item.get("ingredients") if isinstance(item.get("ingredients"), list) else []
    long_description = _html_to_text(_clean_optional_string(item.get("long_description")))
    return ProductDetail(
        sku=str(item.get("sku") or "").strip(),
        slug=slug,
        name=str(item.get("name") or "").strip(),
        brand=_clean_optional_string(item.get("brand_name")),
        product_type=_clean_optional_string(item.get("product_type_name")),
        category=_clean_optional_string(item.get("category_name")),
        short_description=_html_to_text(_clean_optional_string(item.get("short_description"))),
        long_description=long_description,
        ingredients=[str(value).strip() for value in ingredients if str(value).strip()],
        variants=variants,
        is_prescription_drug=bool(item.get("is_prescription_drug")),
        source_url=build_detail_url(slug),
        raw=item,
    )


def _html_to_text(value: str | None) -> str | None:
    if not value:
        return None
    try:
        from bs4 import BeautifulSoup
    except ImportError:
        text = re.sub(r"(?i)<br\s*/?>", "\n", value)
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


def _clean_optional_string(value: Any) -> str | None:
    if value is None:
        return None
    cleaned = str(value).strip()
    return cleaned or None
