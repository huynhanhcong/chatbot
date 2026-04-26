from __future__ import annotations

import json
from pathlib import Path

from Flow_code.pharmacity_client import PharmacityApiClient


FIXTURE_DIR = Path(__file__).parent / "fixtures"


class FakeResponse:
    def __init__(self, payload: dict, status_code: int = 200) -> None:
        self._payload = payload
        self.status_code = status_code
        self.text = json.dumps(payload, ensure_ascii=False)

    def json(self) -> dict:
        return self._payload


class FakeHttpClient:
    def __init__(self) -> None:
        self.requests: list[tuple[str, dict]] = []
        self.search_payload = _load_fixture("pharmacity_search_oresol.json")
        self.detail_payload = _load_fixture("pharmacity_detail_oresol.json")

    def get(self, url: str, params: dict) -> FakeResponse:
        self.requests.append((url, params))
        if "search/index" in url:
            return FakeResponse(self.search_payload)
        if "product/with-slug" in url:
            return FakeResponse(self.detail_payload)
        raise AssertionError(f"Unexpected URL: {url}")


def test_search_products_maps_api_items_to_options() -> None:
    http_client = FakeHttpClient()
    client = PharmacityApiClient(http_client=http_client)

    options = client.search_products("Oresol", max_options=5)

    assert len(options) == 2
    assert options[0].sku == "P00219"
    assert "Oresol 245" in options[0].name
    assert options[0].brand == "DHG Pharma"
    assert options[0].price == "1.350 VND/Gói; 27.000 VND/Hộp"
    assert options[0].detail_url.endswith(".html")
    assert http_client.requests[0][1]["platform"] == 1
    assert http_client.requests[0][1]["keyword"] == "Oresol"


def test_fetch_product_detail_parses_html_description() -> None:
    http_client = FakeHttpClient()
    client = PharmacityApiClient(http_client=http_client)
    option = client.search_products("Oresol", max_options=1)[0]

    detail = client.fetch_product_detail(option)

    assert detail.sku == "P00219"
    assert detail.product_type == "Thuốc không kê đơn"
    assert "Dùng trong điều trị mất nước" in (detail.long_description or "")
    assert "Pha 1 gói vào 200 ml" in (detail.long_description or "")
    assert detail.source_url and detail.source_url.endswith(".html")
    assert detail.raw["api_response"]["data"]["item"]["sku"] == "P00219"
    assert detail.raw["item"]["long_description"].startswith("<p>")


def _load_fixture(name: str) -> dict:
    return json.loads((FIXTURE_DIR / name).read_text(encoding="utf-8"))
