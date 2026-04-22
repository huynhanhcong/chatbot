from __future__ import annotations

from Flow_code.cached_pharmacity_client import CachedPharmacityClient
from Flow_code.drug_extraction import extract_drug_name_local
from Flow_code.drug_summary import answer_product_follow_up_template, render_product_summary
from Flow_code.models import ProductDetail, ProductOption
from Flow_code.pharmacity_index import PharmacityProductIndex, rank_product_options


class CountingApi:
    def __init__(self) -> None:
        self.search_calls = 0
        self.detail_calls = 0

    def search_products(self, keyword: str, max_options: int = 5) -> list[ProductOption]:
        self.search_calls += 1
        return [
            ProductOption(
                index=1,
                sku="P00219",
                slug="oresol-245",
                name="Thuoc bot Oresol 245 DHG",
                brand="DHG Pharma",
                price="1.350 VND/Goi",
                detail_url="https://www.pharmacity.vn/oresol-245.html",
                image_url="https://example.test/oresol.jpg",
                ingredients=["Oresol"],
            )
        ][:max_options]

    def fetch_product_detail(self, product: ProductOption) -> ProductDetail:
        self.detail_calls += 1
        return ProductDetail(
            sku=product.sku,
            slug=product.slug,
            name=product.name,
            brand=product.brand,
            product_type="Thuoc khong ke don",
            long_description="Dung de bo sung nuoc va dien giai.",
            ingredients=["Oresol"],
            variants=[{"price": 1350, "unit_name": "Goi"}],
            source_url=product.detail_url,
        )


def test_local_drug_extractor_handles_common_names_and_sku() -> None:
    assert extract_drug_name_local("Hay tim thuoc oresol") == "Oresol"
    assert extract_drug_name_local("paracetamol 500mg") == "Paracetamol"
    assert extract_drug_name_local("sku P00219") == "P00219"


def test_cached_pharmacity_search_hit_skips_api(tmp_path) -> None:
    api = CountingApi()
    index = PharmacityProductIndex(tmp_path / "products.sqlite")
    client = CachedPharmacityClient(api_client=api, index=index, cache_ttl_seconds=900)

    first = client.search_products("Oresol", max_options=4)
    second = client.search_products("Oresol", max_options=4)

    assert [item.sku for item in first] == ["P00219"]
    assert [item.sku for item in second] == ["P00219"]
    assert api.search_calls == 1


def test_cached_pharmacity_detail_hit_skips_api(tmp_path) -> None:
    api = CountingApi()
    client = CachedPharmacityClient(
        api_client=api,
        index=PharmacityProductIndex(tmp_path / "products.sqlite"),
        cache_ttl_seconds=900,
    )
    option = client.search_products("Oresol", max_options=1)[0]

    first = client.fetch_product_detail(option)
    second = client.fetch_product_detail(option)

    assert first.sku == second.sku == "P00219"
    assert api.detail_calls == 1


def test_cached_pharmacity_detail_preserves_search_image(tmp_path) -> None:
    api = CountingApi()
    client = CachedPharmacityClient(
        api_client=api,
        index=PharmacityProductIndex(tmp_path / "products.sqlite"),
        cache_ttl_seconds=900,
    )
    option = client.search_products("Oresol", max_options=1)[0]

    client.fetch_product_detail(option)
    cached = client.search_products("Oresol", max_options=1)

    assert cached[0].image_url == "https://example.test/oresol.jpg"
    assert api.search_calls == 1


def test_cached_pharmacity_search_refreshes_options_with_missing_images(tmp_path) -> None:
    api = CountingApi()
    index = PharmacityProductIndex(tmp_path / "products.sqlite")
    index.save_search(
        "Oresol",
        [
            ProductOption(
                index=1,
                sku="P00219",
                slug="oresol-245",
                name="Thuoc bot Oresol 245 DHG",
                brand="DHG Pharma",
                price="1.350 VND/Goi",
                detail_url="https://www.pharmacity.vn/oresol-245.html",
                image_url=None,
                ingredients=["Oresol"],
            )
        ],
    )
    client = CachedPharmacityClient(api_client=api, index=index, cache_ttl_seconds=900)

    refreshed = client.search_products("Oresol", max_options=1)

    assert refreshed[0].image_url == "https://example.test/oresol.jpg"
    assert api.search_calls == 1


def test_product_ranking_prefers_name_and_ingredient_match() -> None:
    options = [
        ProductOption(index=1, sku="P1", slug=None, name="Vitamin C", ingredients=["acid ascorbic"]),
        ProductOption(index=2, sku="P2", slug=None, name="Thuoc bot Oresol", ingredients=["oresol"]),
    ]

    ranked = rank_product_options("oresol", options)

    assert ranked[0].sku == "P2"


def test_template_summary_and_followup_do_not_need_llm() -> None:
    detail = ProductDetail(
        sku="P00219",
        slug="oresol-245",
        name="Thuoc bot Oresol",
        ingredients=["Oresol"],
        variants=[{"price": 1350, "unit_name": "Goi"}],
        long_description="Dung de bo sung nuoc va dien giai.",
        source_url="https://www.pharmacity.vn/oresol-245.html",
    )

    summary = render_product_summary(detail)
    follow_up = answer_product_follow_up_template(detail, "gia bao nhieu?")

    assert "Oresol" in summary
    assert "1.350 VND/Goi" in (follow_up or "")
