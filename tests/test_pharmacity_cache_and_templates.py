from __future__ import annotations

from Flow_code.cached_pharmacity_client import CachedPharmacityClient
from Flow_code.drug_extraction import extract_drug_name_local
from Flow_code.drug_summary import answer_product_follow_up_template, render_product_summary
from Flow_code.models import ProductDetail, ProductOption
from Flow_code.pharmacity_export import rebuild_extracted_file_from_raw_export
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
    source = answer_product_follow_up_template(detail, "cho toi xin nguon")

    assert "Pharmacity" not in summary
    assert summary.startswith("Thuoc bot Oresol là một sản phẩm thuốc.")
    assert "Giá hiện tại: 1.350 VND/Goi." in summary
    assert follow_up == 'Sản phẩm "Thuoc bot Oresol" có giá là: 1.350 VND/Goi'
    assert source == "Bạn có thể xem thêm thông tin sản phẩm tại đây: https://www.pharmacity.vn/oresol-245.html"


def test_template_uses_raw_crawled_pharmacity_sections() -> None:
    detail = ProductDetail(
        sku="P28294",
        slug="panadol-vien-sui",
        name="Vien sui Panadol",
        variants=[{"price": 84000, "unit_name": "Hop"}],
        raw={
            "item": {
                "attributes": [
                    {
                        "items": [
                            {
                                "code": "thanh-phan",
                                "name": "Thành phần",
                                "value": {"value": "<p>Paracetamol 500 mg</p>"},
                            },
                            {
                                "code": "huong-dan-su-dung",
                                "name": "Cách Sử Dụng",
                                "value": {"value": "<p>Hòa tan 1 viên trong nước.</p>"},
                            },
                            {
                                "code": "than-trong",
                                "name": "Thận trọng",
                                "value": {"value": "<p>Không dùng quá liều chỉ định.</p>"},
                            },
                        ]
                    }
                ]
            }
        },
    )

    ingredients = answer_product_follow_up_template(detail, "thanh phan la gi")
    usage = answer_product_follow_up_template(detail, "cach dung nhu the nao")
    warning = answer_product_follow_up_template(detail, "can luu y gi")
    full = answer_product_follow_up_template(detail, "day du thong tin")

    assert ingredients and "Paracetamol 500 mg" in ingredients
    assert usage and "Hòa tan 1 viên trong nước" in usage
    assert warning and "Không dùng quá liều" in warning
    assert full and "Thành phần" in full and "Cách Sử Dụng" in full


def test_ingredient_answer_filters_duplicate_active_ingredient_list() -> None:
    detail = ProductDetail(
        sku="P-EVADAYS",
        slug="evadays",
        name="Evadays 500mg Mediplantex (Hộp 1 vi x 10 viên)",
        raw={
            "item": {
                "attributes": [
                    {
                        "items": [
                            {
                                "code": "thanh-phan",
                                "name": "Thành phần",
                                "value": {
                                    "value": (
                                        "<h2>Thành phần</h2>"
                                        "<p>Metronidazole: 500mg<br>"
                                        "Neomycin sulfat: 65000IU<br>"
                                        "Nystatin: 100000IU<br><br>"
                                        "Metronidazole<br>,<br>Neomycin<br>sulfat,<br>Nystatin</p>"
                                    )
                                },
                            }
                        ]
                    }
                ]
            }
        },
    )

    answer = answer_product_follow_up_template(detail, "Thành phần trong thuốc")

    assert answer == (
        'Thành phần của "Evadays 500mg Mediplantex (Hộp 1 vi x 10 viên)":\n'
        "Metronidazole: 500mg\n\n"
        "Neomycin sulfat: 65000IU\n\n"
        "Nystatin: 100000IU"
    )


def test_rebuild_extracted_file_from_pharmacity_export(tmp_path) -> None:
    source = tmp_path / "pharmacity.txt"
    destination = tmp_path / "pharmacity_1.txt"
    source.write_text(
        """
============================================================
PHARMACITY_PRODUCT_SKU: P1
============================================================
PARSED_PRODUCT_DETAIL_JSON
--------------------------
{
  "sku": "P1",
  "slug": "thuoc-a",
  "name": "Thuoc A",
  "brand": "Brand A",
  "product_type": "Thuoc khong ke don",
  "category": "Giam dau",
  "short_description": null,
  "long_description": null,
  "ingredients": [],
  "variants": [{"price": 1000, "unit_name": "Hop"}],
  "is_prescription_drug": false,
  "source_url": "https://example.test/thuoc-a.html",
  "raw": {
    "item": {
      "sku": "P1",
      "name": "Thuoc A",
      "attributes": [
        {
          "items": [
            {
              "code": "chi-dinh",
              "name": "Chỉ định",
              "value": {"value": "<p>Dùng để giảm đau.</p>"}
            }
          ]
        }
      ]
    }
  }
}
""",
        encoding="utf-8",
    )

    count = rebuild_extracted_file_from_raw_export(
        source_path=source,
        destination_path=destination,
    )

    output = destination.read_text(encoding="utf-8")
    assert count == 1
    assert "EXTRACTED_DRUG_INFO" in output
    assert "Ten thuoc: Thuoc A" in output
    assert "[chi-dinh] Chỉ định" in output
    assert "Dùng để giảm đau" in output
