from __future__ import annotations

import pytest

from Flow_code.models import ProductDetail, ProductOption
from Flow_code.pharmacity_flow import FlowNotFoundError, FlowValidationError, PharmacityFlow
from Flow_code.response_formatter import format_user_answer
from Flow_code.session_store import InMemorySessionStore


class FakeClient:
    def __init__(self) -> None:
        self.options = [
            ProductOption(
                index=1,
                sku="P00219",
                slug="oresol-245",
                name="Thuoc bot Oresol 245 DHG",
                brand="DHG Pharma",
                price="1.350 VND/Goi",
                detail_url="https://www.pharmacity.vn/oresol-245.html",
                image_url="https://example.test/oresol.jpg",
            ),
            ProductOption(
                index=2,
                sku="P16404",
                slug="oresol-new-hop-20-goi",
                name="Bot pha uong Oresol new Bidiphar",
                brand="Bidiphar",
                price="1.450 VND/Goi",
                detail_url="https://www.pharmacity.vn/oresol-new-hop-20-goi.html",
                image_url="https://example.test/oresol-new.jpg",
            ),
        ]

    def search_products(self, keyword: str, max_options: int = 4) -> list[ProductOption]:
        assert keyword == "Oresol"
        return self.options[:max_options]

    def fetch_product_detail(self, product: ProductOption) -> ProductDetail:
        return ProductDetail(
            sku=product.sku,
            slug=product.slug,
            name=product.name,
            brand=product.brand,
            product_type="Thuoc khong ke don",
            long_description="Dung trong dieu tri mat nuoc.",
            variants=[
                {"price": 1350, "unit_name": "Goi"},
                {"price": 27000, "unit_name": "Hop"},
            ],
            source_url=product.detail_url,
            raw={
                "item": {
                    "sku": product.sku,
                    "name": product.name,
                    "long_description": "<p>Dung trong dieu tri mat nuoc.</p>",
                    "extra_web_field": {"dosage": "Pha voi nuoc"},
                }
            },
        )


class FakeAssistant:
    def extract_drug_name(self, message: str) -> str | None:
        return "Oresol"

    def summarize_product(self, detail: ProductDetail) -> str:
        return f"Tom tat {detail.name}. Thong tin chi mang tinh tham khao."

    def answer_follow_up(
        self,
        detail: ProductDetail,
        question: str,
        previous_answer: str | None = None,
    ) -> str:
        return f"{detail.name} dung trong dieu tri mat nuoc. Cau hoi: {question}"


def make_flow(
    store: InMemorySessionStore | None = None,
    max_options: int = 4,
    export_path: str | None = None,
    extracted_export_path: str | None = None,
) -> PharmacityFlow:
    return PharmacityFlow(
        client=FakeClient(),
        assistant=FakeAssistant(),
        session_store=store or InMemorySessionStore(),
        max_options=max_options,
        export_path=export_path,
        extracted_export_path=extracted_export_path,
    )


def test_start_flow_returns_selection_options() -> None:
    response = make_flow().handle_message("Hay cho toi biet thong tin ve thuoc Oresol")

    assert response["status"] == "need_selection"
    assert response["conversation_id"]
    assert len(response["options"]) == 2
    assert response["options"][0]["sku"] == "P00219"
    assert response["options"][0]["image_url"] == "https://example.test/oresol.jpg"


@pytest.mark.parametrize(
    ("payload", "expected_sku", "expected_name"),
    [
        ({"selected_index": 1}, "P00219", "Thuoc bot Oresol 245 DHG"),
        ({"selected_sku": "P16404"}, "P16404", "Bot pha uong Oresol new Bidiphar"),
        ({"message": "thuoc dau tien"}, "P00219", "Thuoc bot Oresol 245 DHG"),
    ],
)
def test_select_product_by_index_sku_or_text(
    payload: dict,
    expected_sku: str,
    expected_name: str,
) -> None:
    flow = make_flow()
    start = flow.handle_message("Hay cho toi biet thong tin ve thuoc Oresol")

    request = {
        "message": payload.get("message", "1"),
        "conversation_id": start["conversation_id"],
        "selected_index": payload.get("selected_index"),
        "selected_sku": payload.get("selected_sku"),
    }
    response = flow.handle_message(**request)

    assert response["status"] == "answered"
    assert response["selected_product"]["sku"] == expected_sku
    assert response["answer"].startswith(f"Tom tat {expected_name}.")


def test_select_product_uses_original_price_question() -> None:
    flow = make_flow()
    start = flow.handle_message("Cho toi xin gia cua thuoc Oresol")

    response = flow.handle_message(
        "1",
        conversation_id=start["conversation_id"],
        selected_index=1,
    )

    assert response["status"] == "answered"
    assert response["answer"] == 'Sản phẩm "Thuoc bot Oresol 245 DHG" có giá là: 1.350 VND/Goi; 27.000 VND/Hop'


def test_pending_selection_is_reused_for_contextual_follow_up() -> None:
    flow = make_flow()
    start = flow.handle_message("Cho toi xin gia cua thuoc Oresol")

    response = flow.handle_message(
        "gia bao nhieu",
        conversation_id=start["conversation_id"],
    )

    assert response["status"] == "need_selection"
    assert response["conversation_id"] == start["conversation_id"]
    assert len(response["options"]) == 2


def test_select_invalid_index_raises_validation_error() -> None:
    flow = make_flow()
    start = flow.handle_message("Hay cho toi biet thong tin ve thuoc Oresol")

    with pytest.raises(FlowValidationError):
        flow.handle_message("10", conversation_id=start["conversation_id"], selected_index=10)


def test_follow_up_uses_selected_product_context() -> None:
    flow = make_flow()
    start = flow.handle_message("Hay cho toi biet thong tin ve thuoc Oresol")
    selected = flow.handle_message(
        "1",
        conversation_id=start["conversation_id"],
        selected_index=1,
    )

    response = flow.handle_message(
        "san pham tren co cong dung la gi",
        conversation_id=selected["conversation_id"],
    )

    assert response["status"] == "answered"
    assert response["selected_product"]["sku"] == "P00219"
    assert "dieu tri mat nuoc" in response["answer"]
    assert "Pharmacity" not in response["answer"]


def test_select_product_exports_crawled_drug_detail_to_text_file(tmp_path) -> None:
    export_path = tmp_path / "pharmacity.txt"
    extracted_path = tmp_path / "pharmacity_1.txt"
    flow = make_flow(export_path=str(export_path), extracted_export_path=str(extracted_path))
    start = flow.handle_message("Hay cho toi biet thong tin ve thuoc Oresol")

    flow.handle_message(
        "1",
        conversation_id=start["conversation_id"],
        selected_index=1,
    )

    exported = export_path.read_text(encoding="utf-8")
    assert "PHARMACITY_PRODUCT_SKU: P00219" in exported
    assert "Ten thuoc: Thuoc bot Oresol 245 DHG" in exported
    assert "Loai san pham: Thuoc khong ke don" in exported
    assert "Thong tin chi tiet: Dung trong dieu tri mat nuoc." in exported
    assert "PARSED_PRODUCT_DETAIL_JSON" in exported
    assert "RAW_CRAWLED_JSON" in exported
    assert '"extra_web_field"' in exported
    assert '"dosage": "Pha voi nuoc"' in exported
    assert "Hay cho toi biet" not in exported

    extracted = extracted_path.read_text(encoding="utf-8")
    assert "EXTRACTED_DRUG_INFO" in extracted
    assert "PHARMACITY_PRODUCT_SKU: P00219" in extracted
    assert "Ten thuoc: Thuoc bot Oresol 245 DHG" in extracted
    assert "ALL_CRAWLED_TEXT_FOR_FLOW" in extracted


def test_expired_session_raises_not_found() -> None:
    now = {"value": 0.0}
    store = InMemorySessionStore(ttl_seconds=1, time_func=lambda: now["value"])
    flow = make_flow(store)
    start = flow.handle_message("Hay cho toi biet thong tin ve thuoc Oresol")
    now["value"] = 2.0

    with pytest.raises(FlowNotFoundError):
        flow.handle_message("1", conversation_id=start["conversation_id"], selected_index=1)


def test_user_answer_formatter_removes_noise_and_duplicate_lines() -> None:
    answer = format_user_answer(
        'Thành phần của "Evadays":\n'
        "Thành phần\n\n"
        "Metronidazole: 500mg\n\n"
        "Neomycin sulfat: 65000IU\n\n"
        "Nystatin: 100000IU\n\n"
        ",\n\n"
        "Nystatin\n\n"
        "Nystatin\n"
    )

    assert "," not in answer.splitlines()
    assert answer.count("Nystatin") == 2
    assert "\n\n\n" not in answer
