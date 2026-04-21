from __future__ import annotations

import pytest

from Flow_code.models import ProductDetail, ProductOption
from Flow_code.pharmacity_flow import FlowNotFoundError, FlowValidationError, PharmacityFlow
from Flow_code.session_store import InMemorySessionStore


class FakeClient:
    def __init__(self) -> None:
        self.options = [
            ProductOption(
                index=1,
                sku="P00219",
                slug="oresol-245",
                name="Thuốc bột Oresol 245 DHG",
                brand="DHG Pharma",
                price="1.350 VND/Gói",
                detail_url="https://www.pharmacity.vn/oresol-245.html",
                image_url="https://example.test/oresol.jpg",
            ),
            ProductOption(
                index=2,
                sku="P16404",
                slug="oresol-new-hop-20-goi",
                name="Bột pha uống Oresol new Bidiphar",
                brand="Bidiphar",
                price="1.450 VND/Gói",
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
            product_type="Thuốc không kê đơn",
            long_description="Dùng trong điều trị mất nước.",
            source_url=product.detail_url,
        )


class FakeAssistant:
    def extract_drug_name(self, message: str) -> str | None:
        return "Oresol"

    def summarize_product(self, detail: ProductDetail) -> str:
        return f"Tóm tắt {detail.name}. Thông tin chỉ mang tính tham khảo."

    def answer_follow_up(
        self,
        detail: ProductDetail,
        question: str,
        previous_answer: str | None = None,
    ) -> str:
        return f"{detail.name} dùng trong điều trị mất nước. Câu hỏi: {question}"


def make_flow(
    store: InMemorySessionStore | None = None,
    max_options: int = 4,
) -> PharmacityFlow:
    return PharmacityFlow(
        client=FakeClient(),
        assistant=FakeAssistant(),
        session_store=store or InMemorySessionStore(),
        max_options=max_options,
    )


def test_start_flow_returns_selection_options() -> None:
    response = make_flow().handle_message("Hãy cho tôi biết thông tin về thuốc Oresol")

    assert response["status"] == "need_selection"
    assert response["conversation_id"]
    assert len(response["options"]) == 2
    assert response["options"][0]["sku"] == "P00219"
    assert response["options"][0]["image_url"] == "https://example.test/oresol.jpg"


@pytest.mark.parametrize(
    ("payload", "expected_sku"),
    [
        ({"selected_index": 1}, "P00219"),
        ({"selected_sku": "P16404"}, "P16404"),
        ({"message": "thuốc đầu tiên"}, "P00219"),
    ],
)
def test_select_product_by_index_sku_or_text(payload: dict, expected_sku: str) -> None:
    flow = make_flow()
    start = flow.handle_message("Hãy cho tôi biết thông tin về thuốc Oresol")

    request = {
        "message": payload.get("message", "1"),
        "conversation_id": start["conversation_id"],
        "selected_index": payload.get("selected_index"),
        "selected_sku": payload.get("selected_sku"),
    }
    response = flow.handle_message(**request)

    assert response["status"] == "answered"
    assert response["selected_product"]["sku"] == expected_sku
    assert "Thông tin chỉ mang tính tham khảo" in response["answer"]


def test_select_invalid_index_raises_validation_error() -> None:
    flow = make_flow()
    start = flow.handle_message("Hãy cho tôi biết thông tin về thuốc Oresol")

    with pytest.raises(FlowValidationError):
        flow.handle_message("10", conversation_id=start["conversation_id"], selected_index=10)


def test_follow_up_uses_selected_product_context() -> None:
    flow = make_flow()
    start = flow.handle_message("Hãy cho tôi biết thông tin về thuốc Oresol")
    selected = flow.handle_message(
        "1",
        conversation_id=start["conversation_id"],
        selected_index=1,
    )

    response = flow.handle_message(
        "sản phẩm trên có công dụng là gì",
        conversation_id=selected["conversation_id"],
    )

    assert response["status"] == "answered"
    assert response["selected_product"]["sku"] == "P00219"
    assert "điều trị mất nước" in response["answer"]


def test_expired_session_raises_not_found() -> None:
    now = {"value": 0.0}
    store = InMemorySessionStore(ttl_seconds=1, time_func=lambda: now["value"])
    flow = make_flow(store)
    start = flow.handle_message("Hãy cho tôi biết thông tin về thuốc Oresol")
    now["value"] = 2.0

    with pytest.raises(FlowNotFoundError):
        flow.handle_message("1", conversation_id=start["conversation_id"], selected_index=1)
