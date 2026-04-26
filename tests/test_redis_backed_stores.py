from __future__ import annotations

from Flow_code.conversation_memory import RedisConversationStore
from Flow_code.dialogue_state import RedisDialogueStateStore
from Flow_code.hospital_session import RedisHospitalSessionStore
from Flow_code.models import ProductDetail, ProductOption
from Flow_code.service_contracts import ActiveEntity
from Flow_code.session_store import RedisSessionStore


class FakeRedis:
    def __init__(self) -> None:
        self.data: dict[str, str] = {}
        self.expirations: dict[str, int | None] = {}

    def get(self, key: str):
        return self.data.get(key)

    def set(self, key: str, value: str, ex: int | None = None) -> None:
        self.data[key] = value
        self.expirations[key] = ex

    def delete(self, key: str) -> None:
        self.data.pop(key, None)
        self.expirations.pop(key, None)


def test_redis_conversation_store_persists_turns_and_summary() -> None:
    client = FakeRedis()
    store = RedisConversationStore(
        "redis://test",
        ttl_seconds=90,
        max_recent_turns=2,
        max_summary_chars=1000,
        client=client,
    )

    session = store.get_or_create("conv-1")
    store.save_turn(
        conversation_id=session.conversation_id,
        route="pharmacity",
        user_message="Hoi gia Panadol",
        assistant_message="Panadol co gia 84.000 VND/Hop.",
        metadata={"selected_product": {"sku": "P001", "name": "Panadol Extra"}},
    )
    store.save_turn(
        conversation_id=session.conversation_id,
        route="hospital_rag",
        user_message="Goi IVF Standard gom gi?",
        assistant_message="Goi gom tu van va xet nghiem.",
        sources=[{"title": "Goi IVF Standard", "url": "https://example.test"}],
    )
    store.save_turn(
        conversation_id=session.conversation_id,
        route="hospital_rag",
        user_message="Goi nay bao nhieu tien?",
        assistant_message="Chi phi can lien he benh vien de xac nhan.",
        sources=[{"title": "Goi IVF Standard", "url": "https://example.test"}],
    )

    reloaded = RedisConversationStore("redis://test", ttl_seconds=90, client=client)
    loaded = reloaded.get("conv-1")

    assert loaded is not None
    assert loaded.active_route == "hospital_rag"
    assert len(loaded.turns) == 2
    assert "Hoi gia Panadol" in loaded.summary
    assert loaded.turns[-1].sources[0]["title"] == "Goi IVF Standard"
    assert client.expirations["chatbot:conversation:conv-1"] == 90


def test_redis_hospital_session_store_persists_turns_across_instances() -> None:
    client = FakeRedis()
    store = RedisHospitalSessionStore(
        "redis://test",
        ttl_seconds=120,
        max_turns=1,
        max_summary_chars=1000,
        client=client,
    )

    session = store.get_or_create("conv-2")
    store.save_turn(
        conversation_id=session.conversation_id,
        question="Goi IVF Standard gom gi?",
        standalone_question="Goi IVF Standard gom gi?",
        answer="Goi gom tu van va xet nghiem.",
        sources=[{"title": "Goi IVF Standard", "url": "https://example.test"}],
    )
    store.save_turn(
        conversation_id=session.conversation_id,
        question="Goi nay bao nhieu tien?",
        standalone_question="Goi IVF Standard bao nhieu tien?",
        answer="Can lien he benh vien de xac nhan.",
        sources=[{"title": "Goi IVF Standard", "url": "https://example.test"}],
    )

    reloaded = RedisHospitalSessionStore("redis://test", ttl_seconds=120, client=client)
    loaded = reloaded.get("conv-2")

    assert loaded is not None
    assert len(loaded.turns) == 1
    assert "Goi IVF Standard gom gi?" in loaded.summary
    assert loaded.turns[0].standalone_question == "Goi IVF Standard bao nhieu tien?"
    assert client.expirations["chatbot:hospital:conv-2"] == 120


def test_redis_drug_session_store_persists_selected_detail_and_history() -> None:
    client = FakeRedis()
    store = RedisSessionStore(
        "redis://test",
        ttl_seconds=150,
        max_turns=2,
        max_summary_chars=1000,
        client=client,
    )

    option = ProductOption(
        index=1,
        sku="P00219",
        slug="panadol-extra",
        name="Panadol Extra",
        brand="GSK",
        price="84.000 VND/Hop",
        detail_url="https://example.test/panadol-extra",
    )
    detail = ProductDetail(
        sku="P00219",
        slug="panadol-extra",
        name="Panadol Extra",
        brand="GSK",
        long_description="Dung giam dau, ha sot.",
        variants=[{"price": 84000, "unit_name": "Hop"}],
        source_url="https://example.test/panadol-extra",
    )

    session = store.save_search(
        "Panadol",
        [option],
        question="Cho toi xin gia cua thuoc Panadol",
        conversation_id="conv-3",
    )
    store.save_selected_detail(
        conversation_id=session.conversation_id,
        selected_detail=detail,
        last_answer='Sản phẩm "Panadol Extra" có giá là: 84.000 VND/Hop',
        question="Cho toi xin gia cua thuoc Panadol",
    )

    reloaded = RedisSessionStore("redis://test", ttl_seconds=150, client=client)
    loaded = reloaded.get("conv-3")

    assert loaded is not None
    assert loaded.requested_question == "Cho toi xin gia cua thuoc Panadol"
    assert loaded.selected_detail is not None
    assert loaded.selected_detail.name == "Panadol Extra"
    assert loaded.turns[0].answer == 'Sản phẩm "Panadol Extra" có giá là: 84.000 VND/Hop'
    assert client.expirations["chatbot:drug:conv-3"] == 150


def test_redis_dialogue_state_store_persists_and_deduplicates_entities() -> None:
    client = FakeRedis()
    store = RedisDialogueStateStore(
        "redis://test",
        ttl_seconds=180,
        max_entities=2,
        client=client,
    )

    entity = ActiveEntity(
        entity_type="product",
        entity_id="P00219",
        name="Panadol Extra",
        source_url="https://example.test/panadol-extra",
    )
    state = store.get_or_create("conv-4")
    state.active_domain = "pharmacity"
    state.last_intent = "drug_followup"
    state.active_entity = entity
    state.mentioned_entities = [entity, entity]
    store.save(state)

    reloaded = RedisDialogueStateStore("redis://test", ttl_seconds=180, client=client)
    loaded = reloaded.get_or_create("conv-4")

    assert loaded.active_domain == "pharmacity"
    assert loaded.active_entity is not None
    assert loaded.active_entity.name == "Panadol Extra"
    assert loaded.mentioned_entities == [entity]
    assert client.expirations["chatbot:state:conv-4"] == 180
