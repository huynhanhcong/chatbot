from __future__ import annotations

from Flow_code.dialogue_state import (
    InMemoryDialogueStateStore,
    displayed_items_from_sources,
    entity_from_pharmacity_response,
    entity_from_sources,
    update_state_after_turn,
)
from Flow_code.service_contracts import ActiveEntity


def test_dialogue_state_stores_selected_pharmacity_product() -> None:
    store = InMemoryDialogueStateStore()
    state = store.get_or_create("conv-1")
    entity = entity_from_pharmacity_response(
        {
            "selected_product": {
                "sku": "P00219",
                "name": "Thuoc bot Oresol 245 DHG",
                "detail_url": "https://www.pharmacity.vn/oresol-245.html",
            }
        }
    )

    saved = store.save(
        update_state_after_turn(
            state,
            domain="pharmacity",
            intent="drug_followup",
            active_entity=entity,
        )
    )

    assert saved.active_domain == "pharmacity"
    assert saved.active_entity
    assert saved.active_entity.entity_id == "P00219"
    assert saved.active_entity.source_url == "https://www.pharmacity.vn/oresol-245.html"
    assert saved.mentioned_entities[0].name == "Thuoc bot Oresol 245 DHG"


def test_dialogue_state_stores_hospital_source_entity() -> None:
    store = InMemoryDialogueStateStore()
    state = store.get_or_create("conv-1")
    entity = entity_from_sources(
        [{"id": "package_goi-ivf-standard", "title": "Goi IVF Standard", "url": "https://example.test"}]
    )

    saved = store.save(
        update_state_after_turn(
            state,
            domain="hospital",
            intent="package_search",
            active_entity=entity,
        )
    )

    assert saved.active_domain == "hospital"
    assert saved.active_entity
    assert saved.active_entity.entity_id == "package_goi-ivf-standard"


def test_dialogue_state_deduplicates_mentioned_entities() -> None:
    store = InMemoryDialogueStateStore()
    state = store.get_or_create("conv-1")
    entity = ActiveEntity(entity_type="package", entity_id="pkg-1", name="Goi A")

    state.mentioned_entities.extend([entity, entity])
    saved = store.save(state)

    assert saved.mentioned_entities == [entity]


def test_dialogue_state_stores_last_shown_hospital_items() -> None:
    store = InMemoryDialogueStateStore()
    state = store.get_or_create("conv-1")
    shown_items = displayed_items_from_sources(
        [
            {"id": "package-a", "title": "Goi kham thai", "url": "https://example.test/a"},
            {"id": "doctor-b", "title": "Bac si B", "url": "https://example.test/b"},
        ]
    )

    saved = store.save(
        update_state_after_turn(
            state,
            domain="hospital",
            intent="package_search",
            last_shown_items=shown_items,
        )
    )

    assert saved.last_shown_items[0].title == "Goi kham thai"
    assert saved.last_shown_items[0].entity_type == "package"
    assert saved.last_shown_items[1].entity_type == "doctor"
