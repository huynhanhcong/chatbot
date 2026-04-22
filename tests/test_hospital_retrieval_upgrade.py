from __future__ import annotations

import json

from RAG_app.entity_linker import HanhPhucEntityLinker
from RAG_app.models import SearchResult
from RAG_app.pipeline import _needs_llm_rewrite
from RAG_app.retriever import diversify_results, rrf_fuse


def _result(
    item_id: str,
    *,
    parent_id: str | None = None,
    score: float = 0.0,
    vector_score: float = 0.0,
    bm25_score: float = 0.0,
    title: str = "Goi IVF Standard",
    text: str = "goi ivf standard gom kham va tu van",
) -> SearchResult:
    return SearchResult(
        id=item_id,
        parent_id=parent_id,
        score=score,
        vector_score=vector_score,
        bm25_score=bm25_score,
        title=title,
        entity_type="package",
        category="IVF",
        source_url="https://example.test",
        text=text,
        payload={"source": "hanhphuc", "entity_id": item_id, "parent_id": parent_id or item_id},
    )


def test_rrf_fusion_rewards_items_seen_by_bm25_and_vector() -> None:
    vector = [
        _result("vector-only", vector_score=0.99),
        _result("shared", vector_score=0.40),
    ]
    bm25 = [
        _result("shared", bm25_score=0.50),
        _result("bm25-only", bm25_score=1.0),
    ]

    fused = rrf_fuse(vector_results=vector, bm25_results=bm25, candidate_limit=3)

    assert fused[0].id == "shared"
    assert fused[0].vector_score == 0.40
    assert fused[0].bm25_score == 0.50


def test_entity_linker_matches_package_alias_and_price_intent(tmp_path) -> None:
    path = tmp_path / "entities.jsonl"
    row = {
        "entity_id": "package_goi-ivf-standard",
        "entity_type": "package",
        "canonical_name": "Goi IVF Standard",
        "aliases": ["IVF Standard", "goi thu tinh trong ong nghiem standard"],
        "category": "Ho tro sinh san",
        "source_url": "https://example.test",
        "search_text": "Goi IVF Standard",
    }
    path.write_text(json.dumps(row, ensure_ascii=False) + "\n", encoding="utf-8")

    links = HanhPhucEntityLinker.from_jsonl(path).link("Goi IVF Standard bao nhieu tien?")

    assert links
    assert links[0].entity_id == "package_goi-ivf-standard"
    assert links[0].price_intent is True
    assert links[0].confidence >= 0.9


def test_mmr_limits_repeated_parent_chunks() -> None:
    candidates = [
        _result(f"p1-chunk-{index}", parent_id="parent-1", score=1.0 - index * 0.01)
        for index in range(4)
    ]
    candidates.append(_result("p2-chunk-1", parent_id="parent-2", score=0.8, text="chi phi goi kham"))

    selected = diversify_results("goi ivf gom gi", candidates, top_k=4, max_per_parent=2)

    assert sum(1 for item in selected if item.parent_id == "parent-1") == 2
    assert any(item.parent_id == "parent-2" for item in selected)


def test_conditional_rewrite_skips_independent_questions() -> None:
    assert _needs_llm_rewrite("Goi IVF Standard gom gi?", "") is False
    assert _needs_llm_rewrite("Goi nay bao nhieu tien?", "User: Goi IVF Standard gom gi?") is True
