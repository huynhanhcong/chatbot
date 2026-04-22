from __future__ import annotations

from pathlib import Path
from typing import Any

from .drug_extraction import normalize_search_text
from .models import ProductDetail, ProductOption
from .pharmacity_client import PharmacityApiClient
from .pharmacity_index import PharmacityProductIndex, rank_product_options

try:
    from rapidfuzz import fuzz
except ImportError:  # pragma: no cover
    fuzz = None


class CachedPharmacityClient:
    def __init__(
        self,
        *,
        api_client: Any | None = None,
        index: PharmacityProductIndex | None = None,
        index_path: Path | None = None,
        cache_ttl_seconds: int = 900,
    ) -> None:
        self.api_client = api_client or PharmacityApiClient(timeout_seconds=6.0)
        self.index = index or PharmacityProductIndex(
            index_path or Path("Data_RAG/index/pharmacity_products.sqlite")
        )
        self.cache_ttl_seconds = cache_ttl_seconds

    def search_products(self, keyword: str, max_options: int = 5) -> list[ProductOption]:
        cached = self.index.search_cached(
            keyword,
            max_options=max_options,
            ttl_seconds=self.cache_ttl_seconds,
        )
        if cached and not _has_missing_images(cached):
            return cached
        fallback = cached

        local = self.index.search_local(keyword, max_options=max_options)
        local_is_relevant = bool(local) and _best_match_score(keyword, local) >= 82
        if local_is_relevant and not _has_missing_images(local):
            return local
        if local_is_relevant and not fallback:
            fallback = local

        try:
            options = self.api_client.search_products(keyword, max_options=max_options)
        except Exception:
            if fallback:
                return fallback[:max_options]
            raise
        ranked = rank_product_options(keyword, options)
        self.index.save_search(keyword, ranked)
        return ranked[:max_options]

    def fetch_product_detail(self, product: ProductOption) -> ProductDetail:
        cached = self.index.get_detail(product.sku)
        if cached is not None:
            return cached
        detail = self.api_client.fetch_product_detail(product)
        self.index.save_detail(detail, image_url=product.image_url)
        return detail

    def close(self) -> None:
        if hasattr(self.api_client, "close"):
            self.api_client.close()
        self.index.close()


def _best_match_score(keyword: str, options: list[ProductOption]) -> float:
    query = normalize_search_text(keyword)
    best = 0.0
    for option in options:
        target = normalize_search_text(" ".join([option.name, option.brand or "", *option.ingredients]))
        if query and query in target:
            best = max(best, 100.0)
        elif fuzz is not None:
            best = max(best, float(fuzz.token_set_ratio(query, target)))
        else:
            query_tokens = set(query.split())
            target_tokens = set(target.split())
            if query_tokens and target_tokens:
                best = max(best, len(query_tokens & target_tokens) / len(query_tokens | target_tokens) * 100)
    return best


def _has_missing_images(options: list[ProductOption]) -> bool:
    return any(not option.image_url for option in options)
