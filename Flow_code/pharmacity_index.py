from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import asdict
from pathlib import Path
from threading import RLock
from typing import Any

from .drug_extraction import normalize_search_text
from .models import ProductDetail, ProductOption

try:
    from rapidfuzz import fuzz
except ImportError:  # pragma: no cover
    fuzz = None


class PharmacityProductIndex:
    def __init__(self, path: Path, *, time_func: Any | None = None) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._time_func = time_func or time.time
        self._lock = RLock()
        self._conn = sqlite3.connect(str(path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._fts_available = True
        self._init_schema()

    def search_cached(
        self,
        keyword: str,
        *,
        max_options: int,
        ttl_seconds: int,
    ) -> list[ProductOption]:
        normalized = normalize_search_text(keyword)
        if not normalized:
            return []
        with self._lock:
            row = self._conn.execute(
                "SELECT skus_json, cached_at FROM search_cache WHERE keyword = ?",
                (normalized,),
            ).fetchone()
            if row is None:
                return []
            if ttl_seconds > 0 and float(row["cached_at"]) + ttl_seconds < self._time_func():
                return []
            skus = json.loads(row["skus_json"])
            options = [self._get_option_locked(str(sku)) for sku in skus]
        ranked = rank_product_options(keyword, [option for option in options if option])
        return ranked[:max_options]

    def search_local(self, keyword: str, *, max_options: int) -> list[ProductOption]:
        normalized = normalize_search_text(keyword)
        if not normalized:
            return []
        with self._lock:
            rows = self._local_rows_locked(normalized)
            if not rows:
                rows = self._conn.execute("SELECT option_json, image_url FROM products").fetchall()
            options = [self._option_from_row(row) for row in rows]
        ranked = rank_product_options(keyword, [option for option in options if option])
        return ranked[:max_options]

    def save_search(self, keyword: str, options: list[ProductOption]) -> None:
        normalized = normalize_search_text(keyword)
        if not normalized:
            return
        now = self._time_func()
        with self._lock:
            for option in options:
                self._upsert_option_locked(option, now)
            self._conn.execute(
                """
                INSERT INTO search_cache(keyword, skus_json, cached_at)
                VALUES (?, ?, ?)
                ON CONFLICT(keyword) DO UPDATE SET
                    skus_json = excluded.skus_json,
                    cached_at = excluded.cached_at
                """,
                (normalized, json.dumps([option.sku for option in options]), now),
            )
            self._conn.commit()

    def get_detail(self, sku: str) -> ProductDetail | None:
        sku = sku.strip().upper()
        if not sku:
            return None
        with self._lock:
            row = self._conn.execute(
                "SELECT detail_json FROM products WHERE sku = ?",
                (sku,),
            ).fetchone()
        if row is None or not row["detail_json"]:
            return None
        try:
            return ProductDetail(**json.loads(row["detail_json"]))
        except (TypeError, json.JSONDecodeError):
            return None

    def save_detail(self, detail: ProductDetail, image_url: str | None = None) -> None:
        if not detail.sku:
            return
        sku = detail.sku.strip().upper()
        now = self._time_func()
        with self._lock:
            option = ProductOption(
                index=1,
                sku=sku,
                slug=detail.slug,
                name=detail.name,
                brand=detail.brand,
                price=detail_price(detail),
                detail_url=detail.source_url,
                image_url=image_url or self._get_image_url_locked(sku),
                ingredients=detail.ingredients,
                is_prescription_drug=detail.is_prescription_drug,
            )
            self._upsert_option_locked(option, now)
            self._conn.execute(
                """
                UPDATE products
                SET detail_json = ?, detail_updated_at = ?, updated_at = ?
                WHERE sku = ?
                """,
                (json.dumps(asdict(detail), ensure_ascii=False), now, now, sku),
            )
            self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def _init_schema(self) -> None:
        with self._lock:
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS products (
                    sku TEXT PRIMARY KEY,
                    slug TEXT,
                    name TEXT NOT NULL,
                    brand TEXT,
                    price TEXT,
                    detail_url TEXT,
                    image_url TEXT,
                    ingredients_json TEXT NOT NULL DEFAULT '[]',
                    is_prescription_drug INTEGER NOT NULL DEFAULT 0,
                    option_json TEXT NOT NULL,
                    detail_json TEXT,
                    updated_at REAL NOT NULL,
                    detail_updated_at REAL
                )
                """
            )
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS search_cache (
                    keyword TEXT PRIMARY KEY,
                    skus_json TEXT NOT NULL,
                    cached_at REAL NOT NULL
                )
                """
            )
            try:
                self._conn.execute(
                    "CREATE VIRTUAL TABLE IF NOT EXISTS product_fts USING fts5(sku, name, brand, ingredients)"
                )
            except sqlite3.OperationalError:
                self._fts_available = False
            self._conn.commit()

    def _local_rows_locked(self, normalized: str) -> list[sqlite3.Row]:
        if self._fts_available:
            fts_query = " OR ".join(token for token in normalized.split() if len(token) >= 2)
            if fts_query:
                try:
                    rows = self._conn.execute(
                        """
                        SELECT p.option_json, p.image_url
                        FROM product_fts f
                        JOIN products p ON p.sku = f.sku
                        WHERE product_fts MATCH ?
                        LIMIT 50
                        """,
                        (fts_query,),
                    ).fetchall()
                    if rows:
                        return rows
                except sqlite3.OperationalError:
                    pass

        like_value = f"%{normalized}%"
        return self._conn.execute(
            """
            SELECT option_json, image_url
            FROM products
            WHERE lower(name) LIKE ? OR lower(brand) LIKE ? OR lower(ingredients_json) LIKE ?
            LIMIT 50
            """,
            (like_value, like_value, like_value),
        ).fetchall()

    def _upsert_option_locked(self, option: ProductOption, now: float) -> None:
        sku = option.sku.strip().upper()
        option_data = asdict(option)
        option_data["sku"] = sku
        option_data["image_url"] = option.image_url or self._get_image_url_locked(sku)
        ingredients_json = json.dumps(option.ingredients, ensure_ascii=False)
        self._conn.execute(
            """
            INSERT INTO products(
                sku, slug, name, brand, price, detail_url, image_url, ingredients_json,
                is_prescription_drug, option_json, updated_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(sku) DO UPDATE SET
                slug = excluded.slug,
                name = excluded.name,
                brand = excluded.brand,
                price = excluded.price,
                detail_url = excluded.detail_url,
                image_url = excluded.image_url,
                ingredients_json = excluded.ingredients_json,
                is_prescription_drug = excluded.is_prescription_drug,
                option_json = excluded.option_json,
                updated_at = excluded.updated_at
            """,
            (
                sku,
                option.slug,
                option.name,
                option.brand,
                option.price,
                option.detail_url,
                option_data["image_url"],
                ingredients_json,
                1 if option.is_prescription_drug else 0,
                json.dumps(option_data, ensure_ascii=False),
                now,
            ),
        )
        if self._fts_available:
            self._conn.execute("DELETE FROM product_fts WHERE sku = ?", (sku,))
            self._conn.execute(
                "INSERT INTO product_fts(sku, name, brand, ingredients) VALUES (?, ?, ?, ?)",
                (sku, option.name, option.brand or "", " ".join(option.ingredients)),
            )

    def _get_option_locked(self, sku: str) -> ProductOption | None:
        row = self._conn.execute(
            "SELECT option_json, image_url FROM products WHERE sku = ?",
            (sku.strip().upper(),),
        ).fetchone()
        return self._option_from_row(row) if row else None

    @staticmethod
    def _option_from_row(row: sqlite3.Row | None) -> ProductOption | None:
        if row is None:
            return None
        try:
            option_data = json.loads(row["option_json"])
            if isinstance(option_data, dict) and not option_data.get("image_url"):
                stored_image_url = _row_value(row, "image_url")
                if stored_image_url:
                    option_data["image_url"] = stored_image_url
            return ProductOption(**option_data)
        except (TypeError, json.JSONDecodeError):
            return None

    def _get_image_url_locked(self, sku: str) -> str | None:
        row = self._conn.execute(
            "SELECT image_url, option_json FROM products WHERE sku = ?",
            (sku.strip().upper(),),
        ).fetchone()
        if row is None:
            return None
        stored_image_url = _row_value(row, "image_url")
        if stored_image_url:
            return stored_image_url
        try:
            option_data = json.loads(row["option_json"])
        except (TypeError, json.JSONDecodeError):
            return None
        if not isinstance(option_data, dict):
            return None
        image_url = option_data.get("image_url")
        if image_url is None:
            return None
        return str(image_url).strip() or None


def rank_product_options(keyword: str, options: list[ProductOption]) -> list[ProductOption]:
    normalized_keyword = normalize_search_text(keyword)

    def score(option: ProductOption) -> float:
        name = normalize_search_text(option.name)
        brand = normalize_search_text(option.brand or "")
        ingredients = normalize_search_text(" ".join(option.ingredients))
        value = 0.0
        if normalized_keyword and normalized_keyword in name:
            value += 45.0
        if normalized_keyword and normalized_keyword in ingredients:
            value += 20.0
        if normalized_keyword and normalized_keyword in brand:
            value += 10.0
        value += max(_ratio(normalized_keyword, name), _ratio(normalized_keyword, ingredients)) * 0.45
        value += max(0, 18 - option.index)
        if option.price:
            value += 2.0
        if option.detail_url:
            value += 1.0
        if option.is_prescription_drug:
            value -= 1.0
        return value

    ranked = sorted(options, key=score, reverse=True)
    return [
        ProductOption(
            index=index,
            sku=option.sku,
            slug=option.slug,
            name=option.name,
            brand=option.brand,
            price=option.price,
            detail_url=option.detail_url,
            image_url=option.image_url,
            ingredients=option.ingredients,
            is_prescription_drug=option.is_prescription_drug,
        )
        for index, option in enumerate(ranked, start=1)
    ]


def detail_price(detail: ProductDetail) -> str | None:
    from .models import format_variants_price

    return format_variants_price(detail.variants)


def _ratio(left: str, right: str) -> float:
    if not left or not right:
        return 0.0
    if fuzz is not None:
        return float(fuzz.token_set_ratio(left, right))
    left_tokens = set(left.split())
    right_tokens = set(right.split())
    if not left_tokens or not right_tokens:
        return 0.0
    return len(left_tokens & right_tokens) / len(left_tokens | right_tokens) * 100


def _row_value(row: sqlite3.Row, key: str) -> Any | None:
    return row[key] if key in row.keys() else None
