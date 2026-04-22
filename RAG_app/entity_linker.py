from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .data_loader import read_jsonl
from .text import clean_text, flatten_list

try:
    from rapidfuzz import fuzz, process
except ImportError:  # pragma: no cover - exercised only without optional dependency
    fuzz = None
    process = None


@dataclass(frozen=True)
class EntityLink:
    entity_id: str
    entity_type: str
    title: str
    category: str | None
    confidence: float
    matched_text: str
    match_type: str
    price_intent: bool = False


@dataclass(frozen=True)
class EntityRecord:
    entity_id: str
    entity_type: str
    title: str
    category: str | None
    aliases: tuple[str, ...]
    normalized_aliases: tuple[str, ...]


class HanhPhucEntityLinker:
    def __init__(self, records: list[EntityRecord]) -> None:
        self.records = records
        self._alias_to_records: dict[str, list[EntityRecord]] = {}
        self._alias_choices: list[str] = []
        for record in records:
            for alias in record.normalized_aliases:
                if len(alias) < 3:
                    continue
                self._alias_to_records.setdefault(alias, []).append(record)
        self._alias_choices = list(self._alias_to_records)

    @classmethod
    def from_jsonl(cls, path: Path) -> "HanhPhucEntityLinker":
        records: list[EntityRecord] = []
        for row in read_jsonl(path):
            title = clean_text(row.get("canonical_name"))
            aliases = [
                title,
                *flatten_list(row.get("aliases")),
                clean_text(row.get("department")),
                clean_text(row.get("category")),
            ]
            aliases.extend(flatten_list(row.get("subspecialties")))
            aliases.extend(flatten_list(row.get("service_lines")))
            unique_aliases = tuple(dict.fromkeys(item for item in aliases if item))
            normalized_aliases = tuple(
                dict.fromkeys(normalize_for_linking(item) for item in unique_aliases if item)
            )
            records.append(
                EntityRecord(
                    entity_id=str(row["entity_id"]),
                    entity_type=str(row["entity_type"]),
                    title=title,
                    category=clean_text(row.get("department") or row.get("category")) or None,
                    aliases=unique_aliases,
                    normalized_aliases=normalized_aliases,
                )
            )
        return cls(records)

    def link(self, query: str, *, limit: int = 3) -> list[EntityLink]:
        normalized_query = normalize_for_linking(query)
        if not normalized_query:
            return []

        price_intent = has_price_intent(normalized_query)
        matches: dict[str, EntityLink] = {}

        for alias, records in self._alias_to_records.items():
            if alias and alias in normalized_query:
                confidence = 1.0 if len(alias) >= 8 else 0.88
                for record in records:
                    self._keep_best(
                        matches,
                        record,
                        confidence=confidence,
                        matched_text=alias,
                        match_type="exact",
                        price_intent=price_intent,
                    )

        if len(matches) < limit and self._alias_choices:
            for alias, score, _ in self._fuzzy_extract(normalized_query, limit=limit * 4):
                if score < 82:
                    continue
                for record in self._alias_to_records.get(alias, []):
                    self._keep_best(
                        matches,
                        record,
                        confidence=float(score) / 100,
                        matched_text=alias,
                        match_type="fuzzy",
                        price_intent=price_intent,
                    )

        links = sorted(matches.values(), key=lambda item: item.confidence, reverse=True)
        return links[:limit]

    def _fuzzy_extract(self, query: str, *, limit: int) -> list[tuple[str, float, int]]:
        if process is not None and fuzz is not None:
            return [
                (str(alias), float(score), int(index))
                for alias, score, index in process.extract(
                    query,
                    self._alias_choices,
                    scorer=fuzz.token_set_ratio,
                    limit=limit,
                )
            ]

        scored: list[tuple[str, float, int]] = []
        query_tokens = set(query.split())
        for index, alias in enumerate(self._alias_choices):
            alias_tokens = set(alias.split())
            if not alias_tokens:
                continue
            overlap = len(query_tokens & alias_tokens) / len(alias_tokens)
            scored.append((alias, overlap * 100, index))
        return sorted(scored, key=lambda item: item[1], reverse=True)[:limit]

    @staticmethod
    def _keep_best(
        matches: dict[str, EntityLink],
        record: EntityRecord,
        *,
        confidence: float,
        matched_text: str,
        match_type: str,
        price_intent: bool,
    ) -> None:
        current = matches.get(record.entity_id)
        if current and current.confidence >= confidence:
            return
        matches[record.entity_id] = EntityLink(
            entity_id=record.entity_id,
            entity_type=record.entity_type,
            title=record.title,
            category=record.category,
            confidence=confidence,
            matched_text=matched_text,
            match_type=match_type,
            price_intent=price_intent,
        )


def has_price_intent(normalized_query: str) -> bool:
    return any(
        keyword in normalized_query
        for keyword in ["gia", "chi phi", "bao nhieu tien", "bao nhieu", "phi"]
    )


def normalize_for_linking(value: Any) -> str:
    text = clean_text(value).lower()
    repaired = _repair_mojibake(text)
    if repaired and repaired != text:
        text = f"{text} {repaired}"
    text = unicodedata.normalize("NFD", text)
    text = "".join(char for char in text if unicodedata.category(char) != "Mn")
    text = text.replace("đ", "d").replace("Đ", "d")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _repair_mojibake(value: str) -> str:
    try:
        return value.encode("cp1252").decode("utf-8")
    except UnicodeError:
        return ""
