from __future__ import annotations

import re

from .models import RagAnswer, SearchResult
from .text import clean_text


LEADING_BOILERPLATE_PATTERNS = [
    r"^(dua tren (?:context|thong tin|du lieu) [^:]*:\s*)",
    r"^(theo (?:context|thong tin|du lieu) [^:]*:\s*)",
    r"^(theo thong tin hien co,\s*)",
]


def build_sources(results: list[SearchResult]) -> list[dict[str, str | None]]:
    sources = []
    seen = set()
    for result in results:
        key = (result.id, result.source_url)
        if key in seen:
            continue
        seen.add(key)
        sources.append(
            {
                "id": result.id,
                "title": result.title,
                "url": result.source_url,
            }
        )
    return sources


def compose_answer_text(answer: str, *, max_paragraphs: int = 2) -> str:
    blocks = [block.strip() for block in str(answer or "").replace("\r\n", "\n").split("\n\n") if block.strip()]
    if not blocks:
        return ""

    composed: list[str] = []
    for block in blocks:
        block = clean_text(block)
        for pattern in LEADING_BOILERPLATE_PATTERNS:
            block = re.sub(pattern, "", block, flags=re.IGNORECASE)
        if block:
            composed.append(block)

    if not composed:
        return ""

    if not any(part.lstrip().startswith(("-", "*", "•")) for part in composed):
        composed = composed[:max_paragraphs]

    return "\n\n".join(composed).strip()


def render_answer(answer: RagAnswer) -> str:
    lines = [answer.answer.strip()]
    if answer.sources:
        lines.append("\nNguồn:")
        for index, source in enumerate(answer.sources, start=1):
            url = source.get("url") or ""
            title = source.get("title") or source.get("id") or "Nguồn"
            lines.append(f"{index}. {title}: {url}")
    lines.append(f"\nIntent: {answer.intent} | Confidence: {answer.confidence}")
    return "\n".join(lines)
